"""
UNISON: Unified Framework for Learning from Scored Bags

This module implements the complete UNISON framework for learning from scored bags,
as described in the paper "UNISON: A Unified Framework for Learning from Scored Bags".

Framework Overview:
    UNISON bridges multiple instance learning and recommendation systems by treating
    both as instances of learning from scored bag collections of (item, score) pairs.
    The framework decomposes learning into three interconnected stages:

    Stage 1 (Item Embedding): Learn universal item representations
        f: X → R^d

    Stage 2 (Preference Modeling): Model bag-specific preference patterns
        z_tilde = phi(z_i)                    # Shared MLP projection
        score = <theta_j, z_tilde> + b_j      # Linear scoring per bag

    Stage 3 (Bag Classification): Infer bag-level characteristics
        y_j = psi([theta_j; b_j])             # Functional embeddings as features

Key Design Principles:
    - Linear scoring suffices when embeddings are discriminative (Theorem 1)
    - Inference-time optimization for cold-start scenarios (Theorem 2)
    - Functional embeddings eliminate need for auxiliary features
    - Domain-agnostic: works across entertainment, immunology, ecology

Supported Scenarios:
    - Warm Bag, Warm Items (WB-WI): Standard training scenario
    - Warm Bag, Cold Items (WB-CI): New items for known bags
    - Cold Bag, Warm Items (CB-WI): New bags with known item universe
    - Cold Bag, Cold Items (CB-CI): Fully cold-start scenario

Task Types:
    - Item Scoring: Regression (continuous scores) or Binary Classification
    - Bag Classification: Regression, Binary, or Multiclass

Typical Usage:
    # Initialize framework with item embeddings from Stage 1
    model = UNISON(
        train_loader=train_loader,
        d_in=128,              # Item embedding dimension (from Stage 1)
        d_model=64,            # Latent preference space dimension
        mlp_hidden=(256, 128), # Shared MLP architecture
        item_task="regression",
        is_user_attr_head=True,
        user_attr_task="classification"
    )

    # Train Stage 2 (preference modeling) on warm bags
    for batch in train_loader:
        result = model.train_step_warm_user_warm_item(
            optimizer, batch['user_id'], batch['items'], batch['scores']
        )

    # Cold-start inference: fit linear parameters on support set
    result = model.predict_cold_user_items(
        items_sup=support_items,
        scores_sup=support_scores,
        items_qry=query_items
    )

    # Stage 3: Predict bag-level attributes from functional embeddings
    attr_pred = model.predict_cold_user_attribute(
        items_sup=support_items,
        scores_sup=support_scores
    )

Implementation Notes:
    - "user" terminology in method names reflects recommendation domain convention
      but applies generally to any "bag" (e.g., medical patient, ecological sample)
    - Normalization to [-1, 1] for regression ensures stable optimization
    - Class imbalance handling for binary classification via negative downsampling
    - Early stopping prevents overfitting during cold-start adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict, Any, Literal, List, Union, Sequence, Tuple

from src.utils.metrics import regression_metrics, binary_classification_metrics, multiclass_classification_metrics
from src.model.stage2.user_scorer import UserScorer
from src.model.stage2.cold_start_adapter import fit_cold_start_batch
from src.model.stage3.heads import UserAttrHead

from torch.utils.data import DataLoader


class UNISON(nn.Module):
    """
    UNISON: Unified framework for learning from scored bags.

    This class integrates all three stages of the UNISON framework:
    - Stage 1: Item embedding (external, provided via d_in parameter)
    - Stage 2: Preference modeling via UserScorer
    - Stage 3: Bag classification via UserAttrHead (optional)

    The framework supports both warm (training) and cold (inference) scenarios,
    per-batch adaptation for cold bags, and label normalization for stable training.

    Parameters
    ----------
    Data Source (choose one):
        train_loader : DataLoader, optional
            Training DataLoader from which bag IDs are extracted automatically
        user_id_mapping : Dict[Any, int], optional
            Pre-computed mapping from external bag IDs to internal indices [0, num_bags)
        user_id_key : str, default='user_id'
            Key in batch dictionary containing bag IDs (used with train_loader)

    Stage 2 - UserScorer Configuration:
        d_in : int, required
            Dimensionality of item features from Stage 1 encoder
        d_model : int, required
            Dimensionality of shared latent space (z_tilde and theta_j)
        mlp_hidden : Sequence[int], default=(128,)
            Hidden layer sizes for shared MLP phi
        mlp_dropout : float, default=0.0
            Dropout probability in shared MLP
        mlp_activation : str, default="relu"
            Activation function: "relu", "gelu", "silu", "tanh"
        use_bias : bool, default=True
            Learn per-bag bias terms b_j
        mlp_use_batchnorm : bool, default=False
            Apply BatchNorm in shared MLP
        mlp_use_layernorm : bool, default=False
            Apply LayerNorm in shared MLP
        normalize_user : bool, default=True
            L2-normalize bag embeddings theta_j for stability
        normalize_item : bool, default=False
            L2-normalize projected items z_tilde
        init_std : float, default=0.02
            Standard deviation for bag embedding initialization
        dtype : torch.dtype, default=torch.float32
            Data type for parameters
        device : torch.device or str, optional
            Device placement ("cuda" or "cpu")

    Item Scoring Task Configuration:
        item_task : {"regression", "binary_classification"}, default="regression"
            Type of item-level prediction task
        binarization_threshold : float, optional
            Threshold for converting continuous scores to binary labels.
            If set, scores >= threshold → 1, else → 0

    Stage 3 - UserAttrHead Configuration:
        is_user_attr_head : bool, default=False
            Whether to create Stage 3 classifier for bag-level attributes
        user_head_out_dim : int, default=1
            Output dimension: 1 for binary/scalar, num_classes for multiclass
        user_head_hidden : Sequence[int], default=(128, 64)
            Hidden layer sizes for attribute classifier
        user_head_dropout : float or Sequence[float], default=0.0
            Dropout probabilities (uniform or per-layer)
        user_head_activation : str, default="relu"
            Activation function for attribute head
        user_head_use_batchnorm : bool, default=False
            Apply BatchNorm in attribute head
        user_head_use_layernorm : bool, default=False
            Apply LayerNorm in attribute head

    Cold-Start Configuration:
        cold_steps : int, default=20
            Maximum optimization iterations for cold-start adaptation
        cold_lr : float, default=1e-2
            Learning rate for cold-start optimization
        cold_wd : float, default=0.0
            Weight decay for cold-start optimization
        cold_patience : int, default=10
            Early stopping patience for cold-start

    Normalization Configuration:
        label_scale : {"minmax"}, default="minmax"
            Score normalization method (currently only minmax to [-1, 1])
        user_attr_task : {"regression", "classification"}, default="regression"
            Task type for Stage 3 bag-level prediction
        scores_range : Tuple[float, float], optional
            (min, max) score range for normalization, e.g., (1.0, 5.0)

    Attributes
    ----------
    user_scorer : UserScorer
        Stage 2 preference modeling module
    user_attr_head : UserAttrHead or None
        Stage 3 bag classification module (if enabled)
    user_id_mapping : Dict[Any, int]
        Mapping from external bag IDs to internal indices
    reverse_mapping : Dict[int, Any]
        Inverse mapping from internal indices to external IDs
    y_min, y_max : float or None
        Score range bounds for normalization

    Raises
    ------
    ValueError
        If required parameters (d_in, d_model) are missing
        If neither train_loader nor user_id_mapping is provided
        If scores_range is invalid

    Notes
    -----
    The term "user" in method names is a convention from the recommendation domain
    but applies generally to any "bag" concept (e.g., medical patient, biological sample).

    Examples
    --------
    Initialize for recommendation task:
        >>> model = UNISON(
        ...     train_loader=train_loader,
        ...     d_in=768,
        ...     d_model=128,
        ...     mlp_hidden=(512, 256),
        ...     item_task="regression",
        ...     scores_range=(1.0, 5.0),
        ...     is_user_attr_head=True,
        ...     user_attr_task="classification"
        ... )

    Initialize for immunology task:
        >>> model = UNISON(
        ...     train_loader=tcr_loader,
        ...     d_in=75,
        ...     d_model=64,
        ...     item_task="binary_classification",
        ...     binarization_threshold=1e-4,
        ...     is_user_attr_head=True
        ... )
    """

    def __init__(
            self,
            # ===== Data Source (choose one) =====
            train_loader: Optional[DataLoader] = None,
            user_id_mapping: Optional[Dict[Any, int]] = None,
            user_id_key: str = 'user_id',

            # ===== UserScorer parameters (Stage 2) =====
            d_in: int = None,
            d_model: int = None,
            mlp_hidden: Sequence[int] = (128,),
            mlp_dropout: float = 0.0,
            mlp_activation: str = "relu",
            use_bias: bool = True,
            mlp_use_batchnorm: bool = False,
            mlp_use_layernorm: bool = False,
            normalize_user: bool = True,
            normalize_item: bool = False,
            init_std: float = 0.02,
            dtype: torch.dtype = torch.float32,
            device: Optional[Union[torch.device, str]] = None,

            # ===== Item prediction task config =====
            item_task: Literal["regression", "binary_classification"] = "regression",
            binarization_threshold: Optional[float] = None,

            # ===== UserAttrHead config (Stage 3) =====
            is_user_attr_head: bool = False,
            user_head_out_dim: int = 1,
            user_head_hidden: Sequence[int] = (128, 64),
            user_head_dropout: Union[float, Sequence[float]] = 0.0,
            user_head_activation: str = "relu",
            user_head_use_batchnorm: bool = False,
            user_head_use_layernorm: bool = False,

            # ===== Cold-start config =====
            cold_steps: int = 20,
            cold_lr: float = 1e-2,
            cold_wd: float = 0.0,
            cold_patience: int = 10,

            # ===== Normalization config =====
            label_scale: Literal["minmax"] = "minmax",
            user_attr_task: Literal["regression", "classification"] = "regression",
            scores_range: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()

        # Validate required parameters
        if d_in is None or d_model is None:
            raise ValueError("d_in and d_model are required parameters")

        # ===== STEP 1: Determine num_users and build user_id_mapping =====
        if train_loader is not None:
            # Option 1: Extract bag IDs from DataLoader
            print("Extracting bag IDs from DataLoader...")
            all_user_ids = self._extract_user_ids_from_loader(train_loader, user_id_key)
            num_users_extracted = len(all_user_ids)
            self.user_id_mapping = {uid: idx for idx, uid in enumerate(sorted(all_user_ids))}
            self.reverse_mapping = {v: k for k, v in self.user_id_mapping.items()}
            num_users_final = num_users_extracted
            print(f"✓ Found {num_users_final} unique bags")

        elif user_id_mapping is not None:
            # Option 2: Use provided mapping
            self.user_id_mapping = user_id_mapping
            self.reverse_mapping = {v: k for k, v in user_id_mapping.items()}
            num_users_final = len(user_id_mapping)
            print(f"✓ Using provided mapping with {num_users_final} bags")

        else:
            raise ValueError(
                "Must provide one of: train_loader or user_id_mapping"
            )

        # ===== STEP 2: Create UserScorer (Stage 2) with all parameters =====
        self.user_scorer = UserScorer(
            num_users=num_users_final,
            d_in=d_in,
            d_model=d_model,
            mlp_hidden=mlp_hidden,
            mlp_dropout=mlp_dropout,
            mlp_activation=mlp_activation,
            use_batchnorm=mlp_use_batchnorm,
            use_layernorm=mlp_use_layernorm,
            use_bias=use_bias,
            normalize_user=normalize_user,
            normalize_item=normalize_item,
            init_std=init_std,
            dtype=dtype,
            device=device,
        )

        # Cold-start configuration
        self.cold_steps = cold_steps
        self.cold_lr = cold_lr
        self.cold_wd = cold_wd
        self.cold_patience = cold_patience

        # Task configuration
        self.label_scale = label_scale
        self.item_task = item_task
        self.binarization_threshold = binarization_threshold

        # Validate item task configuration
        if self.item_task == "binary_classification":
            if scores_range is not None and binarization_threshold is None:
                print("⚠ Warning: scores_range provided for binary classification without binarization_threshold. "
                      "Assuming labels are already 0/1.")
            if binarization_threshold is not None:
                print(f"✓ Binary classification with threshold: scores >= {binarization_threshold} → 1, else → 0")
        elif self.item_task == "regression":
            if binarization_threshold is not None:
                raise ValueError("binarization_threshold should only be set for binary_classification task")

        # Score normalization parameters (for regression tasks)
        if scores_range is not None:
            if len(scores_range) != 2:
                raise ValueError("scores_range must be a tuple of (min, max)")
            self.y_min = float(scores_range[0])
            self.y_max = float(scores_range[1])
            if self.y_max <= self.y_min:
                raise ValueError(f"scores_range max ({self.y_max}) must be > min ({self.y_min})")
            print(f"✓ Score range set to [{self.y_min}, {self.y_max}]")
        else:
            self.y_min = None
            self.y_max = None

        self.user_attr_task = user_attr_task

        # Store Stage 3 configuration (for saving/loading)
        self.user_head_out_dim = user_head_out_dim
        self.user_head_hidden = tuple(user_head_hidden)
        self.user_head_dropout = user_head_dropout
        self.user_head_activation = user_head_activation
        self.user_head_use_batchnorm = user_head_use_batchnorm
        self.user_head_use_layernorm = user_head_use_layernorm

        # ===== STEP 3: Create UserAttrHead (Stage 3) if enabled =====
        if is_user_attr_head:
            self.user_attr_head = UserAttrHead(
                d_model=d_model,
                out_dim=user_head_out_dim,
                hidden=tuple(user_head_hidden),
                dropout=user_head_dropout,
                activation=user_head_activation,
                use_batchnorm=user_head_use_batchnorm,
                use_layernorm=user_head_use_layernorm,
            )
            print(f"✓ Created UserAttrHead (Stage 3): d_model={d_model} → out_dim={user_head_out_dim}")
        else:
            self.user_attr_head = None

        # Storage for training/eval configurations
        self.loss_cfg: Dict[str, Any] = {}
        self.metric_cfg: Dict[str, Any] = {}

    # ==================== Internal Utilities ==================== #

    def _extract_user_ids_from_loader(self, dataloader: DataLoader, key: str) -> List:
        """
        Iterate through DataLoader once to collect all unique bag IDs.

        This method is used during initialization to automatically determine the
        number of bags in the training set without requiring manual counting.

        Args:
            dataloader: The training DataLoader
            key: Key in batch dictionary containing bag IDs (e.g., 'user_id')

        Returns:
            List of unique bag IDs found in the dataset

        Raises:
            TypeError: If batch is not a dictionary with the specified key
        """
        all_user_ids = set()

        try:
            from tqdm import tqdm
            iterator = tqdm(dataloader, desc="Extracting bag IDs")
        except ImportError:
            iterator = dataloader
            print("Iterating through DataLoader to extract bag IDs...")

        for batch in iterator:
            # Extract bag IDs from batch
            if isinstance(batch, dict):
                user_ids = batch[key]
            else:
                raise TypeError(
                    f"Expected batch to be a dict with key '{key}', "
                    f"got {type(batch)}"
                )

            # Handle both list and tensor formats
            if isinstance(user_ids, torch.Tensor):
                user_ids = user_ids.tolist()

            all_user_ids.update(user_ids)

        return list(all_user_ids)

    def _to_internal_ids(self, user_ids) -> torch.LongTensor:
        """
        Convert external bag IDs to internal indices [0, num_bags-1].

        This mapping enables efficient embedding table lookup in Stage 2 while
        allowing users to work with arbitrary bag identifiers (strings, UUIDs, etc.).

        Args:
            user_ids: List/array of external bag IDs, or LongTensor of internal indices

        Returns:
            LongTensor of internal indices for embedding table lookup

        Raises:
            KeyError: If an external ID was not seen during training (cold bag)
            TypeError: If tensor is passed when mapping exists (ambiguous intent)
        """
        if self.user_id_mapping is None:
            # No mapping: assume already internal indices
            if isinstance(user_ids, torch.Tensor):
                return user_ids.long()
            return torch.tensor(user_ids, dtype=torch.long)

        # Mapping exists: translate external → internal
        if isinstance(user_ids, torch.Tensor):
            raise TypeError(
                "When using ID mapping, pass user_ids as a list/array of external IDs, "
                "not as a tensor of internal indices."
            )

        try:
            internal = [self.user_id_mapping[uid] for uid in user_ids]
        except KeyError as e:
            raise KeyError(
                f"Unknown bag ID: {e}. This bag was not in the training data. "
                f"Use cold-start methods for new bags."
            )

        return torch.tensor(internal, dtype=torch.long)

    def binarize_labels(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous scores to binary labels {0, 1} using threshold.

        This is used when item_task="binary_classification" and raw scores
        need to be converted to binary format (e.g., for TCR binding prediction).

        Args:
            scores: Tensor of scores (any shape)

        Returns:
            Binary labels: 1 if score >= threshold, 0 otherwise.
            If threshold is None, assumes labels are already binary and returns as-is.
        """
        if self.binarization_threshold is None:
            # Assume already binary (0/1)
            return scores
        else:
            # Convert: score >= threshold → 1, else → 0
            return (scores >= self.binarization_threshold).float()

    def _check_normalizer_ready(self) -> None:
        """
        Validate that score normalization parameters are configured.

        For regression tasks, UNISON normalizes scores to [-1, 1] for stable
        optimization. This method ensures the normalization range is set before
        training or inference.

        Raises:
            NotImplementedError: If normalization mode is not "minmax"
            RuntimeError: If scores_range was not provided during initialization
        """
        if self.item_task == "binary_classification":
            return  # No normalization needed for classification

        if self.label_scale != "minmax":
            raise NotImplementedError(f"Normalization mode '{self.label_scale}' not supported.")
        if self.y_min is None or self.y_max is None:
            raise RuntimeError(
                "Target normalizer not set. "
                "Pass scores_range=(min, max) to __init__."
            )

    def prepare_targets(self, y: torch.Tensor) -> torch.Tensor:
        """
        Transform targets based on task type for stable optimization.

        Preprocessing strategy:
        - Regression: Normalize scores to [-1, 1] using min-max scaling
        - Binary classification: Binarize using threshold (or keep as 0/1)

        This normalization is critical for convergence during cold-start optimization
        (Theorem 2) and for balanced gradient magnitudes across different score ranges.

        Args:
            y: Raw target values (any shape)

        Returns:
            Preprocessed targets:
            - Regression: values in [-1, 1]
            - Binary classification: values in {0, 1}
        """
        if self.item_task == "binary_classification":
            return self.binarize_labels(y)

        # Regression path: normalize to [-1, 1]
        self._check_normalizer_ready()
        y = y.float()
        # Map [y_min, y_max] → [0, 1] → [-1, 1]
        y0 = (y - self.y_min) / (self.y_max - self.y_min)
        return y0 * 2.0 - 1.0

    def post_prepare_predictions(self, y_scaled: torch.Tensor) -> torch.Tensor:
        """
        Post-process model outputs to interpretable predictions.

        Postprocessing strategy:
        - Regression: Denormalize from [-1, 1] back to real score scale
        - Binary classification: Apply sigmoid to logits → probabilities [0, 1]

        Args:
            y_scaled: Model outputs (logits for classification, normalized for regression)

        Returns:
            Interpretable predictions:
            - Regression: real-scale scores
            - Binary classification: probabilities in [0, 1]
        """
        if self.item_task == "binary_classification":
            # Apply sigmoid to logits to get probabilities
            return torch.sigmoid(y_scaled)

        # Regression path: denormalize from [-1, 1] to real scale
        self._check_normalizer_ready()
        # Map [-1, 1] → [0, 1] → [y_min, y_max]
        y0 = (y_scaled + 1.0) * 0.5
        return y0 * (self.y_max - self.y_min) + self.y_min

    # ==================== Inference Methods ==================== #

    def predict_warm_user_cold_item(
            self,
            user_ids: torch.LongTensor,
            items: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict scores for warm bags on cold items (WB-CI scenario).

        This is the standard Stage 2 forward pass: use learned bag parameters
        (theta_j, b_j) to score new items that were not seen during training.

        Use case examples:
        - Recommend newly released movies to existing users
        - Predict binding to new peptides for known TCR repertoires

        Args:
            user_ids: Bag identifiers [B]
            items: Item features from Stage 1 encoder [B, Q, d_in]

        Returns:
            scores: Item scores [B, Q] in real scale

        Notes:
            This method uses eval mode and no_grad for inference efficiency.
        """
        user_idx = self._to_internal_ids(user_ids)

        self.eval()
        with torch.no_grad():
            # Stage 2 forward: <theta_j, z_tilde> + b_j
            logits = self.user_scorer(user_idx, items)
            # Post-process: sigmoid for classification, denormalize for regression
            scores = self.post_prepare_predictions(logits)

        return scores

    def predict_cold_user_items(
            self,
            items_sup: torch.Tensor,
            scores_sup: torch.Tensor,
            items_qry: torch.Tensor,
    ) -> dict:
        """
        Cold-bag inference for item scoring (CB-WI / CB-CI scenarios).

        This implements the cold-start procedure from Section 3.3 and Theorem 2:
        1. Normalize support set scores to [-1, 1]
        2. Optimize (theta_hat, bias_hat) on support set via gradient descent
        3. Score query items using learned parameters
        4. Denormalize predictions to real scale

        The lightweight linear optimization (O(d' * |S|) operations) dramatically
        outperforms gradient-based meta-learning while providing convergence guarantees.

        Args:
            items_sup: Support set item features [B, N_sup, d_in]
            scores_sup: Support set scores [B, N_sup] in REAL scale
            items_qry: Query set item features [B, Q, d_in]

        Returns:
            Dictionary containing:
            - scores: Query predictions [B, Q] in real scale
            - theta_hat: Optimized linear weights [B, d_model]
            - bias_hat: Optimized bias terms [B, 1]
            - loss_curve: Training loss per iteration [T] (normalized space)

        Notes:
            Early stopping prevents overfitting on small support sets, which is
            critical for the superior cold-start performance shown in Table 1.
        """
        # Ensure normalizer is fitted for regression tasks
        if self.item_task == "regression":
            self._check_normalizer_ready()

        self.eval()

        # Step 1: Normalize support targets to [-1, 1] for stable optimization
        with torch.no_grad():
            scores_sup_scaled = self.prepare_targets(scores_sup)

        # Step 2: Per-batch cold-start adaptation (stateless, no weight updates to shared MLP)
        theta_hat, bias_hat, loss_curve = fit_cold_start_batch(
            user_scorer=self.user_scorer,
            items_sup=items_sup,
            scores_sup=scores_sup_scaled,
            steps=self.cold_steps,
            lr=self.cold_lr,
            weight_decay=self.cold_wd,
            patience=self.cold_patience,
            task_type=self.item_task,
        )

        # Step 3: Score queries in normalized space using <theta_hat, z_tilde> + bias_hat
        with torch.no_grad():
            items_qry_tilde = self.user_scorer.compute_z_i_tilde(items_qry).detach()  # [B, Q, d_model]
            logits = torch.einsum("bqd,bd->bq", items_qry_tilde, theta_hat) + bias_hat

            # Step 4: Denormalize to REAL scale for interpretable predictions
            scores_real = self.post_prepare_predictions(logits)

        return {
            "scores": scores_real,
            "theta_hat": theta_hat,
            "bias_hat": bias_hat,
            "loss_curve": torch.tensor(loss_curve, device=scores_real.device),
        }

    def predict_cold_user_attribute(
            self,
            items_sup: torch.Tensor,
            scores_sup: torch.Tensor,
    ) -> dict:
        """
        Cold-bag attribute inference via Stage 3 functional embeddings.

        This implements the key insight from Section 3.1: bag-specific parameters
        (theta_j, b_j) learned for item scoring also encode bag-level characteristics.

        Procedure:
        1. Fit (theta_hat, bias_hat) via cold-start optimization on support set
        2. Use these as "functional embeddings" for Stage 3 classifier
        3. Predict bag attributes (e.g., demographics, disease state)

        This approach eliminates the need for auxiliary features while achieving
        strong classification performance across diverse domains (Table 1, Stage 3 AUC).

        Args:
            items_sup: Support set item features [B, N_sup, d_in]
            scores_sup: Support set scores [B, N_sup] in REAL scale

        Returns:
            Dictionary containing:
            - attr_pred: Attribute predictions [B, A] (logits or values depending on task)
            - theta_hat: Functional embeddings [B, d_model]
            - bias_hat: Functional bias terms [B, 1]
            - loss_curve: Cold-start optimization curve [T]

        Raises:
            AssertionError: If UserAttrHead (Stage 3) was not initialized

        Notes:
            The same (theta_hat, bias_hat) parameters that enable accurate item scoring
            also serve as discriminative features for bag classification, validating
            the unified framework design.
        """
        assert self.user_attr_head is not None, "UserAttrHead is required for cold-bag attribute prediction."
        if self.item_task == "regression":
            self._check_normalizer_ready()

        self.eval()

        # Step 1: Normalize support targets for stable per-batch fitting
        with torch.no_grad():
            scores_sup_scaled = self.prepare_targets(scores_sup)

        # Step 2: Per-batch adaptation to obtain functional embeddings
        theta_hat, bias_hat, loss_curve = fit_cold_start_batch(
            user_scorer=self.user_scorer,
            items_sup=items_sup,
            scores_sup=scores_sup_scaled,
            steps=self.cold_steps,
            lr=self.cold_lr,
            weight_decay=self.cold_wd,
            patience=self.cold_patience,
            task_type=self.item_task
        )

        # Step 3: Attribute prediction from functional embeddings (Eq. 6)
        with torch.no_grad():
            attr_pred = self.user_attr_head(theta_hat, bias_hat)

        return {
            "attr_pred": attr_pred,
            "theta_hat": theta_hat,
            "bias_hat": bias_hat,
            "loss_curve": torch.tensor(loss_curve, device=attr_pred.device),
        }

    # ==================== Training Methods ==================== #

    def train_step_warm_user_warm_item(
            self,
            optimizer: torch.optim.Optimizer,
            user_ids: torch.LongTensor,
            items_sup: torch.Tensor,
            scores_sup: torch.Tensor,
    ) -> dict:
        """
        Single optimization step for Stage 2 on warm bags (WB-WI scenario).

        This is the core training procedure for learning the shared MLP (phi) and
        bag-specific parameters (theta_j, b_j). The objective is to minimize:

        - Regression: MSE(predicted_scores, normalized_targets)
        - Binary classification: BCE with class imbalance correction

        The normalization to [-1, 1] for regression ensures stable gradients and
        is inverted during evaluation for interpretable metrics.

        Args:
            optimizer: Torch optimizer (typically Adam)
            user_ids: Bag identifiers [B] (external IDs or internal indices)
            items_sup: Item features [B, N, d_in]
            scores_sup: Item scores [B, N] in REAL scale

        Returns:
            Dictionary containing:
            - loss_scaled: Training loss (scalar) in normalized space
            - pred_real: Predictions [B, N] in real scale (for logging)
            - target_real: Ground truth [B, N] in real scale

        Notes:
            For binary classification, automatic negative downsampling (10:1 ratio)
            addresses class imbalance common in biological datasets (e.g., TCR binding).
        """
        user_idx = self._to_internal_ids(user_ids)
        self.train()

        # Prepare targets: normalize for regression, binarize for classification
        if self.item_task == "regression":
            self._check_normalizer_ready()
        scores_in_prepared = self.prepare_targets(scores_sup)

        # Forward pass: Stage 2 scoring
        optimizer.zero_grad(set_to_none=True)
        pred_logits = self.user_scorer(user_idx, items_sup)  # [B, N]

        if self.item_task == "binary_classification":
            # ========== Class Imbalance Handling ==========
            # Strategy: Include ALL positives + downsample negatives to maintain
            # a controlled pos:neg ratio (default 1:10). This prevents the model
            # from learning a trivial "always negative" solution while exposing
            # sufficient negative examples for discrimination.
            NEG_POS_RATIO = 10.0

            logits_flat = pred_logits.view(-1)
            labels_flat = scores_in_prepared.view(-1)  # already 0/1

            pos_mask = (labels_flat == 1)
            neg_mask = (labels_flat == 0)

            num_pos = int(pos_mask.sum().item())
            num_neg = int(neg_mask.sum().item())

            if num_pos > 0 and num_neg > 0:
                pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
                neg_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)

                max_neg = min(int(NEG_POS_RATIO * num_pos), num_neg)
                if max_neg > 0:
                    perm = torch.randperm(num_neg, device=labels_flat.device)
                    chosen_neg = neg_idx[perm[:max_neg]]
                    chosen = torch.cat([pos_idx, chosen_neg], dim=0)
                else:
                    # Edge case: only positives
                    chosen = pos_idx

                loss = F.binary_cross_entropy_with_logits(
                    logits_flat[chosen], labels_flat[chosen]
                )
            else:
                # Fallback: no positives or no negatives in this batch
                loss = F.binary_cross_entropy_with_logits(
                    logits_flat, labels_flat
                )

        else:  # regression
            loss = F.mse_loss(pred_logits, scores_in_prepared)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Convert predictions back to real scale for logging
        with torch.no_grad():
            pred_real = self.post_prepare_predictions(pred_logits)

        return {
            "loss_scaled": loss.detach(),
            "pred_real": pred_real.detach(),
            "target_real": scores_sup.detach(),
        }

    def train_step_warm_user_item_and_attribute(
            self,
            optimizer: torch.optim.Optimizer,
            user_ids,
            items_sup: torch.Tensor,
            scores_sup: torch.Tensor,
            target_attr: torch.Tensor,
            lambda_attr: float = 1.0,
    ) -> dict:
        """
        Joint optimization of Stage 2 (item scoring) + Stage 3 (bag attributes).

        This implements joint training as described in Section 3.2, Equation 10:
            L_total = L_score + lambda_attr * L_bag

        The joint objective encourages functional embeddings (theta_j, b_j) to
        simultaneously capture item-level preferences and bag-level characteristics.
        Empirically, joint training shows no significant difference from sequential
        training (Figure 2F), suggesting the objectives are well-aligned.

        Args:
            optimizer: Torch optimizer for both Stage 2 and Stage 3 parameters
            user_ids: Bag identifiers [B]
            items_sup: Item features [B, N_sup, d_in]
            scores_sup: Item scores [B, N_sup] in REAL scale
            target_attr: Bag-level attributes [B] or [B, 1]
            lambda_attr: Weight for attribute loss relative to scoring loss

        Returns:
            Dictionary containing:
            - loss_scaled: Total weighted loss (scalar)
            - loss_ratings: Item scoring loss component
            - loss_attr: Attribute prediction loss component
            - pred_real_ratings: Item score predictions [B, N_sup] in real scale
            - target_ratings_real: Ground truth scores [B, N_sup]
            - pred_attr: Attribute predictions (format depends on task)
            - target_attr: Ground truth attributes

        Raises:
            RuntimeError: If UserAttrHead was not initialized

        Notes:
            Lambda_attr controls the trade-off between objectives. Default 1.0 works
            well across domains, but can be tuned for specific applications.
        """
        if self.user_attr_head is None:
            raise RuntimeError("user_attr_head is not initialized. Set is_user_attr_head=True in __init__.")

        user_idx = self._to_internal_ids(user_ids)
        self.train()

        # ========== Stage 2: Item Scoring Loss ==========
        if self.item_task == "regression":
            self._check_normalizer_ready()

        scores_in_prepared = self.prepare_targets(scores_sup)

        optimizer.zero_grad(set_to_none=True)

        pred_logits_items = self.user_scorer(user_idx, items_sup)

        if self.item_task == "binary_classification":
            # Class imbalance handling (same as train_step_warm_user_warm_item)
            NEG_POS_RATIO = 4

            logits_flat = pred_logits_items.view(-1)
            labels_flat = scores_in_prepared.view(-1)

            pos_mask = (labels_flat == 1)
            neg_mask = (labels_flat == 0)

            num_pos = int(pos_mask.sum().item())
            num_neg = int(neg_mask.sum().item())

            if num_pos > 0 and num_neg > 0:
                pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
                neg_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)

                max_neg = min(int(NEG_POS_RATIO * num_pos), num_neg)
                if max_neg > 0:
                    perm = torch.randperm(num_neg, device=labels_flat.device)
                    chosen_neg = neg_idx[perm[:max_neg]]
                    chosen = torch.cat([pos_idx, chosen_neg], dim=0)
                else:
                    chosen = pos_idx

                rating_loss = F.binary_cross_entropy_with_logits(
                    logits_flat[chosen], labels_flat[chosen]
                )
            else:
                rating_loss = F.binary_cross_entropy_with_logits(
                    logits_flat, labels_flat
                )

        else:  # regression
            rating_loss = F.mse_loss(pred_logits_items, scores_in_prepared)

        # ========== Stage 3: Attribute Loss using Warm Bag Embeddings ==========
        # Extract warm bag embeddings with gradient tracking (for joint training)
        theta_hat, bias_hat = self.user_scorer.get_user_embeddings(user_idx)

        logits_attr = self.user_attr_head(theta_hat, bias_hat)  # [B, A]
        B, A = logits_attr.shape

        if self.user_attr_task == "regression":
            # Regression: MSE loss
            pred_attr = logits_attr.squeeze(-1) if A == 1 else logits_attr
            t_attr = (
                target_attr.squeeze(-1)
                if (target_attr.dim() == 2 and target_attr.size(1) == 1)
                else target_attr
            )
            attr_loss = F.mse_loss(pred_attr, t_attr.float())

        else:
            # Classification
            if A == 1:
                # Binary classification: BCE with logits
                y = target_attr.squeeze(-1) if target_attr.dim() > 1 else target_attr
                y = y.float()
                logits_flat = logits_attr.squeeze(1)
                attr_loss = F.binary_cross_entropy_with_logits(logits_flat, y)
                pred_attr = torch.sigmoid(logits_flat)

            else:
                # Multiclass classification: Cross-entropy
                y = (
                    target_attr.squeeze(-1).long()
                    if target_attr.dim() > 1
                    else target_attr.long()
                )
                attr_loss = F.cross_entropy(logits_attr, y)
                pred_attr = torch.softmax(logits_attr, dim=-1)

        # ========== Total Loss and Optimization ==========
        total_loss = rating_loss + lambda_attr * attr_loss
        total_loss.backward()
        optimizer.step()

        # Convert rating predictions back to real scale for logging
        with torch.no_grad():
            pred_real_ratings = self.post_prepare_predictions(pred_logits_items)

        return {
            "loss_scaled": total_loss.detach(),
            "loss_ratings": rating_loss.detach(),
            "loss_attr": attr_loss.detach(),
            "pred_real_ratings": pred_real_ratings.detach(),
            "target_ratings_real": scores_sup.detach(),
            "pred_attr": pred_attr.detach(),
            "target_attr": target_attr.detach(),
        }

    def train_step_warm_user_attribute(
            self,
            optimizer: torch.optim.Optimizer,
            user_ids,
            items_sup: torch.Tensor,
            scores_sup: torch.Tensor,
            target_attr: torch.Tensor
    ) -> dict:
        """
        Stage 3 ONLY optimization step on warm bags.

        This trains the attribute classifier (Stage 3) using frozen bag embeddings
        from Stage 2. Useful when Stage 2 is pre-trained and fixed, and only the
        downstream attribute prediction needs to be learned.

        Use cases:
        - Transfer learning: adapt pre-trained scoring model to new attribute tasks
        - Two-stage training: first train Stage 2, then train Stage 3 independently
        - Ablation studies: isolate Stage 3 performance

        Args:
            optimizer: Torch optimizer (only updates Stage 3 parameters)
            user_ids: Bag identifiers [B]
            items_sup: Item features [B, N_sup, d_in] (not used, kept for API consistency)
            scores_sup: Item scores [B, N_sup] (not used, kept for API consistency)
            target_attr: Bag-level attributes [B] or [B, 1]

        Returns:
            Dictionary containing:
            - loss_attr: Attribute prediction loss (scalar)
            - loss_scaled: Same as loss_attr (for generic training loops)
            - pred_attr: Attribute predictions (format depends on task)
            - target_attr: Ground truth attributes
            - attr_accuracy: Accuracy (for classification tasks)
            - attr_auc: AUC (for binary classification only)

        Raises:
            RuntimeError: If UserAttrHead was not initialized

        Notes:
            Bag embeddings are detached from Stage 2, so gradients only flow through
            Stage 3. This prevents catastrophic forgetting of Stage 2 parameters.
        """
        if self.user_attr_head is None:
            raise RuntimeError(
                "user_attr_head is not initialized. "
                "Set is_user_attr_head=True in __init__ before calling train_step_warm_user_attribute."
            )

        user_idx = self._to_internal_ids(user_ids)
        self.train()

        # Get warm-bag embeddings WITHOUT gradients to Stage 2
        theta_hat, bias_hat = self.user_scorer.get_user_embeddings(user_idx)

        # Forward through Stage 3 attribute head
        logits_attr = self.user_attr_head(theta_hat, bias_hat)  # [B, A]
        B, A = logits_attr.shape

        # Task-dependent loss and predictions
        attr_loss: torch.Tensor
        pred_attr: torch.Tensor
        attr_accuracy = None
        attr_auc = None

        if self.user_attr_task == "regression":
            # Regression: MSE loss
            pred_attr = logits_attr.squeeze(-1) if A == 1 else logits_attr
            t_attr = (
                target_attr.squeeze(-1)
                if (target_attr.dim() == 2 and target_attr.size(1) == 1)
                else target_attr
            )
            t_attr = t_attr.float()
            attr_loss = F.mse_loss(pred_attr, t_attr)

        else:
            # Classification
            if A == 1:
                # Binary classification
                y = target_attr.squeeze(-1) if target_attr.dim() > 1 else target_attr
                y = y.float()
                logits_flat = logits_attr.squeeze(1)

                attr_loss = F.binary_cross_entropy_with_logits(logits_flat, y)
                probs = torch.sigmoid(logits_flat)
                pred_attr = probs

                # Compute metrics without gradients
                with torch.no_grad():
                    # Accuracy
                    preds_bin = (probs >= 0.5).long()
                    labels_long = y.long()
                    attr_accuracy = (preds_bin == labels_long).float().mean().item()

                    # AUC (approximate via Mann-Whitney U statistic)
                    pos = probs[y == 1]
                    neg = probs[y == 0]
                    if pos.numel() > 0 and neg.numel() > 0:
                        auc_tensor = (pos[:, None] > neg[None, :]).float().mean()
                        attr_auc = float(auc_tensor.item())
                    else:
                        attr_auc = float("nan")

            else:
                # Multiclass classification
                y = (
                    target_attr.squeeze(-1).long()
                    if target_attr.dim() > 1
                    else target_attr.long()
                )

                attr_loss = F.cross_entropy(logits_attr, y)
                probs = torch.softmax(logits_attr, dim=-1)
                pred_attr = probs

                # Accuracy metric
                with torch.no_grad():
                    preds_cls = probs.argmax(dim=-1)
                    attr_accuracy = (preds_cls == y).float().mean().item()
                    attr_auc = None  # Not defined for general multiclass

        # Backward pass and optimization (only Stage 3 parameters updated)
        optimizer.zero_grad(set_to_none=True)
        attr_loss.backward()
        optimizer.step()

        out = {
            "loss_attr": attr_loss.detach(),
            "loss_scaled": attr_loss.detach(),
            "pred_attr": pred_attr.detach(),
            "target_attr": target_attr.detach(),
        }

        # Add classification metrics if available
        if self.user_attr_task != "regression":
            if attr_accuracy is not None:
                out["attr_accuracy"] = attr_accuracy
            if attr_auc is not None:
                out["attr_auc"] = attr_auc

        return out

    # ==================== Evaluation Methods ==================== #

    def evaluate_warm_user_cold_item(
            self,
            user_ids: torch.LongTensor,
            items_qry: torch.Tensor,
            scores_qry: torch.Tensor,
            mask_qry: torch.Tensor,
    ) -> dict:
        """
        Evaluate Stage 2 on warm bags, cold items (WB-CI scenario).

        This corresponds to the standard evaluation setting where the model has
        learned bag preferences during training and must generalize to new items.

        Metrics are computed on real scale for interpretability, while loss is
        computed in normalized space ([-1, 1]) for consistency with training.

        Args:
            user_ids: Bag identifiers [B]
            items_qry: Query item features [B, Q, d_in]
            scores_qry: Query scores [B, Q] in REAL scale
            mask_qry: Query mask [B, Q] indicating which items to evaluate

        Returns:
            Dictionary containing:
            - loss_scaled: Evaluation loss in normalized space
            - pred_real: Predictions [B, Q] in real scale
            - Regression: mae_WU_CI, mse_WU_CI, spearman_WU_CI
            - Classification: auc, accuracy

        Notes:
            The WB-CI scenario tests the model's ability to generalize the learned
            preference function to new items, which is critical for real-world
            applications (e.g., recommending newly released content).
        """
        user_idx = self._to_internal_ids(user_ids)
        self.eval()

        if self.item_task == "regression":
            self._check_normalizer_ready()

        with torch.no_grad():
            # Forward pass: predict item scores
            pred_logits = self.user_scorer(user_idx, items_qry)  # [B, Q]
            tgt_prepared = self.prepare_targets(scores_qry)

            # Compute loss in normalized space
            if self.item_task == "binary_classification":
                loss_scaled = F.binary_cross_entropy_with_logits(pred_logits, tgt_prepared)
            else:  # regression
                loss_scaled = F.mse_loss(pred_logits, tgt_prepared)

            # Post-process predictions to real scale
            pred_real = self.post_prepare_predictions(pred_logits)

            # Compute task-specific metrics
            if self.item_task == "binary_classification":
                metrics = binary_classification_metrics(pred_logits, tgt_prepared)
            else:  # regression
                metrics = regression_metrics(pred_real, scores_qry, mask=mask_qry, classes=(3,))
                # Add aliases for readability
                metrics["mae_WU_CI"] = metrics.get("mae_3", torch.tensor(float("nan"), device=pred_real.device))
                metrics["mse_WU_CI"] = metrics.get("mse_3", torch.tensor(float("nan"), device=pred_real.device))
                metrics["spearman_WU_CI"] = metrics.get("spearman_3",
                                                        torch.tensor(float("nan"), device=pred_real.device))

        return {
            "loss_scaled": loss_scaled,
            "pred_real": pred_real,
            **metrics,
        }

    def evaluate_cold_user_items(
            self,
            items_sup: torch.Tensor,
            scores_sup: torch.Tensor,
            items_qry: torch.Tensor,
            scores_qry: torch.Tensor,
            mask_qry: torch.Tensor,
    ) -> dict:
        """
        Cold-bag item evaluation (CB-WI / CB-CI scenarios).

        This is the most challenging evaluation setting, corresponding to the
        CU-CI (Cold User-Cold Item) results in Table 1. The model must:
        1. Learn bag preferences from a small support set
        2. Generalize to new query items

        The mask enables separate evaluation of CB-WI (query items seen during training)
        and CB-CI (completely new query items), providing fine-grained analysis of
        generalization capabilities.

        Args:
            items_sup: Support set item features [B, N_sup, d_in]
            scores_sup: Support set scores [B, N_sup] in REAL scale
            items_qry: Query set item features [B, Q, d_in]
            scores_qry: Query set scores [B, Q] in REAL scale
            mask_qry: Query mask [B, Q] with values:
                      0 = padding, 1 = warm item (CB-WI), 2 = cold item (CB-CI)

        Returns:
            Dictionary containing:
            - pred_real: Query predictions [B, Q] in real scale
            - pred_logits: Raw model outputs [B, Q]
            - theta_hat: Learned bag parameters [B, d_model]
            - bias_hat: Learned bias terms [B, 1]
            - loss_scaled: Evaluation loss in normalized space
            - loss_curve: Cold-start optimization curve [T]
            - Regression: mae_CU_WI, mse_CU_WI, spearman_CU_WI (class 1)
                          mae_CU_CI, mse_CU_CI, spearman_CU_CI (class 2)
            - Classification: auc_CU_WI, accuracy_CU_WI (class 1)
                              auc_CU_CI, accuracy_CU_CI (class 2)

        Notes:
            The separation into CB-WI and CB-CI metrics reveals how much of the model's
            performance comes from item-level memorization versus true generalization.
            Strong CB-CI performance (as shown in Table 1) validates the framework's
            ability to learn transferable preference patterns.
        """
        if self.item_task == "regression":
            self._check_normalizer_ready()
        self.eval()

        # Step 1: Normalize support targets
        with torch.no_grad():
            scores_sup_prepared = self.prepare_targets(scores_sup)

        # Step 2: Cold-start adaptation
        theta_hat, bias_hat, loss_curve = fit_cold_start_batch(
            user_scorer=self.user_scorer,
            items_sup=items_sup,
            scores_sup=scores_sup_prepared,
            steps=self.cold_steps,
            lr=self.cold_lr,
            weight_decay=self.cold_wd,
            patience=self.cold_patience,
            task_type=self.item_task,
        )

        # Step 3: Predict on query set
        with torch.no_grad():
            items_qry_tilde = self.user_scorer.compute_z_i_tilde(items_qry).detach()
            pred_logits = torch.einsum("bqd,bd->bq", items_qry_tilde, theta_hat) + bias_hat

            tgt_prepared = self.prepare_targets(scores_qry)

            # Compute loss
            if self.item_task == "binary_classification":
                loss_scaled = F.binary_cross_entropy_with_logits(pred_logits, tgt_prepared)
            else:  # regression
                loss_scaled = F.mse_loss(pred_logits, tgt_prepared)

            # Post-process predictions
            pred_real = self.post_prepare_predictions(pred_logits)

            # Compute metrics with mask-based separation
            if self.item_task == "binary_classification":
                metrics = binary_classification_metrics(
                    logits=pred_logits,
                    labels=tgt_prepared,
                    mask=mask_qry,
                    classes=(1, 2),  # 1=CB-WI, 2=CB-CI
                )
                # Add readable aliases
                if "auc_1" in metrics:
                    metrics["auc_CU_WI"] = metrics["auc_1"]
                if "accuracy_1" in metrics:
                    metrics["accuracy_CU_WI"] = metrics["accuracy_1"]
                if "auc_2" in metrics:
                    metrics["auc_CU_CI"] = metrics["auc_2"]
                if "accuracy_2" in metrics:
                    metrics["accuracy_CU_CI"] = metrics["accuracy_2"]
            else:  # regression
                metrics = regression_metrics(pred_real, scores_qry, mask_qry, classes=(1, 2))
                # Add readable aliases
                if "mae_1" in metrics: metrics["mae_CU_WI"] = metrics["mae_1"]
                if "mse_1" in metrics: metrics["mse_CU_WI"] = metrics["mse_1"]
                if "spearman_1" in metrics: metrics["spearman_CU_WI"] = metrics["spearman_1"]
                if "mae_2" in metrics: metrics["mae_CU_CI"] = metrics["mae_2"]
                if "mse_2" in metrics: metrics["mse_CU_CI"] = metrics["mse_2"]
                if "spearman_2" in metrics: metrics["spearman_CU_CI"] = metrics["spearman_2"]

        return {
            "pred_real": pred_real,
            "pred_logits": pred_logits,
            "theta_hat": theta_hat,
            "bias_hat": bias_hat,
            "loss_scaled": loss_scaled,
            "loss_curve": torch.tensor(loss_curve, device=pred_real.device),
            **metrics,
        }

    def evaluate_cold_user_attribute(
            self,
            items_sup: torch.Tensor,
            scores_sup: torch.Tensor,
            target_attr: torch.Tensor,
    ) -> dict:
        """
        Cold-bag attribute evaluation via Stage 3 functional embeddings.

        This evaluates the key insight from Section 3.1: parameters learned for
        item scoring can predict bag-level characteristics without auxiliary features.

        The evaluation procedure mirrors cold-start inference:
        1. Fit (theta_hat, bias_hat) on support set
        2. Use as functional embeddings for Stage 3 classifier
        3. Compute task-specific metrics (regression: MAE/MSE, classification: AUC/accuracy)

        Strong performance here (Table 1, Stage 3 AUC > 0.7 across domains) validates
        that scoring behavior encodes higher-level bag characteristics.

        Args:
            items_sup: Support set item features [B, N_sup, d_in]
            scores_sup: Support set scores [B, N_sup] in REAL scale
            target_attr: Bag-level attributes [B] or [B, 1]

        Returns:
            Dictionary containing:
            - theta_hat: Functional embeddings [B, d_model]
            - bias_hat: Functional bias terms [B, 1]
            - loss_curve: Cold-start optimization curve [T]
            - Regression: pred (predictions), mae, mse, spearman
            - Binary classification: logits, loss_bce, auc, accuracy, probs
            - Multiclass classification: logits, loss_ce, accuracy, probs, preds

        Raises:
            AssertionError: If UserAttrHead was not initialized

        Notes:
            This evaluation setting is particularly important for domains where
            bag-level labels are expensive or sparse (e.g., medical diagnosis from
            repertoire data), as it enables few-shot prediction without manual features.
        """
        assert self.user_attr_head is not None, "UserAttrHead is required."
        if self.item_task == "regression":
            self._check_normalizer_ready()
        self.eval()

        # Step 1: Cold-start adaptation to obtain functional embeddings
        with torch.no_grad():
            scores_sup_scaled = self.prepare_targets(scores_sup)

        theta_hat, bias_hat, loss_curve = fit_cold_start_batch(
            user_scorer=self.user_scorer,
            items_sup=items_sup,
            scores_sup=scores_sup_scaled,
            steps=self.cold_steps,
            lr=self.cold_lr,
            weight_decay=self.cold_wd,
            patience=self.cold_patience,
            task_type=self.item_task,
        )

        # Step 2: Attribute prediction from functional embeddings
        with torch.no_grad():
            logits = self.user_attr_head(theta_hat, bias_hat)  # [B, A]
        B, A = logits.shape

        out = {
            "theta_hat": theta_hat,
            "bias_hat": bias_hat,
            "loss_curve": torch.tensor(loss_curve, device=logits.device),
        }

        # Step 3: Task-dependent evaluation
        if self.user_attr_task == "regression":
            # Regression: compute MAE, MSE, Spearman
            pred = logits.squeeze(-1) if A == 1 else logits
            t = target_attr.squeeze(-1) if (target_attr.dim() == 2 and target_attr.size(1) == 1) else target_attr
            metrics = regression_metrics(pred, t, mask=None)
            out.update({"pred": pred, **metrics})
            return out

        # Classification path
        if A == 1:
            # Binary classification: AUC, accuracy, BCE loss
            y = target_attr.squeeze(-1) if target_attr.dim() > 1 else target_attr
            metrics = binary_classification_metrics(logits.squeeze(1), y)
            out.update({"logits": logits.squeeze(1), **metrics})
        else:
            # Multiclass classification: accuracy, cross-entropy loss
            y = target_attr.squeeze(-1).long() if target_attr.dim() > 1 else target_attr.long()
            metrics = multiclass_classification_metrics(logits, y)
            out.update({"logits": logits, **metrics})

        return out
