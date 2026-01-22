# ============================================================
# unison.py
# ============================================================
# UNISON

# A unified meta-learning framework integrating:
#   - Stage 2: Preference Modeling (warm/cold user, warm/cold item  - item scoring)
#   - Stage 3: Bag Classification (attribute prediction)
#
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict, Any, Literal, List, Union, Sequence, Tuple

from src.utils.metrics import (
    regression_metrics,
    binary_classification_metrics,
    multiclass_classification_metrics,
)
from src.model.stage2.user_scorer import UserScorer
from src.model.stage2.cold_start_adapter import fit_cold_start_batch
from src.model.stage3.heads import UserAttrHead

from torch.utils.data import DataLoader


class Unison(nn.Module):
    """
    UNISON

    A meta-learning framework that unifies preference modeling (Stage 2) and
    bag classification (Stage 3) into a single system. Supports both warm and
    cold evaluation modes with per-batch adaptation.

    Architecture Overview
    ---------------------
    **Stage 2 - Preference Modeling (Warm Users)**:
        - Learn per-user embeddings θ_u ∈ R^{d_model}
        - Transform items: z_hat = MLP(z_in)
        - Score: <θ_u, z_hat> + b_u

    **Stage 3 - Bag Classification (Cold Users)**:
        - Adapt user embeddings from support set (few-shot learning)
        - Option A: Score new items with adapted embeddings
        - Option B: Predict user attributes from adapted embeddings

    Parameters
    ----------
    train_loader : DataLoader, optional
        Training data loader for extracting user IDs automatically.
    user_id_mapping : dict, optional
        Explicit mapping from external user IDs to internal indices [0, num_users).
        Provide either train_loader OR user_id_mapping (not both).
    user_id_key : str, default='user_id'
        Key in batch dictionary containing user IDs.

    d_in : int, required
        Dimensionality of input item features.
    d_model : int, required
        Dimensionality of user/item embedding space.
    mlp_hidden : Sequence[int], default=(128,)
        Hidden layer sizes for item feature MLP.
    mlp_dropout : float, default=0.0
        Dropout probability in MLP.
    mlp_activation : str, default='relu'
        Activation function for MLP.
    use_bias : bool, default=True
        Learn per-user bias terms.
    mlp_use_batchnorm : bool, default=False
        Apply batch normalization in MLP.
    mlp_use_layernorm : bool, default=False
        Apply layer normalization in MLP.
    normalize_user : bool, default=True
        L2-normalize user embeddings (recommended).
    normalize_item : bool, default=False
        L2-normalize item features after MLP.
    init_std : float, default=0.02
        Standard deviation for user embedding initialization.
    dtype : torch.dtype, default=torch.float32
        Data type for parameters.
    device : torch.device or str, optional
        Device for model placement.

    item_task : {'regression', 'binary_classification'}, default='regression'
        Task type for item scoring.
    binarization_threshold : float, optional
        Threshold for converting continuous scores to binary labels.
        Only used when item_task='binary_classification'.

    is_user_attr_head : bool, default=False
        Whether to include user attribute prediction head (Stage 3).
    user_head_out_dim : int, default=1
        Output dimension for user attribute head.
    user_head_hidden : Sequence[int], default=(128, 64)
        Hidden layers for user attribute MLP.
    user_head_dropout : float or Sequence[float], default=0.0
        Dropout for user attribute MLP.
    user_head_activation : str, default='relu'
        Activation for user attribute MLP.
    user_head_use_batchnorm : bool, default=False
        Batch normalization for user attribute MLP.
    user_head_use_layernorm : bool, default=False
        Layer normalization for user attribute MLP.

    cold_steps : int, default=20
        Maximum optimization steps for cold-start adaptation.
    cold_lr : float, default=1e-2
        Learning rate for cold-start adaptation.
    cold_wd : float, default=0.0
        Weight decay for cold-start adaptation.
    cold_patience : int, default=10
        Early stopping patience for cold-start adaptation.

    label_scale : {'minmax'}, default='minmax'
        Label normalization method for regression.
    user_attr_task : {'regression', 'classification'}, default='regression'
        Task type for user attribute prediction.
    scores_range : tuple of (float, float), optional
        (min, max) range for score normalization. E.g., (1.0, 5.0) for ratings.

    Examples
    --------
    >>> # Create model from DataLoader
    >>> model = Unison(
    ...     train_loader=train_loader,
    ...     d_in=64,
    ...     d_model=128,
    ...     item_task='regression',
    ...     scores_range=(1.0, 5.0),
    ...     is_user_attr_head=True,
    ...     user_head_out_dim=1
    ... )

    >>> # Warm-user prediction
    >>> scores = model.predict_warm_user_cold_item(
    ...     user_ids=[0, 1, 2],
    ...     items=items  # [3, 10, 64]
    ... )

    >>> # Cold-user adaptation
    >>> result = model.predict_cold_user_items(
    ...     items_sup=support_items,    # [B, N_sup, D_in]
    ...     scores_sup=support_scores,  # [B, N_sup]
    ...     items_qry=query_items        # [B, Q, D_in]
    ... )
    """

    def __init__(
            self,
            # ===== Data Source (choose one) =====
            train_loader: Optional[DataLoader] = None,
            user_id_mapping: Optional[Dict[Any, int]] = None,
            user_id_key: str = "user_id",
            # ===== UserScorer parameters =====
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
            # ===== UserAttrHead config =====
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
            all_user_ids = self._extract_user_ids_from_loader(train_loader, user_id_key)
            num_users_extracted = len(all_user_ids)
            self.user_id_mapping = {
                uid: idx for idx, uid in enumerate(sorted(all_user_ids))
            }
            self.reverse_mapping = {v: k for k, v in self.user_id_mapping.items()}
            num_users_final = num_users_extracted

        elif user_id_mapping is not None:
            self.user_id_mapping = user_id_mapping
            self.reverse_mapping = {v: k for k, v in user_id_mapping.items()}
            num_users_final = len(user_id_mapping)

        else:
            raise ValueError("Must provide one of: train_loader or user_id_mapping")

        # ===== STEP 2: Create UserScorer =====
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
                pass  # Assume labels are already 0/1
        elif self.item_task == "regression":
            if binarization_threshold is not None:
                raise ValueError(
                    "binarization_threshold should only be set for binary_classification task"
                )

        # Score normalization parameters
        if scores_range is not None:
            if len(scores_range) != 2:
                raise ValueError("scores_range must be a tuple of (min, max)")
            self.y_min = float(scores_range[0])
            self.y_max = float(scores_range[1])
            if self.y_max <= self.y_min:
                raise ValueError(
                    f"scores_range max ({self.y_max}) must be > min ({self.y_min})"
                )
        else:
            self.y_min = None
            self.y_max = None

        self.user_attr_task = user_attr_task

        # Store user attribute head config (for saving/loading)
        self.user_head_out_dim = user_head_out_dim
        self.user_head_hidden = tuple(user_head_hidden)
        self.user_head_dropout = user_head_dropout
        self.user_head_activation = user_head_activation
        self.user_head_use_batchnorm = user_head_use_batchnorm
        self.user_head_use_layernorm = user_head_use_layernorm

        # Create user attribute head if requested (Stage 3)
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
        else:
            self.user_attr_head = None

        # Storage for training/eval configs
        self.loss_cfg: Dict[str, Any] = {}
        self.metric_cfg: Dict[str, Any] = {}

    def _extract_user_ids_from_loader(
            self, dataloader: DataLoader, key: str
    ) -> List:
        """
        Iterate through DataLoader to collect all unique user IDs.

        Parameters
        ----------
        dataloader : DataLoader
            The training DataLoader.
        key : str
            Key in batch dictionary containing user IDs.

        Returns
        -------
        list
            Unique user IDs found in the dataset.
        """
        all_user_ids = set()

        try:
            from tqdm import tqdm

            iterator = tqdm(dataloader, desc="Extracting user IDs")
        except ImportError:
            iterator = dataloader

        for batch in iterator:
            if isinstance(batch, dict):
                user_ids = batch[key]
            else:
                raise TypeError(
                    f"Expected batch to be a dict with key '{key}', got {type(batch)}"
                )

            # Handle both list and tensor formats
            if isinstance(user_ids, torch.Tensor):
                user_ids = user_ids.tolist()

            all_user_ids.update(user_ids)

        return list(all_user_ids)

    def _to_internal_ids(self, user_ids) -> torch.LongTensor:
        """
        Convert external user IDs to internal indices [0, num_users).

        Parameters
        ----------
        user_ids : list, array, or LongTensor
            External user IDs or internal indices.

        Returns
        -------
        torch.LongTensor
            Internal user indices.

        Raises
        ------
        KeyError
            If user ID not found in mapping (unknown user).
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
                f"Unknown user ID: {e}. This user was not in the training data. "
                f"Use cold-start methods for new users."
            )

        return torch.tensor(internal, dtype=torch.long)

    def binarize_labels(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous scores to binary labels using threshold.

        Parameters
        ----------
        scores : Tensor
            Continuous scores (any shape).

        Returns
        -------
        Tensor
            Binary labels (0 or 1).
        """
        if self.binarization_threshold is None:
            # Assume already binary
            return scores
        else:
            # score >= threshold → 1, else → 0
            return (scores >= self.binarization_threshold).float()

    def _check_normalizer_ready(self) -> None:
        """
        Verify that score normalization parameters are set.

        Only required for regression tasks. Classification tasks skip this check.

        Raises
        ------
        RuntimeError
            If normalization parameters not set for regression task.
        """
        if self.item_task == "binary_classification":
            return

        if self.label_scale != "minmax":
            raise NotImplementedError(
                f"Normalization mode '{self.label_scale}' not supported."
            )
        if self.y_min is None or self.y_max is None:
            raise RuntimeError(
                "Target normalizer not set. "
                "Pass scores_range=(min, max) to __init__."
            )

    def prepare_targets(self, y: torch.Tensor) -> torch.Tensor:
        """
        Prepare targets for training/evaluation based on task type.

        - Regression: Normalize to [-1, 1] range using min-max scaling
        - Classification: Binarize using threshold (or keep as 0/1)

        Parameters
        ----------
        y : Tensor
            Raw target values.

        Returns
        -------
        Tensor
            Prepared targets (normalized or binarized).
        """
        if self.item_task == "binary_classification":
            return self.binarize_labels(y)

        # Regression: normalize to [-1, 1]
        self._check_normalizer_ready()
        y = y.float()
        y0 = (y - self.y_min) / (self.y_max - self.y_min)  # [0, 1]
        return y0 * 2.0 - 1.0  # [-1, 1]

    def post_prepare_predictions(self, y_scaled: torch.Tensor) -> torch.Tensor:
        """
        Post-process model outputs to interpretable predictions.

        - Regression: Denormalize from [-1, 1] to original score range
        - Classification: Apply sigmoid to logits for probabilities [0, 1]

        Parameters
        ----------
        y_scaled : Tensor
            Model outputs (logits or normalized values).

        Returns
        -------
        Tensor
            Interpretable predictions (real-scale scores or probabilities).
        """
        if self.item_task == "binary_classification":
            # Logits → probabilities
            return torch.sigmoid(y_scaled)

        # Regression: denormalize from [-1, 1]
        self._check_normalizer_ready()
        y0 = (y_scaled + 1.0) * 0.5  # [-1, 1] → [0, 1]
        return y0 * (self.y_max - self.y_min) + self.y_min  # [0, 1] → [y_min, y_max]

        # ============================================================
        # PREDICTION METHODS
        # ============================================================

    def predict_warm_user_cold_item(
            self,
            user_ids: Union[List, torch.LongTensor],
            items: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict scores for warm users on cold items (Stage 2).

        This uses the learned user embeddings from training to score
        new items that were not seen during training.

        Parameters
        ----------
        user_ids : list or LongTensor [B]
            User identifiers (external IDs or internal indices).
        items : Tensor [B, Q, D_in]
            Item features from encoder.

        Returns
        -------
        scores : Tensor [B, Q]
            Predicted scores in original scale (real-valued for regression,
            probabilities for classification).
        """
        user_idx = self._to_internal_ids(user_ids)

        self.eval()
        with torch.no_grad():
            logits = self.user_scorer(user_idx, items)
            scores = self.post_prepare_predictions(logits)

        return scores

    def predict_cold_user_items(
            self,
            items_sup: torch.Tensor,
            scores_sup: torch.Tensor,
            items_qry: torch.Tensor,
    ) -> dict:
        """
        Predict item scores for cold users (Stage 3 - Item Scoring).

        Adapts user embeddings from a support set, then scores query items.
        This handles both CU-WI (cold user, warm item) and CU-CI (cold user,
        cold item) scenarios.

        Parameters
        ----------
        items_sup : Tensor [B, N_sup, D_in]
            Support set item features.
        scores_sup : Tensor [B, N_sup]
            Support set scores (real scale).
        items_qry : Tensor [B, Q, D_in]
            Query item features to score.

        Returns
        -------
        dict
            - 'scores': Tensor [B, Q] - Predictions in real scale
            - 'theta_hat': Tensor [B, d_model] - Adapted user embeddings
            - 'bias_hat': Tensor [B, 1] - Adapted bias terms
            - 'loss_curve': Tensor [T] - Adaptation loss history
        """
        if self.item_task == "regression":
            self._check_normalizer_ready()

        self.eval()

        # Normalize support targets
        with torch.no_grad():
            scores_sup_scaled = self.prepare_targets(scores_sup)

        # Adapt user embeddings from support set
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

        # Score query items using adapted embeddings
        with torch.no_grad():
            items_qry_tilde = self.user_scorer.compute_z_i_tilde(items_qry).detach()
            logits = torch.einsum("bqd,bd->bq", items_qry_tilde, theta_hat) + bias_hat
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
        Predict user attributes for cold users (Stage 3 - Attribute Prediction).

        Adapts user embeddings from a support set of item ratings, then
        predicts user-level attributes using the UserAttrHead.

        Parameters
        ----------
        items_sup : Tensor [B, N_sup, D_in]
            Support set item features.
        scores_sup : Tensor [B, N_sup]
            Support set scores (real scale).

        Returns
        -------
        dict
            - 'attr_pred': Tensor [B, A] - Attribute predictions
            - 'theta_hat': Tensor [B, d_model] - Adapted user embeddings
            - 'bias_hat': Tensor [B, 1] - Adapted bias terms
            - 'loss_curve': Tensor [T] - Adaptation loss history

        Raises
        ------
        AssertionError
            If UserAttrHead is not initialized.
        """
        assert (
                self.user_attr_head is not None
        ), "UserAttrHead required. Set is_user_attr_head=True in __init__."

        if self.item_task == "regression":
            self._check_normalizer_ready()

        self.eval()

        # Normalize support targets for adaptation
        with torch.no_grad():
            scores_sup_scaled = self.prepare_targets(scores_sup)

        # Adapt user embeddings
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

        # Predict attributes from adapted embeddings
        with torch.no_grad():
            attr_pred = self.user_attr_head(theta_hat, bias_hat)

        return {
            "attr_pred": attr_pred,
            "theta_hat": theta_hat,
            "bias_hat": bias_hat,
            "loss_curve": torch.tensor(loss_curve, device=attr_pred.device),
        }

    # ============================================================
    # TRAINING METHODS
    # ============================================================

    def train_step_warm_user_warm_item(
            self,
            optimizer: torch.optim.Optimizer,
            user_ids: Union[List, torch.LongTensor],
            items_sup: torch.Tensor,
            scores_sup: torch.Tensor,
    ) -> dict:
        """
        Single training step for Stage 2 (warm users, warm items).

        Optimizes user embeddings and MLP parameters using labeled
        item-score pairs.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer for model parameters.
        user_ids : list or LongTensor [B]
            User identifiers.
        items_sup : Tensor [B, N_sup, D_in]
            Item features.
        scores_sup : Tensor [B, N_sup]
            Target scores (real scale).

        Returns
        -------
        dict
            - 'loss_scaled': Tensor (scalar) - Loss value
            - 'pred_real': Tensor [B, N_sup] - Predictions (real scale)
            - 'target_real': Tensor [B, N_sup] - Targets (real scale)
        """
        user_idx = self._to_internal_ids(user_ids)

        self.train()

        # Prepare targets (normalize/binarize)
        if self.item_task == "regression":
            self._check_normalizer_ready()
        scores_in_prepared = self.prepare_targets(scores_sup)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        pred_logits = self.user_scorer(user_idx, items_sup)

        # Compute loss based on task type
        if self.item_task == "binary_classification":
            # Handle class imbalance via negative sampling
            neg_pos_ratio = 10.0

            logits_flat = pred_logits.view(-1)
            labels_flat = scores_in_prepared.view(-1)

            pos_mask = labels_flat == 1
            neg_mask = labels_flat == 0

            num_pos = int(pos_mask.sum().item())
            num_neg = int(neg_mask.sum().item())

            if num_pos > 0 and num_neg > 0:
                pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
                neg_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)

                max_neg = min(int(neg_pos_ratio * num_pos), num_neg)
                if max_neg > 0:
                    perm = torch.randperm(num_neg, device=labels_flat.device)
                    chosen_neg = neg_idx[perm[:max_neg]]
                    chosen = torch.cat([pos_idx, chosen_neg], dim=0)
                else:
                    chosen = pos_idx

                loss = F.binary_cross_entropy_with_logits(
                    logits_flat[chosen], labels_flat[chosen]
                )
            else:
                # Fallback: use all samples
                loss = F.binary_cross_entropy_with_logits(logits_flat, labels_flat)

        else:  # regression
            loss = F.mse_loss(pred_logits, scores_in_prepared)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Post-process predictions for logging
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
            user_ids: Union[List, torch.LongTensor],
            items_sup: torch.Tensor,
            scores_sup: torch.Tensor,
            target_attr: torch.Tensor,
            lambda_attr: float = 1.0,
    ) -> dict:
        """
        Joint training step for Stage 2 (items) + Stage 3 (attributes).

        Simultaneously optimizes:
        - Item scoring loss (Stage 2)
        - User attribute prediction loss (Stage 3)

        Total loss = rating_loss + lambda_attr * attr_loss

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer for model parameters.
        user_ids : list or LongTensor [B]
            User identifiers.
        items_sup : Tensor [B, N_sup, D_in]
            Item features.
        scores_sup : Tensor [B, N_sup]
            Target scores (real scale).
        target_attr : Tensor [B] or [B, 1]
            User attribute targets.
        lambda_attr : float, default=1.0
            Weight for attribute loss relative to rating loss.

        Returns
        -------
        dict
            - 'loss_scaled': Total loss
            - 'loss_ratings': Rating prediction loss
            - 'loss_attr': Attribute prediction loss
            - 'pred_real_ratings': Item score predictions
            - 'target_ratings_real': Item score targets
            - 'pred_attr': Attribute predictions
            - 'target_attr': Attribute targets
        """
        if self.user_attr_head is None:
            raise RuntimeError(
                "UserAttrHead not initialized. Set is_user_attr_head=True."
            )

        user_idx = self._to_internal_ids(user_ids)

        self.train()

        # ===== Stage 2: Rating loss =====
        if self.item_task == "regression":
            self._check_normalizer_ready()

        scores_in_prepared = self.prepare_targets(scores_sup)

        optimizer.zero_grad(set_to_none=True)

        pred_logits_items = self.user_scorer(user_idx, items_sup)

        if self.item_task == "binary_classification":
            # Handle class imbalance
            neg_pos_ratio = 4

            logits_flat = pred_logits_items.view(-1)
            labels_flat = scores_in_prepared.view(-1)

            pos_mask = labels_flat == 1
            neg_mask = labels_flat == 0

            num_pos = int(pos_mask.sum().item())
            num_neg = int(neg_mask.sum().item())

            if num_pos > 0 and num_neg > 0:
                pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
                neg_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)

                max_neg = min(int(neg_pos_ratio * num_pos), num_neg)
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

        # ===== Stage 3: Attribute loss =====
        theta_hat, bias_hat = self.user_scorer.get_user_embeddings(user_idx)

        logits_attr = self.user_attr_head(theta_hat, bias_hat)
        B, A = logits_attr.shape

        if self.user_attr_task == "regression":
            # Regression
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
                # Binary
                y = target_attr.squeeze(-1) if target_attr.dim() > 1 else target_attr
                y = y.float()
                logits_flat = logits_attr.squeeze(1)
                attr_loss = F.binary_cross_entropy_with_logits(logits_flat, y)
                pred_attr = torch.sigmoid(logits_flat)
            else:
                # Multiclass
                y = (
                    target_attr.squeeze(-1).long()
                    if target_attr.dim() > 1
                    else target_attr.long()
                )
                attr_loss = F.cross_entropy(logits_attr, y)
                pred_attr = torch.softmax(logits_attr, dim=-1)

        # ===== Total loss and optimization =====
        total_loss = rating_loss + lambda_attr * attr_loss
        total_loss.backward()
        optimizer.step()

        # Post-process for logging
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

    # ============================================================
    # EVALUATION METHODS
    # ============================================================

    def evaluate_warm_user_cold_item(
            self,
            user_ids: Union[List, torch.LongTensor],
            items_qry: torch.Tensor,
            scores_qry: torch.Tensor,
            mask_qry: torch.Tensor,
    ) -> dict:
        """
        Evaluate Stage 2 on warm users with cold items (WU-CI).

        Parameters
        ----------
        user_ids : list or LongTensor [B]
            User identifiers.
        items_qry : Tensor [B, Q, D_in]
            Query item features.
        scores_qry : Tensor [B, Q]
            True scores (real scale).
        mask_qry : Tensor [B, Q]
            Item mask (0=pad, 1=warm item, 2=cold item, 3=query).

        Returns
        -------
        dict
            - 'loss_scaled': Loss value
            - 'pred_real': Predictions (real scale)
            - Plus task-specific metrics (MAE/MSE/Spearman or AUC/Accuracy)
        """
        user_idx = self._to_internal_ids(user_ids)

        self.eval()
        if self.item_task == "regression":
            self._check_normalizer_ready()

        with torch.no_grad():
            # Predict
            pred_logits = self.user_scorer(user_idx, items_qry)
            tgt_prepared = self.prepare_targets(scores_qry)

            # Compute loss
            if self.item_task == "binary_classification":
                loss_scaled = F.binary_cross_entropy_with_logits(
                    pred_logits, tgt_prepared
                )
            else:  # regression
                loss_scaled = F.mse_loss(pred_logits, tgt_prepared)

            # Post-process predictions
            pred_real = self.post_prepare_predictions(pred_logits)

            # Compute metrics
            if self.item_task == "binary_classification":
                metrics = binary_classification_metrics(pred_logits, tgt_prepared)
            else:  # regression
                metrics = regression_metrics(
                    pred_real, scores_qry, mask=mask_qry, classes=(3,)
                )
                # Add WU-CI specific metric aliases
                metrics["mae_WU_CI"] = metrics.get(
                    "mae_3", torch.tensor(float("nan"), device=pred_real.device)
                )
                metrics["mse_WU_CI"] = metrics.get(
                    "mse_3", torch.tensor(float("nan"), device=pred_real.device)
                )
                metrics["spearman_WU_CI"] = metrics.get(
                    "spearman_3", torch.tensor(float("nan"), device=pred_real.device)
                )

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
        Evaluate cold-user item prediction (CU-WI, CU-CI).

        Adapts user embeddings from support set, then evaluates on query items.
        Computes separate metrics for warm items (class 1) and cold items (class 2).

        Parameters
        ----------
        items_sup : Tensor [B, N_sup, D_in]
            Support set item features.
        scores_sup : Tensor [B, N_sup]
            Support set scores (real scale).
        items_qry : Tensor [B, Q, D_in]
            Query item features.
        scores_qry : Tensor [B, Q]
            True query scores (real scale).
        mask_qry : Tensor [B, Q]
            Query item mask (0=pad, 1=warm item, 2=cold item).

        Returns
        -------
        dict
            - 'pred_real': Predictions (real scale)
            - 'pred_logits': Raw model outputs
            - 'theta_hat': Adapted user embeddings
            - 'bias_hat': Adapted bias terms
            - 'loss_scaled': Loss value
            - 'loss_curve': Adaptation loss history
            - Plus task-specific metrics with CU-WI and CU-CI breakdowns
        """
        if self.item_task == "regression":
            self._check_normalizer_ready()
        self.eval()

        with torch.no_grad():
            scores_sup_prepared = self.prepare_targets(scores_sup)

        # Adapt user embeddings
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

        with torch.no_grad():
            # Score query items
            items_qry_tilde = self.user_scorer.compute_z_i_tilde(items_qry).detach()
            pred_logits = (
                    torch.einsum("bqd,bd->bq", items_qry_tilde, theta_hat) + bias_hat
            )

            tgt_prepared = self.prepare_targets(scores_qry)

            # Compute loss
            if self.item_task == "binary_classification":
                loss_scaled = F.binary_cross_entropy_with_logits(
                    pred_logits, tgt_prepared
                )
            else:  # regression
                loss_scaled = F.mse_loss(pred_logits, tgt_prepared)

            pred_real = self.post_prepare_predictions(pred_logits)

            # Compute metrics with class breakdown
            if self.item_task == "binary_classification":
                metrics = binary_classification_metrics(
                    logits=pred_logits,
                    labels=tgt_prepared,
                    mask=mask_qry,
                    classes=(1, 2),
                )
                # Add CU-WI / CU-CI aliases
                if "auc_1" in metrics:
                    metrics["auc_CU_WI"] = metrics["auc_1"]
                if "accuracy_1" in metrics:
                    metrics["accuracy_CU_WI"] = metrics["accuracy_1"]
                if "auc_2" in metrics:
                    metrics["auc_CU_CI"] = metrics["auc_2"]
                if "accuracy_2" in metrics:
                    metrics["accuracy_CU_CI"] = metrics["accuracy_2"]
            else:  # regression
                metrics = regression_metrics(
                    pred_real, scores_qry, mask_qry, classes=(1, 2)
                )
                # Add CU-WI / CU-CI aliases
                if "mae_1" in metrics:
                    metrics["mae_CU_WI"] = metrics["mae_1"]
                if "mse_1" in metrics:
                    metrics["mse_CU_WI"] = metrics["mse_1"]
                if "spearman_1" in metrics:
                    metrics["spearman_CU_WI"] = metrics["spearman_1"]
                if "mae_2" in metrics:
                    metrics["mae_CU_CI"] = metrics["mae_2"]
                if "mse_2" in metrics:
                    metrics["mse_CU_CI"] = metrics["mse_2"]
                if "spearman_2" in metrics:
                    metrics["spearman_CU_CI"] = metrics["spearman_2"]

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
        Evaluate cold-user attribute prediction.

        Adapts user embeddings from support set, then predicts user attributes.
        Evaluation metrics depend on user_attr_task (regression vs classification).

        Parameters
        ----------
        items_sup : Tensor [B, N_sup, D_in]
            Support set item features.
        scores_sup : Tensor [B, N_sup]
            Support set scores (real scale).
        target_attr : Tensor [B] or [B, 1]
            True user attribute values.

        Returns
        -------
        dict
            - 'theta_hat': Adapted user embeddings
            - 'bias_hat': Adapted bias terms
            - 'loss_curve': Adaptation loss history
            - For regression: 'pred', MAE, MSE, etc.
            - For classification: 'logits', 'probs', AUC, accuracy, etc.
        """
        assert self.user_attr_head is not None, "UserAttrHead required."

        if self.item_task == "regression":
            self._check_normalizer_ready()
        self.eval()

        # Adapt user embeddings from support set
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

        # Predict attributes
        with torch.no_grad():
            logits = self.user_attr_head(theta_hat, bias_hat)
        B, A = logits.shape

        out = {
            "theta_hat": theta_hat,
            "bias_hat": bias_hat,
            "loss_curve": torch.tensor(loss_curve, device=logits.device),
        }

        # Task-dependent evaluation
        if self.user_attr_task == "regression":
            pred = logits.squeeze(-1) if A == 1 else logits
            t = (
                target_attr.squeeze(-1)
                if (target_attr.dim() == 2 and target_attr.size(1) == 1)
                else target_attr
            )
            metrics = regression_metrics(pred, t, mask=None)
            out.update({"pred": pred, **metrics})
            return out

        # Classification path
        if A == 1:
            # Binary classification
            y = target_attr.squeeze(-1) if target_attr.dim() > 1 else target_attr
            metrics = binary_classification_metrics(logits.squeeze(1), y)
            out.update({"logits": logits.squeeze(1), **metrics})
        else:
            # Multiclass classification
            y = (
                target_attr.squeeze(-1).long()
                if target_attr.dim() > 1
                else target_attr.long()
            )
            metrics = multiclass_classification_metrics(logits, y)
            out.update({"logits": logits, **metrics})

        return out

