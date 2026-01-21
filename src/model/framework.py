"""
Unified Recommender System - Stage 2 & 3
A meta-learning framework integrating warm-user scoring and cold-start adaptation
for both item recommendation and user attribute prediction.
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, Any, Literal, List, Union, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Internal imports (assuming these are the anonymized versions)
from model.stage2.user_scorer import UserScorer
from model.stage2.cold_start_adapter import fit_cold_start_batch
from model.stage3.heads import UserAttrHead
from utils.metrics import regression_metrics, binary_classification_metrics

logger = logging.getLogger(__name__)


class UnifiedRecommender(nn.Module):
    """
    Integrates Stage 2 (UserScorer) and Stage 3 (Cold-start & Attributes).

    Supports:
    - Warm-user evaluation (via learned embeddings).
    - Cold-start adaptation (via local optimization on support sets).
    - Joint multi-task learning (ratings + user attributes).
    """

    def __init__(
            self,
            d_in: int,
            d_model: int,
            train_loader: Optional[DataLoader] = None,
            user_id_mapping: Optional[Dict[Any, int]] = None,
            user_id_key: str = 'user_id',
            mlp_hidden: Sequence[int] = (128,),
            mlp_dropout: float = 0.0,
            mlp_activation: str = "relu",
            use_bias: bool = True,
            mlp_use_batchnorm: bool = False,
            mlp_use_layernorm: bool = False,
            normalize_user: bool = True,
            normalize_item: bool = False,
            init_std: float = 0.02,
            item_task: Literal["regression", "binary_classification"] = "regression",
            binarization_threshold: Optional[float] = None,
            is_user_attr_head: bool = False,
            user_head_params: Optional[Dict[str, Any]] = None,
            cold_params: Optional[Dict[str, Any]] = None,
            scores_range: Optional[Tuple[float, float]] = None,
            user_attr_task: Literal["regression", "classification"] = "regression",
            dtype: torch.dtype = torch.float32,
            device: Optional[Union[torch.device, str]] = None,
    ):
        super().__init__()

        # 1. Initialize Identity and Mapping
        self._init_user_mapping(train_loader, user_id_mapping, user_id_key)

        # 2. Initialize Core Scorer (Stage 2)
        self.user_scorer = UserScorer(
            num_users=self.num_users,
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

        # 3. Configure Task Logic
        self.item_task = item_task
        self.binarization_threshold = binarization_threshold
        self.user_attr_task = user_attr_task
        self._init_label_normalization(scores_range)

        # 4. Cold-start Configuration
        cp = cold_params or {}
        self.cold_steps = cp.get("steps", 20)
        self.cold_lr = cp.get("lr", 1e-2)
        self.cold_wd = cp.get("wd", 0.0)
        self.cold_patience = cp.get("patience", 10)

        # 5. User Attribute Head (Stage 3)
        self.user_attr_head = None
        if is_user_attr_head:
            self._init_user_attr_head(d_model, user_head_params or {})

    def _init_user_mapping(self, loader, mapping, key):
        """Sets up the mapping from external user IDs to internal indices."""
        if loader is not None:
            logger.info("Extracting user IDs from DataLoader...")
            all_ids = set()
            for batch in loader:
                ids = batch[key]
                all_ids.update(ids.tolist() if isinstance(ids, torch.Tensor) else ids)
            self.user_id_mapping = {uid: i for i, uid in enumerate(sorted(all_ids))}
        elif mapping is not None:
            self.user_id_mapping = mapping
        else:
            raise ValueError("Must provide train_loader or user_id_mapping")

        self.reverse_mapping = {v: k for k, v in self.user_id_mapping.items()}
        self.num_users = len(self.user_id_mapping)
        logger.info(f"Initialized with {self.num_users} unique users.")

    def _init_label_normalization(self, scores_range):
        """Configures scaling for regression targets."""
        self.y_min = float(scores_range[0]) if scores_range else None
        self.y_max = float(scores_range[1]) if scores_range else None

        if self.item_task == "regression" and (self.y_min is None or self.y_max is None):
            logger.warning("Regression task enabled without scores_range. Ensure labels are pre-normalized.")

    def _init_user_attr_head(self, d_model, params):
        """Initializes the attribute prediction head."""
        from model.stage3.heads import UserAttrHead  # Lazy import
        self.user_attr_head = UserAttrHead(
            d_model=d_model,
            out_dim=params.get("out_dim", 1),
            hidden=params.get("hidden", (128, 64)),
            dropout=params.get("dropout", 0.0),
            activation=params.get("activation", "relu"),
            use_batchnorm=params.get("use_batchnorm", False),
            use_layernorm=params.get("use_layernorm", False)
        )

    def _to_internal_ids(self, user_ids: Union[List, torch.Tensor]) -> torch.LongTensor:
        """Translates external IDs to internal indices."""
        if isinstance(user_ids, torch.Tensor):
            return user_ids.long()  # Assume already internal if passed as tensor

        try:
            indices = [self.user_id_mapping[uid] for uid in user_ids]
            return torch.tensor(indices, dtype=torch.long)
        except KeyError as e:
            raise KeyError(f"User ID {e} not found in warm-user mapping. Use cold-start methods.")

    def prepare_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Normalizes or binarizes ground truth labels."""
        if self.item_task == "binary_classification":
            return (y >= self.binarization_threshold).float() if self.binarization_threshold is not None else y

        if self.y_min is not None and self.y_max is not None:
            # Map [min, max] -> [-1, 1]
            y_norm = (y - self.y_min) / (self.y_max - self.y_min)
            return y_norm * 2.0 - 1.0
        return y

    def post_prepare_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Converts internal model outputs to real-world scale/probabilities."""
        if self.item_task == "binary_classification":
            return torch.sigmoid(logits)

        if self.y_min is not None and self.y_max is not None:
            # Map [-1, 1] -> [min, max]
            y_01 = (logits + 1.0) * 0.5
            return y_01 * (self.y_max - self.y_min) + self.y_min
        return logits

    def _compute_balanced_loss(self, logits: torch.Tensor, targets: torch.Tensor, ratio: float = 4.0):
        """Unified helper for handling class imbalance in binary classification."""
        logits_flat = logits.view(-1)
        labels_flat = targets.view(-1)

        pos_mask = (labels_flat == 1)
        neg_mask = (labels_flat == 0)

        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())

        if n_pos > 0 and n_neg > 0:
            pos_idx = torch.nonzero(pos_mask).squeeze(1)
            neg_idx = torch.nonzero(neg_mask).squeeze(1)
            n_chosen_neg = min(int(ratio * n_pos), n_neg)

            chosen_neg = neg_idx[torch.randperm(n_neg, device=logits.device)[:n_chosen_neg]]
            indices = torch.cat([pos_idx, chosen_neg])
            return F.binary_cross_entropy_with_logits(logits_flat[indices], labels_flat[indices])

        return F.binary_cross_entropy_with_logits(logits_flat, labels_flat)

    def predict_warm_user(self, user_ids: List, items: torch.Tensor) -> torch.Tensor:
        """Standard inference for known users."""
        self.eval()
        u_idx = self._to_internal_ids(user_ids).to(items.device)
        with torch.no_grad():
            logits = self.user_scorer(u_idx, items)
            return self.post_prepare_predictions(logits)

    def predict_cold_user(self, items_sup: torch.Tensor, scores_sup: torch.Tensor, items_qry: torch.Tensor) -> Dict[
        str, torch.Tensor]:
        """Inference for new users using on-the-fly optimization."""
        self.eval()
        scores_sup_scaled = self.prepare_targets(scores_sup)

        theta_hat, bias_hat, _ = fit_cold_start_batch(
            user_scorer=self.user_scorer,
            items_sup=items_sup,
            scores_sup=scores_sup_scaled,
            steps=self.cold_steps,
            lr=self.cold_lr,
            weight_decay=self.cold_wd,
            patience=self.cold_patience,
            task_type=self.item_task
        )

        with torch.no_grad():
            items_qry_tilde = self.user_scorer.compute_transformed_items(items_qry)
            logits = torch.einsum("bqd,bd->bq", items_qry_tilde, theta_hat) + bias_hat
            return {
                "scores": self.post_prepare_predictions(logits),
                "theta": theta_hat,
                "bias": bias_hat
            }

    def train_step_joint(
            self,
            optimizer: torch.optim.Optimizer,
            user_ids: List,
            items_sup: torch.Tensor,
            scores_sup: torch.Tensor,
            target_attr: Optional[torch.Tensor] = None,
            lambda_attr: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Performs a joint training step for ratings and user attributes.
        """
        self.train()
        optimizer.zero_grad(set_to_none=True)
        u_idx = self._to_internal_ids(user_ids).to(items_sup.device)

        # 1. Rating Loss
        prep_scores = self.prepare_targets(scores_sup)
        logits_items = self.user_scorer(u_idx, items_sup)

        if self.item_task == "binary_classification":
            rating_loss = self._compute_balanced_loss(logits_items, prep_scores)
        else:
            rating_loss = F.mse_loss(logits_items, prep_scores)

        # 2. Attribute Loss (Optional)
        total_loss = rating_loss
        metrics = {"rating_loss": rating_loss.detach()}

        if target_attr is not None and self.user_attr_head is not None:
            theta, bias = self.user_scorer.get_user_embeddings_with_grad(u_idx)
            logits_attr = self.user_attr_head(theta, bias)

            if self.user_attr_task == "regression":
                attr_loss = F.mse_loss(logits_attr.squeeze(), target_attr.float())
            else:
                attr_loss = F.cross_entropy(logits_attr, target_attr.long())

            total_loss += lambda_attr * attr_loss
            metrics["attr_loss"] = attr_loss.detach()

        total_loss.backward()
        optimizer.step()

        metrics["total_loss"] = total_loss.detach()
        return metrics