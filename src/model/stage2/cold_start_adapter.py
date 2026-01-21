"""
Cold Start Adapter - Stage 2
Handles per-batch optimization for new users (Cold Start) by fitting
a user embedding and bias to a small set of support items.
"""

from __future__ import annotations
import logging
from typing import List, Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def fit_cold_start_batch(
        user_scorer: nn.Module,
        items_sup: torch.Tensor,
        scores_sup: torch.Tensor,
        *,
        steps: int = 100,
        lr: float = 1e-2,
        weight_decay: float = 0.05,
        patience: int = 10,
        task_type: Literal["regression", "binary_classification"] = "regression",
        neg_pos_ratio: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Performs a local optimization (fine-tuning) to estimate user parameters
    for a cold-start batch.

    Args:
        user_scorer: The UserScorer model used to project items.
        items_sup: Support items tensor [Batch, N_support, D_in].
        scores_sup: Ground truth labels/scores [Batch, N_support].
        steps: Maximum number of gradient descent steps.
        lr: Learning rate for the local Adam optimizer.
        weight_decay: L2 penalty for the learned parameters.
        patience: Early stopping patience.
        task_type: Type of loss function to use.
        neg_pos_ratio: Ratio of negative to positive samples for imbalanced classification.

    Returns:
        A tuple containing:
            - best_theta: Optimized user embeddings [Batch, D_model].
            - best_bias: Optimized user biases [Batch, 1].
            - loss_history: List of loss values recorded during optimization.
    """
    device = items_sup.device
    dtype = items_sup.dtype

    # 1. Project support items into the shared latent space
    # We use compute_transformed_items (updated name from previous step)
    with torch.no_grad():
        # Note: calling the method we renamed in the previous file
        z_sup_tilde = user_scorer.compute_transformed_items(items_sup)

    z_sup_tilde = z_sup_tilde.detach().requires_grad_(False)
    batch_size, num_support, d_model = z_sup_tilde.shape

    # 2. Initialize fresh parameters for this specific batch
    # Local parameters are optimized to minimize error on the support set
    theta = nn.Parameter(torch.randn(batch_size, d_model, device=device, dtype=dtype) * 0.02)
    bias = nn.Parameter(torch.zeros(batch_size, 1, device=device, dtype=dtype))

    optimizer = torch.optim.Adam([theta, bias], lr=lr, weight_decay=weight_decay)

    # State tracking for early stopping
    best_loss = float("inf")
    best_theta = theta.detach().clone()
    best_bias = bias.detach().clone()
    no_improve_counter = 0
    loss_history: List[float] = []

    for t in range(steps):
        optimizer.zero_grad(set_to_none=True)

        # Compute predicted scores: <z_sup_tilde, theta> + bias
        # (B, N, D) x (B, D) -> (B, N)
        predictions = torch.einsum("bnd,bd->bn", z_sup_tilde, theta) + bias

        if task_type == "binary_classification":
            loss = _compute_balanced_bce(predictions, scores_sup, neg_pos_ratio)
        else:
            loss = F.mse_loss(predictions, scores_sup)

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        loss_history.append(current_loss)

        # Early stopping logic
        if current_loss < best_loss - 1e-8:
            best_loss = current_loss
            best_theta = theta.detach().clone()
            best_bias = bias.detach().clone()
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            if no_improve_counter >= patience:
                break

    return best_theta, best_bias, loss_history


def _compute_balanced_bce(
        logits: torch.Tensor,
        labels: torch.Tensor,
        neg_pos_ratio: int
) -> torch.Tensor:
    """Helper to compute BCE loss with negative downsampling to handle imbalance."""
    logits_flat = logits.view(-1)
    labels_flat = labels.view(-1)

    pos_mask = (labels_flat == 1)
    neg_mask = (labels_flat == 0)

    num_pos = int(pos_mask.sum().item())
    num_neg = int(neg_mask.sum().item())

    if num_pos > 0 and num_neg > 0:
        pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
        neg_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)

        max_neg = min(int(neg_pos_ratio * num_pos), num_neg)
        if max_neg > 0:
            perm = torch.randperm(num_neg, device=labels_flat.device)
            chosen_neg = neg_idx[perm[:max_neg]]
            indices = torch.cat([pos_idx, chosen_neg], dim=0)
            return F.binary_cross_entropy_with_logits(logits_flat[indices], labels_flat[indices])

    # Fallback for cases with no positives or negatives
    return F.binary_cross_entropy_with_logits(logits_flat, labels_flat)