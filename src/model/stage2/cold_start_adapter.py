# model/stage2/cold_start_adapter.py
"""
Cold-start adaptation module for UNISON Stage 2.

This module implements inference-time optimization for cold (unseen) bags by fitting
lightweight linear parameters (theta, bias) on a support set of known item-score pairs.
The optimization procedure is detailed in Theorem 2 of the paper, with convergence
guarantees for convex objectives.

Key features:
- Early stopping to prevent overfitting on small support sets
- Task-agnostic: supports both regression (MSE) and binary classification (BCE)
- Imbalanced data handling: automatic negative sampling for binary tasks
- Batch-wise optimization: processes multiple cold bags in parallel

Typical usage:
    theta_hat, bias_hat, loss_curve = fit_cold_start_batch(
        user_scorer=scorer,
        items_sup=support_items,
        scores_sup=support_scores,
        steps=100,
        lr=1e-2,
        weight_decay=0.05,
        patience=10,
        task_type="regression"
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def fit_cold_start_batch(
        user_scorer: nn.Module,
        items_sup: torch.Tensor,  # [B, N_sup, D_in]
        scores_sup: torch.Tensor,  # [B, N_sup]
        *,
        steps: int = 100,
        lr: float = 1e-2,
        weight_decay: float = 0.05,
        patience: int = 10,
        task_type: str = "regression",  # "regression" or "binary_classification"
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """
    Perform per-batch cold-start adaptation by optimizing linear scoring parameters.

    This function implements the inference-time optimization described in Section 3.3
    and Theorem 2. Given a support set of (item, score) pairs for unseen bags, it
    optimizes bag-specific parameters (theta, bias) while keeping the shared MLP frozen.

    The optimization minimizes:
    - Regression: MSE between predicted and true scores
    - Binary classification: BCE with automatic class imbalance correction

    Args:
        user_scorer: Stage 2 UserScorer module with frozen shared MLP (phi)
        items_sup: Support set item features [B, N_sup, D_in]
                   B = batch size (number of cold bags)
                   N_sup = support set size per bag
                   D_in = item embedding dimension
        scores_sup: Support set scores [B, N_sup]
                    For regression: continuous values
                    For binary_classification: binary labels {0, 1}
        steps: Maximum number of optimization iterations
        lr: Learning rate for Adam optimizer
        weight_decay: L2 regularization coefficient for (theta, bias)
        patience: Early stopping patience (stops if no improvement for this many steps)
        task_type: "regression" or "binary_classification"

    Returns:
        theta_hat: Optimized linear weights [B, D_model] (best checkpoint)
        bias_hat: Optimized bias terms [B, 1] (best checkpoint)
        loss_history: Training loss curve (length <= steps due to early stopping)

    Notes:
        - The shared MLP projection (phi) is frozen during optimization
        - Early stopping prevents overfitting on small support sets (Theorem 2)
        - For binary classification, automatic negative downsampling addresses class imbalance
    """
    device = items_sup.device
    dtype = items_sup.dtype

    # Step 1: Project support items through frozen shared MLP to get z_tilde
    # This is the representation space where linear scoring is applied (Eq. 5 in paper)
    with torch.no_grad():
        z_sup_tilde = user_scorer.compute_z_i_tilde(items_sup)  # [B, N_sup, D_model]

    # Detach from computation graph but enable gradient tracking for optimization
    z_sup_tilde = z_sup_tilde.detach().requires_grad_(True)
    B, N, Dm = z_sup_tilde.shape

    # Step 2: Initialize fresh trainable parameters for each bag in the batch
    # Small random init (0.02 std) helps with convergence stability
    theta = nn.Parameter(torch.randn(B, Dm, device=device, dtype=dtype) * 0.02)
    bias = nn.Parameter(torch.zeros(B, 1, device=device, dtype=dtype))

    # Step 3: Set up optimizer with weight decay for implicit regularization
    optim = torch.optim.Adam([theta, bias], lr=lr, weight_decay=weight_decay)

    # Early stopping state: track best parameters to avoid overfitting
    best_loss = float("inf")
    best_theta = theta.detach().clone()
    best_bias = bias.detach().clone()
    no_improve = 0

    loss_history: list[float] = []

    # Step 4: Optimization loop with early stopping
    for t in range(steps):
        optim.zero_grad(set_to_none=True)

        # Compute predictions: score = theta^T * z_tilde + bias (Eq. 5)
        # Einstein summation: [B,N,D] x [B,D] -> [B,N]
        pred = torch.einsum("bnd,bd->bn", z_sup_tilde, theta) + bias  # [B, N]

        if task_type == "binary_classification":
            # ========== Imbalanced Binary Classification Handling ==========
            # Problem: Real-world scored bags often have severe class imbalance
            # (e.g., many negative items, few positive items in TCR repertoires)
            #
            # Solution: For each batch, include ALL positive examples but downsample
            # negatives to maintain a controlled pos:neg ratio (default 1:4).
            # This prevents the optimizer from learning a trivial "always predict negative"
            # solution while still exposing the model to sufficient negative examples.
            #
            # Implementation:
            # 1. Identify all positive and negative indices in the flattened batch
            # 2. Keep all positives
            # 3. Randomly sample up to (NEG_POS_RATIO * num_positives) negatives
            # 4. Compute BCE only on this balanced subset

            NEG_POS_RATIO = 4  # Allow up to 4x negatives per positive

            logits_flat = pred.view(-1)  # [B*N]
            labels_flat = scores_sup.view(-1)  # [B*N], binary labels {0, 1}

            # Separate positive and negative indices
            pos_mask = (labels_flat == 1)
            neg_mask = (labels_flat == 0)

            num_pos = int(pos_mask.sum().item())
            num_neg = int(neg_mask.sum().item())

            # Only apply downsampling if both classes are present
            if num_pos > 0 and num_neg > 0:
                pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
                neg_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)

                # Compute max negatives to sample (capped by available negatives)
                max_neg = min(int(NEG_POS_RATIO * num_pos), num_neg)
                if max_neg > 0:
                    # Random permutation for unbiased negative sampling
                    perm = torch.randperm(num_neg, device=labels_flat.device)
                    chosen_neg = neg_idx[perm[:max_neg]]
                    chosen = torch.cat([pos_idx, chosen_neg], dim=0)
                else:
                    # Edge case: only positives (shouldn't happen with typical data)
                    chosen = pos_idx

                loss = F.binary_cross_entropy_with_logits(
                    logits_flat[chosen], labels_flat[chosen]
                )
            else:
                # Fallback: no positives or no negatives in this batch
                # Use full BCE (degenerates to trivial solution, but rare in practice)
                loss = F.binary_cross_entropy_with_logits(
                    logits_flat, labels_flat
                )
        else:  # regression
            # Standard MSE loss for continuous scores
            loss = F.mse_loss(pred, scores_sup)

        # Backpropagate and update parameters
        loss.backward()
        optim.step()

        curr = loss.item()
        loss_history.append(curr)

        # Early stopping: track best checkpoint and stop if no improvement
        if curr < best_loss - 1e-8:  # Small margin to avoid numerical noise
            best_loss = curr
            best_theta = theta.detach().clone()
            best_bias = bias.detach().clone()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                # No improvement for 'patience' steps -> stop early
                break

    # Return best parameters (not final parameters) to avoid overfitting
    return best_theta, best_bias, loss_history

# Example usage:
# theta_hat, bias_hat, curve = fit_cold_start_batch(
#     user_scorer=scorer,
#     items_sup=items_sup,
#     scores_sup=scores_sup,
#     steps=100,
#     lr=1e-2,
#     weight_decay=0.05,
#     patience=10,
#     task_type="regression"
# )