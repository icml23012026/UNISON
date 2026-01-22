# model/stage2/cold_start_adapter.py
# ------------------------------------------------------------
# Cold-start adaptation for new users.
#
# This module provides functionality to adapt the warm-user scorer
# to new (cold-start) users by fitting user embeddings on a small
# support set of labeled items.
#
# The adaptation process optimizes user-specific parameters (θ, b)
# to minimize prediction loss on the support set, with early stopping
# to prevent overfitting.
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


def fit_cold_start_batch(
    user_scorer: nn.Module,
    items_sup: torch.Tensor,
    scores_sup: torch.Tensor,
    *,
    steps: int = 100,
    lr: float = 1e-2,
    weight_decay: float = 0.05,
    patience: int = 10,
    task_type: str = "regression",
    neg_pos_ratio: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """
    Fit user embeddings for cold-start users using a support set.

    This function adapts a trained UserScorer to new users by optimizing
    user-specific parameters (θ, b) on a small labeled support set. The
    optimization uses gradient descent with early stopping.

    For each user in the batch, we learn:
        - θ ∈ R^{d_model}: user embedding vector
        - b ∈ R: scalar bias term

    These parameters are optimized to minimize prediction loss on the
    support items for that user.

    Parameters
    ----------
    user_scorer : nn.Module
        Trained UserScorer module (must have `compute_z_i_tilde` method).
    items_sup : Tensor [B, N_sup, D_in]
        Support set item features for B cold-start users.
    scores_sup : Tensor [B, N_sup]
        Target scores/labels for the support items.
        - For regression: continuous values
        - For classification: binary labels (0 or 1)
    steps : int, default=100
        Maximum number of optimization steps.
    lr : float, default=1e-2
        Learning rate for Adam optimizer.
    weight_decay : float, default=0.05
        L2 regularization strength (prevents overfitting on small support sets).
    patience : int, default=10
        Early stopping patience. Training stops if loss doesn't improve
        for this many consecutive steps.
    task_type : str, default='regression'
        Task formulation. Options:
        - 'regression': Minimize MSE loss
        - 'binary_classification': Minimize BCE loss with logits
    neg_pos_ratio : int, default=4
        For binary classification: maximum ratio of negative to positive
        samples used in each training step. Helps handle class imbalance.

    Returns
    -------
    theta_hat : Tensor [B, d_model]
        Learned user embeddings for the cold-start users.
    bias_hat : Tensor [B, 1]
        Learned bias terms for the cold-start users.
    loss_history : list[float]
        Training loss at each step (length <= steps due to early stopping).

    Notes
    -----
    - The function uses early stopping to prevent overfitting, which is
      crucial when adapting on small support sets (e.g., 5-20 items).
    - For binary classification, negative sampling is used to handle
      class imbalance in the support set.
    - The learned parameters (θ, b) can be used with the UserScorer
      to score query items for the cold-start users.

    Examples
    --------
    >>> # Adapt to 4 cold-start users with 10 support items each
    >>> items_sup = torch.randn(4, 10, 64)      # [B, N_sup, D_in]
    >>> scores_sup = torch.randn(4, 10)         # [B, N_sup]
    >>> theta_hat, bias_hat, loss_curve = fit_cold_start_batch(
    ...     user_scorer=scorer,
    ...     items_sup=items_sup,
    ...     scores_sup=scores_sup,
    ...     steps=100,
    ...     lr=1e-2,
    ...     patience=10,
    ...     task_type='regression'
    ... )
    >>> theta_hat.shape
    torch.Size([4, 128])
    """
    device = items_sup.device
    dtype = items_sup.dtype

    # Transform support items through the scorer's MLP
    with torch.no_grad():
        z_sup_tilde = user_scorer.compute_z_i_tilde(items_sup)  # [B, N_sup, D_model]

    # Detach from computation graph (we only optimize θ and b, not the MLP)
    z_sup_tilde = z_sup_tilde.detach()
    B, N_sup, D_model = z_sup_tilde.shape

    # Initialize learnable parameters for this batch of cold-start users
    theta = nn.Parameter(torch.randn(B, D_model, device=device, dtype=dtype) * 0.02)
    bias = nn.Parameter(torch.zeros(B, 1, device=device, dtype=dtype))

    # Optimizer for the user parameters
    optim = torch.optim.Adam([theta, bias], lr=lr, weight_decay=weight_decay)

    # Early stopping state
    best_loss = float("inf")
    best_theta = theta.detach().clone()
    best_bias = bias.detach().clone()
    no_improve = 0

    loss_history: list[float] = []

    for step in range(steps):
        optim.zero_grad(set_to_none=True)

        # Compute predictions: <θ, z_tilde> + b
        # Einstein sum: [B,N,D] × [B,D] -> [B,N]
        pred = torch.einsum("bnd,bd->bn", z_sup_tilde, theta) + bias

        # Compute loss based on task type
        if task_type == "binary_classification":
            # Handle class imbalance via negative sampling
            logits_flat = pred.view(-1)  # [B*N_sup]
            labels_flat = scores_sup.view(-1)  # [B*N_sup]

            pos_mask = labels_flat == 1
            neg_mask = labels_flat == 0

            num_pos = int(pos_mask.sum().item())
            num_neg = int(neg_mask.sum().item())

            # Sample negatives to balance the classes
            if num_pos > 0 and num_neg > 0:
                pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
                neg_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)

                # Use at most neg_pos_ratio × num_positives negatives
                max_neg = min(neg_pos_ratio * num_pos, num_neg)
                if max_neg > 0:
                    perm = torch.randperm(num_neg, device=device)
                    chosen_neg = neg_idx[perm[:max_neg]]
                    chosen = torch.cat([pos_idx, chosen_neg], dim=0)
                else:
                    # Edge case: only positives available
                    chosen = pos_idx

                loss = F.binary_cross_entropy_with_logits(
                    logits_flat[chosen], labels_flat[chosen]
                )
            else:
                # Fallback: use all samples if no positives or no negatives
                loss = F.binary_cross_entropy_with_logits(logits_flat, labels_flat)

        else:  # regression
            loss = F.mse_loss(pred, scores_sup)

        loss.backward()
        optim.step()

        curr_loss = loss.item()
        loss_history.append(curr_loss)

        # Early stopping: keep best parameters seen so far
        if curr_loss < best_loss - 1e-8:  # Small margin to avoid numerical noise
            best_loss = curr_loss
            best_theta = theta.detach().clone()
            best_bias = bias.detach().clone()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                # No improvement for 'patience' steps → stop early
                break

    return best_theta, best_bias, loss_history


# Example usage:
# >>> theta_hat, bias_hat, loss_curve = fit_cold_start_batch(
# ...     user_scorer=scorer,
# ...     items_sup=support_items,      # [B, N_sup, D_in]
# ...     scores_sup=support_scores,    # [B, N_sup]
# ...     steps=100,
# ...     lr=1e-2,
# ...     weight_decay=0.05,
# ...     patience=10,
# ...     task_type='regression'
# ... )