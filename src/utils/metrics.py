"""
UNISON Framework - Evaluation Metrics
Provides masked evaluation for regression and classification tasks,
supporting per-group analysis (e.g., Warm-Item vs. Cold-Item).
"""

from __future__ import annotations
import logging
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


@torch.no_grad()
def _spearman_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes Spearman's Rank Correlation Coefficient using double-argsort.
    Fast PyTorch implementation suitable for GPU/CPU.
    """
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()

    if x.numel() < 2:
        return torch.tensor(float("nan"), device=x.device)

    # Double argsort provides the rank of each element
    rx = torch.argsort(torch.argsort(x)).float()
    ry = torch.argsort(torch.argsort(y)).float()

    # Center the ranks
    rx = rx - rx.mean()
    ry = ry - ry.mean()

    std_x = rx.std(unbiased=False)
    std_y = ry.std(unbiased=False)

    if std_x <= 0 or std_y <= 0:
        return torch.tensor(float("nan"), device=x.device)

    return (rx * ry).mean() / (std_x * std_y)


@torch.no_grad()
def regression_metrics(
        pred_real: torch.Tensor,
        target_real: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        classes: Iterable[int] = (1, 2, 3),
) -> Dict[str, torch.Tensor]:
    """
    Computes regression metrics (MAE, MSE, Spearman) with group-aware masking.

    Args:
        pred_real: Predicted values in original scale.
        target_real: Ground truth values in original scale.
        mask: Tensor indicating groups (0=Ignore/Padding, 1=Group A, 2=Group B, etc.)
        classes: The group IDs to calculate specific metrics for.
    """
    assert pred_real.shape == target_real.shape
    device = pred_real.device
    results: Dict[str, torch.Tensor] = {}

    # Define valid indices (ignore padding)
    valid_mask = (mask != 0) if mask is not None else torch.ones_like(pred_real, dtype=torch.bool)

    # Global Metrics
    if valid_mask.any():
        p_valid = pred_real[valid_mask]
        t_valid = target_real[valid_mask]
        results["mae"] = (p_valid - t_valid).abs().mean()
        results["mse"] = (p_valid - t_valid).pow(2).mean()
        results["spearman"] = _spearman_1d(p_valid, t_valid)
    else:
        nan_val = torch.tensor(float("nan"), device=device)
        results.update({"mae": nan_val, "mse": nan_val, "spearman": nan_val})

    # Per-Group Metrics (e.g., Cold-Start vs. Warm-Start)
    if mask is not None:
        for c in classes:
            group_mask = (mask == int(c))
            if group_mask.any():
                p_group = pred_real[group_mask]
                t_group = target_real[group_mask]
                results[f"mae_{c}"] = (p_group - t_group).abs().mean()
                results[f"mse_{c}"] = (p_group - t_group).pow(2).mean()
                results[f"spearman_{c}"] = _spearman_1d(p_group, t_group)
            else:
                nan_val = torch.tensor(float("nan"), device=device)
                results.update({f"mae_{c}": nan_val, f"mse_{c}": nan_val, f"spearman_{c}": nan_val})

    return results


@torch.no_grad()
def binary_classification_metrics(
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        classes: Iterable[int] = (),
) -> Dict[str, torch.Tensor]:
    """
    Computes binary classification metrics including AUC and Accuracy.
    """
    assert logits.shape == labels.shape
    device = logits.device
    labels = labels.float()
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()

    valid_mask = (mask != 0) if mask is not None else torch.ones_like(labels, dtype=torch.bool)
    results: Dict[str, torch.Tensor] = {"probs": probs, "preds": preds}

    if valid_mask.any():
        l_v, t_v, p_v = logits[valid_mask], labels[valid_mask], probs[valid_mask]

        results["loss_bce"] = F.binary_cross_entropy_with_logits(l_v, t_v)
        results["accuracy"] = (preds[valid_mask] == t_v.long()).float().mean()

        # AUC calculation (requires both classes to be present)
        try:
            y_true = t_v.cpu().numpy().flatten()
            y_pred = p_v.cpu().numpy().flatten()
            results["auc"] = torch.tensor(roc_auc_score(y_true, y_pred), device=device)
        except Exception:
            results["auc"] = torch.tensor(0.5, device=device)
    else:
        nan_val = torch.tensor(float("nan"), device=device)
        results.update({"loss_bce": nan_val, "accuracy": nan_val, "auc": nan_val})

    # Per-Group Classification Metrics
    if mask is not None and classes:
        for c in classes:
            group_sel = (mask.view(-1) == int(c))
            if group_sel.any():
                t_g = labels.view(-1)[group_sel]
                p_g = probs.view(-1)[group_sel]

                results[f"accuracy_{c}"] = (preds.view(-1)[group_sel] == t_g.long()).float().mean()
                try:
                    results[f"auc_{c}"] = torch.tensor(roc_auc_score(t_g.cpu(), p_g.cpu()), device=device)
                except Exception:
                    results[f"auc_{c}"] = torch.tensor(float("nan"), device=device)
            else:
                nan_val = torch.tensor(float("nan"), device=device)
                results.update({f"accuracy_{c}": nan_val, f"auc_{c}": nan_val})

    return results


@torch.no_grad()
def multiclass_classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Standard multi-class evaluation."""
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)

    return {
        "loss_ce": F.cross_entropy(logits, labels),
        "accuracy": (preds == labels).float().mean(),
        "probs": probs,
        "preds": preds,
    }