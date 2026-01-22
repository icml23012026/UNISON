"""
UNISON Framework - Training Module

Orchestrates training and evaluation for:
- Stage 2: Preference Modeling (warm users on items)
- Stage 3: Bag Classification (cold users and attribute prediction)

Evaluation scenarios:
- WU-CI: Warm User, Cold Item
- CU-WI: Cold User, Warm Item
- CU-CI: Cold User, Cold Item
- CU-Attr: Cold User, Attribute Prediction
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Framework Imports
from src.data_prep.bag_loader_book_crossing import (
    BookCrossingBagDataset,
    collate_bags,
)
from src.model.framework import Unison


def load_config(path="configs/config_book_crossing.json"):
    """
    Load experiment configuration from JSON file.

    Parameters
    ----------
    path : str
        Path to configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(path, "r") as f:
        return json.load(f)


def log_epoch(epoch, tr_metrics, wu_metrics, cu_metrics):
    """
    Print epoch metrics in a clean, structured format.

    Parameters
    ----------
    epoch : int
        Current epoch number.
    tr_metrics : dict
        Training metrics.
    wu_metrics : dict
        Warm user evaluation metrics.
    cu_metrics : dict
        Cold user evaluation metrics.
    """
    print(f"\n{'=' * 60}")
    print(f"EPOCH {epoch:03d}")
    print(f"{'=' * 60}")

    # Training metrics
    print(f"\n[TRAINING]")
    print(f"  Total Loss:   {tr_metrics['loss']:.4f}")
    print(f"  Rating Loss:  {tr_metrics['loss_ratings']:.4f}")
    print(f"  Attr Loss:    {tr_metrics.get('loss_attr', 0):.4f}")
    if 'attr_acc' in tr_metrics:
        print(f"  Attr Acc:     {tr_metrics['attr_acc']:.2%}")

    # Stage 2: Warm User - Cold Item
    print(f"\n[STAGE 2] Item Scoring")
    print(f"  WU-CI MAE:          {wu_metrics.get('mae', 0):.4f}")
    print(f"  WU-CI Spearman:     {wu_metrics.get('spearman', 0):.4f}")
    print(f"  CU-WI MAE:          {cu_metrics.get('mae_wi', 0):.4f}")
    print(f"  CU-WI Spear:        {cu_metrics.get('spearman_wi', 0):.4f}")
    print(f"  CU-CI MAE:          {cu_metrics.get('mae_ci', 0):.4f}")
    print(f"  CU-CI Spear:        {cu_metrics.get('spearman_ci', 0):.4f}")

    # Stage 3: Cold User - Items & Attributes
    print(f"\n[STAGE 3: COLD USER Attribute]")
    print(f"  Attr AUC:     {cu_metrics.get('attr_auc', 0):.4f}")
    print(f"  Attr Acc:     {cu_metrics.get('attr_acc', 0):.2%}")

    print(f"\n{'=' * 60}\n")


def train_epoch(model, loader, optimizer, config, device):
    """
    Execute one training epoch for Stage 2 + Stage 3.

    Jointly optimizes:
    - User embeddings and item scoring (Stage 2)
    - User attribute prediction head (Stage 3)

    Parameters
    ----------
    model : UNISON
        Model instance.
    loader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer.
    config : dict
        Configuration dictionary.
    device : torch.device
        Device for computation.

    Returns
    -------
    dict
        Aggregated training metrics.
    """
    model.train()
    metrics = {
        "loss": 0,
        "loss_ratings": 0,
        "loss_attr": 0,
        "attr_acc": [],
    }

    for batch in tqdm(loader, desc="Training", leave=False):
        # Joint Stage 2 + Stage 3 training step
        res = model.train_step_warm_user_item_and_attribute(
            optimizer=optimizer,
            user_ids=batch["id"],
            items_sup=batch["items_sup"].to(device),
            scores_sup=batch["scores_sup"].to(device),
            target_attr=batch["target_attr"].to(device),
            lambda_attr=config["model"]["lambda_attr"],
        )

        metrics["loss"] += res["loss_scaled"].item()
        metrics["loss_ratings"] += res["loss_ratings"].item()
        metrics["loss_attr"] += res["loss_attr"].item()

        # Compute attribute accuracy (binary classification)
        if "pred_attr" in res:
            preds = (res["pred_attr"] >= 0.5).float()
            acc = (preds == batch["target_attr"].to(device)).float().mean().item()
            metrics["attr_acc"].append(acc)

    # Average over batches
    metrics["loss"] /= len(loader)
    metrics["loss_ratings"] /= len(loader)
    metrics["loss_attr"] /= len(loader)
    metrics["attr_acc"] = np.mean(metrics["attr_acc"]) if metrics["attr_acc"] else 0

    return metrics


def evaluate_warm_user(model, loader, device):
    """
    Evaluate Stage 2: Warm User, Cold Item (WU-CI).

    Uses learned user embeddings to score items not seen during training.
    Computes global metrics across all items (not averaged per-user).

    Parameters
    ----------
    model : UNISON
        Model instance.
    loader : DataLoader
        Evaluation data loader.
    device : torch.device
        Device for computation.

    Returns
    -------
    dict
        WU-CI metrics (MAE, Spearman).
    """
    model.eval()

    all_preds_ci = []
    all_targets_ci = []

    for batch in tqdm(loader, desc="Eval WU-CI", leave=False):
        with torch.no_grad():
            # Evaluate using the model's built-in evaluation method
            res = model.evaluate_warm_user_cold_item(
                user_ids=batch["id"],
                items_qry=batch["items_qry"].to(device),
                scores_qry=batch["scores_qry"].to(device),
                mask_qry=batch["mask_qry"].to(device),
            )

            preds = res["pred_real"]  # [B, Q]
            targets = batch["scores_qry"].to(device)  # [B, Q]
            masks = batch["mask_qry"].to(device)  # [B, Q]

            # Extract only cold items (mask == 3 for WU-CI)
            mask_ci = masks == 3
            if mask_ci.any():
                all_preds_ci.append(preds[mask_ci])
                all_targets_ci.append(targets[mask_ci])

    # Concatenate all predictions and targets into single vectors
    if all_preds_ci:
        all_preds_ci = torch.cat(all_preds_ci, dim=0)  # [N_total_items]
        all_targets_ci = torch.cat(all_targets_ci, dim=0)  # [N_total_items]

        # Global MAE
        mae = torch.abs(all_preds_ci - all_targets_ci).mean().item()

        # Global Spearman correlation
        if len(all_preds_ci) > 1:
            try:
                from scipy.stats import spearmanr
                corr, _ = spearmanr(
                    all_preds_ci.cpu().numpy(),
                    all_targets_ci.cpu().numpy()
                )
                spearman = corr if not np.isnan(corr) else 0.0
            except:
                spearman = 0.0
        else:
            spearman = 0.0
    else:
        mae = 0.0
        spearman = 0.0

    return {
        "mae": mae,
        "spearman": spearman,
    }


def evaluate_cold_user(model, loader, device):
    """
    Evaluate Stage 3: Cold User scenarios.

    Evaluates:
    - CU-WI: Cold User, Warm Item (items seen during training)
    - CU-CI: Cold User, Cold Item (items not seen during training)
    - CU-Attr: Cold User, Attribute Prediction

    Computes global metrics across all items (not averaged per-user).

    Parameters
    ----------
    model : UNISON
        Model instance.
    loader : DataLoader
        Evaluation data loader.
    device : torch.device
        Device for computation.

    Returns
    -------
    dict
        Cold user metrics (MAE, Spearman for WI/CI, AUC/Acc for attributes).
    """
    model.eval()

    # Storage for item prediction metrics
    all_preds_wi = []
    all_targets_wi = []
    all_preds_ci = []
    all_targets_ci = []

    # Storage for attribute prediction metrics
    all_attr_preds = []
    all_attr_targets = []

    for batch in tqdm(loader, desc="Eval CU", leave=False):

        # Evaluate item prediction (CU-WI, CU-CI)
        res_items = model.evaluate_cold_user_items(
            items_sup=batch["items_sup"].to(device),
            scores_sup=batch["scores_sup"].to(device),
            items_qry=batch["items_qry"].to(device),
            scores_qry=batch["scores_qry"].to(device),
            mask_qry=batch["mask_qry"].to(device),
        )

        preds = res_items["pred_real"]  # [B, Q]
        targets = batch["scores_qry"].to(device)  # [B, Q]
        masks = batch["mask_qry"].to(device)  # [B, Q]

        # Collect warm items (mask == 1)
        mask_wi = masks == 1
        if mask_wi.any():
            all_preds_wi.append(preds[mask_wi])
            all_targets_wi.append(targets[mask_wi])

        # Collect cold items (mask == 2)
        mask_ci = masks == 2
        if mask_ci.any():
            all_preds_ci.append(preds[mask_ci])
            all_targets_ci.append(targets[mask_ci])

        # Evaluate attribute prediction (CU-Attr)
        res_attr = model.evaluate_cold_user_attribute(
            items_sup=batch["items_sup"].to(device),
            scores_sup=batch["scores_sup"].to(device),
            target_attr=batch["target_attr"].to(device),
        )

        # Extract probabilities or logits for AUC computation
        if "probs" in res_attr:
            all_attr_preds.append(res_attr["probs"])
        else:
            # Binary classification: apply sigmoid to logits
            all_attr_preds.append(torch.sigmoid(res_attr["logits"]))

        all_attr_targets.append(batch["target_attr"].to(device))

    # ===== Compute CU-WI metrics =====
    if all_preds_wi:
        all_preds_wi = torch.cat(all_preds_wi, dim=0)
        all_targets_wi = torch.cat(all_targets_wi, dim=0)

        mae_wi = torch.abs(all_preds_wi - all_targets_wi).mean().item()

        if len(all_preds_wi) > 1:
            try:
                from scipy.stats import spearmanr
                corr, _ = spearmanr(
                    all_preds_wi.cpu().numpy(),
                    all_targets_wi.cpu().numpy()
                )
                spearman_wi = corr if not np.isnan(corr) else 0.0
            except:
                spearman_wi = 0.0
        else:
            spearman_wi = 0.0
    else:
        mae_wi = 0.0
        spearman_wi = 0.0

    # ===== Compute CU-CI metrics =====
    if all_preds_ci:
        all_preds_ci = torch.cat(all_preds_ci, dim=0)
        all_targets_ci = torch.cat(all_targets_ci, dim=0)

        mae_ci = torch.abs(all_preds_ci - all_targets_ci).mean().item()

        if len(all_preds_ci) > 1:
            try:
                from scipy.stats import spearmanr
                corr, _ = spearmanr(
                    all_preds_ci.cpu().numpy(),
                    all_targets_ci.cpu().numpy()
                )
                spearman_ci = corr if not np.isnan(corr) else 0.0
            except:
                spearman_ci = 0.0
        else:
            spearman_ci = 0.0
    else:
        mae_ci = 0.0
        spearman_ci = 0.0

    # ===== Compute attribute metrics =====
    attr_preds = torch.cat(all_attr_preds, dim=0)
    attr_targets = torch.cat(all_attr_targets, dim=0)

    # Flatten if needed
    if attr_preds.dim() > 1 and attr_preds.shape[1] == 1:
        attr_preds = attr_preds.squeeze(1)
    if attr_targets.dim() > 1:
        attr_targets = attr_targets.squeeze(1)

    # Accuracy
    attr_preds_binary = (attr_preds >= 0.5).float()
    attr_acc = (attr_preds_binary == attr_targets).float().mean().item()

    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        attr_auc = roc_auc_score(
            attr_targets.cpu().numpy(),
            attr_preds.cpu().numpy()
        )
    except:
        # Fallback: Mann-Whitney approximation
        pos = attr_preds[attr_targets == 1]
        neg = attr_preds[attr_targets == 0]
        if pos.numel() > 0 and neg.numel() > 0:
            attr_auc = (pos[:, None] > neg[None, :]).float().mean().item()
        else:
            attr_auc = 0.0

    return {
        "mae_wi": mae_wi,
        "spearman_wi": spearman_wi,
        "mae_ci": mae_ci,
        "spearman_ci": spearman_ci,
        "attr_auc": attr_auc,
        "attr_acc": attr_acc,
    }


def main():
    """Main training loop."""
    cfg = load_config()
    device = torch.device(cfg["device"])

    # ===== 1. Dataset Initialization =====
    train_loader = DataLoader(
        BookCrossingBagDataset(cfg["data"]["train_path"], cfg["data"]["embeddings"]),
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        collate_fn=collate_bags,
    )
    wu_loader = DataLoader(
        BookCrossingBagDataset(cfg["data"]["val_wu_path"], cfg["data"]["embeddings"]),
        batch_size=cfg["data"]["batch_size"],
        collate_fn=collate_bags,
    )
    cu_loader = DataLoader(
        BookCrossingBagDataset(cfg["data"]["val_cu_path"], cfg["data"]["embeddings"]),
        batch_size=cfg["data"]["batch_size"],
        collate_fn=collate_bags,
    )

    # ===== 2. Model Initialization =====
    model = Unison(
        train_loader=train_loader,
        user_id_key="id",
        d_in=cfg["model"]["d_in"],
        d_model=cfg["model"]["d_model"],
        mlp_hidden=cfg["model"]["mlp_hidden"],
        mlp_dropout=cfg["model"]["mlp_dropout"],
        item_task=cfg["model"]["item_task"],
        scores_range=tuple(cfg["model"]["scores_range"]),
        is_user_attr_head=cfg["model"]["is_user_attr_head"],
        user_attr_task=cfg["model"]["user_attr_task"],
        user_head_out_dim=cfg["model"]["user_head_out_dim"],
        user_head_hidden=cfg["model"]["user_head_hidden"],
        user_head_dropout=cfg["model"]["user_head_dropout"],
        cold_steps=cfg["adaptation"]["steps"],
        cold_lr=cfg["adaptation"]["lr"],
        cold_wd=cfg["adaptation"]["weight_decay"],
        cold_patience=cfg["adaptation"]["patience"],
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ===== 3. Training Loop =====
    print(f"\nStarting Experiment: {cfg['experiment_name']}\n")
    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {cfg['training']['num_epochs']}")
    print(f"  Learning Rate: {cfg['training']['lr']}")
    print(f"  Lambda Attr: {cfg['model']['lambda_attr']}")
    print(f"  Cold Steps: {cfg['adaptation']['steps']}\n")

    for epoch in range(1, cfg["training"]["num_epochs"] + 1):
        # Training
        tr_metrics = train_epoch(model, train_loader, optimizer, cfg, device)

        # Evaluation
        wu_metrics = evaluate_warm_user(model, wu_loader, device)
        cu_metrics = evaluate_cold_user(model, cu_loader, device)

        # Logging
        log_epoch(epoch, tr_metrics, wu_metrics, cu_metrics)


if __name__ == "__main__":
    main()