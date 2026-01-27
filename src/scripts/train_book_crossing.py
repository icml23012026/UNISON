"""
UNISON Training Script for Book-Crossing Dataset

This script trains the UNISON framework on the Book-Crossing recommendation
dataset with both Stage 2 (item scoring) and Stage 3 (user attribute prediction).

Training Strategy:
    - Stage 2 + Stage 3 joint training on WU-WI episodes
    - Evaluation on WU-CI (warm user, cold items)
    - Evaluation on CU-Mixed (cold user, mixed items) with separate metrics:
        * CU-WI: Cold user, warm items
        * CU-CI: Cold user, cold items

Metrics:
    - Stage 2: MAE (Mean Absolute Error), Spearman correlation
    - Stage 3: AUC (for classification), Accuracy

Usage:
    python src/scripts/train_book_crossing.py --config configs/config_book_crossing.json

Directory Structure (relative to project root):
    configs/
        config_book_crossing.json       # Configuration file
    data/
        embeddings/
            item2vec_books_qwen.pkl     # Pre-computed item embeddings
        episodes_books/
            N_SUP_50/
                wu_wi/                  # Training episodes
                wu_ci/                  # Warm-user evaluation episodes
                cu_mixed/               # Cold-user evaluation episodes
    src/
        scripts/
            train_book_crossing.py      # This file
    checkpoints/
        book_crossing/                  # Saved models
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data_prep.episode_dataset_book_crossing import EpisodesDataset, collate_fn
from src.model.framework import UNISON


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_epoch(
    model: UNISON,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    lambda_attr: float,
    device: str,
) -> Dict[str, float]:
    """
    Train one epoch on WU-WI episodes (Stage 2 + Stage 3 jointly).

    Args:
        model: UNISON model instance
        optimizer: PyTorch optimizer
        train_loader: DataLoader for WU-WI training episodes
        lambda_attr: Weight for Stage 3 loss (relative to Stage 2)
        device: Device string ("cuda" or "cpu")

    Returns:
        Dictionary with training metrics:
        - train_loss: Total loss
        - train_loss_ratings: Stage 2 loss
        - train_loss_attr: Stage 3 loss
        - train_attr_accuracy: Stage 3 accuracy (if classification)
    """
    model.train()

    total_loss = 0.0
    total_rating_loss = 0.0
    total_attr_loss = 0.0
    attr_correct = 0
    attr_total = 0
    n_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        # Move data to device
        items_sup = batch["items_sup"].to(device)
        scores_sup = batch["scores_sup"].to(device)
        target_attr = batch["target_attr"].to(device)
        user_ids = batch["id"]

        # Forward + backward
        result = model.train_step_warm_user_item_and_attribute(
            optimizer=optimizer,
            user_ids=user_ids,
            items_sup=items_sup,
            scores_sup=scores_sup,
            target_attr=target_attr,
            lambda_attr=lambda_attr,
        )

        # Accumulate losses
        total_loss += float(result["loss_scaled"])
        total_rating_loss += float(result["loss_ratings"])
        total_attr_loss += float(result["loss_attr"])
        n_batches += 1

        # Compute accuracy for classification
        if model.user_attr_task == "classification":
            with torch.no_grad():
                pred_attr = result["pred_attr"]  # Probabilities
                tgt_attr = result["target_attr"]

                # Binary classification: threshold at 0.5
                if pred_attr.dim() == 1 or pred_attr.size(-1) == 1:
                    preds = (pred_attr >= 0.5).long()
                    labels = tgt_attr.squeeze(-1).long() if tgt_attr.dim() > 1 else tgt_attr.long()
                else:
                    # Multiclass: argmax
                    preds = pred_attr.argmax(dim=-1)
                    labels = tgt_attr.squeeze(-1).long() if tgt_attr.dim() > 1 else tgt_attr.long()

                attr_correct += (preds == labels).sum().item()
                attr_total += labels.numel()

    # Compute averages
    metrics = {
        "train_loss": total_loss / max(n_batches, 1),
        "train_loss_ratings": total_rating_loss / max(n_batches, 1),
        "train_loss_attr": total_attr_loss / max(n_batches, 1),
    }

    if attr_total > 0:
        metrics["train_attr_accuracy"] = attr_correct / attr_total

    return metrics


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================
def evaluate_warm_user_cold_item(
    model: UNISON,
    eval_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """
    Evaluate Stage 2 on WU-CI episodes (warm users, cold items).

    Args:
        model: UNISON model instance
        eval_loader: DataLoader for WU-CI evaluation episodes
        device: Device string

    Returns:
        Dictionary with evaluation metrics:
        - eval_loss: Average loss
        - mae: Mean Absolute Error
        - spearman: Spearman correlation
    """
    model.eval()

    total_loss = 0.0
    mae_list = []
    spearman_list = []
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Eval WU-CI", leave=False):
            items_qry = batch["items_qry"].to(device)
            scores_qry = batch["scores_qry"].to(device)
            mask_qry = batch["mask_qry"].to(device)
            user_ids = batch["id"]

            result = model.evaluate_warm_user_cold_item(
                user_ids=user_ids,
                items_qry=items_qry,
                scores_qry=scores_qry,
                mask_qry=mask_qry,
            )

            total_loss += float(result["loss_scaled"])

            # Extract metrics
            if "mae_WU_CI" in result:
                mae_list.append(float(result["mae_WU_CI"]))
            if "spearman_WU_CI" in result:
                spearman_list.append(float(result["spearman_WU_CI"]))

            n_batches += 1

    return {
        "eval_loss": total_loss / max(n_batches, 1),
        "mae": np.nanmean(mae_list) if mae_list else float("nan"),
        "spearman": np.nanmean(spearman_list) if spearman_list else float("nan"),
    }


def evaluate_cold_user_items(
    model: UNISON,
    eval_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """
    Evaluate Stage 2 on CU-Mixed episodes (cold users, mixed items).

    Returns separate metrics for CU-WI (cold user, warm item) and
    CU-CI (cold user, cold item) using the mask codes.

    Args:
        model: UNISON model instance
        eval_loader: DataLoader for CU-Mixed evaluation episodes
        device: Device string

    Returns:
        Dictionary with evaluation metrics:
        - eval_loss: Average loss
        - mae_cu_wi: MAE for cold-user warm-item queries
        - spearman_cu_wi: Spearman for CU-WI
        - mae_cu_ci: MAE for cold-user cold-item queries
        - spearman_cu_ci: Spearman for CU-CI
    """
    model.eval()

    total_loss = 0.0
    mae_wi_list = []
    spearman_wi_list = []
    mae_ci_list = []
    spearman_ci_list = []
    n_batches = 0


    for batch in tqdm(eval_loader, desc="Eval CU-Mixed", leave=False):
        items_sup = batch["items_sup"].to(device)
        scores_sup = batch["scores_sup"].to(device)
        items_qry = batch["items_qry"].to(device)
        scores_qry = batch["scores_qry"].to(device)
        mask_qry = batch["mask_qry"].to(device)

        result = model.evaluate_cold_user_items(
            items_sup=items_sup,
            scores_sup=scores_sup,
            items_qry=items_qry,
            scores_qry=scores_qry,
            mask_qry=mask_qry,
        )

        total_loss += float(result["loss_scaled"])

        # Extract CU-WI metrics (mask code 1)
        if "mae_CU_WI" in result:
            mae_wi_list.append(float(result["mae_CU_WI"]))
        if "spearman_CU_WI" in result:
            spearman_wi_list.append(float(result["spearman_CU_WI"]))

        # Extract CU-CI metrics (mask code 2)
        if "mae_CU_CI" in result:
            mae_ci_list.append(float(result["mae_CU_CI"]))
        if "spearman_CU_CI" in result:
            spearman_ci_list.append(float(result["spearman_CU_CI"]))

        n_batches += 1

    return {
        "eval_loss": total_loss / max(n_batches, 1),
        "mae_cu_wi": np.nanmean(mae_wi_list) if mae_wi_list else float("nan"),
        "spearman_cu_wi": np.nanmean(spearman_wi_list) if spearman_wi_list else float("nan"),
        "mae_cu_ci": np.nanmean(mae_ci_list) if mae_ci_list else float("nan"),
        "spearman_cu_ci": np.nanmean(spearman_ci_list) if spearman_ci_list else float("nan"),
    }


def evaluate_cold_user_attributes(
    model: UNISON,
    eval_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """
    Evaluate Stage 3 on CU-Mixed episodes (cold users, user attributes).

    Args:
        model: UNISON model instance
        eval_loader: DataLoader for CU-Mixed evaluation episodes
        device: Device string

    Returns:
        Dictionary with Stage 3 metrics:
        - attr_eval_loss: Average loss
        - attr_auc: AUC (for classification)
        - attr_accuracy: Accuracy (for classification)
    """
    model.eval()

    loss_list = []
    auc_list = []
    acc_list = []

    for batch in tqdm(eval_loader, desc="Eval CU Attributes", leave=False):
        items_sup = batch["items_sup"].to(device)
        scores_sup = batch["scores_sup"].to(device)
        target_attr = batch["target_attr"].to(device)

        result = model.evaluate_cold_user_attribute(
            items_sup=items_sup,
            scores_sup=scores_sup,
            target_attr=target_attr,
        )

        # Extract metrics based on task type
        if model.user_attr_task == "classification":
            if "loss_bce" in result:
                loss_list.append(float(result["loss_bce"]))
            if "auc" in result:
                auc_list.append(float(result["auc"]))
            if "accuracy" in result:
                acc_list.append(float(result["accuracy"]))

    metrics = {}
    if loss_list:
        metrics["attr_eval_loss"] = float(np.mean(loss_list))
    if auc_list:
        metrics["attr_auc"] = float(np.mean(auc_list))
    if acc_list:
        metrics["attr_accuracy"] = float(np.mean(acc_list))

    return metrics


# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train UNISON on Book-Crossing dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_book_crossing.json",
        help="Path to configuration JSON file"
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Set device
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ========== Create DataLoaders ==========
    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    train_dataset = EpisodesDataset(
        config["data"]["train_data_path"],
        config["data"]["embedding_path"]
    )
    val_wu_dataset = EpisodesDataset(
        config["data"]["val_wu_data_path"],
        config["data"]["embedding_path"]
    )
    val_cu_dataset = EpisodesDataset(
        config["data"]["val_cu_data_path"],
        config["data"]["embedding_path"]
    )

    print(f"WU-WI (train):     {len(train_dataset)} episodes")
    print(f"WU-CI (eval):      {len(val_wu_dataset)} episodes")
    print(f"CU-Mixed (eval):   {len(val_cu_dataset)} episodes\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
    )
    val_wu_loader = DataLoader(
        val_wu_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
    )
    val_cu_loader = DataLoader(
        val_cu_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
    )

    # ========== Create Model ==========
    print("=" * 60)
    print("CREATING MODEL")
    print("=" * 60)

    model = UNISON(
        train_loader=train_loader,
        user_id_key="id",
        d_in=config["model"]["d_in"],
        d_model=config["model"]["d_model"],
        mlp_hidden=tuple(config["model"]["mlp_hidden"]),
        mlp_dropout=config["model"]["mlp_dropout"],
        mlp_use_batchnorm=config["model"]["mlp_use_batchnorm"],
        mlp_use_layernorm=config["model"]["mlp_use_layernorm"],
        normalize_user=config["model"]["normalize_user"],
        normalize_item=config["model"]["normalize_item"],
        item_task=config["task"]["item_task"],
        binarization_threshold=config["task"]["binarization_threshold"],
        scores_range=tuple(config["task"]["scores_range"]),
        is_user_attr_head=config["stage3"]["is_user_attr_head"],
        user_attr_task=config["stage3"]["user_attr_task"],
        user_head_out_dim=config["stage3"]["user_head_out_dim"],
        user_head_hidden=tuple(config["stage3"]["user_head_hidden"]),
        user_head_dropout=config["stage3"]["user_head_dropout"],
        user_head_activation=config["stage3"]["user_head_activation"],
        user_head_use_batchnorm=config["stage3"]["user_head_use_batchnorm"],
        user_head_use_layernorm=config["stage3"]["user_head_use_layernorm"],
        cold_steps=config["cold_start"]["cold_steps"],
        cold_lr=config["cold_start"]["cold_lr"],
        cold_wd=config["cold_start"]["cold_wd"],
        cold_patience=config["cold_start"]["cold_patience"],
        device=device,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # ========== Create Optimizer ==========
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # ========== Training Loop ==========
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Lambda (Stage 3): {config['stage3']['lambda_attr']}\n")

    best_wu_ci_loss = float("inf")

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model,
            optimizer,
            train_loader,
            lambda_attr=config["stage3"]["lambda_attr"],
            device=device,
        )

        # Evaluate WU-CI (Stage 2)
        wu_ci_metrics = evaluate_warm_user_cold_item(model, val_wu_loader, device)

        # Evaluate CU-Mixed (Stage 2)
        cu_mixed_metrics = evaluate_cold_user_items(model, val_cu_loader, device)

        # Evaluate CU-Mixed (Stage 3)
        cu_attr_metrics = evaluate_cold_user_attributes(model, val_cu_loader, device)

        # ========== Print Results ==========
        print("\n" + "-" * 60)
        print("TRAINING METRICS")
        print("-" * 60)
        print(f"Total Loss:       {train_metrics['train_loss']:.4f}")
        print(f"Rating Loss:      {train_metrics['train_loss_ratings']:.4f}")
        print(f"Attribute Loss:   {train_metrics['train_loss_attr']:.4f}")
        if "train_attr_accuracy" in train_metrics:
            print(f"Attribute Acc:    {train_metrics['train_attr_accuracy']:.4f}")

        print("\n" + "-" * 60)
        print("EVALUATION: WU-CI (Warm User, Cold Item)")
        print("-" * 60)
        print(f"Loss:       {wu_ci_metrics['eval_loss']:.4f}")
        print(f"MAE:        {wu_ci_metrics['mae']:.4f}")
        print(f"Spearman:   {wu_ci_metrics['spearman']:.4f}")

        print("\n" + "-" * 60)
        print("EVALUATION: CU-WI / CU-CI (Cold User, Mixed Items)")
        print("-" * 60)
        print(f"Loss:       {cu_mixed_metrics['eval_loss']:.4f}")
        print(f"\nCU-WI (Cold User, Warm Item):")
        print(f"  MAE:        {cu_mixed_metrics['mae_cu_wi']:.4f}")
        print(f"  Spearman:   {cu_mixed_metrics['spearman_cu_wi']:.4f}")
        print(f"\nCU-CI (Cold User, Cold Item):")
        print(f"  MAE:        {cu_mixed_metrics['mae_cu_ci']:.4f}")
        print(f"  Spearman:   {cu_mixed_metrics['spearman_cu_ci']:.4f}")

        print("\n" + "-" * 60)
        print("EVALUATION: Stage 3 (User Attributes)")
        print("-" * 60)
        if cu_attr_metrics:
            if "attr_eval_loss" in cu_attr_metrics:
                print(f"Loss:       {cu_attr_metrics['attr_eval_loss']:.4f}")
            if "attr_auc" in cu_attr_metrics:
                print(f"AUC:        {cu_attr_metrics['attr_auc']:.4f}")
            if "attr_accuracy" in cu_attr_metrics:
                print(f"Accuracy:   {cu_attr_metrics['attr_accuracy']:.4f}")
        else:
            print("No Stage 3 metrics available")


if __name__ == "__main__":
    main()