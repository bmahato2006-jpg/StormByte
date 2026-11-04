"""Cyclone Detector V3 Training Pipeline.

Strict Scientific Standards:
1. Pure PyTorch Dataset Loader (zero torchvision dependency).
2. Data augmentations implemented in pure PyTorch/NumPy (flips, rotations, subtle contrast/brightness).
3. Class-weighted CrossEntropyLoss for balanced gradient updates.
4. AdamW optimizer + Cosine Annealing learning rate schedule.
5. Early stopping monitored on validation loss.
6. Checkpoint persistence with complete hyperparameter and training metadata.
"""

import os
import sys
import json
import time
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from backend.app.ml.detection.model_v3 import CycloneDetectorCNN_V3, CLASS_NAMES

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_DIR = DATA_DIR / "processed" / "satellite" / "detection_v3"
CHECKPOINT_DIR = PROJECT_ROOT / "backend" / "app" / "ml" / "checkpoints"

# ImageNet normalization constants
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class SatelliteDatasetV3(Dataset):
    """Pure PyTorch Dataset for authentic North Indian Ocean satellite imagery."""

    def __init__(self, manifest_csv: Path, is_training: bool = False):
        self.manifest_path = Path(manifest_csv)
        self.is_training = is_training

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        self.records = self.df.to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def _apply_augmentations(self, img_arr: np.ndarray) -> np.ndarray:
        """Pure NumPy/PyTorch geometric and photometric augmentations."""
        # Random horizontal flip
        if random.random() > 0.5:
            img_arr = np.fliplr(img_arr)

        # Random vertical flip
        if random.random() > 0.5:
            img_arr = np.flipud(img_arr)

        # Random 90/180/270 rotation
        k = random.randint(0, 3)
        if k > 0:
            img_arr = np.rot90(img_arr, k)

        # Random subtle contrast / brightness jitter (+/- 10%)
        if random.random() > 0.5:
            factor = random.uniform(0.9, 1.1)
            img_arr = np.clip(img_arr * factor, 0.0, 255.0)

        return np.ascontiguousarray(img_arr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        rel_path = rec["file_path"]
        abs_path = PROJECT_ROOT / rel_path

        # Robust image loading with PIL
        try:
            with Image.open(abs_path) as img:
                img_rgb = img.convert("RGB")
                img_arr = np.array(img_rgb, dtype=np.float32)
        except Exception:
            # Fallback for unexpected I/O error
            img_arr = np.zeros((256, 256, 3), dtype=np.float32)

        if self.is_training:
            img_arr = self._apply_augmentations(img_arr)

        # HWC -> CHW and scale to [0, 1]
        tensor = img_arr.transpose(2, 0, 1) / 255.0
        # Normalize with ImageNet standard
        tensor = (tensor - NORM_MEAN) / NORM_STD
        image_tensor = torch.from_numpy(tensor).float()

        label = int(rec["label"])
        image_id = str(rec.get("image_id", abs_path.stem))

        return image_tensor, label, image_id


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Computes standard scientific evaluation metrics."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "f1": round(float(f1), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def train_cyclone_detector_v3(
    batch_size: int = 32,
    num_epochs: int = 6,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 3,
) -> Dict[str, Any]:
    """Executes full training pipeline for Cyclone Detector V3."""
    torch.set_num_threads(os.cpu_count() or 12)
    train_manifest = MANIFEST_DIR / "train.csv"
    val_manifest = MANIFEST_DIR / "validation.csv"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now(timezone.utc).isoformat()}] Initializing Cyclone Detector V3 Training Pipeline...", flush=True)
    print(f"  Train manifest: {train_manifest}", flush=True)
    print(f"  Val manifest:   {val_manifest}", flush=True)

    train_dataset = SatelliteDatasetV3(train_manifest, is_training=True)
    val_dataset = SatelliteDatasetV3(val_manifest, is_training=False)

    print(f"  Training samples:   {len(train_dataset):,}", flush=True)
    print(f"  Validation samples: {len(val_dataset):,}", flush=True)

    # Compute class weights for loss function
    train_labels = [r["label"] for r in train_dataset.records]
    n_total = len(train_labels)
    n_pos = sum(train_labels)
    n_neg = n_total - n_pos
    w_neg = n_total / (2.0 * max(1, n_neg))
    w_pos = n_total / (2.0 * max(1, n_pos))
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32)
    print(f"  Class balance: {n_pos} Cyclone, {n_neg} No-Cyclone | Weights: [{w_neg:.2f}, {w_pos:.2f}]", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single-process data loading for Windows stability
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Initialize model
    model = CycloneDetectorCNN_V3()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        t_epoch_start = time.time()

        # Training Phase
        model.train()
        train_loss_sum = 0.0
        train_preds_list: List[int] = []
        train_targets_list: List[int] = []

        for batch_idx, (images, targets, _) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(targets)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            train_preds_list.extend(preds)
            train_targets_list.extend(targets.numpy())

            if (batch_idx + 1) % 30 == 0 or (batch_idx + 1) == len(train_loader):
                elapsed_b = time.time() - t_epoch_start
                rate_b = ((batch_idx + 1) * batch_size) / max(0.1, elapsed_b)
                print(f"  [Epoch {epoch}/{num_epochs}] Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | Speed: {rate_b:.1f} imgs/s", flush=True)

        scheduler.step()

        train_loss = train_loss_sum / len(train_dataset)
        train_metrics = compute_metrics(
            np.array(train_targets_list),
            np.array(train_preds_list),
            np.zeros(len(train_targets_list)),
        )

        # Validation Phase
        model.eval()
        val_loss_sum = 0.0
        val_preds_list: List[int] = []
        val_targets_list: List[int] = []
        val_probs_list: List[float] = []

        with torch.no_grad():
            for images, targets, _ in val_loader:
                logits = model(images)
                loss = criterion(logits, targets)
                val_loss_sum += loss.item() * len(targets)

                probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                preds = (probs >= 0.50).astype(int)

                val_probs_list.extend(probs)
                val_preds_list.extend(preds)
                val_targets_list.extend(targets.numpy())

        val_loss = val_loss_sum / len(val_dataset)
        val_metrics = compute_metrics(
            np.array(val_targets_list),
            np.array(val_preds_list),
            np.array(val_probs_list),
        )

        epoch_elapsed = time.time() - t_epoch_start
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": round(val_loss, 4),
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_recall": val_metrics["recall"],
            "val_precision": val_metrics["precision"],
            "lr": round(scheduler.get_last_lr()[0], 6),
            "time_seconds": round(epoch_elapsed, 1),
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_metrics['accuracy']:.4f} F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1']:.4f} | "
            f"Elapsed: {epoch_elapsed:.1f}s"
        )

        # Checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            epochs_no_improve = 0

            checkpoint_path = CHECKPOINT_DIR / "cyclone_detector_cnn_v3.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best checkpoint to {checkpoint_path} (val_loss: {val_loss:.4f}, val_f1: {val_metrics['f1']:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  --> Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
                break

    total_training_time = time.time() - t_start

    # Save comprehensive metadata
    metadata = {
        "model_name": "CycloneDetectorCNN_V3",
        "version": "v3.0.0",
        "architecture": "5-Stage Deep Residual Convolutional Neural Network",
        "parameters": model.count_parameters(),
        "input_dimensions": [3, 256, 256],
        "classes": CLASS_NAMES,
        "class_weights": [round(float(w_neg), 4), round(float(w_pos), 4)],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "training": {
            "total_epochs_trained": len(history),
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "best_val_f1": round(best_val_f1, 4),
            "total_time_seconds": round(total_training_time, 1),
            "hyperparameters": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
            },
            "history": history,
        },
        "checkpoint_file": "cyclone_detector_cnn_v3.pt",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    meta_path = CHECKPOINT_DIR / "cyclone_detector_cnn_v3_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[{datetime.now(timezone.utc).isoformat()}] Training finished in {total_training_time/60:.1f} minutes.")
    print(f"  Best epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f} | Best Val F1: {best_val_f1:.4f}")
    print(f"  Checkpoint: {CHECKPOINT_DIR / 'cyclone_detector_cnn_v3.pt'}")
    print(f"  Metadata:   {meta_path}")

    return metadata


if __name__ == "__main__":
    train_cyclone_detector_v3(num_epochs=7, batch_size=32)
