"""CycloneViT Training Pipeline on Real Satellite Imagery.

Trains CycloneViT (Vision Transformer) on the 3,723 authentic North Indian Ocean
satellite tiles (Kaggle/MOSDAC INSAT-3D) using SatelliteDatasetV3.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ml.detection.config import (
    CHECKPOINTS_DIR,
    RANDOM_SEED,
)
from backend.app.ml.detection.trainer_v3 import SatelliteDatasetV3
from backend.app.ml.training_framework.extended_models import CycloneViT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cyclone_x.vit_trainer")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_DIR = DATA_DIR / "processed" / "satellite" / "detection_v3"
TRAIN_MANIFEST = MANIFEST_DIR / "train.csv"
VAL_MANIFEST = MANIFEST_DIR / "validation.csv"
TEST_MANIFEST = MANIFEST_DIR / "test.csv"
VIT_CHECKPOINT_PATH = CHECKPOINTS_DIR / "cyclone_vit.pt"
VIT_METADATA_PATH = CHECKPOINTS_DIR / "cyclone_vit_metadata.json"


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ViTTrainer:
    """Manages training and evaluation of CycloneViT on real INSAT-3D imagery."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.model = CycloneViT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            num_classes=2,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            dropout=0.1,
        ).to(self.device)
        logger.info(f"Initialized CycloneViT on device: {self.device}")

    def train(
        self,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        evaluate: bool = True,
    ) -> Dict[str, Any]:
        set_seed()
        logger.info(f"Starting CycloneViT training: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")

        train_ds = SatelliteDatasetV3(TRAIN_MANIFEST, is_training=True)
        val_ds = SatelliteDatasetV3(VAL_MANIFEST, is_training=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )

        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        history = []
        t0 = time.time()

        for epoch in range(1, num_epochs + 1):
            # Train step
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for images, labels, _ in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                out = self.model(images)
                logits = out["logits"]
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)

            scheduler.step()
            avg_train_loss = train_loss / max(1, total)
            train_acc = correct / max(1, total)

            # Validation step
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    out = self.model(images)
                    logits = out["logits"]
                    loss = criterion(logits, labels)

                    val_loss += loss.item() * len(labels)
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += len(labels)

            avg_val_loss = val_loss / max(1, val_total)
            val_acc = val_correct / max(1, val_total)

            logger.info(
                f"Epoch [{epoch}/{num_epochs}] Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}"
            )

            history.append({
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 5),
                "train_acc": round(train_acc, 4),
                "val_loss": round(avg_val_loss, 5),
                "val_acc": round(val_acc, 4),
            })

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

        elapsed = time.time() - t0

        # Save best model
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "architecture": "CycloneViT",
                "epochs_trained": num_epochs,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "img_size": 256,
                "patch_size": 16,
                "embed_dim": 192,
                "depth": 4,
                "num_heads": 6,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            VIT_CHECKPOINT_PATH,
        )
        logger.info(f"Saved trained CycloneViT checkpoint to {VIT_CHECKPOINT_PATH}")

        test_metrics = {}
        if evaluate and TEST_MANIFEST.exists():
            test_metrics = self.evaluate(TEST_MANIFEST)

        metadata = {
            "model_name": "cyclone_vit",
            "architecture": "Vision Transformer (ViT)",
            "status": "TRAINED",
            "epochs": num_epochs,
            "best_epoch": best_epoch,
            "training_time_seconds": round(elapsed, 2),
            "best_val_loss": round(best_val_loss, 5),
            "test_metrics": test_metrics,
            "dataset": "INSAT-3D Real Satellite Granules (detection_v3)",
            "checkpoint_path": str(VIT_CHECKPOINT_PATH),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(VIT_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def evaluate(self, test_manifest: Path = TEST_MANIFEST) -> Dict[str, Any]:
        test_ds = SatelliteDatasetV3(test_manifest, is_training=False)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

        self.model.eval()
        y_true = []
        y_pred = []
        y_prob = []

        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(self.device)
                out = self.model(images)
                probs = out["probabilities"][:, 1].cpu().numpy()
                preds = out["logits"].argmax(dim=-1).cpu().numpy()

                y_true.extend(labels.numpy())
                y_pred.extend(preds)
                y_prob.extend(probs)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
        brier = float(brier_score_loss(y_true, y_prob))

        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "brier_score": round(brier, 4),
            "total_samples": len(y_true),
        }
        logger.info(f"CycloneViT Out-of-Sample Test Metrics: {metrics}")
        return metrics


def train_vit(epochs: int = 3, batch_size: int = 16, lr: float = 1e-4) -> Dict[str, Any]:
    trainer = ViTTrainer()
    return trainer.train(num_epochs=epochs, batch_size=batch_size, learning_rate=lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycloneViT Training CLI")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train_vit(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
