"""Deterministic PyTorch Training Pipeline for Cyclone-X Detection (Phase 9).

Strict Governance:
- Trained on authentic partitioned satellite image tiles.
- Augmentation applied strictly to the training split.
- Checkpoints saved with complete architecture specifications and SHA-256 provenance.
- Reproducible random seed setting across random, numpy, and torch.
"""

from datetime import datetime, timezone
from pathlib import Path
import random
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.app.data.provenance import compute_file_sha256
from backend.app.ml.detection.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CHECKPOINT_V2_PATH,
    EARLY_STOPPING_PATIENCE,
    IMAGE_SIZE,
    INPUT_CHANNELS,
    LEARNING_RATE,
    MAX_EPOCHS,
    NUM_CLASSES,
    RANDOM_SEED,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
    WEIGHT_DECAY,
)
from backend.app.ml.detection.dataset import (
    CycloneImageDataset,
    DetectionDatasetBuilder,
)
from backend.app.ml.detection.model import CycloneDetectorCNN


def set_seed(seed: int = RANDOM_SEED):
    """Sets random seeds across random, numpy, and torch for 100% reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DetectionTrainer:
    """Manages training, validation, early stopping, and checkpointing for CycloneDetectorCNN."""

    def __init__(
        self,
        train_path: Optional[Path] = None,
        val_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        model_name: str = "cyclone_detector_cnn_v2",
        seed: int = RANDOM_SEED,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.train_path = Path(train_path or TRAIN_MANIFEST_PATH)
        self.val_path = Path(val_path or VAL_MANIFEST_PATH)
        default_ckpt = CHECKPOINT_V2_PATH if "v2" in model_name else CHECKPOINT_PATH
        self.checkpoint_path = Path(checkpoint_path or default_ckpt)
        self.seed = seed
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        set_seed(self.seed)

    def train(
        self,
        epochs: int = MAX_EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        patience: int = EARLY_STOPPING_PATIENCE,
    ) -> Dict[str, Any]:
        """Executes full training and validation loop."""
        # 1. Ensure manifests and tile dataset exist
        DetectionDatasetBuilder.build_dataset_if_needed()

        print(f"[{datetime.now(timezone.utc).isoformat()}] Initializing CycloneDetectorCNN training...")
        print(f"  Device: {self.device}")
        print(f"  Train Manifest: {self.train_path}")
        print(f"  Val Manifest:   {self.val_path}")

        # 2. Datasets & Loaders
        train_dataset = CycloneImageDataset(
            csv_path=self.train_path, is_training=True
        )
        val_dataset = CycloneImageDataset(
            csv_path=self.val_path, is_training=False
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        print(f"  Train samples: {len(train_dataset)} ({len(train_loader)} batches)")
        print(f"  Val samples:   {len(val_dataset)} ({len(val_loader)} batches)")

        # 3. Model & Optimizer
        model = CycloneDetectorCNN(
            in_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES, dropout=0.3
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        # 4. Training Loop
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        start_time = datetime.now(timezone.utc)

        for epoch in range(1, epochs + 1):
            # Train Phase
            model.train()
            train_losses = []
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            epoch_train_loss = float(np.mean(train_losses))
            epoch_train_acc = float(train_correct / train_total) if train_total > 0 else 0.0

            # Val Phase
            model.eval()
            val_losses = []
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device)

                    logits = model(images)
                    loss = criterion(logits, labels)
                    val_losses.append(loss.item())

                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            epoch_val_acc = float(val_correct / val_total) if val_total > 0 else 0.0
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(round(epoch_train_loss, 5))
            history["val_loss"].append(round(epoch_val_loss, 5))
            history["train_acc"].append(round(epoch_train_acc, 4))
            history["val_acc"].append(round(epoch_val_acc, 4))
            history["lr"].append(current_lr)

            scheduler.step(epoch_val_loss)

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {epoch_train_loss:.4f} (Acc: {epoch_train_acc * 100:.1f}%) | "
                f"Val Loss: {epoch_val_loss:.4f} (Acc: {epoch_val_acc * 100:.1f}%) | "
                f"LR: {current_lr:.6f}"
            )

            # Checkpoint saving & Early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                patience_counter = 0

                checkpoint = {
                    "model_name": self.model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "val_acc": epoch_val_acc,
                    "train_loss": epoch_train_loss,
                    "train_acc": epoch_train_acc,
                    "hyperparameters": {
                        "in_channels": INPUT_CHANNELS,
                        "num_classes": NUM_CLASSES,
                        "image_size": IMAGE_SIZE,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "weight_decay": weight_decay,
                        "random_seed": self.seed,
                    },
                    "training_provenance": {
                        "train_manifest": str(self.train_path),
                        "train_manifest_sha256": compute_file_sha256(str(self.train_path)),
                        "val_manifest": str(self.val_path),
                        "val_manifest_sha256": compute_file_sha256(str(self.val_path)),
                        "trained_at": datetime.now(timezone.utc).isoformat(),
                        "device": str(self.device),
                        "torch_version": torch.__version__,
                    },
                }
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, self.checkpoint_path)
                print(f"  --> Saved new best checkpoint at Epoch {epoch} (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping triggered at Epoch {epoch}.")
                    break

        duration_sec = (datetime.now(timezone.utc) - start_time).total_seconds()
        print(f"Training completed in {duration_sec:.2f}s. Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")

        return {
            "model_name": self.model_name,
            "status": "TRAINED",
            "epochs_completed": len(history["train_loss"]),
            "best_epoch": best_epoch,
            "final_train_loss": history["train_loss"][-1],
            "best_val_loss": best_val_loss,
            "best_val_acc": history["val_acc"][best_epoch - 1],
            "checkpoint_path": str(self.checkpoint_path),
            "training_duration_seconds": round(duration_sec, 2),
            "history": history,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Cyclone-X CycloneDetectorCNN")
    parser.add_argument("--model", type=str, default="cyclone_detector_cnn_v2", help="Model name (v1 or v2)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    args = parser.parse_args()

    trainer = DetectionTrainer(model_name=args.model)
    res = trainer.train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience)
    print("\nTraining summary:")
    for k, v in res.items():
        if k != "history":
            print(f"  {k}: {v}")

