"""Deterministic PyTorch Training Pipeline for Cyclone-X TrajectoryLSTM (Phase 8).

Strict Governance:
- Trains SOLELY on data/processed/manifests/trajectory_train.csv.
- Uses validation set (2022) SOLELY for early stopping and model checkpoint selection.
- Does NOT touch test data.
- Full reproducibility with explicit seed setting.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.app.data.provenance import compute_file_sha256
from backend.app.ml.trajectory.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    EARLY_STOPPING_PATIENCE,
    HIDDEN_DIM,
    INPUT_DIM,
    LEARNING_RATE,
    MAX_EPOCHS,
    NUM_LAYERS,
    OUTPUT_DIM,
    RANDOM_SEED,
    SCALER_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
    WEIGHT_DECAY,
)
from backend.app.ml.trajectory.dataset import CycloneSequenceDataset
from backend.app.ml.trajectory.features import TrajectoryFeatureScaler
from backend.app.ml.trajectory.model import TrajectoryLSTM


def set_seed(seed: int = RANDOM_SEED):
    """Sets random seeds across random, numpy, and torch for 100% reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrajectoryTrainer:
    """Manages training, validation, early stopping, and checkpointing for TrajectoryLSTM."""

    def __init__(
        self,
        train_path: Optional[Path] = None,
        val_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        seed: int = RANDOM_SEED,
        device: Optional[str] = None,
    ):
        self.train_path = Path(train_path or TRAIN_MANIFEST_PATH)
        self.val_path = Path(val_path or VAL_MANIFEST_PATH)
        self.checkpoint_path = Path(checkpoint_path or CHECKPOINT_PATH)
        self.scaler_path = Path(scaler_path or SCALER_PATH)
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        set_seed(self.seed)

    def train(
        self,
        epochs: int = MAX_EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        patience: int = EARLY_STOPPING_PATIENCE,
    ) -> Dict[str, Any]:
        """Executes full model training and validation loop."""
        print(f"[{datetime.now(timezone.utc).isoformat()}] Initializing TrajectoryLSTM training...")
        print(f"  Device: {self.device}")
        print(f"  Train Manifest: {self.train_path}")
        print(f"  Val Manifest: {self.val_path}")

        # 1. Prepare Datasets & Scaler (fit scaler SOLELY on train)
        scaler = TrajectoryFeatureScaler()
        train_dataset = CycloneSequenceDataset(
            csv_path=self.train_path,
            scaler=scaler,
            fit_scaler=True,
            require_all_horizons=True,
        )
        # Save fitted scaler
        scaler.save(self.scaler_path)

        val_dataset = CycloneSequenceDataset(
            csv_path=self.val_path,
            scaler=scaler,
            fit_scaler=False,
            require_all_horizons=True,
        )

        def cyclone_collate_fn(batch_items):
            return {
                "input": torch.stack([b["input"] for b in batch_items]),
                "target": torch.stack([b["target"] for b in batch_items]),
            }

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=cyclone_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=cyclone_collate_fn,
        )

        print(f"  Train samples: {len(train_dataset)} ({len(train_loader)} batches)")
        print(f"  Val samples:   {len(val_dataset)} ({len(val_loader)} batches)")

        # 2. Instantiate Model
        model = TrajectoryLSTM(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            output_dim=OUTPUT_DIM,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        # 3. Training Loop
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "learning_rate": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        start_time = datetime.now(timezone.utc)

        for epoch in range(1, epochs + 1):
            # Train Phase
            model.train()
            train_batch_losses = []
            for batch in train_loader:
                x = batch["input"].to(self.device)
                y = batch["target"].to(self.device)

                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                # Gradient clipping to prevent exploding gradients in RNNs
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_batch_losses.append(loss.item())

            epoch_train_loss = float(np.mean(train_batch_losses))

            # Validation Phase
            model.eval()
            val_batch_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["input"].to(self.device)
                    y = batch["target"].to(self.device)
                    pred = model(x)
                    loss = criterion(pred, y)
                    val_batch_losses.append(loss.item())

            epoch_val_loss = float(np.mean(val_batch_losses)) if val_batch_losses else float("inf")
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)
            history["learning_rate"].append(current_lr)

            scheduler.step(epoch_val_loss)

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss (MSE): {epoch_train_loss:.5f} | "
                f"Val Loss (MSE): {epoch_val_loss:.5f} | "
                f"LR: {current_lr:.6f}"
            )

            # Early stopping & best checkpoint selection
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                patience_counter = 0

                # Checkpoint payload
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "train_loss": epoch_train_loss,
                    "hyperparameters": {
                        "input_dim": INPUT_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "num_layers": NUM_LAYERS,
                        "output_dim": OUTPUT_DIM,
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
                print(f"  --> Saved new best checkpoint at Epoch {epoch} (Val Loss: {best_val_loss:.5f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping triggered at Epoch {epoch} (No improvement for {patience} epochs).")
                    break

        duration_sec = (datetime.now(timezone.utc) - start_time).total_seconds()
        print(f"Training completed in {duration_sec:.2f}s. Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.5f}")

        return {
            "model_name": "trajectory_lstm_v1",
            "status": "TRAINED",
            "epochs_completed": len(history["train_loss"]),
            "best_epoch": best_epoch,
            "final_train_loss": history["train_loss"][-1],
            "best_val_loss": best_val_loss,
            "checkpoint_path": str(self.checkpoint_path),
            "scaler_path": str(self.scaler_path),
            "training_duration_seconds": round(duration_sec, 2),
            "history": history,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Cyclone-X TrajectoryLSTM")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    args = parser.parse_args()

    trainer = TrajectoryTrainer()
    res = trainer.train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience)
    print("\nTraining summary:")
    for k, v in res.items():
        if k != "history":
            print(f"  {k}: {v}")
