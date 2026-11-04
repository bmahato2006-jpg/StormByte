"""Training engine for Cyclone Intensity Regression."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.app.ml.intensity.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CV_FOLDS_PATH,
    EARLY_STOPPING_PATIENCE,
    HUBER_DELTA,
    LEARNING_RATE_HEAD,
    METADATA_PATH,
    NUM_EPOCHS,
    SEED,
    TARGET_MEAN,
    TARGET_STD,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
    WEIGHT_DECAY,
)
from backend.app.ml.intensity.dataset import CycloneIntensityDataset
from backend.app.ml.intensity.evaluator import IntensityEvaluator
from backend.app.ml.intensity.model import CycloneIntensityResNet18

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cyclone_x.intensity_trainer")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IntensityTrainer:
    """Orchestrates model training, cross-validation, checkpointing, and evaluation."""

    def __init__(
        self,
        model: Optional[CycloneIntensityResNet18] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model or CycloneIntensityResNet18()
        self.model.to(self.device)
        self.evaluator = IntensityEvaluator()

    def train(
        self,
        train_manifest: Path = TRAIN_MANIFEST_PATH,
        val_manifest: Path = VAL_MANIFEST_PATH,
        test_manifest: Path = TEST_MANIFEST_PATH,
        num_epochs: int = NUM_EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: Optional[float] = None,
        epochs: Optional[int] = None,
        checkpoint_path: Path = CHECKPOINT_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> Dict[str, Any]:
        """Trains intensity model on primary train/validation splits, evaluates on holdout test set."""
        if epochs is not None:
            num_epochs = epochs
        set_seed()
        logger.info(f"Initializing Intensity Training on device: {self.device}")
        logger.info(f"Train manifest: {train_manifest} | Val: {val_manifest} | Test: {test_manifest}")

        train_dataset = CycloneIntensityDataset(train_manifest, is_training=True)
        val_dataset = CycloneIntensityDataset(val_manifest, is_training=False)
        test_dataset = CycloneIntensityDataset(test_manifest, is_training=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        effective_lr = lr if lr is not None else LEARNING_RATE_HEAD
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=effective_lr,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-5
        )
        criterion = nn.HuberLoss(delta=HUBER_DELTA)

        best_val_mae = float("inf")
        best_epoch = 0
        best_state = None
        patience_counter = 0
        history = []

        start_time = datetime.now(timezone.utc)

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            train_loss = 0.0

            for tensors, norm_targets, _ in train_loader:
                tensors = tensors.to(self.device)
                norm_targets = norm_targets.to(self.device)

                optimizer.zero_grad()
                preds = self.model(tensors)
                loss = criterion(preds, norm_targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(tensors)

            scheduler.step()
            train_loss /= len(train_dataset)

            # Validation
            val_metrics = self.evaluator.evaluate_dataloader(self.model, val_loader, self.device)
            val_mae = val_metrics["mae_knots"]
            val_rmse = val_metrics["rmse_knots"]
            val_r2 = val_metrics["r2_score"]

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "lr": scheduler.get_last_lr()[0],
            })

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:02d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val MAE: {val_mae:.2f} kt | Val RMSE: {val_rmse:.2f} kt | Val R2: {val_r2:.3f}"
                )

            # Checkpoint on best validation MAE
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE and epoch >= 25:
                logger.info(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
                break

        # Load best model weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final Evaluation on Unseen Test Holdout
        test_metrics = self.evaluator.evaluate_dataloader(self.model, test_loader, self.device)
        logger.info("=== FINAL UNSEEN TEST EVALUATION ===")
        logger.info(f"Test MAE:  {test_metrics['mae_knots']} kt")
        logger.info(f"Test RMSE: {test_metrics['rmse_knots']} kt")
        logger.info(f"Test R2:   {test_metrics['r2_score']}")
        logger.info(f"Test MedAE:{test_metrics['medae_knots']} kt")
        logger.info(f"Improvement vs Naive Baseline: +{test_metrics['mae_improvement_vs_naive_percent']}%")

        # Save Checkpoint
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": "cyclone_intensity_resnet18_v1",
                "model_version": "1.0.0",
                "model_state_dict": self.model.state_dict(),
                "architecture": "ResNet18",
                "target_mean": TARGET_MEAN,
                "target_std": TARGET_STD,
                "hyperparameters": {
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "learning_rate": LEARNING_RATE_HEAD,
                    "weight_decay": WEIGHT_DECAY,
                    "huber_delta": HUBER_DELTA,
                },
                "training_metadata": {
                    "best_epoch": best_epoch,
                    "best_val_mae": best_val_mae,
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "test_samples": len(test_dataset),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            },
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Save Metadata JSON
        meta_dict = {
            "model_name": "cyclone_intensity_resnet18_v1",
            "model_version": "1.0.0",
            "status": "TRAINED",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "best_epoch": best_epoch,
            "test_metrics": test_metrics,
            "training_history": history,
            "dataset_sizes": {
                "train": len(train_dataset),
                "val": len(val_dataset),
                "test": len(test_dataset),
            },
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

        return meta_dict


def run_training_pipeline() -> Dict[str, Any]:
    """CLI / programmatic entrypoint for training."""
    trainer = IntensityTrainer()
    results = trainer.train()
    return results


if __name__ == "__main__":
    run_training_pipeline()
