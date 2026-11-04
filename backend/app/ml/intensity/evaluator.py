"""Evaluation engine for Cyclone Intensity Regression."""

from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.app.ml.intensity.config import (
    TARGET_MEAN,
    TARGET_STD,
    knots_to_imd_category,
)


class IntensityEvaluator:
    """Computes comprehensive regression metrics and slice-based category breakdowns."""

    def __init__(self, target_mean: float = TARGET_MEAN, target_std: float = TARGET_STD):
        self.target_mean = target_mean
        self.target_std = target_std

    def evaluate_dataloader(
        self, model: torch.nn.Module, dataloader: DataLoader, device: torch.device
    ) -> Dict[str, Any]:
        """Evaluates model across an entire DataLoader."""
        model.eval()
        predictions_kt: List[float] = []
        ground_truth_kt: List[float] = []

        with torch.no_grad():
            for tensors, norm_targets, raw_targets in dataloader:
                tensors = tensors.to(device)
                pred_norm = model(tensors)
                pred_kt = pred_norm * self.target_std + self.target_mean

                for p, g in zip(pred_kt.squeeze(-1).cpu().numpy(), raw_targets):
                    predictions_kt.append(float(p))
                    ground_truth_kt.append(float(g))

        return self.compute_metrics(np.array(predictions_kt), np.array(ground_truth_kt))

    def compute_metrics(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> Dict[str, Any]:
        """Calculates regression metrics given predicted and true wind speeds (knots)."""
        residuals = predictions - ground_truth
        abs_errors = np.abs(residuals)
        sq_errors = residuals ** 2

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(sq_errors)))
        medae = float(np.median(abs_errors))
        bias = float(np.mean(residuals))
        max_err = float(np.max(abs_errors))

        # Coefficient of Determination R2
        ss_res = float(np.sum(sq_errors))
        ss_tot = float(np.sum((ground_truth - np.mean(ground_truth)) ** 2))
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

        # Benchmarks
        naive_mean_preds = np.full_like(ground_truth, np.mean(ground_truth))
        naive_mean_mae = float(np.mean(np.abs(naive_mean_preds - ground_truth)))

        naive_median_preds = np.full_like(ground_truth, np.median(ground_truth))
        naive_median_mae = float(np.mean(np.abs(naive_median_preds - ground_truth)))

        # Slice-based breakdown by IMD Category
        categories = [knots_to_imd_category(kt) for kt in ground_truth]
        unique_cats = sorted(set(categories))
        slice_breakdown = {}

        for cat in unique_cats:
            mask = np.array([c == cat for c in categories])
            if np.any(mask):
                cat_abs_err = abs_errors[mask]
                cat_sq_err = sq_errors[mask]
                slice_breakdown[cat] = {
                    "count": int(np.sum(mask)),
                    "mae": round(float(np.mean(cat_abs_err)), 2),
                    "rmse": round(float(np.sqrt(np.mean(cat_sq_err))), 2),
                    "medae": round(float(np.median(cat_abs_err)), 2),
                }

        return {
            "mae_knots": round(mae, 2),
            "rmse_knots": round(rmse, 2),
            "medae_knots": round(medae, 2),
            "r2_score": round(r2, 4),
            "mean_bias_error_knots": round(bias, 2),
            "max_error_knots": round(max_err, 2),
            "sample_count": len(predictions),
            "naive_mean_baseline_mae": round(naive_mean_mae, 2),
            "naive_median_baseline_mae": round(naive_median_mae, 2),
            "mae_improvement_vs_naive_percent": round(
                ((naive_mean_mae - mae) / (naive_mean_mae + 1e-8)) * 100.0, 1
            ),
            "slice_breakdown_by_category": slice_breakdown,
        }
