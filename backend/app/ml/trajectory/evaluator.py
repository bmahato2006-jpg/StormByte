"""Benchmark and validation evaluator for Cyclone-X ML Trajectory Forecasting.

Rigorous Scientific Evaluation:
- Evaluates solely on held-out test storms (years >= 2023) from data/processed/manifests/trajectory_test.csv.
- Benchmarks TrajectoryTransformer and TrajectoryLSTM side-by-side with physics_guided_advection.
- Computes genuine Haversine distance errors (km), MAE, RMSE, median, 90th percentile, and per-storm breakdowns.
- Saves full transparent evaluation report to data/reports/trajectory_evaluation_metrics.json
  and data/reports/trajectory_transformer_evaluation.json.
- Strictly NO data fabrication.
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from backend.app.data.provenance import compute_file_sha256
from backend.app.ml.trajectory.config import (
    CHECKPOINT_PATH,
    EVALUATION_METRICS_PATH,
    HORIZONS,
    INPUT_DIM,
    NUM_LAYERS,
    OUTPUT_DIM,
    SCALER_PATH,
    SEQ_LENGTH,
    TEST_MANIFEST_PATH,
    TRANSFORMER_CHECKPOINT_PATH,
    TRANSFORMER_EVALUATION_METRICS_PATH,
)
from backend.app.ml.trajectory.dataset import CycloneSequenceDataset
from backend.app.ml.trajectory.features import (
    haversine_distance,
    TrajectoryFeatureScaler,
)
from backend.app.ml.trajectory.model import TrajectoryLSTM
from backend.app.ml.trajectory.physics import trajectory_forecaster
from backend.app.ml.trajectory.transformer_model import TrajectoryTransformer


class TrajectoryEvaluator:
    """Computes transparent, reproducible evaluation metrics comparing ML models and Physics."""

    def __init__(
        self,
        test_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        transformer_checkpoint_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        metrics_output_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.test_path = Path(test_path or TEST_MANIFEST_PATH)
        self.checkpoint_path = Path(checkpoint_path or CHECKPOINT_PATH)
        self.transformer_checkpoint_path = Path(transformer_checkpoint_path or TRANSFORMER_CHECKPOINT_PATH)
        self.scaler_path = Path(scaler_path or SCALER_PATH)
        self.metrics_output_path = Path(metrics_output_path or EVALUATION_METRICS_PATH)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def evaluate(self) -> Dict[str, Any]:
        """Runs comparative evaluation across all test samples for available ML models & Physics."""
        if not self.test_path.exists():
            raise FileNotFoundError(f"Test manifest not found: {self.test_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler parameters not found: {self.scaler_path}")

        has_lstm = self.checkpoint_path.exists()
        has_transformer = self.transformer_checkpoint_path.exists()

        if not has_lstm and not has_transformer:
            raise FileNotFoundError(
                f"No model checkpoint found. Neither LSTM ({self.checkpoint_path}) "
                f"nor Transformer ({self.transformer_checkpoint_path}) exists. Train a model first."
            )

        # 1. Load Scaler & Dataset
        scaler = TrajectoryFeatureScaler.load(self.scaler_path)
        test_dataset = CycloneSequenceDataset(
            csv_path=self.test_path,
            scaler=scaler,
            fit_scaler=False,
            require_all_horizons=True,
        )

        num_samples = len(test_dataset)
        if num_samples == 0:
            raise ValueError(f"No valid test sequences found in {self.test_path}")

        # 2. Load Models
        lstm_model: Optional[TrajectoryLSTM] = None
        lstm_ckpt: Dict[str, Any] = {}
        if has_lstm:
            lstm_ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            lstm_model = TrajectoryLSTM(
                input_dim=lstm_ckpt.get("hyperparameters", {}).get("input_dim", INPUT_DIM),
                hidden_dim=lstm_ckpt.get("hyperparameters", {}).get("hidden_dim", 64),
                num_layers=lstm_ckpt.get("hyperparameters", {}).get("num_layers", NUM_LAYERS),
                output_dim=lstm_ckpt.get("hyperparameters", {}).get("output_dim", OUTPUT_DIM),
            ).to(self.device)
            lstm_model.load_state_dict(lstm_ckpt["model_state_dict"])
            lstm_model.eval()

        transformer_model: Optional[TrajectoryTransformer] = None
        transformer_ckpt: Dict[str, Any] = {}
        if has_transformer:
            transformer_ckpt = torch.load(self.transformer_checkpoint_path, map_location=self.device, weights_only=False)
            thparams = transformer_ckpt.get("hyperparameters", {})
            transformer_model = TrajectoryTransformer(
                input_dim=thparams.get("input_dim", INPUT_DIM),
                d_model=thparams.get("d_model", 64),
                nhead=thparams.get("nhead", 4),
                num_layers=thparams.get("num_layers", NUM_LAYERS),
                dim_feedforward=thparams.get("dim_feedforward", 128),
                output_dim=thparams.get("output_dim", OUTPUT_DIM),
                seq_len=thparams.get("seq_len", SEQ_LENGTH),
            ).to(self.device)
            transformer_model.load_state_dict(transformer_ckpt["model_state_dict"])
            transformer_model.eval()

        # 3. Collect Per-Horizon Errors and Sample Details
        lstm_errors_by_h: Dict[int, List[float]] = {h: [] for h in HORIZONS}
        transformer_errors_by_h: Dict[int, List[float]] = {h: [] for h in HORIZONS}
        physics_errors_by_h: Dict[int, List[float]] = {h: [] for h in HORIZONS}

        per_storm_data: Dict[str, Dict[str, Any]] = {}
        sample_results: List[Dict[str, Any]] = []

        with torch.no_grad():
            for idx in range(num_samples):
                sample = test_dataset[idx]
                sid = sample["storm_id"]
                sname = sample["storm_name"]
                init_time = sample["init_time"]
                origin_lat = sample["origin_lat"]
                origin_lon = sample["origin_lon"]
                actual_coords = sample["actual_coords"]
                track_history = sample["track_history"]

                if sid not in per_storm_data:
                    per_storm_data[sid] = {
                        "storm_id": sid,
                        "storm_name": sname,
                        "sample_count": 0,
                        "lstm_errors_by_h": {h: [] for h in HORIZONS},
                        "transformer_errors_by_h": {h: [] for h in HORIZONS},
                        "physics_errors_by_h": {h: [] for h in HORIZONS},
                    }
                per_storm_data[sid]["sample_count"] += 1

                input_tensor = sample["input"].unsqueeze(0).to(self.device)

                # Predictions
                lstm_coords = (
                    lstm_model.predict_coordinates(input_tensor, origin_lat, origin_lon)
                    if lstm_model else None
                )
                trans_coords = (
                    transformer_model.predict_coordinates(input_tensor, origin_lat, origin_lon)
                    if transformer_model else None
                )
                physics_preds = trajectory_forecaster.forecast(
                    track_history=track_history,
                    forecast_horizons_hours=HORIZONS,
                )
                physics_pred_coords = {
                    item["forecast_lead_hours"]: (item["latitude"], item["longitude"])
                    for item in physics_preds
                }

                sample_entry = {
                    "sample_index": idx,
                    "storm_id": sid,
                    "storm_name": sname,
                    "init_time": init_time,
                    "origin": {"latitude": origin_lat, "longitude": origin_lon},
                    "horizons": {},
                }

                for h in HORIZONS:
                    act_lat = actual_coords[h]["latitude"]
                    act_lon = actual_coords[h]["longitude"]

                    horizon_dict: Dict[str, Any] = {
                        "actual": {"latitude": act_lat, "longitude": act_lon},
                    }

                    # Physics error
                    phys_lat, phys_lon = physics_pred_coords.get(h, (origin_lat, origin_lon))
                    phys_err = haversine_distance(act_lat, act_lon, phys_lat, phys_lon)
                    physics_errors_by_h[h].append(phys_err)
                    per_storm_data[sid]["physics_errors_by_h"][h].append(phys_err)
                    horizon_dict["physics_prediction"] = {
                        "latitude": phys_lat,
                        "longitude": phys_lon,
                        "error_km": round(phys_err, 2),
                    }

                    # LSTM error
                    if lstm_coords:
                        l_lat, l_lon = lstm_coords[h]
                        l_err = haversine_distance(act_lat, act_lon, l_lat, l_lon)
                        lstm_errors_by_h[h].append(l_err)
                        per_storm_data[sid]["lstm_errors_by_h"][h].append(l_err)
                        horizon_dict["lstm_prediction"] = {
                            "latitude": l_lat,
                            "longitude": l_lon,
                            "error_km": round(l_err, 2),
                        }

                    # Transformer error
                    if trans_coords:
                        t_lat, t_lon = trans_coords[h]
                        t_err = haversine_distance(act_lat, act_lon, t_lat, t_lon)
                        transformer_errors_by_h[h].append(t_err)
                        per_storm_data[sid]["transformer_errors_by_h"][h].append(t_err)
                        horizon_dict["transformer_prediction"] = {
                            "latitude": t_lat,
                            "longitude": t_lon,
                            "error_km": round(t_err, 2),
                        }

                    # Legacy ml_prediction alias (prefers transformer if available, otherwise lstm)
                    if trans_coords:
                        horizon_dict["ml_prediction"] = horizon_dict["transformer_prediction"]
                    elif lstm_coords:
                        horizon_dict["ml_prediction"] = horizon_dict["lstm_prediction"]

                    sample_entry["horizons"][str(h)] = horizon_dict

                sample_results.append(sample_entry)

        # 4. Compute Aggregate Statistics
        def calc_stats(errs: List[float]) -> Dict[str, float]:
            if not errs:
                return {
                    "mae_km": 0.0,
                    "rmse_km": 0.0,
                    "median_km": 0.0,
                    "p90_km": 0.0,
                    "min_km": 0.0,
                    "max_km": 0.0,
                }
            arr = np.array(errs)
            return {
                "mae_km": round(float(np.mean(arr)), 2),
                "rmse_km": round(float(np.sqrt(np.mean(arr ** 2))), 2),
                "median_km": round(float(np.median(arr)), 2),
                "p90_km": round(float(np.percentile(arr, 90)), 2),
                "min_km": round(float(np.min(arr)), 2),
                "max_km": round(float(np.max(arr)), 2),
            }

        all_phys_errs = [e for h in HORIZONS for e in physics_errors_by_h[h]]
        physics_summary = {
            "model_name": "physics_guided_advection",
            "overall": calc_stats(all_phys_errs),
            "by_horizon": {f"{h}h": calc_stats(physics_errors_by_h[h]) for h in HORIZONS},
        }

        lstm_summary: Optional[Dict[str, Any]] = None
        if has_lstm:
            all_l_errs = [e for h in HORIZONS for e in lstm_errors_by_h[h]]
            lstm_summary = {
                "model_name": "trajectory_lstm_v1",
                "overall": calc_stats(all_l_errs),
                "by_horizon": {f"{h}h": calc_stats(lstm_errors_by_h[h]) for h in HORIZONS},
            }

        transformer_summary: Optional[Dict[str, Any]] = None
        if has_transformer:
            all_t_errs = [e for h in HORIZONS for e in transformer_errors_by_h[h]]
            transformer_summary = {
                "model_name": "trajectory_transformer_v1",
                "overall": calc_stats(all_t_errs),
                "by_horizon": {f"{h}h": calc_stats(transformer_errors_by_h[h]) for h in HORIZONS},
            }

        # Primary ML summary (prefers transformer)
        primary_ml_summary = transformer_summary or lstm_summary

        # 5. Master Comparison Table
        comparison_table = []
        for h in HORIZONS:
            phys_mae = physics_summary["by_horizon"][f"{h}h"]["mae_km"]
            l_mae = lstm_summary["by_horizon"][f"{h}h"]["mae_km"] if lstm_summary else None
            t_mae = transformer_summary["by_horizon"][f"{h}h"]["mae_km"] if transformer_summary else None

            # Primary ML metric for backward compatibility
            p_mae = t_mae if t_mae is not None else (l_mae if l_mae is not None else 0.0)
            diff_km = round(p_mae - phys_mae, 2)
            pct_diff = round(((p_mae - phys_mae) / phys_mae) * 100.0, 1) if phys_mae > 0 else 0.0

            candidates = [("physics_guided_advection", phys_mae)]
            if l_mae is not None:
                candidates.append(("trajectory_lstm_v1", l_mae))
            if t_mae is not None:
                candidates.append(("trajectory_transformer_v1", t_mae))
            best_model_name = min(candidates, key=lambda c: c[1])[0]

            row: Dict[str, Any] = {
                "horizon": f"+{h}h",
                "ml_mae_km": p_mae,
                "ml_rmse_km": primary_ml_summary["by_horizon"][f"{h}h"]["rmse_km"] if primary_ml_summary else 0.0,
                "ml_median_km": primary_ml_summary["by_horizon"][f"{h}h"]["median_km"] if primary_ml_summary else 0.0,
                "ml_p90_km": primary_ml_summary["by_horizon"][f"{h}h"]["p90_km"] if primary_ml_summary else 0.0,
                "physics_mae_km": phys_mae,
                "physics_rmse_km": physics_summary["by_horizon"][f"{h}h"]["rmse_km"],
                "physics_median_km": physics_summary["by_horizon"][f"{h}h"]["median_km"],
                "physics_p90_km": physics_summary["by_horizon"][f"{h}h"]["p90_km"],
                "difference_km": diff_km,
                "relative_diff_pct": pct_diff,
                "better_model": best_model_name,
            }
            if t_mae is not None:
                row["transformer_mae_km"] = t_mae
                row["transformer_rmse_km"] = transformer_summary["by_horizon"][f"{h}h"]["rmse_km"]
            if l_mae is not None:
                row["lstm_mae_km"] = l_mae
                row["lstm_rmse_km"] = lstm_summary["by_horizon"][f"{h}h"]["rmse_km"]

            comparison_table.append(row)

        # Overall Mean Row
        overall_phys_mae = physics_summary["overall"]["mae_km"]
        overall_p_mae = primary_ml_summary["overall"]["mae_km"] if primary_ml_summary else 0.0
        overall_diff = round(overall_p_mae - overall_phys_mae, 2)
        overall_pct = round(((overall_p_mae - overall_phys_mae) / overall_phys_mae) * 100.0, 1)

        overall_candidates = [("physics_guided_advection", overall_phys_mae)]
        if lstm_summary:
            overall_candidates.append(("trajectory_lstm_v1", lstm_summary["overall"]["mae_km"]))
        if transformer_summary:
            overall_candidates.append(("trajectory_transformer_v1", transformer_summary["overall"]["mae_km"]))
        overall_best = min(overall_candidates, key=lambda c: c[1])[0]

        overall_row: Dict[str, Any] = {
            "horizon": "Overall Mean",
            "ml_mae_km": overall_p_mae,
            "ml_rmse_km": primary_ml_summary["overall"]["rmse_km"] if primary_ml_summary else 0.0,
            "ml_median_km": primary_ml_summary["overall"]["median_km"] if primary_ml_summary else 0.0,
            "ml_p90_km": primary_ml_summary["overall"]["p90_km"] if primary_ml_summary else 0.0,
            "physics_mae_km": overall_phys_mae,
            "physics_rmse_km": physics_summary["overall"]["rmse_km"],
            "physics_median_km": physics_summary["overall"]["median_km"],
            "physics_p90_km": physics_summary["overall"]["p90_km"],
            "difference_km": overall_diff,
            "relative_diff_pct": overall_pct,
            "better_model": overall_best,
        }
        if transformer_summary:
            overall_row["transformer_mae_km"] = transformer_summary["overall"]["mae_km"]
            overall_row["transformer_rmse_km"] = transformer_summary["overall"]["rmse_km"]
        if lstm_summary:
            overall_row["lstm_mae_km"] = lstm_summary["overall"]["mae_km"]
            overall_row["lstm_rmse_km"] = lstm_summary["overall"]["rmse_km"]
        comparison_table.append(overall_row)

        # 6. Per-Storm Breakdown
        storms_summary = []
        for sid, pdata in per_storm_data.items():
            storm_phys_all = [e for h in HORIZONS for e in pdata["physics_errors_by_h"][h]]
            storm_t_all = [e for h in HORIZONS for e in pdata["transformer_errors_by_h"][h]]
            storm_l_all = [e for h in HORIZONS for e in pdata["lstm_errors_by_h"][h]]

            p_err_all = storm_t_all if storm_t_all else storm_l_all

            storm_entry: Dict[str, Any] = {
                "storm_id": sid,
                "storm_name": pdata["storm_name"],
                "test_samples": pdata["sample_count"],
                "ml_mae_km": round(float(np.mean(p_err_all)), 2) if p_err_all else 0.0,
                "physics_mae_km": round(float(np.mean(storm_phys_all)), 2) if storm_phys_all else 0.0,
                "by_horizon": {},
            }
            if storm_t_all:
                storm_entry["transformer_mae_km"] = round(float(np.mean(storm_t_all)), 2)
            if storm_l_all:
                storm_entry["lstm_mae_km"] = round(float(np.mean(storm_l_all)), 2)

            for h in HORIZONS:
                horizon_stat: Dict[str, float] = {
                    "physics_mae_km": round(float(np.mean(pdata["physics_errors_by_h"][h])), 2)
                    if pdata["physics_errors_by_h"][h] else 0.0,
                }
                if pdata["transformer_errors_by_h"][h]:
                    t_h_mae = round(float(np.mean(pdata["transformer_errors_by_h"][h])), 2)
                    horizon_stat["transformer_mae_km"] = t_h_mae
                    horizon_stat["ml_mae_km"] = t_h_mae
                elif pdata["lstm_errors_by_h"][h]:
                    l_h_mae = round(float(np.mean(pdata["lstm_errors_by_h"][h])), 2)
                    horizon_stat["lstm_mae_km"] = l_h_mae
                    horizon_stat["ml_mae_km"] = l_h_mae

                if pdata["lstm_errors_by_h"][h]:
                    horizon_stat["lstm_mae_km"] = round(float(np.mean(pdata["lstm_errors_by_h"][h])), 2)

                storm_entry["by_horizon"][f"{h}h"] = horizon_stat

            storms_summary.append(storm_entry)

        storms_summary.sort(key=lambda s: s["test_samples"], reverse=True)

        # 7. Compile Final Models Evaluated list
        models_evaluated_list = [
            {
                "name": "physics_guided_advection",
                "type": "Physics-Guided Dynamical Steering + Beta Drift",
                "version": "0.5.0",
            }
        ]
        if has_lstm:
            models_evaluated_list.append({
                "name": "trajectory_lstm_v1",
                "type": "Neural Network (2-Layer LSTM)",
                "checkpoint": str(self.checkpoint_path),
                "checkpoint_epoch": lstm_ckpt.get("epoch"),
                "training_provenance": lstm_ckpt.get("training_provenance"),
            })
        if has_transformer:
            models_evaluated_list.append({
                "name": "trajectory_transformer_v1",
                "type": "Spatio-Temporal Transformer Encoder (Pre-LN)",
                "checkpoint": str(self.transformer_checkpoint_path),
                "checkpoint_epoch": transformer_ckpt.get("epoch"),
                "training_provenance": transformer_ckpt.get("training_provenance"),
            })

        evaluation_payload = {
            "evaluation_id": f"eval_trajectory_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "test_dataset": {
                "manifest_path": str(self.test_path),
                "manifest_sha256": compute_file_sha256(str(self.test_path)),
                "num_test_sequences": num_samples,
                "num_test_storms": len(per_storm_data),
                "horizons_evaluated": HORIZONS,
            },
            "models_evaluated": models_evaluated_list,
            "comparison_table": comparison_table,
            "ml_model_metrics": primary_ml_summary,
            "transformer_model_metrics": transformer_summary,
            "lstm_model_metrics": lstm_summary,
            "physics_model_metrics": physics_summary,
            "per_storm_breakdown": storms_summary,
            "sample_results": sample_results,
            "disclaimer": (
                "SCIENTIFIC AUDIT NOTICE: All metrics are computed strictly on held-out test storms (years >= 2023) "
                "from authentic NOAA NCEI IBTrACS North Indian Ocean records. No synthetic values or observations were added. "
                "This research model is for scientific exploration and is NOT an official operational forecasting system of the "
                "India Meteorological Department (IMD)."
            ),
        }

        # Write to JSON reports
        self.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_output_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_payload, f, indent=2)

        TRANSFORMER_EVALUATION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRANSFORMER_EVALUATION_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(evaluation_payload, f, indent=2)

        print(f"[{datetime.now(timezone.utc).isoformat()}] Evaluation complete!")
        print(f"  Test Samples: {num_samples} across {len(per_storm_data)} storms")
        if transformer_summary:
            print(f"  trajectory_transformer_v1 MAE: {transformer_summary['overall']['mae_km']:.2f} km")
        if lstm_summary:
            print(f"  trajectory_lstm_v1 MAE:        {lstm_summary['overall']['mae_km']:.2f} km")
        print(f"  physics_guided_advection MAE:  {overall_phys_mae:.2f} km")
        print(f"  Metrics saved to: {self.metrics_output_path}")

        return evaluation_payload


evaluator = TrajectoryEvaluator()


if __name__ == "__main__":
    ev = TrajectoryEvaluator()
    res = ev.evaluate()
    print("\n--- MASTER COMPARISON TABLE ---")
    for row in res["comparison_table"]:
        t_mae = row.get("transformer_mae_km", "N/A")
        l_mae = row.get("lstm_mae_km", "N/A")
        p_mae = row.get("physics_mae_km", "N/A")
        print(f"Horizon: {row['horizon']:12} | Transformer: {t_mae} km | LSTM: {l_mae} km | Physics: {p_mae} km | Winner: {row['better_model']}")
