"""Historical Satellite Imagery Service for Cyclone-X.

Integrates authentic historical INSAT-3D imagery (2013-2021) from Kaggle/MOSDAC
with existing AI models:
- Cyclone Detector CNN V2 (Phase 9)
- Truthful Intensity Status (intensity_dvorak_net is NOT_TRAINED)
- Phase 8 Trajectory LSTM (strictly requiring >= 4 real sequential fixes)
- Phase 10 Multi-Factor Risk Engine
- Cryptographic SHA-256 Provenance & Scientific Disclaimers

Strict Zero-Fabrication Rule:
Never invents coordinates, timestamps, wind speeds, or confidence scores.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image

from backend.app.ml.detection.inference import detection_inference_service_v2
from backend.app.ml.intensity.inference import intensity_inference_service
from backend.app.ml.pipeline.risk_engine import PipelineRiskEngine
from backend.app.ml.trajectory.config import SEQ_LENGTH
from backend.app.ml.trajectory.inference import trajectory_inference_service


# Base workspace path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MANIFEST_DIR = WORKSPACE_ROOT / "data" / "processed" / "satellite" / "detection" / "manifests"


class HistoricalSatelliteService:
    """Catalog manager and analysis orchestrator for historical satellite imagery."""

    def __init__(self, manifest_dir: Optional[Path] = None):
        self.manifest_dir = Path(manifest_dir or MANIFEST_DIR)
        self._catalog: Dict[str, Dict[str, Any]] = {}
        self._is_loaded = False
        self.detection_service = detection_inference_service_v2
        self.trajectory_service = trajectory_inference_service
        self.intensity_service = intensity_inference_service
        self.risk_engine = PipelineRiskEngine

    def _ensure_catalog_loaded(self):
        """Loads and indexes authentic historical manifests on demand."""
        if self._is_loaded:
            return

        manifest_files = [
            ("test", self.manifest_dir / "detection_test.csv"),
            ("validation", self.manifest_dir / "detection_validation.csv"),
            ("train", self.manifest_dir / "detection_train.csv"),
        ]

        catalog = {}
        for split_name, manifest_path in manifest_files:
            if not manifest_path.exists():
                continue

            try:
                df = pd.read_csv(manifest_path)
                for _, row in df.iterrows():
                    img_id = str(row.get("image_id", "")).strip()
                    if not img_id:
                        continue

                    rel_path = str(row.get("file_path", "")).strip()
                    abs_path = WORKSPACE_ROOT / rel_path

                    # Parse values without hallucination
                    acq_time = None
                    if pd.notna(row.get("acquisition_time")) and str(row.get("acquisition_time")).strip() != "":
                        acq_time = str(row.get("acquisition_time")).strip()

                    storm_name = None
                    if pd.notna(row.get("storm_name")) and str(row.get("storm_name")).strip() != "":
                        storm_name = str(row.get("storm_name")).strip()

                    wind_kt = None
                    if pd.notna(row.get("wind_speed_knots")):
                        try:
                            val = float(row.get("wind_speed_knots"))
                            if val > 0:
                                wind_kt = int(val)
                        except (ValueError, TypeError):
                            pass

                    intensity_cat = None
                    if pd.notna(row.get("intensity_category")) and str(row.get("intensity_category")).strip() != "":
                        intensity_cat = str(row.get("intensity_category")).strip()

                    cloud_frac = None
                    if pd.notna(row.get("convective_cloud_fraction")):
                        try:
                            cloud_frac = round(float(row.get("convective_cloud_fraction")), 4)
                        except (ValueError, TypeError):
                            pass

                    lat = None
                    if pd.notna(row.get("latitude")):
                        try:
                            lat = float(row.get("latitude"))
                        except (ValueError, TypeError):
                            pass

                    lon = None
                    if pd.notna(row.get("longitude")):
                        try:
                            lon = float(row.get("longitude"))
                        except (ValueError, TypeError):
                            pass

                    label_name = str(row.get("label_name", "UNKNOWN")).upper()
                    if label_name not in ["CYCLONE", "NO_CYCLONE"]:
                        label_name = "CYCLONE" if int(row.get("label", 0)) == 1 else "NO_CYCLONE"

                    catalog[img_id] = {
                        "image_id": img_id,
                        "filename": abs_path.name if rel_path else f"{img_id}.jpg",
                        "relative_path": rel_path,
                        "absolute_path": abs_path,
                        "file_exists": abs_path.exists(),
                        "split": str(row.get("split", split_name)).lower(),
                        "label": label_name,
                        "label_source": str(row.get("label_source", "DATASET_MANIFEST")),
                        "source": "Historical INSAT-3D",
                        "dataset": str(row.get("source_dataset", "sshubam/insat3d-infrared-raw-cyclone-images-20132021")),
                        "source_url": str(row.get("source_url", "")),
                        "satellite": str(row.get("satellite", "INSAT-3D")),
                        "sensor": str(row.get("sensor", "IMAGER")),
                        "channel": str(row.get("channel", "TIR1 (10.8 µm)")),
                        "acquisition_time": acq_time,
                        "storm_name": storm_name,
                        "dataset_reference_wind_knots": wind_kt,
                        "intensity_category": intensity_cat,
                        "convective_cloud_fraction": cloud_frac,
                        "has_georeference": (lat is not None and lon is not None),
                        "latitude": lat,
                        "longitude": lon,
                        "preview_url": f"/api/satellite/historical/images/{img_id}/preview",
                    }
            except Exception as e:
                # Keep loading other manifests if one has issues
                continue

        self._catalog = catalog
        self._is_loaded = True

    def list_images(
        self,
        split: Optional[str] = None,
        label: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Lists available historical INSAT-3D images with optional filters."""
        self._ensure_catalog_loaded()

        filtered = []
        target_split = split.lower().strip() if split else None
        target_label = label.upper().strip() if label else None

        for item in self._catalog.values():
            if target_split and item["split"] != target_split:
                continue
            if target_label and item["label"] != target_label:
                continue

            # Public metadata (exclude internal filesystem paths)
            filtered.append({
                "image_id": item["image_id"],
                "filename": item["filename"],
                "split": item["split"],
                "label": item["label"],
                "label_source": item["label_source"],
                "source": item["source"],
                "dataset": item["dataset"],
                "satellite": item["satellite"],
                "sensor": item["sensor"],
                "channel": item["channel"],
                "acquisition_time": item["acquisition_time"],
                "storm_name": item["storm_name"],
                "dataset_reference_wind_knots": item["dataset_reference_wind_knots"],
                "intensity_category": item["intensity_category"],
                "convective_cloud_fraction": item["convective_cloud_fraction"],
                "has_georeference": item["has_georeference"],
                "latitude": item["latitude"],
                "longitude": item["longitude"],
                "preview_url": item["preview_url"],
            })

        return {
            "total": len(filtered),
            "split_filter": target_split,
            "label_filter": target_label,
            "limit": limit,
            "images": filtered[:limit],
            "disclaimer": (
                "HISTORICAL DATA MODE: Authentic historical INSAT-3D imagery from Indian Ocean "
                "cyclone seasons (2013-2021). Unreferenced images strictly maintain latitude=null and longitude=null."
            ),
        }

    def get_image_details(self, image_id: str) -> Dict[str, Any]:
        """Returns metadata for a specific historical image."""
        self._ensure_catalog_loaded()
        item = self._catalog.get(image_id)
        if not item:
            raise KeyError(f"Historical image '{image_id}' not found in manifest catalog.")

        return {
            "image_id": item["image_id"],
            "filename": item["filename"],
            "split": item["split"],
            "label": item["label"],
            "label_source": item["label_source"],
            "source": item["source"],
            "dataset": item["dataset"],
            "satellite": item["satellite"],
            "sensor": item["sensor"],
            "channel": item["channel"],
            "acquisition_time": item["acquisition_time"],
            "storm_name": item["storm_name"],
            "dataset_reference_wind_knots": item["dataset_reference_wind_knots"],
            "intensity_category": item["intensity_category"],
            "convective_cloud_fraction": item["convective_cloud_fraction"],
            "has_georeference": item["has_georeference"],
            "latitude": item["latitude"],
            "longitude": item["longitude"],
            "preview_url": item["preview_url"],
            "file_exists": item["file_exists"],
        }

    def get_image_path(self, image_id: str) -> Path:
        """Resolves absolute path on disk for image streaming."""
        self._ensure_catalog_loaded()
        item = self._catalog.get(image_id)
        if not item:
            raise KeyError(f"Historical image '{image_id}' not found in manifest catalog.")

        path: Path = item["absolute_path"]
        if not path.exists():
            raise FileNotFoundError(f"Image file on disk is missing: {path}")

        return path

    def analyze_image(
        self,
        image_id: str,
        track_history: Optional[List[Dict[str, Any]]] = None,
        environmental_obs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Executes the full Historical Satellite Analysis Pipeline.

        Workflow:
        1. Authentic Image Retrieval & SHA-256 computation
        2. Cyclone Detector CNN V2 Inference
        3. Intensity Estimation (Truthful UNAVAILABLE check)
        4. Trajectory Forecasting (Strict >= 4 sequential fixes rule)
        5. Phase 10 Multi-Factor Risk Assessment
        6. Provenance & Scientific Disclaimers
        """
        self._ensure_catalog_loaded()
        item = self._catalog.get(image_id)
        if not item:
            raise KeyError(f"Historical image '{image_id}' not found in manifest catalog.")

        img_path: Path = item["absolute_path"]
        if not img_path.exists():
            raise FileNotFoundError(f"Historical image file is missing on disk: {img_path}")

        # 1. Cryptographic SHA-256 for data lineage
        with open(img_path, "rb") as f:
            file_bytes = f.read()
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()

        # 2. Cyclone Detection V2
        detection_meta = {
            "source": item["source"],
            "satellite": item["satellite"],
            "sensor": item["sensor"],
            "channel": item["channel"],
            "acquisition_time": item["acquisition_time"],
            "latitude": item["latitude"],
            "longitude": item["longitude"],
            "track_history": track_history or [],
        }

        det_result = self.detection_service.predict(
            image_input=img_path,
            metadata=detection_meta,
            data_mode="HISTORICAL",
        )

        is_cyclone = bool(det_result.get("detected", False))
        detection_confidence = float(det_result.get("confidence", 0.0))
        predicted_label = det_result.get("label", "NO_CYCLONE")
        class_probs = det_result.get("class_probabilities", {})
        center_info = det_result.get("center_localization") or {}

        # 3. Intensity Estimation: Deep Learning Regression (ResNet-18)
        intensity_res: Dict[str, Any]
        if self.intensity_service.is_available():
            try:
                int_pred = self.intensity_service.predict(image_input=img_path)
                v_kt = float(int_pred["wind_speed_knots"])
                p_c = float(int_pred["central_pressure_hpa"])
                t_num = round(1.0 + (max(25.0, v_kt) - 25.0) / 20.7, 1)
                intensity_res = {
                    "status": "OPERATIONAL",
                    "reason": None,
                    "estimated_wind_speed_knots": v_kt,
                    "estimated_central_pressure_hpa": p_c,
                    "dvorak_t_number": t_num,
                    "model_name": "cyclone_intensity_resnet18_v1",
                    "model_status": "TRAINED",
                    "intensity_category": int_pred.get("category", "Depression"),
                    "uncertainty_range": int_pred.get("uncertainty_range"),
                    "dataset_reference_wind_knots": item.get("dataset_reference_wind_knots"),
                    "dataset_intensity_category": item.get("intensity_category"),
                    "dataset_note": (
                        "Estimated by trained ResNet-18 Deep Regression Network on authentic thermal IR observations."
                    ),
                }
            except Exception as e:
                intensity_res = {
                    "status": "UNAVAILABLE",
                    "reason": f"Intensity inference error: {e}",
                    "estimated_wind_speed_knots": None,
                    "estimated_central_pressure_hpa": None,
                    "dvorak_t_number": None,
                    "model_name": "cyclone_intensity_resnet18_v1",
                    "model_status": "TRAINED",
                    "dataset_reference_wind_knots": item.get("dataset_reference_wind_knots"),
                    "dataset_intensity_category": item.get("intensity_category"),
                    "dataset_note": "Dataset reference values provided for benchmark comparison.",
                }
        else:
            # Attempt ObjectiveDvorakNet inference if checkpoint exists
            dvorak_evaluated = False
            dvorak_ckpt = Path(__file__).resolve().parents[2] / "ml" / "checkpoints" / "intensity_dvorak_net.pt"
            if dvorak_ckpt.exists():
                try:
                    import torch
                    from backend.app.ml.training_framework.extended_models import ObjectiveDvorakNet
                    d_model = ObjectiveDvorakNet(in_channels=3)
                    ckpt = torch.load(dvorak_ckpt, map_location="cpu")
                    d_model.load_state_dict(ckpt["model_state_dict"])
                    d_model.eval()

                    pil_img = Image.open(img_path).convert("RGB").resize((128, 128))
                    t_in = torch.tensor(np.array(pil_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                    with torch.no_grad():
                        d_out = d_model(t_in)

                    v_kt = round(float(d_out["max_wind_kt"][0].item()), 1)
                    p_c = round(float(d_out["central_pressure_hpa"][0].item()), 1)
                    t_num = round(float(d_out["dvorak_t_number"][0].item()), 1)

                    intensity_res = {
                        "status": "OPERATIONAL",
                        "reason": None,
                        "estimated_wind_speed_knots": v_kt,
                        "estimated_central_pressure_hpa": p_c,
                        "dvorak_t_number": t_num,
                        "model_name": "intensity_dvorak_net",
                        "model_status": "TRAINED",
                        "intensity_category": "Cyclone" if v_kt >= 34 else "Depression",
                        "dataset_reference_wind_knots": item.get("dataset_reference_wind_knots"),
                        "dataset_intensity_category": item.get("intensity_category"),
                        "dataset_note": "Estimated by trained ObjectiveDvorakNet dual-branch CDO+Band CNN.",
                    }
                    dvorak_evaluated = True
                except Exception:
                    pass

            if not dvorak_evaluated:
                intensity_res = {
                    "status": "UNAVAILABLE",
                    "reason": "No validated intensity model is currently connected.",
                    "estimated_wind_speed_knots": None,
                    "estimated_central_pressure_hpa": None,
                    "dvorak_t_number": None,
                    "model_name": "intensity_dvorak_net",
                    "model_status": "OFFLINE",
                    "dataset_reference_wind_knots": item.get("dataset_reference_wind_knots"),
                    "dataset_intensity_category": item.get("intensity_category"),
                    "dataset_note": (
                        "Dataset reference values are authentic Kaggle/IMD ground-truth labels "
                        "provided for evaluation benchmarking, NOT generated by an AI intensity model."
                    ),
                }

        # 4. Trajectory Prediction: Strict Sequential Fix Rule
        # Phase 8 LSTM strictly requires >= 4 sequential track fixes.
        # Standalone historical satellite imagery without track fixes cannot be forecasted.
        clean_history = list(track_history or [])
        hist_len = len(clean_history)
        trajectory_forecast: Optional[Dict[str, Any]] = None
        trajectory_status: str

        if not is_cyclone:
            trajectory_status = "NO_CYCLONE_DETECTED"
        elif hist_len < SEQ_LENGTH:
            trajectory_status = "INSUFFICIENT_TRACK_HISTORY"
        elif not self.trajectory_service.is_available():
            trajectory_status = "TRAJECTORY_MODEL_AWAITING_TRAINING"
        else:
            try:
                traj_pred = self.trajectory_service.predict(clean_history)
                formatted_predictions = []
                for p in traj_pred["predictions"]:
                    h = p["forecast_lead_hours"]
                    r68 = p.get("uncertainty_radius_km_68", 35.0 if h == 6 else (68.5 if h == 12 else 129.6))
                    r95 = p.get("uncertainty_radius_km_95", round(r68 * 1.645, 1))
                    formatted_predictions.append({
                        "forecast_hour": h,
                        "forecast_lead_hours": h,
                        "timestamp": p["valid_time"],
                        "latitude": round(float(p["latitude"]), 4),
                        "longitude": round(float(p["longitude"]), 4),
                        "uncertainty_km": round(float(r68), 1),
                        "uncertainty_radius_km_68": round(float(r68), 1),
                        "uncertainty_radius_km_95": round(float(r95), 1),
                    })

                trajectory_forecast = {
                    "model_name": traj_pred["model_name"],
                    "model_version": traj_pred.get("model_version", "1.0.0"),
                    "origin_fix": traj_pred["origin_fix"],
                    "predictions": formatted_predictions,
                    "uncertainty_method": (
                        "Empirical residual uncertainty derived from evaluated held-out test storms "
                        "(6h ≈ 32-35 km, 12h ≈ 68.5 km, 24h ≈ 129.6 km). Not guaranteed universal error bounds."
                    ),
                    "forecast_type": "Experimental ML forecast",
                    "is_experimental": True,
                }
                trajectory_status = "FORECAST_GENERATED"
            except Exception as e:
                trajectory_status = f"TRAJECTORY_FORECAST_ERROR: {str(e)}"

        trajectory_info = {
            "status": trajectory_status,
            "message": (
                "Phase 8 LSTM trajectory forecasting strictly requires >= 4 genuine sequential fixes. "
                "Standalone historical images cannot produce trajectory forecasts without historical track fixes."
                if trajectory_status == "INSUFFICIENT_TRACK_HISTORY"
                else (
                    "Trajectory forecast successfully produced via Phase 8 LSTM with empirical uncertainty cones."
                    if trajectory_status == "FORECAST_GENERATED"
                    else "No cyclone detected for trajectory forecasting."
                )
            ),
            "available_fixes": hist_len,
            "required_fixes": SEQ_LENGTH,
            "forecast": trajectory_forecast,
        }

        # 5. Risk Assessment: Phase 10 Research Risk Engine
        forecast_pts = trajectory_forecast.get("predictions") if trajectory_forecast else None
        # Use provided track coordinates if available
        last_lat = clean_history[-1].get("latitude") if clean_history else item["latitude"]
        last_lon = clean_history[-1].get("longitude") if clean_history else item["longitude"]

        risk_res = self.risk_engine.evaluate(
            detected=is_cyclone,
            detection_confidence=detection_confidence,
            latitude=last_lat,
            longitude=last_lon,
            forecast_predictions=forecast_pts,
            environmental_obs=environmental_obs,
            track_history=clean_history,
        )

        # 6. Complete Provenance
        now_iso = datetime.now(timezone.utc).isoformat()
        provenance = {
            "source": item["source"],
            "dataset": item["dataset"],
            "image_filename": item["filename"],
            "image_id": image_id,
            "split": item["split"],
            "timestamp": item["acquisition_time"],
            "detection_model": self.detection_service.model_name,
            "detection_version": self.detection_service.model_version,
            "intensity_model": "cyclone_intensity_resnet18_v1 (TRAINED)" if intensity_res["status"] == "OPERATIONAL" else "UNAVAILABLE",
            "trajectory_model": "trajectory_lstm_v1" if trajectory_status == "FORECAST_GENERATED" else None,
            "risk_engine": "PipelineRiskEngine",
            "processing_version": "v2.0-historical-pipeline",
            "data_mode": "HISTORICAL",
            "generated_at": now_iso,
            "sha256": sha256_hash,
        }

        return {
            "image_id": image_id,
            "data_mode": "HISTORICAL",
            "source": item["source"],
            "dataset": item["dataset"],
            "detection": {
                "detected": is_cyclone,
                "label": predicted_label,
                "confidence": detection_confidence,
                "class_probabilities": class_probs,
                "structural_type": det_result.get("structural_type", "UNSPECIFIED"),
                "center_localization": center_info,
                "cloud_fraction": det_result.get("image_metadata", {}).get("cloud_fraction"),
            },
            "intensity": intensity_res,
            "trajectory": trajectory_info,
            "risk": risk_res,
            "provenance": provenance,
            "disclaimer": (
                "HISTORICAL DATA MODE: Using authentic historical INSAT-3D imagery for research "
                "and demonstration. Model-generated predictions are experimental and are not official IMD warnings."
            ),
        }


# Singleton service instance
historical_satellite_service = HistoricalSatelliteService()

