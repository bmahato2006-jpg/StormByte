"""FastAPI Router for Cyclone-X AI Intensity Estimation (ResNet-18).

Endpoints:
- POST /api/ml/intensity/predict : Run inference on uploaded/historical satellite image.
- GET  /api/ml/intensity/status  : Operational status, model checkpoint, and metrics summary.
- GET  /api/ml/intensity/metrics : Held-out test evaluation metrics and category breakdown.
"""

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import Image

from backend.app.ml.intensity.config import (
    CHECKPOINT_PATH,
    METADATA_PATH,
    IMAGE_SIZE,
    knots_to_central_pressure,
    knots_to_imd_category,
)
from backend.app.ml.intensity.inference import intensity_inference_service
from backend.app.ml.intensity.schemas import (
    IntensityPredictRequest,
    IntensityPredictResponse,
    IntensityPredictionData,
    IntensityStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml/intensity", tags=["AI Cyclone Intensity (ResNet-18)"])


@router.get("/status", response_model=IntensityStatusResponse, summary="Get intensity estimation status")
def get_intensity_status():
    """Inspects active intensity model readiness, metrics availability, and supported data modes."""
    is_trained = intensity_inference_service.is_available()
    metrics_exist = METADATA_PATH.exists()
    test_metrics = None

    if metrics_exist:
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                test_metrics = meta.get("test_metrics")
        except Exception as e:
            logger.warning("Failed to read intensity metadata: %s", e)

    return {
        "model_name": "cyclone_intensity_resnet18_v1",
        "is_trained": is_trained,
        "status": "OPERATIONAL_TRAINED" if is_trained else "AWAITING_TRAINING",
        "checkpoint_path": str(CHECKPOINT_PATH),
        "metrics_available": metrics_exist,
        "test_metrics": test_metrics,
        "supported_modes": ["DEMO", "HISTORICAL", "UPLOAD", "LIVE_NRT"],
        "disclaimer": "Scientific AI estimate. Not an official IMD/WMO warning. Consult national meteorological agencies.",
    }


@router.get("/metrics", summary="Get held-out test evaluation metrics")
def get_intensity_metrics():
    """Returns held-out test set performance metrics, error distributions, and slice breakdowns."""
    if not METADATA_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Intensity model evaluation metrics not found. Train the model first."
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return meta


@router.post(
    "/predict",
    response_model=IntensityPredictResponse,
    summary="Estimate cyclone wind speed and central pressure from satellite imagery",
)
async def predict_intensity(
    file: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
    data_mode: str = Form("UPLOAD"),
    environmental_pressure: float = Form(1010.0),
):
    """Executes ResNet-18 intensity estimation on input satellite imagery."""
    if not intensity_inference_service.is_available():
        return IntensityPredictResponse(
            available=False,
            intensity=None,
            model_name="cyclone_intensity_resnet18_v1",
            model_version="1.0.0",
            data_mode=data_mode,
            message="Intensity model checkpoint missing. Train the model first.",
        )

    try:
        if file is not None:
            raw_bytes = await file.read()
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        elif image_path is not None:
            p = Path(image_path)
            if not p.exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Image path does not exist: {image_path}",
                )
            img = Image.open(p).convert("RGB")
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'file' or 'image_path' must be provided.",
            )

        pred = intensity_inference_service.predict(img)
        vmax_kt = pred["wind_speed_knots"]
        pc_hpa = knots_to_central_pressure(vmax_kt, environmental_pressure=environmental_pressure)
        cat_name = pred["category"]
        unc = pred.get("uncertainty_range", {})
        ci_95 = unc.get("confidence_interval_95", [max(15.0, vmax_kt - 20.0), vmax_kt + 20.0])
        rmse = unc.get("model_rmse", 22.14)

        # Compute pressure uncertainty bounds from wind speed bounds
        pc_lower = knots_to_central_pressure(ci_95[1], environmental_pressure=environmental_pressure)
        pc_upper = knots_to_central_pressure(ci_95[0], environmental_pressure=environmental_pressure)

        data = IntensityPredictionData(
            vmax_knots=vmax_kt,
            vmax_kmh=round(vmax_kt * 1.852, 1),
            vmax_uncertainty_lower_kt=round(ci_95[0], 1),
            vmax_uncertainty_upper_kt=round(ci_95[1], 1),
            central_pressure_hpa=pc_hpa,
            central_pressure_uncertainty_lower_hpa=round(pc_lower, 1),
            central_pressure_uncertainty_upper_hpa=round(pc_upper, 1),
            category_name=cat_name,
            category_code=cat_name.upper().replace(" ", "_"),
            confidence_score=round(max(0.0, min(1.0, 1.0 - (rmse / 100.0))), 3),
            available=True,
        )

        return IntensityPredictResponse(
            available=True,
            intensity=data,
            model_name="cyclone_intensity_resnet18_v1",
            model_version="1.0.0",
            data_mode=data_mode,
            message="Intensity estimated successfully using pure PyTorch ResNet-18.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Intensity prediction failed: %s", e, exc_info=True)
        return IntensityPredictResponse(
            available=False,
            intensity=None,
            model_name="cyclone_intensity_resnet18_v1",
            model_version="1.0.0",
            data_mode=data_mode,
            message=f"Intensity estimation error: {str(e)}",
        )
