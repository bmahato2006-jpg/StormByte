"""Intensity estimation package for Cyclone-X."""

from typing import Any, Optional
from backend.app.ml.intensity.config import (
    CHECKPOINT_PATH,
    IMAGE_SIZE,
    METADATA_PATH,
    knots_to_central_pressure,
    knots_to_imd_category,
)
from backend.app.ml.intensity.inference import (
    IntensityInferenceService,
    intensity_inference_service,
)
from backend.app.ml.intensity.model import CycloneIntensityResNet18
from backend.app.schemas.analysis import IntensityResult


class IntensityEstimator:
    """Production intensity estimator integrating ResNet-18 model with zero-fabrication safety."""

    def __init__(self, model_checkpoint: Optional[str] = None):
        self.service = intensity_inference_service
        self.model_checkpoint = model_checkpoint

    @property
    def is_model_loaded(self) -> bool:
        return self.service.is_available()

    def estimate(
        self,
        gray_array: Any,
        center_x: float,
        center_y: float,
        classification_category: str,
    ) -> IntensityResult:
        if not self.service.is_available() or classification_category == "No Active Cyclone Detected":
            return IntensityResult(
                available=False,
                wind_speed_knots=None,
                central_pressure_hpa=None,
                category=None,
                uncertainty_range=None,
                status_message=(
                    "Intensity model (ResNet-18 v1.0.0) operational & standing by. Activates upon detection of cyclonic circulation."
                    if classification_category == "No Active Cyclone Detected"
                    else "Intensity estimation unavailable — trained intensity model required."
                ),
            )

        if gray_array is None:
            return IntensityResult(
                available=False,
                wind_speed_knots=None,
                central_pressure_hpa=None,
                category=None,
                uncertainty_range=None,
                status_message="Intensity estimation unavailable — trained intensity model required for valid imagery.",
            )

        try:
            # Crop box around storm center if valid coordinates provided
            crop_box = None
            if gray_array is not None and hasattr(gray_array, "shape"):
                h, w = gray_array.shape[:2]
                r = int(min(h, w) * 0.35)
                crop_box = (
                    max(0, int(center_x - r)),
                    max(0, int(center_y - r)),
                    min(w, int(center_x + r)),
                    min(h, int(center_y + r)),
                )

            res = self.service.predict(gray_array, crop_box=crop_box)
            unc = res.get("uncertainty_range")
            if isinstance(unc, dict):
                ci = unc.get("confidence_interval_95", [0.0, 0.0])
                unc_str = f"±{unc.get('margin_error_95ci', 0.0):.1f} kt (95% CI: {ci[0]:.1f}-{ci[1]:.1f} kt, RMSE: {unc.get('model_rmse', 0.0):.1f} kt)"
            else:
                unc_str = str(unc) if unc else None

            return IntensityResult(
                available=True,
                wind_speed_knots=res["wind_speed_knots"],
                central_pressure_hpa=res["central_pressure_hpa"],
                category=res["category"],
                uncertainty_range=unc_str,
                status_message=res["status_message"],
            )
        except Exception as e:
            return IntensityResult(
                available=False,
                wind_speed_knots=None,
                central_pressure_hpa=None,
                category=None,
                uncertainty_range=None,
                status_message=f"Intensity estimation error: {str(e)}",
            )


intensity_estimator = IntensityEstimator()

__all__ = [
    "CycloneIntensityResNet18",
    "IntensityInferenceService",
    "intensity_inference_service",
    "IntensityEstimator",
    "intensity_estimator",
    "knots_to_central_pressure",
    "knots_to_imd_category",
    "CHECKPOINT_PATH",
    "METADATA_PATH",
    "IMAGE_SIZE",
]
