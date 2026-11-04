"""Inference Service for Tropical Cyclone Intensity Estimation."""

from datetime import datetime, timezone
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

from backend.app.ml.intensity.config import (
    CHECKPOINT_PATH,
    IMAGE_SIZE,
    METADATA_PATH,
    knots_to_central_pressure,
    knots_to_imd_category,
)
from backend.app.ml.intensity.dataset import IntensityTransform
from backend.app.ml.intensity.model import CycloneIntensityResNet18


class IntensityInferenceService:
    """Production inference service for Cyclone Intensity Estimation."""

    def __init__(
        self,
        checkpoint_path: Path = CHECKPOINT_PATH,
        metadata_path: Path = METADATA_PATH,
        device: Optional[torch.device] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.metadata_path = Path(metadata_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "cyclone_intensity_resnet18_v1"
        self.model_version = "1.0.0"
        self._model: Optional[CycloneIntensityResNet18] = None
        self._transform = IntensityTransform(target_size=IMAGE_SIZE, is_training=False)
        self._test_rmse: float = 8.5  # fallback uncertainty estimate

    def is_available(self) -> bool:
        """Returns True if trained checkpoint exists on disk."""
        return self.checkpoint_path.exists()

    def _lazy_load(self):
        if not self.is_available():
            raise FileNotFoundError(
                f"Intensity model checkpoint missing at {self.checkpoint_path}. "
                "Trigger training first."
            )

        if self._model is None:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self._model = CycloneIntensityResNet18().to(self.device)
            self._model.load_state_dict(state_dict)
            self._model.eval()

            # Read metadata for empirical uncertainty interval
            if self.metadata_path.exists():
                try:
                    with open(self.metadata_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        self._test_rmse = float(meta.get("test_metrics", {}).get("rmse_knots", 8.5))
                except Exception:
                    self._test_rmse = 8.5

    def predict(
        self,
        image_input: Union[bytes, str, Path, Image.Image, np.ndarray],
        crop_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """Predicts tropical cyclone intensity (wind speed and pressure) from satellite imagery."""
        self._lazy_load()

        if isinstance(image_input, (bytes, bytearray)):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                image_input = (np.clip(image_input, 0.0, 1.0) * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Optional crop around candidate vortex center
        if crop_box is not None:
            x0, y0, x1, y1 = crop_box
            w, h = pil_img.size
            x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
            y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
            if x1 > x0 and y1 > y0:
                pil_img = pil_img.crop((x0, y0, x1, y1))

        tensor = self._transform(pil_img).unsqueeze(0).to(self.device)
        pred_dict = self._model.predict_single(tensor)

        v_max = pred_dict["wind_speed_knots"]
        p_c = pred_dict["central_pressure_hpa"]
        category = pred_dict["category"]

        # 95% Confidence Interval based on holdout validation RMSE
        margin = round(float(1.96 * self._test_rmse), 1)
        ci_lower = round(float(max(15.0, v_max - margin)), 1)
        ci_upper = round(float(v_max + margin), 1)

        return {
            "available": True,
            "wind_speed_knots": v_max,
            "central_pressure_hpa": p_c,
            "category": category,
            "uncertainty_range": {
                "metric": "knots",
                "margin_error_95ci": margin,
                "confidence_interval_95": [ci_lower, ci_upper],
                "model_rmse": self._test_rmse,
            },
            "model_name": self.model_name,
            "model_version": self.model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status_message": (
                f"Intensity estimated using {self.model_name} (ResNet-18) "
                f"trained on historical INSAT-3D thermal infrared benchmark."
            ),
        }


intensity_inference_service = IntensityInferenceService()
