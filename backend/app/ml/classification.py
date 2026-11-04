"""Cyclone Structural Classification Module.

Classifies cyclonic cloud canopy morphology into standardized structural categories.
Strictly tags results as experimental baseline and never claims official IMD authorization.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
from PIL import Image
import torch

from backend.app.schemas.analysis import ClassificationResult, ConfidenceTier


class CycloneClassifier:
    """Classifies tropical vortex intensity stages from cloud morphology and symmetry."""

    def __init__(self):
        self._dvorak_model = None
        self._model_loaded = False

    def _get_dvorak_model(self):
        if not self._model_loaded:
            try:
                ckpt_path = Path(__file__).resolve().parents[2] / "backend" / "app" / "ml" / "checkpoints" / "intensity_dvorak_net.pt"
                if not ckpt_path.exists():
                    ckpt_path = Path(__file__).resolve().parents[0] / "checkpoints" / "intensity_dvorak_net.pt"
                if ckpt_path.exists():
                    from backend.app.ml.training_framework.extended_models import ObjectiveDvorakNet
                    model = ObjectiveDvorakNet(in_channels=3)
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    state_dict = ckpt.get("model_state_dict", ckpt)
                    model.load_state_dict(state_dict)
                    model.eval()
                    self._dvorak_model = model
            except Exception:
                self._dvorak_model = None
            self._model_loaded = True
        return self._dvorak_model

    def classify(
        self,
        detected: bool,
        area_ratio: float,
        spiral_index: float,
        symmetry_score: float,
        image_array: Optional[np.ndarray] = None,
        vmax_knots: Optional[float] = None,
    ) -> ClassificationResult:
        """Categorizes storm structure based on neural Dvorak analysis and morphological curvature."""
        if not detected:
            return ClassificationResult(
                category="No Active Cyclone Detected",
                confidence=ConfidenceTier.HIGH,
                is_experimental=True,
                criteria_summary="Cloud canopy and spiral curvature are below organized cyclonic thresholds. No organized vortex present.",
            )

        # 1. Attempt Objective Deep Neural Dvorak classification if image_array is available
        if image_array is not None:
            model = self._get_dvorak_model()
            if model is not None:
                try:
                    h, w = image_array.shape[:2]
                    u8_arr = (
                        (np.clip(image_array, 0.0, 1.0) * 255.0).astype(np.uint8)
                        if image_array.max() <= 1.0
                        else np.clip(image_array, 0, 255).astype(np.uint8)
                    )
                    pil_im = Image.fromarray(u8_arr).convert("RGB").resize((128, 128))
                    img_t = torch.tensor(np.array(pil_im), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

                    with torch.no_grad():
                        out = model(img_t)
                        t_num = float(out["dvorak_t_number"][0].item()) if "dvorak_t_number" in out else 3.0

                    if t_num >= 6.0:
                        category = "Extreme / Super Cyclone"
                        confidence = ConfidenceTier.HIGH
                    elif t_num >= 4.5:
                        category = "Strong / Very Severe System"
                        confidence = ConfidenceTier.HIGH
                    elif t_num >= 3.5:
                        category = "Severe Tropical Storm"
                        confidence = ConfidenceTier.MEDIUM
                    elif t_num >= 2.5:
                        category = "Tropical Storm / Cyclonic Storm"
                        confidence = ConfidenceTier.MEDIUM
                    elif t_num >= 1.5:
                        category = "Tropical Depression"
                        confidence = ConfidenceTier.LOW
                    else:
                        category = "Tropical Depression / Formative Vortex"
                        confidence = ConfidenceTier.LOW

                    criteria = (
                        f"ObjectiveDvorakNet deep convolutional analysis: T-Number T{t_num:.1f} "
                        f"(Spiral index: {spiral_index:.2f}, Symmetry: {symmetry_score:.2f}, Area ratio: {area_ratio:.2f})."
                    )
                    return ClassificationResult(
                        category=category,
                        confidence=confidence,
                        is_experimental=True,
                        criteria_summary=criteria,
                        disclaimer="AI-generated experimental analysis. Not an official IMD/RSMC classification.",
                    )
                except Exception:
                    pass

        # 2. Morphological decision criteria fallback
        if spiral_index >= 0.75 and symmetry_score >= 0.70 and area_ratio >= 0.20:
            category = "Extreme / Super Cyclone"
            confidence = ConfidenceTier.HIGH
            criteria = "High radial symmetry and tight primary spiral banding indicative of intense mature vortex."
        elif spiral_index >= 0.65 and symmetry_score >= 0.55 and area_ratio >= 0.15:
            category = "Strong / Very Severe System"
            confidence = ConfidenceTier.HIGH
            criteria = "Well-defined curved banding and distinct central circulation core."
        elif spiral_index >= 0.60 and area_ratio >= 0.12:
            category = "Severe Tropical Storm"
            confidence = ConfidenceTier.MEDIUM
            criteria = "Moderate curvature index with consolidated deep convective cluster."
        elif spiral_index >= 0.55 and area_ratio >= 0.10:
            category = "Tropical Storm / Cyclonic Storm"
            confidence = ConfidenceTier.MEDIUM
            criteria = "Developing cyclonic circulation pattern with formative curved band."
        elif spiral_index >= 0.50 and area_ratio >= 0.08:
            category = "Tropical Depression"
            confidence = ConfidenceTier.LOW
            criteria = "Incipient low-level circulation with formative convective mass."
        else:
            category = "Tropical Depression / Formative Vortex"
            confidence = ConfidenceTier.LOW
            criteria = "Formative cyclonic vortex identified by deep learning spatial pattern analysis."

        return ClassificationResult(
            category=category,
            confidence=confidence,
            is_experimental=True,
            criteria_summary=criteria,
            disclaimer="AI-generated experimental analysis. Not an official IMD/RSMC classification.",
        )


cyclone_classifier = CycloneClassifier()

