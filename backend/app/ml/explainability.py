"""Explainable AI (XAI) & Visual Detection Overlay Module.

Renders transparent RGBA visual attribution overlays showing detected storm bounds,
estimated circulation center crosshairs, and convective core radius.
Provides deterministic natural-language reasoning of visual features.
"""

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw

from backend.app.schemas.analysis import ExplainabilityResult


class ExplainabilityEngine:
    """Generates visual explanation overlays and scientific feature reasoning."""

    def __init__(self, overlay_dir: Optional[str] = None):
        if overlay_dir:
            self.overlay_dir = Path(overlay_dir)
        else:
            base = Path(__file__).resolve().parent.parent.parent
            self.overlay_dir = base / "uploads" / "overlays"
        self.overlay_dir.mkdir(parents=True, exist_ok=True)

    def generate_overlay_image(
        self,
        dimensions: Tuple[int, int],
        bbox: Optional[Dict[str, int]],
        center_x: float,
        center_y: float,
        radius: float,
        detected: bool,
        cloud_mask: Optional[np.ndarray] = None
    ) -> Tuple[Image.Image, str]:
        """Create a transparent RGBA image with annotated detection overlays."""
        w, h = dimensions
        # Create transparent canvas
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        if not detected:
            # Draw subtle gray frame indicating no organized candidate
            draw.rectangle([10, 10, w - 10, h - 10], outline=(148, 163, 184, 100), width=2)
        else:
            # 1. Semi-transparent convective canopy highlight
            if cloud_mask is not None:
                # Tint segmented cloud region with a subtle cyan glow
                mask_img = Image.fromarray((cloud_mask * 40).astype(np.uint8), mode="L")
                tint = Image.new("RGBA", (w, h), (0, 240, 255, 0))
                overlay.paste(tint, (0, 0), mask_img)

            # 2. Draw Storm Candidate Bounding Box (Cyan)
            if bbox:
                x0, y0 = max(0, bbox["x_min"]), max(0, bbox["y_min"])
                x1, y1 = min(w - 1, bbox["x_max"]), min(h - 1, bbox["y_max"])
                draw.rectangle([x0, y0, x1, y1], outline=(0, 240, 255, 200), width=2)

                # Corner accent brackets
                corner_len = min(20, bbox.get("width", 50) // 4)
                # Top-left
                draw.line([(x0, y0), (x0 + corner_len, y0)], fill=(0, 240, 255, 255), width=3)
                draw.line([(x0, y0), (x0, y0 + corner_len)], fill=(0, 240, 255, 255), width=3)
                # Bottom-right
                draw.line([(x1, y1), (x1 - corner_len, y1)], fill=(0, 240, 255, 255), width=3)
                draw.line([(x1, y1), (x1, y1 - corner_len)], fill=(0, 240, 255, 255), width=3)

            # 3. Draw Convective Core Radius (Dashed Amber Circle)
            r = max(10.0, radius)
            draw.ellipse(
                [center_x - r, center_y - r, center_x + r, center_y + r],
                outline=(245, 158, 11, 220),
                width=2
            )

            # 4. Draw Center Crosshair Target (Amber / Gold)
            cross_len = 16
            draw.line(
                [(center_x - cross_len, center_y), (center_x + cross_len, center_y)],
                fill=(251, 191, 36, 255),
                width=2
            )
            draw.line(
                [(center_x, center_y - cross_len), (center_x, center_y + cross_len)],
                fill=(251, 191, 36, 255),
                width=2
            )
            # Center target bullseye
            draw.ellipse(
                [center_x - 4, center_y - 4, center_x + 4, center_y + 4],
                fill=(251, 191, 36, 255),
                outline=(255, 255, 255, 255),
                width=1
            )

        # Convert to Base64 data URI
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64_str}"

        return overlay, data_uri

    def explain(
        self,
        analysis_id: str,
        dimensions: Tuple[int, int],
        bbox: Optional[Dict[str, int]],
        center_x: float,
        center_y: float,
        radius: float,
        detected: bool,
        spiral_score: float,
        symmetry_score: float,
        cloud_mask: Optional[np.ndarray] = None,
        model_version: str = "0.4.0"
    ) -> ExplainabilityResult:
        """Synthesize explainability reasoning and generate detection overlay."""
        overlay_img, b64_data = self.generate_overlay_image(
            dimensions=dimensions,
            bbox=bbox,
            center_x=center_x,
            center_y=center_y,
            radius=radius,
            detected=detected,
            cloud_mask=cloud_mask,
        )

        # Save static file for direct HTTP retrieval
        file_name = f"overlay_{analysis_id}.png"
        file_path = self.overlay_dir / file_name
        overlay_img.save(file_path, format="PNG")
        url = f"/api/analysis/{analysis_id}/overlay"

        if detected:
            explanation = (
                f"The detector identified a coherent convective cloud mass exhibiting a spiral curvature "
                f"index of {spiral_score:.2f} and radial symmetry convergence of {symmetry_score:.2f}. "
                f"The estimated circulation center at pixel ({center_x:.0f}, {center_y:.0f}) correlates with "
                f"the convergence of tangential infrared brightness temperature gradients."
            )
        else:
            explanation = (
                "No organized cyclonic vortex was identified. Cloud structures lack sufficient rotational symmetry "
                "or consolidated core coverage to satisfy tropical cyclone detection criteria."
            )

        regions: List[Dict[str, Any]] = []
        if bbox:
            regions.append({"type": "PRIMARY_CONVECTIVE_MASS", "bounding_box": bbox})
        if detected:
            regions.append({
                "type": "ESTIMATED_VORTEX_CORE",
                "center_pixel": [center_x, center_y],
                "radius_pixels": radius,
            })
        else:
            regions.append({
                "type": "BACKGROUND_ATMOSPHERIC_FIELD",
                "description": "Non-cyclonic cloud cover and background oceanic sector.",
            })

        return ExplainabilityResult(
            explanation_type="Radial Symmetry & Morphological Curvature Gradient",
            explanation_text=explanation,
            key_visual_regions=regions,
            overlay_image_url=url,
            overlay_base64=b64_data,
            model_version=model_version,
        )


explainability_engine = ExplainabilityEngine()

