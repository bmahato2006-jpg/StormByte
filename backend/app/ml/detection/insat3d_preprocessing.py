"""Scientific Image Preprocessing for INSAT-3D Infrared Imagery.

Supports:
- 8-bit & 16-bit raw geostationary satellite rasters (TIR1 / 10.8 µm band)
- Channel mode standardization (Grayscale, Palette, RGB, RGBA)
- Extreme value clipping & contrast enhancement preservation
- Model tensor standardization (3x256x256, ImageNet normalization)
- Convective cloud top fraction calculation
"""

import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageOps

from backend.app.ml.detection.config import IMAGE_SIZE
from backend.app.ml.detection.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DetectionTransform,
    compute_cloud_fraction,
    standardize_channels_to_rgb,
    validate_image_payload,
)


class INSAT3DPreprocessor:
    """Scientific preprocessor specifically calibrated for INSAT-3D TIR1/VIS cyclone imagery."""

    def __init__(self, target_size: Tuple[int, int] = IMAGE_SIZE):
        self.target_size = target_size
        self.transform = DetectionTransform(target_size=target_size, is_training=False)

    def process(
        self,
        image_input: Union[bytes, str, Path, Image.Image],
    ) -> Tuple[Any, Image.Image, Dict[str, Any]]:
        """Processes an INSAT-3D image into a model-ready PyTorch tensor, PIL RGB image, and metadata.

        Returns:
            tensor: torch.Tensor of shape (1, 3, 256, 256)
            pil_rgb: Standardized PIL Image (target_size)
            metadata: Scientific metadata dictionary
        """
        is_valid, err, pil_img = validate_image_payload(image_input)
        if not is_valid or pil_img is None:
            raise ValueError(err or "Corrupted or unreadable INSAT-3D image")

        orig_w, orig_h = pil_img.size
        orig_mode = pil_img.mode

        # 1. Standardize channels to RGB
        rgb_img = standardize_channels_to_rgb(pil_img)

        # 2. Extract convective cloud fraction on full raster before resizing
        cloud_fraction, _ = compute_cloud_fraction(rgb_img, threshold=140)

        # 3. Bilinear resize to standard model dimensions
        resized_rgb = rgb_img.resize(self.target_size, Image.Resampling.BILINEAR)

        # 4. Compute pixel intensity statistics
        gray_arr = np.array(ImageOps.grayscale(resized_rgb), dtype=np.float32)
        mean_intensity = float(np.mean(gray_arr))
        std_intensity = float(np.std(gray_arr))
        min_intensity = float(np.min(gray_arr))
        max_intensity = float(np.max(gray_arr))

        # 5. Convert to normalized PyTorch tensor [1, 3, H, W]
        tensor = self.transform(resized_rgb).unsqueeze(0)

        metadata = {
            "satellite": "INSAT-3D",
            "sensor": "IMAGER",
            "channel": "TIR1 (10.8 µm)",
            "original_resolution": f"{orig_w}x{orig_h}",
            "processed_resolution": f"{self.target_size[0]}x{self.target_size[1]}",
            "original_color_mode": orig_mode,
            "cloud_fraction": round(cloud_fraction, 4),
            "mean_intensity": round(mean_intensity, 2),
            "std_intensity": round(std_intensity, 2),
            "dynamic_range": [round(min_intensity, 1), round(max_intensity, 1)],
            "preprocessing_version": "v2.0-insat3d-scientific",
        }

        return tensor, resized_rgb, metadata

    # Alias for API symmetry
    preprocess_single = process


insat3d_preprocessor = INSAT3DPreprocessor()
