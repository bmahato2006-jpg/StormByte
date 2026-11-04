"""Scientific Image Preprocessing & Quality Assessment Module.

Handles validation, normalization, contrast enhancement, and image quality metrics
for meteorological satellite imagery using Pillow and NumPy.
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from backend.app.schemas.analysis import PreprocessingMetadata

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_IMAGE_SIZE = 15 * 1024 * 1024  # 15 MB
MIN_DIMENSION = 64


def validate_image_bytes(content: bytes, filename: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Verify size, extension, and raster integrity without modifying the original."""
    if len(content) > MAX_IMAGE_SIZE:
        return False, f"File size ({len(content)} bytes) exceeds maximum limit of 15 MB"

    if len(content) < 32:
        return False, "File is too small or empty"

    if filename:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported image extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"

    try:
        with Image.open(io.BytesIO(content)) as img:
            img.verify()
    except Exception as e:
        return False, f"Image file is corrupted or cannot be decoded: {str(e)}"

    return True, None


def compute_laplacian_sharpness(gray_array: np.ndarray) -> float:
    """Compute sharpness via the variance of the 2D discrete Laplacian filter."""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = gray_array.shape
    if h < 3 or w < 3:
        return 0.0

    # 2D convolution for interior pixels
    sub = (
        gray_array[0:-2, 1:-1]
        + gray_array[2:, 1:-1]
        + gray_array[1:-1, 0:-2]
        + gray_array[1:-1, 2:]
        - 4.0 * gray_array[1:-1, 1:-1]
    )
    variance = float(np.var(sub))
    return round(variance * 1000.0, 2)


def preprocess_satellite_image(
    content: bytes,
    target_size: Tuple[int, int] = (512, 512),
    enhance_contrast: bool = True
) -> Tuple[Image.Image, np.ndarray, PreprocessingMetadata]:
    """Preprocess satellite raster: resize, normalize, contrast stretch, and assess quality.

    Preserves original raster data while returning normalized array and quality metadata.
    """
    operations: List[str] = ["LOAD_BYTE_STREAM"]

    with Image.open(io.BytesIO(content)) as orig_img:
        orig_w, orig_h = orig_img.size
        channels = len(orig_img.getbands())

        # Convert to RGB if palette/alpha
        if orig_img.mode in ("RGBA", "P"):
            rgb_img = orig_img.convert("RGB")
            operations.append(f"CONVERT_MODE_{orig_img.mode}_TO_RGB")
        else:
            rgb_img = orig_img.copy()

        # Resize smoothly to target working resolution
        if (orig_w, orig_h) != target_size:
            proc_img = rgb_img.resize(target_size, Image.Resampling.BILINEAR)
            operations.append(f"RESIZE_BILINEAR_{orig_w}x{orig_h}_TO_{target_size[0]}x{target_size[1]}")
        else:
            proc_img = rgb_img

        # Grayscale conversion for feature extraction
        gray_img = proc_img.convert("L")
        operations.append("GRAYSCALE_CONVERSION")

        # Contrast enhancement via histogram equalization if enabled
        if enhance_contrast:
            gray_img = ImageOps.equalize(gray_img)
            operations.append("HISTOGRAM_EQUALIZATION")

        # Convert to normalized float32 array in [0.0, 1.0]
        gray_array = np.asarray(gray_img, dtype=np.float32) / 255.0
        operations.append("MIN_MAX_NORMALIZATION_[0,1]")

        # Quality assessment
        sharpness = compute_laplacian_sharpness(gray_array)
        dyn_range = float(np.max(gray_array) - np.min(gray_array))
        std_dev = float(np.std(gray_array))

        # Composite quality score 0-100 based on dynamic range and sharpness
        quality_score = min(100.0, max(10.0, (dyn_range * 50.0) + min(50.0, sharpness * 2.5)))
        quality_score = round(quality_score, 1)

        metadata = PreprocessingMetadata(
            original_dimensions={"width": orig_w, "height": orig_h},
            processed_dimensions={"width": target_size[0], "height": target_size[1]},
            channels=channels,
            quality_score=quality_score,
            sharpness=sharpness,
            dynamic_range=round(dyn_range, 3),
            operations_applied=operations,
        )

        return proc_img, gray_array, metadata

