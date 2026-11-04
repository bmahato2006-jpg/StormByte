"""Image preprocessing, channel normalization, and augmentation for Cyclone-X Detection (Phase 9).

Strict Robustness:
- Never assumes images have identical channels or color spaces.
- Transparently handles Grayscale (TIR1), RGB (VIS), RGBA, Palette, and 16-bit GeoTIFF.
- Rejects corrupt, empty, or unreadable image payloads with informative diagnostic messages.
- Deterministic, reproducible transformations.
"""

import io
from pathlib import Path
import random
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch

from backend.app.ml.detection.config import IMAGE_SIZE


# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



def validate_image_payload(
    image_input: Union[bytes, str, Path, Image.Image]
) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """Validates whether an image payload is readable, non-empty, and structurally intact.

    Returns:
        is_valid: bool
        error_message: Optional[str]
        pil_image: Optional[Image.Image]
    """
    if isinstance(image_input, Image.Image):
        try:
            image_input.load()
            return True, None, image_input
        except Exception as e:
            return False, f"Invalid image object: {str(e)}", None

    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            return False, f"Image file not found at path: {p}", None
        if p.stat().st_size == 0:
            return False, f"Image file is empty (0 bytes): {p}", None
        try:
            with open(p, "rb") as f:
                raw_bytes = f.read()
        except Exception as e:
            return False, f"Failed to read file from disk: {str(e)}", None
    elif isinstance(image_input, bytes):
        raw_bytes = image_input
        if len(raw_bytes) == 0:
            return False, "Image byte payload is empty (0 bytes).", None
    else:
        return False, f"Unsupported image input type: {type(image_input)}", None

    try:
        buf = io.BytesIO(raw_bytes)
        img = Image.open(buf)
        img.load()  # Force reading entire raster data to detect corruption
        return True, None, img
    except Exception as e:
        return False, f"Corrupted or unsupported image file: {str(e)}", None


def standardize_channels_to_rgb(img: Image.Image) -> Image.Image:
    """Converts arbitrary satellite channel modes (L, P, RGBA, CMYK, 16-bit) to standardized RGB."""
    mode = img.mode

    if mode == "RGB":
        return img

    if mode == "L":
        # Grayscale TIR1 / single-band infrared: replicate across 3 channels
        return img.convert("RGB")

    if mode == "RGBA":
        # Alpha composite over solid dark background (representative of space/ocean)
        background = Image.new("RGB", img.size, (0, 0, 0))
        background.paste(img, mask=img.split()[3])
        return background

    if mode == "P":
        # Palette-indexed image
        return img.convert("RGB")

    if mode in ("I;16", "I", "F"):
        # 16-bit or 32-bit floating point radiometry raster (e.g. raw satellite GeoTIFF)
        arr = np.array(img, dtype=np.float32)
        min_v, max_v = np.min(arr), np.max(arr)
        if max_v > min_v:
            norm_arr = ((arr - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
        else:
            norm_arr = np.zeros(arr.shape, dtype=np.uint8)
        return Image.fromarray(norm_arr).convert("RGB")

    # Fallback to standard RGB conversion
    return img.convert("RGB")


def compute_cloud_fraction(img_rgb: Image.Image, threshold: int = 140) -> Tuple[float, np.ndarray]:
    """Computes convective cloud fraction and binary cloud mask from satellite image.

    In thermal infrared (TIR1), cold cloud tops emit low radiance and appear bright
    in inverted visual products (pixel value > threshold).
    """
    gray = ImageOps.grayscale(img_rgb)
    arr = np.array(gray, dtype=np.uint8)
    mask = (arr >= threshold).astype(np.uint8)
    fraction = float(np.sum(mask)) / float(mask.size)
    return round(fraction, 4), mask


class DetectionTransform:
    """Deterministic PIL and PyTorch image transformation and augmentation pipeline."""

    def __init__(
        self,
        target_size: Tuple[int, int] = IMAGE_SIZE,
        is_training: bool = False,
    ):
        self.target_size = target_size
        self.is_training = is_training
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.size != self.target_size:
            img = img.resize(self.target_size, Image.Resampling.BILINEAR)

        if self.is_training:
            if random.random() > 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if random.random() > 0.5:
                angle = random.uniform(-15.0, 15.0)
                img = img.rotate(angle, resample=Image.Resampling.BILINEAR)
            if random.random() > 0.5:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(random.uniform(0.85, 1.15))
            if random.random() > 0.5:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(random.uniform(0.85, 1.15))

        # Convert PIL Image to normalized PyTorch tensor [C, H, W] in [0.0, 1.0]
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

        # Standardize using ImageNet mean & std
        tensor = (tensor - self.mean) / self.std
        return tensor


def get_detection_transform(
    target_size: Tuple[int, int] = IMAGE_SIZE,
    is_training: bool = False,
) -> DetectionTransform:
    """Creates reproducible PyTorch preprocessing and augmentation pipeline."""
    return DetectionTransform(target_size=target_size, is_training=is_training)


def preprocess_for_inference(
    image_input: Union[bytes, str, Path, Image.Image],
    target_size: Tuple[int, int] = IMAGE_SIZE,
) -> Tuple[torch.Tensor, Image.Image, Dict[str, Any]]:
    """Complete inference preprocessing pipeline.

    Returns:
        tensor: torch.Tensor of shape (1, 3, H, W)
        rgb_image: Standardized PIL Image (target_size)
        metadata: Extraction metadata dictionary
    """
    is_valid, err, pil_img = validate_image_payload(image_input)
    if not is_valid or pil_img is None:
        raise ValueError(err or "Invalid image payload")

    original_size = pil_img.size
    original_mode = pil_img.mode

    # Standardize channels to RGB
    rgb_img = standardize_channels_to_rgb(pil_img)
    resized_rgb = rgb_img.resize(target_size, Image.Resampling.BILINEAR)

    # Calculate cloud fraction
    cloud_frac, _ = compute_cloud_fraction(resized_rgb)

    # Convert to normalized PyTorch tensor
    transform = get_detection_transform(target_size=target_size, is_training=False)
    tensor = transform(resized_rgb).unsqueeze(0)  # Add batch dim: (1, 3, H, W)

    metadata = {
        "original_dimensions": {"width": original_size[0], "height": original_size[1]},
        "processed_dimensions": {"width": target_size[0], "height": target_size[1]},
        "original_mode": original_mode,
        "standardized_mode": "RGB",
        "cloud_fraction": cloud_frac,
    }

    return tensor, resized_rgb, metadata
