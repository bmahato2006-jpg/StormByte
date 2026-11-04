"""Production PyTorch Data Loaders & Dataset Implementations for Cyclone-X.

Implements:
1. SatelliteCenterNetDataset: Ingests satellite raster tiles (GeoTIFF/NetCDF/PNG/NumPy),
   computes ground-truth sub-pixel coordinates, generates 2D Gaussian heatmaps with
   adaptive radius kernels, and extracts sub-pixel coordinate offset vectors.
2. TrajectorySequenceDataset: Parses IBTrACS historical tracks into temporal sliding windows
   with 8 kinematic and atmospheric steering features:
   [lat, lon, wind, psl, sst, shear, u500, v500].
"""

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image


# =============================================================================
# 1. 2D Gaussian Heatmap Generator for CenterNet
# =============================================================================

def gaussian2d(shape: Tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    """Generate a 2D Gaussian kernel array centered at (radius, radius)."""
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def generate_2d_gaussian_heatmap(
    heatmap: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    k: float = 1.0,
) -> np.ndarray:
    """Draw a 2D Gaussian bell curve onto target heatmap in-place using element-wise maximum.

    Args:
        heatmap: Target 2D float32 array [H, W] initialized with zeros.
        center: (center_x, center_y) integer peak location in heatmap space.
        radius: Integer radius of the Gaussian kernel footprint.
        k: Peak magnitude scalar (default 1.0).

    Returns:
        Modified heatmap array.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6.0)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0], heatmap.shape[1]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def compute_gaussian_radius(
    box_height: float, box_width: float, min_overlap: float = 0.7
) -> int:
    """Compute Gaussian radius from cyclone bounding box or estimated circulation radius.

    Based on CornerNet / CenterNet formulation.
    """
    a1 = 1.0
    b1 = box_height + box_width
    c1 = box_width * box_height * (1.0 - min_overlap) / (1.0 + min_overlap)
    sq1 = math.sqrt(max(0.0, b1**2 - 4 * a1 * c1))
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4.0
    b2 = 2.0 * (box_height + box_width)
    c2 = (1.0 - min_overlap) * box_width * box_height
    sq2 = math.sqrt(max(0.0, b2**2 - 4 * a2 * c2))
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (box_height + box_width)
    c3 = (min_overlap - 1.0) * box_width * box_height
    sq3 = math.sqrt(max(0.0, b3**2 - 4 * a3 * c3))
    r3 = (b3 + sq3) / (2 * a3)

    return max(2, int(min(r1, r2, r3)))


# =============================================================================
# 2. SatelliteCenterNetDataset (PyTorch Vision Dataset)
# =============================================================================

class SatelliteCenterNetDataset(Dataset):
    """Dataset for training CycloneCenterNet eye keypoint detection.

    Converts satellite images (or simulated tensors) and ground-truth storm eye coordinates
    into downsampled Gaussian heatmaps and sub-pixel continuous offset targets.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        input_size: Tuple[int, int] = (512, 512),
        downsample_ratio: int = 4,
        default_radius: int = 6,
        transform: Optional[Callable] = None,
        is_training: bool = False,
    ):
        """
        Args:
            samples: List of dicts, each with:
                - 'image': np.ndarray [H, W] or path (str/Path) to raster file
                - 'center_x': float, ground-truth eye x in pixel space
                - 'center_y': float, ground-truth eye y in pixel space
                - 'eye_radius': optional float, estimated RMW or core radius
                - 'has_eye': optional bool, True if organized eye exists (default True)
            input_size: (H, W) model input dimensions.
            downsample_ratio: Output stride (default 4 -> 512x512 to 128x128).
            default_radius: Fallback Gaussian kernel radius in output feature space.
            transform: Optional albumentations or torchvision image transforms.
            is_training: Whether data augmentations are enabled.
        """
        self.samples = samples
        self.input_size = input_size
        self.downsample_ratio = downsample_ratio
        self.output_size = (
            input_size[0] // downsample_ratio,
            input_size[1] // downsample_ratio,
        )
        self.default_radius = default_radius
        self.transform = transform
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, source: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image array and convert to float32 in [0, 1]."""
        if isinstance(source, np.ndarray):
            img = source.copy()
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Satellite raster not found at {path}")
            if path.suffix.lower() in [".tif", ".tiff"]:
                try:
                    import rasterio
                    with rasterio.open(path) as src:
                        img = src.read(1).astype(np.float32)
                except ImportError:
                    pil_img = Image.open(path)
                    img = np.array(pil_img, dtype=np.float32)
            else:
                pil_img = Image.open(path).convert("L")
                img = np.array(pil_img, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported image type: {type(source)}")

        # Rescale / Normalize to [0, 1]
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)  # [H, W, 1]
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)  # Convert to 3-channel for ResNet [H, W, 3]

        if img.max() > 1.0:
            p_min, p_max = float(np.min(img)), float(np.max(img))
            if p_max > p_min:
                img = (img - p_min) / (p_max - p_min)
            else:
                img = np.zeros_like(img)

        # Resize to input_size if necessary
        if (img.shape[0], img.shape[1]) != self.input_size:
            pil_img = Image.fromarray((img[:, :, 0] * 255.0).astype(np.uint8))
            pil_img = pil_img.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            img = np.stack([arr, arr, arr], axis=-1)

        return img.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        raw_img = sample["image"]
        img = self._load_image(raw_img)

        orig_h, orig_w = (
            raw_img.shape[:2] if isinstance(raw_img, np.ndarray) else self.input_size
        )
        scale_x = self.input_size[1] / float(orig_w)
        scale_y = self.input_size[0] / float(orig_h)

        # Scaled eye coordinates in input space
        eye_x = float(sample.get("center_x", orig_w / 2.0) * scale_x)
        eye_y = float(sample.get("center_y", orig_h / 2.0) * scale_y)
        has_eye = sample.get("has_eye", True)

        # Dynamic spatial translation jitter during training
        if self.is_training:
            dx = int(np.random.randint(-28, 29))
            dy = int(np.random.randint(-28, 29))
            # Shift image with edge replication
            img = np.roll(np.roll(img, dx, axis=1), dy, axis=0)
            if dx > 0:
                img[:, :dx, :] = img[:, dx:dx+1, :]
            elif dx < 0:
                img[:, dx:, :] = img[:, dx-1:dx, :]
            if dy > 0:
                img[:dy, :, :] = img[dy:dy+1, :, :]
            elif dy < 0:
                img[dy:, :, :] = img[dy-1:dy, :, :]

            eye_x = max(10.0, min(self.input_size[1] - 10.0, eye_x + dx))
            eye_y = max(10.0, min(self.input_size[0] - 10.0, eye_y + dy))

            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
                eye_x = self.input_size[1] - 1.0 - eye_x

        out_h, out_w = self.output_size

        # Initialize ground truth tensors
        # 1. Keypoint Heatmap [1, out_h, out_w]
        heatmap = np.zeros((out_h, out_w), dtype=np.float32)
        # 2. Sub-pixel Offset [2, out_h, out_w]
        offset = np.zeros((2, out_h, out_w), dtype=np.float32)
        # 3. Offset Mask [1, out_h, out_w] (only 1 at true peak cell)
        offset_mask = np.zeros((1, out_h, out_w), dtype=np.float32)

        if has_eye and (0 <= eye_x < self.input_size[1]) and (0 <= eye_y < self.input_size[0]):
            # Downsampled target coordinates
            ct_x = eye_x / float(self.downsample_ratio)
            ct_y = eye_y / float(self.downsample_ratio)
            ct_int_x = int(ct_x)
            ct_int_y = int(ct_y)

            # Clamp to valid output boundaries
            ct_int_x = min(out_w - 1, max(0, ct_int_x))
            ct_int_y = min(out_h - 1, max(0, ct_int_y))

            # Dynamic radius from eye diameter or default
            eye_rad = sample.get("eye_radius", None)
            if eye_rad is not None and eye_rad > 0:
                kernel_radius = max(2, int((eye_rad * scale_x) / (self.downsample_ratio * 2)))
            else:
                kernel_radius = self.default_radius

            generate_2d_gaussian_heatmap(heatmap, (ct_int_x, ct_int_y), radius=kernel_radius)

            # Sub-pixel continuous discretization offset
            offset[0, ct_int_y, ct_int_x] = float(ct_x - ct_int_x)
            offset[1, ct_int_y, ct_int_x] = float(ct_y - ct_int_y)
            offset_mask[0, ct_int_y, ct_int_x] = 1.0

        # Convert image from [H, W, C] to PyTorch tensor [C, H, W]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).contiguous().float()
        offset_tensor = torch.from_numpy(offset).contiguous().float()
        offset_mask_tensor = torch.from_numpy(offset_mask).contiguous().float()

        return {
            "image": img_tensor,
            "target_heatmap": heatmap_tensor,
            "target_offset": offset_tensor,
            "offset_mask": offset_mask_tensor,
            "true_coords": torch.tensor([eye_x, eye_y], dtype=torch.float32),
        }


# =============================================================================
# 3. TrajectorySequenceDataset (PyTorch Sequence Dataset)
# =============================================================================

# Standard 8-feature order: [lat, lon, wind, psl, sst, shear, u500, v500]
FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "max_wind_kt",
    "central_pressure_hpa",
    "sea_surface_temp_c",
    "vertical_wind_shear_kt",
    "steering_u500_kt",
    "steering_v500_kt",
]


class TrajectorySequenceDataset(Dataset):
    """Dataset for training TrajectoryBiLSTM using multi-feature sliding windows.

    Accepts 8-dimensional atmospheric & kinematic sequences:
    [lat, lon, wind, psl, sst, shear, u500, v500]
    and extracts future displacement targets (delta_lat, delta_lon) for multi-horizon forecast.
    """

    def __init__(
        self,
        storm_tracks: List[List[Dict[str, float]]],
        seq_length: int = 4,
        forecast_steps: int = 3,
        step_hours: int = 6,
        scaler: Optional[Dict[str, Tuple[float, float]]] = None,
        is_training: bool = True,
    ):
        """
        Args:
            storm_tracks: List of storms, each storm being an ordered list of fix dicts.
            seq_length: Input history length (e.g. 4 fixes = 24h).
            forecast_steps: Forecast horizon steps (e.g. 3 fixes = +6h, +12h, +24h).
            step_hours: Hours per observation step (default 6h).
            scaler: Optional pre-computed mean/std dict for normalization.
            is_training: Whether dataset is used for computing scalers.
        """
        self.seq_length = seq_length
        self.forecast_steps = forecast_steps
        self.step_hours = step_hours
        self.is_training = is_training

        # Extract sliding windows
        self.windows: List[Tuple[np.ndarray, np.ndarray]] = []
        raw_sequences = []

        for track in storm_tracks:
            if len(track) < (seq_length + forecast_steps):
                continue

            matrix = np.zeros((len(track), len(FEATURE_COLUMNS)), dtype=np.float32)
            for i, fix in enumerate(track):
                matrix[i, 0] = float(fix.get("latitude", fix.get("lat", 0.0)))
                matrix[i, 1] = float(fix.get("longitude", fix.get("lon", 0.0)))
                matrix[i, 2] = float(fix.get("max_wind_kt", fix.get("wind", 30.0)))
                matrix[i, 3] = float(fix.get("central_pressure_hpa", fix.get("psl", 1000.0)))
                matrix[i, 4] = float(fix.get("sea_surface_temp_c", fix.get("sst", 28.0)))
                matrix[i, 5] = float(fix.get("vertical_wind_shear_kt", fix.get("shear", 15.0)))
                matrix[i, 6] = float(fix.get("steering_u500_kt", fix.get("u500", 0.0)))
                matrix[i, 7] = float(fix.get("steering_v500_kt", fix.get("v500", 0.0)))

            raw_sequences.append(matrix)

        # Compute or apply normalization
        if scaler is not None:
            self.scaler = scaler
        elif len(raw_sequences) > 0 and is_training:
            all_data = np.concatenate(raw_sequences, axis=0)
            mean = np.mean(all_data, axis=0)
            std = np.std(all_data, axis=0)
            std[std < 1e-4] = 1.0
            self.scaler = {
                "mean": tuple(float(m) for m in mean),
                "std": tuple(float(s) for s in std),
            }
        else:
            self.scaler = {
                "mean": tuple(0.0 for _ in range(8)),
                "std": tuple(1.0 for _ in range(8)),
            }

        mean_arr = np.array(self.scaler["mean"], dtype=np.float32)
        std_arr = np.array(self.scaler["std"], dtype=np.float32)

        # Generate (input_seq, target_displacements) pairs
        for matrix in raw_sequences:
            norm_matrix = (matrix - mean_arr) / std_arr
            for start in range(len(matrix) - seq_length - forecast_steps + 1):
                in_seq = norm_matrix[start : start + seq_length]

                # Target displacements in unnormalized degrees relative to current fix
                curr_lat = matrix[start + seq_length - 1, 0]
                curr_lon = matrix[start + seq_length - 1, 1]

                targets = np.zeros((forecast_steps, 2), dtype=np.float32)
                for step in range(forecast_steps):
                    future_idx = start + seq_length + step
                    targets[step, 0] = matrix[future_idx, 0] - curr_lat  # dlat
                    targets[step, 1] = matrix[future_idx, 1] - curr_lon  # dlon

                self.windows.append((in_seq, targets))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        in_seq, targets = self.windows[idx]
        return torch.from_numpy(in_seq).float(), torch.from_numpy(targets).float()


# =============================================================================
# 4. DataLoader Factory Helpers
# =============================================================================

def create_centernet_dataloader(
    samples: List[Dict[str, Any]],
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    input_size: Tuple[int, int] = (512, 512),
    is_training: bool = False,
) -> DataLoader:
    """Build production PyTorch DataLoader for satellite keypoint training."""
    dataset = SatelliteCenterNetDataset(samples, input_size=input_size, is_training=is_training)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def create_trajectory_dataloader(
    storm_tracks: List[List[Dict[str, float]]],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    seq_length: int = 4,
    forecast_steps: int = 3,
    scaler: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[DataLoader, Dict[str, Tuple[float, float]]]:
    """Build production PyTorch DataLoader for trajectory BiLSTM sequence modeling."""
    dataset = TrajectorySequenceDataset(
        storm_tracks,
        seq_length=seq_length,
        forecast_steps=forecast_steps,
        scaler=scaler,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader, dataset.scaler
