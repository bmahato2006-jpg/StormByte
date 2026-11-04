"""Dataset and augmentation pipeline for Cyclone Intensity Regression."""

from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from backend.app.ml.intensity.config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PROJECT_ROOT,
    TARGET_MEAN,
    TARGET_STD,
)


class IntensityTransform:
    """Deterministic & physically safe transform pipeline for infrared satellite imagery."""

    def __init__(self, target_size: Tuple[int, int] = IMAGE_SIZE, is_training: bool = False):
        self.target_size = target_size
        self.is_training = is_training
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.size != self.target_size:
            img = img.resize(self.target_size, Image.Resampling.BICUBIC)

        if self.is_training:
            # Orientation and rotation augmentations (cyclonic circulation is rotation invariant)
            if random.random() > 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if random.random() > 0.5:
                # 90/180/270 degree discrete rotations or continuous rotation
                angle = random.uniform(0.0, 360.0)
                img = img.rotate(angle, resample=Image.Resampling.BICUBIC)

        # Convert to float array normalized to [0.0, 1.0]
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]

        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        tensor = (tensor - self.mean) / self.std
        return tensor


class CycloneIntensityDataset(Dataset):
    """PyTorch Dataset for Tropical Cyclone Intensity Regression."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        is_training: bool = False,
        target_mean: float = TARGET_MEAN,
        target_std: float = TARGET_STD,
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        self.is_training = is_training
        self.target_mean = target_mean
        self.target_std = target_std
        self.transform = IntensityTransform(is_training=is_training)

        # Validate rows
        self.records: List[Dict] = []
        for _, row in self.df.iterrows():
            # Try raw image path first, fallback to processed
            raw_p = PROJECT_ROOT / str(row["image_path"])
            proc_p = PROJECT_ROOT / str(row["processed_image_path"])
            img_path = raw_p if raw_p.exists() else proc_p

            if not img_path.exists():
                continue

            wind_kt = float(row["wind_speed_knots"])
            norm_target = (wind_kt - self.target_mean) / self.target_std

            self.records.append({
                "image_path": img_path,
                "wind_speed_knots": wind_kt,
                "normalized_target": norm_target,
                "sample_id": row.get("sample_id", ""),
                "category": row.get("intensity_category", ""),
            })

        if len(self.records) == 0:
            raise ValueError(f"Zero valid image records found in manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        rec = self.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        tensor = self.transform(img)
        norm_target = torch.tensor([rec["normalized_target"]], dtype=torch.float32)
        raw_target = rec["wind_speed_knots"]
        return tensor, norm_target, raw_target
