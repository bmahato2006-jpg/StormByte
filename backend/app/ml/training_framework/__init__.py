"""Cyclone-X Production Deep Learning and ML Training Framework.

Provides modular, scalable PyTorch and Gradient Boosting training pipelines for:
1. Rapid Intensification (RI) Predictor (Tabular / Statistical 24h RI Classification)
2. CycloneCenterNet (Sub-pixel Cyclone Eye Keypoint Detection with CenterNet Focal Loss)
3. TrajectoryBiLSTM (Environmental Steering Sequence Modeling for Track Prediction)
"""

from backend.app.ml.training_framework.data_loaders import (
    SatelliteCenterNetDataset,
    TrajectorySequenceDataset,
    generate_2d_gaussian_heatmap,
)
from backend.app.ml.training_framework.models import (
    CycloneCenterNet,
    CenterNetFocalLoss,
    TrajectoryBiLSTM,
)
from backend.app.ml.training_framework.train_pipeline import (
    train_ri_predictor,
    train_centernet,
    train_trajectory_bilstm,
)

__all__ = [
    "SatelliteCenterNetDataset",
    "TrajectorySequenceDataset",
    "generate_2d_gaussian_heatmap",
    "CycloneCenterNet",
    "CenterNetFocalLoss",
    "TrajectoryBiLSTM",
    "train_ri_predictor",
    "train_centernet",
    "train_trajectory_bilstm",
]
