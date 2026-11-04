"""Cyclone-X Trajectory Forecasting Package (Physics & ML).

Provides:
- Physics-guided advection baseline (TrajectoryForecaster)
- PyTorch deep learning sequence modeling (TrajectoryLSTM)
- Feature engineering, training, inference, and evaluation engines
"""

from backend.app.ml.trajectory.physics import TrajectoryForecaster, trajectory_forecaster
from backend.app.ml.trajectory.model import TrajectoryLSTM
from backend.app.ml.trajectory.trainer import TrajectoryTrainer
from backend.app.ml.trajectory.evaluator import TrajectoryEvaluator, evaluator
from backend.app.ml.trajectory.inference import TrajectoryInferenceService, trajectory_inference_service

__all__ = [
    "TrajectoryForecaster",
    "trajectory_forecaster",
    "TrajectoryLSTM",
    "TrajectoryTrainer",
    "TrajectoryEvaluator",
    "evaluator",
    "TrajectoryInferenceService",
    "trajectory_inference_service",
]


