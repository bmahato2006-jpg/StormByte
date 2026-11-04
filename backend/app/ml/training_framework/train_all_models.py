"""Master Training Orchestrator for Cyclone-X.

Trains, benchmarks, and exports production checkpoints for all 13 AI/ML models:
1. cyclone_centernet_resnet18: Deep Sub-pixel Cyclone Eye Keypoint Detector
2. cyclone_vit: Synoptic Vision Transformer (16x16 patch embeddings)
3. intensity_dvorak_net: Objective Dvorak T-Number Deep Regression Network
4. trajectory_bilstm_v1: 8-Feature Environmental Steering Recurrent Network
5. trajectory_transformer: Pre-LN Spatio-Temporal Self-Attention Transformer
6. surge_deep_surrogate: Hydrodynamic 2D Coastal Storm Surge Surrogate
7. holland_vortex_pinn: Physics-Informed Neural Network (Navier-Stokes Gradient Wind)
8. physics_advection_net: Neural Advection & Planetary Rossby Beta-Drift Net
9. precipitation_unet: Satellite Infrared to Precipitation Flux U-Net
10. satellite_radar_fusion_net: Multi-Modal Satellite + Doppler Radar Cross-Attention
11. landfall_net: Multi-Task Coastal Crossing Point & Lead-Time Regressor
12. flood_risk_net: Hydrological Runoff & Drainage Basin Inundation Net
13. impact_neural_engine: Multi-Hazard Composite Vulnerability Deep Network
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.app.ml.training_framework.models import (
    CenterNetFocalLoss,
    CenterNetRegLoss,
    CycloneCenterNet,
    TrajectoryBiLSTM,
)
from backend.app.ml.trajectory.transformer_model import TrajectoryTransformer
from backend.app.ml.training_framework.extended_models import (
    CycloneViT,
    ObjectiveDvorakNet,
    PrecipitationUNet,
    SatelliteRadarFusionNet,
    HollandPINN,
    PhysicsAdvectionNet,
    SurgeDeepSurrogate,
    FloodRiskNet,
    LandfallPredictorNet,
    ImpactNeuralEngine,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CycloneX.MasterTrainer")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CHECKPOINT_DIR = PROJECT_ROOT / "backend" / "app" / "ml" / "checkpoints"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_all_models(epochs: int = 3, device: Optional[torch.device] = None) -> Dict[str, Any]:
    dev = device or get_device()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Initiating Cyclone-X Master Deep Learning Training on device: {dev}")
    from backend.app.ml.training_framework.train_authentic_models import train_all_authentic
    manifest = train_all_authentic()
    logger.info(f"Master training complete! Checkpoints saved to {CHECKPOINT_DIR}")
    return manifest


if __name__ == "__main__":
    train_all_models(epochs=3)
