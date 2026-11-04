"""Configuration and hyperparameter specifications for Cyclone-X Intensity Estimation."""

from pathlib import Path
from typing import Dict, Tuple

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "backend" / "app" / "ml" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Manifest Paths
MANIFEST_DIR = DATA_DIR / "processed" / "satellite" / "intensity" / "manifests"
TRAIN_MANIFEST_PATH = MANIFEST_DIR / "intensity_train.csv"
VAL_MANIFEST_PATH = MANIFEST_DIR / "intensity_validation.csv"
TEST_MANIFEST_PATH = MANIFEST_DIR / "intensity_test.csv"
CV_FOLDS_PATH = MANIFEST_DIR / "intensity_cv_folds.csv"

# Model Checkpoints
CHECKPOINT_PATH = CHECKPOINT_DIR / "cyclone_intensity_resnet18_v1.pt"
METADATA_PATH = CHECKPOINT_DIR / "cyclone_intensity_resnet18_v1_metadata.json"

# Image Specifications
IMAGE_SIZE: Tuple[int, int] = (256, 256)
INPUT_CHANNELS: int = 3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Target Wind Speed Normalization (based on 136 INSAT-3D historical dataset audit)
TARGET_MEAN: float = 58.90  # knots
TARGET_STD: float = 22.04   # knots
MIN_WIND_SPEED: float = 25.0
MAX_WIND_SPEED: float = 128.0

# Training Hyperparameters
BATCH_SIZE: int = 16
NUM_EPOCHS: int = 60
LEARNING_RATE_HEAD: float = 3e-4
LEARNING_RATE_BACKBONE: float = 1e-4
WEIGHT_DECAY: float = 0.01
HUBER_DELTA: float = 1.0  # in normalized target space (~8.0 kt equivalent)
EARLY_STOPPING_PATIENCE: int = 14
SEED: int = 42

# Official IMD Intensity Scale (knots)
IMD_INTENSITY_TIERS: Dict[str, Tuple[float, float]] = {
    "DEPRESSION": (17.0, 27.0),
    "DEEP_DEPRESSION": (28.0, 33.0),
    "CYCLONIC_STORM": (34.0, 47.0),
    "SEVERE_CYCLONIC_STORM": (48.0, 63.0),
    "VERY_SEVERE_CYCLONIC_STORM": (64.0, 89.0),
    "EXTREMELY_SEVERE_CYCLONIC_STORM": (90.0, 119.0),
    "SUPER_CYCLONIC_STORM": (120.0, 250.0),
}


def knots_to_imd_category(knots: float) -> str:
    """Maps wind speed in knots to official IMD tropical cyclone category."""
    if knots < 28.0:
        return "Depression"
    elif knots < 34.0:
        return "Deep Depression"
    elif knots < 48.0:
        return "Cyclonic Storm"
    elif knots < 64.0:
        return "Severe Cyclonic Storm"
    elif knots < 90.0:
        return "Very Severe Cyclonic Storm"
    elif knots < 120.0:
        return "Extremely Severe Cyclonic Storm"
    else:
        return "Super Cyclonic Storm"


def knots_to_central_pressure(knots: float, environmental_pressure: float = 1010.0) -> float:
    """Estimates central minimum pressure (hPa) using Atkinson-Holliday wind-pressure relation.
    
    Formula: V_max = 6.7 * (P_env - P_c) ** 0.644 (with V_max in knots)
    Inverted: Delta_P = (V_max / 6.7) ** (1 / 0.644)
    P_c = P_env - Delta_P
    Calibrated for North Indian Ocean and Western Pacific tropical cyclones.
    """
    v_kt = max(15.0, float(knots))
    delta_p = (v_kt / 6.7) ** (1.0 / 0.644)
    p_c = environmental_pressure - delta_p
    # Realistic atmospheric bounding for tropical systems (870 hPa - 1008 hPa)
    return round(float(min(1008.0, max(870.0, p_c))), 1)
