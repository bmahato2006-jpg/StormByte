"""Configuration and hyperparameters for Cyclone-X ML Trajectory Forecasting (Phase 8)."""

from pathlib import Path

# Paths
MODULE_DIR = Path(__file__).resolve().parent
APP_DIR = MODULE_DIR.parent.parent
PROJECT_ROOT = MODULE_DIR.parents[3]  # Root of cyclone-x repo
DATA_DIR = PROJECT_ROOT / "data"
MANIFESTS_DIR = DATA_DIR / "processed" / "manifests"
CHECKPOINTS_DIR = MODULE_DIR.parent / "checkpoints"
REPORTS_DIR = DATA_DIR / "reports"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# Manifest files
TRAIN_MANIFEST_PATH = MANIFESTS_DIR / "trajectory_train.csv"
VAL_MANIFEST_PATH = MANIFESTS_DIR / "trajectory_validation.csv"
TEST_MANIFEST_PATH = MANIFESTS_DIR / "trajectory_test.csv"

# Checkpoint & Scaler artifacts
CHECKPOINT_PATH = CHECKPOINTS_DIR / "trajectory_lstm_v1.pt"
TRANSFORMER_CHECKPOINT_PATH = CHECKPOINTS_DIR / "trajectory_transformer_v1.pt"
TRANSFORMER_METADATA_PATH = CHECKPOINTS_DIR / "trajectory_transformer_v1_metadata.json"
SCALER_PATH = CHECKPOINTS_DIR / "trajectory_scaler_v1.json"
EVALUATION_METRICS_PATH = REPORTS_DIR / "trajectory_evaluation_metrics.json"
TRANSFORMER_EVALUATION_METRICS_PATH = REPORTS_DIR / "trajectory_transformer_evaluation.json"

# Model Architecture
SEQ_LENGTH = 4         # Number of historical observation steps (past 9-12 hours)
HORIZONS = [6, 12, 24] # Forecast horizons in hours
NUM_HORIZONS = len(HORIZONS)

FEATURE_NAMES = [
    "lat",              # Normalized latitude
    "lon",              # Normalized longitude
    "delta_lat",        # Difference from previous fix
    "delta_lon",        # Difference from previous fix
    "displacement_km",  # Haversine step displacement
    "speed_kmh",        # Translation velocity
    "bearing_sin",      # Sine of track bearing
    "bearing_cos",      # Cosine of track bearing
    "time_delta_h",     # Time interval from previous fix
    "month_sin",        # Cyclical month sin
    "month_cos",        # Cyclical month cos
    "hour_sin",         # Diurnal cyclical hour sin
    "hour_cos"          # Diurnal cyclical hour cos
]

INPUT_DIM = len(FEATURE_NAMES)  # 13 features
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.1
OUTPUT_DIM = len(HORIZONS) * 2  # 6 outputs: (delta_lat_6, delta_lon_6, delta_lat_12, delta_lon_12, delta_lat_24, delta_lon_24)

# Training Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 4

# Earth geometry constants
EARTH_RADIUS_KM = 6371.0

