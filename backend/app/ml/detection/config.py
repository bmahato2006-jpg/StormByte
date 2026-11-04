"""Configuration and Hyperparameters for Cyclone-X AI-Based Cyclone Detection (Phase 9)."""

from pathlib import Path

# Module & Project Paths
MODULE_DIR = Path(__file__).resolve().parent
APP_DIR = MODULE_DIR.parent.parent
PROJECT_ROOT = MODULE_DIR.parents[3]  # Root of cyclone-x repo
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = MODULE_DIR.parent / "checkpoints"
REPORTS_DIR = DATA_DIR / "reports"

SATELLITE_DIR = DATA_DIR / "processed" / "satellite"
DETECTION_DATA_DIR = SATELLITE_DIR / "detection"
DETECTION_IMAGES_DIR = DETECTION_DATA_DIR / "images"
DETECTION_MANIFESTS_DIR = DETECTION_DATA_DIR / "manifests"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DETECTION_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DETECTION_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

# Manifest Files
TRAIN_MANIFEST_PATH = DETECTION_MANIFESTS_DIR / "detection_train.csv"
VAL_MANIFEST_PATH = DETECTION_MANIFESTS_DIR / "detection_validation.csv"
TEST_MANIFEST_PATH = DETECTION_MANIFESTS_DIR / "detection_test.csv"

# Model Checkpoints & Artifacts
CHECKPOINT_PATH = CHECKPOINTS_DIR / "cyclone_detector_cnn_v3.pt" if (CHECKPOINTS_DIR / "cyclone_detector_cnn_v3.pt").exists() else (CHECKPOINTS_DIR / "cyclone_detector_cnn_v2.pt" if (CHECKPOINTS_DIR / "cyclone_detector_cnn_v2.pt").exists() else CHECKPOINTS_DIR / "cyclone_detector_cnn_v1.pt")
CHECKPOINT_V1_PATH = CHECKPOINTS_DIR / "cyclone_detector_cnn_v1.pt"
CHECKPOINT_V2_PATH = CHECKPOINTS_DIR / "cyclone_detector_cnn_v2.pt"
CHECKPOINT_V3_PATH = CHECKPOINTS_DIR / "cyclone_detector_cnn_v3.pt"
METADATA_V3_PATH = CHECKPOINTS_DIR / "cyclone_detector_cnn_v3_metadata.json"

METRICS_PATH = REPORTS_DIR / "detection_evaluation_metrics.json"
METRICS_V1_PATH = REPORTS_DIR / "detection_evaluation_metrics.json"
METRICS_V2_PATH = REPORTS_DIR / "insat3d_detection_evaluation.json"
REPORT_V2_PATH = REPORTS_DIR / "insat3d_detection_report.md"
REPORT_V3_PATH = REPORTS_DIR / "detection_v3_training_report.md"
DATASET_REPORT_V3_PATH = REPORTS_DIR / "detection_v3_dataset_report.md"

# Model Hyperparameters
IMAGE_SIZE = (256, 256)
INPUT_CHANNELS = 3  # Standardized 3-channel input
NUM_CLASSES = 2
CLASS_NAMES = ["NO_CYCLONE", "CYCLONE"]
CLASS_TO_IDX = {"NO_CYCLONE": 0, "CYCLONE": 1}
IDX_TO_CLASS = {0: "NO_CYCLONE", 1: "CYCLONE"}

# Training Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 4

# Confidence Calibrations
DETECTION_THRESHOLD = 0.50
HIGH_CONFIDENCE_THRESHOLD = 0.85

