"""Central path resolution for Cyclone-X Scientific Data (Phase 7)."""

from pathlib import Path

# cyclone-x root directory (3 levels up from backend/app/data)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# data directory
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
MANIFESTS_DIR = DATA_ROOT / "manifests"
SAMPLES_DIR = DATA_ROOT / "samples"
CACHE_DIR = DATA_ROOT / "cache"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MANIFESTS_DIR, SAMPLES_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

