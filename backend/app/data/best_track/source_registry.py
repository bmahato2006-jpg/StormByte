"""Source Registry and Constants for Best Track Data."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = PROJECT_ROOT / "data"

RAW_IMD_DIR = DATA_ROOT / "raw" / "best_track" / "imd"
RAW_IBTRACS_DIR = DATA_ROOT / "raw" / "best_track" / "ibtracs"
PROCESSED_DIR = DATA_ROOT / "processed" / "best_track"

DB_PATH = PROCESSED_DIR / "best_track.db"
CANONICAL_CSV = PROCESSED_DIR / "north_indian_ocean_best_track.csv"
CANONICAL_JSON = PROCESSED_DIR / "north_indian_ocean_best_track.json"
STORM_INDEX_CSV = PROCESSED_DIR / "storm_index.csv"
IMD_MANIFEST_CSV = PROCESSED_DIR / "imd_manifest.csv"
VALIDATION_REPORT_JSON = PROCESSED_DIR / "validation_report.json"
QUARANTINE_CSV = PROCESSED_DIR / "quarantine.csv"

# Remote URLs
IMD_RSMC_URL = "https://rsmcnewdelhi.imd.gov.in"
IMD_EXCEL_FILENAME = "78b4b0_Best_Tracks__Data__1982-2026_.xlsx"
IMD_EXCEL_DOWNLOAD_URL = (
    "https://rsmcnewdelhi.imd.gov.in/download.php?path=uploads/best-track/78b4b0_Best_Tracks__Data__1982-2026_.xlsx"
)
IBTRACS_NI_FILENAME = "ibtracs.NI.list.v04r01.csv"
IBTRACS_NI_DOWNLOAD_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
)

# Scientific Thresholds & Constants
TRACK_MATCH_MAX_TIME_HOURS = 3.0
TRACK_MATCH_MAX_DISTANCE_KM = 150.0
REQUIRED_FIXES_FOR_FORECAST = 4
NIO_LAT_BOUNDS = (0.0, 45.0)
NIO_LON_BOUNDS = (35.0, 115.0)
EARTH_RADIUS_KM = 6371.0
