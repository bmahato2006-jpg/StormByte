"""Scientific Geospatial & Satellite Configuration (Phase 11).

Official MOSDAC Reference:
Documentation: https://mosdac.gov.in/downloadapi-manual
Dataset Browser: https://mosdac.gov.in/catalog-app/satellite.php
Authentication: MOSDAC SSO (Username & Password)
"""

import os
from typing import Dict
from backend.app.core.config import settings

# Official MOSDAC Endpoints
MOSDAC_DEFAULT_API_URL = "https://mosdac.gov.in"
MOSDAC_CATALOG_URL = "https://mosdac.gov.in/catalog-app/satellite.php"
MOSDAC_MANUAL_URL = "https://mosdac.gov.in/downloadapi-manual"
MOSDAC_SSO_REALM_URL = "https://mosdac.gov.in/realms/Mosdac"

# Officially Documented MOSDAC Dataset IDs (datasetId in config.json)
MOSDAC_DATASET_IDS: Dict[str, str] = {
    "INSAT-3D_IMAGER_L1B": "3DIMG_L1B_STD",
    "INSAT-3DR_IMAGER_L1B": "3RIMG_L1B_STD",
    "INSAT-3DS_IMAGER_L1B": "3SIMG_L1B_STD",
    "INSAT-3D_HYDRO_ESTIMATOR": "3DIMG_L2B_HEM",
    "INSAT-3D_OLR": "3DIMG_L2B_OLR",
}

# Standard Meteorological Bounding Boxes (minLon, minLat, maxLon, maxLat)
METEOROLOGICAL_SECTOR_BOUNDS: Dict[str, str] = {
    "NORTH_INDIAN_OCEAN": "45.0,-10.0,110.0,38.0",
    "BAY_OF_BENGAL": "80.0,5.0,100.0,25.0",
    "ARABIAN_SEA": "50.0,5.0,78.0,25.0",
}

# Quotas and Safety Constraints
MOSDAC_MAX_DAILY_DOWNLOAD_QUOTA = 5000
MOSDAC_MAX_CONSECUTIVE_AUTH_FAILURES = 3
MOSDAC_AUTH_LOCKOUT_HOURS = 1


def get_mosdac_credentials_status() -> Dict[str, bool]:
    """Inspects whether MOSDAC credentials exist without leaking their values.

    Returns boolean presence flags only.
    """
    username = (
        getattr(settings, "MOSDAC_USERNAME", "")
        or os.getenv("MOSDAC_USERNAME", "")
        or ""
    ).strip()
    password = (
        getattr(settings, "MOSDAC_PASSWORD", "")
        or os.getenv("MOSDAC_PASSWORD", "")
        or ""
    ).strip()

    return {
        "MOSDAC_USERNAME": bool(len(username) > 0),
        "MOSDAC_PASSWORD": bool(len(password) > 0),
        "MOSDAC_DATASET_ID": bool(
            getattr(settings, "MOSDAC_DATASET_ID", "")
            or os.getenv("MOSDAC_DATASET_ID", "")
        ),
    }

