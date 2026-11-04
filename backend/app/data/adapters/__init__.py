"""Cyclone-X Data Adapters Package (Phase 3 & Phase 7)."""

# Phase 3 DataSourceAdapter implementations
from backend.app.data.adapters.base import DataSourceAdapter
from backend.app.data.adapters.mosdac import MosdacAdapter
from backend.app.data.adapters.imd import ImdAdapter
from backend.app.data.adapters.nasa_gpm import NasaGpmAdapter
from backend.app.data.adapters.historical import HistoricalTrackAdapter

# Phase 7 Scientific Dataset Adapters
from backend.app.data.adapters.ibtracs_adapter import IbtracsAdapter
from backend.app.data.adapters.imd_adapter import ImdAdapter as ImdBestTrackAdapter
from backend.app.data.adapters.mosdac_adapter import MosdacAdapter as MosdacSatelliteAdapter
from backend.app.data.adapters.gpm_adapter import GpmAdapter
from backend.app.data.adapters.era5_adapter import Era5Adapter
from backend.app.data.adapters.exposure_adapter import ExposureAdapter

__all__ = [
    # Phase 3
    "DataSourceAdapter",
    "MosdacAdapter",
    "ImdAdapter",
    "NasaGpmAdapter",
    "HistoricalTrackAdapter",
    # Phase 7
    "IbtracsAdapter",
    "ImdBestTrackAdapter",
    "MosdacSatelliteAdapter",
    "GpmAdapter",
    "Era5Adapter",
    "ExposureAdapter",
]
