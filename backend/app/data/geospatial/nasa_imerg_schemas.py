"""Pydantic Schemas for NASA GPM/PPS IMERG Precipitation Ingestion (Phase 11).

Strict Zero-Fabrication:
1. Every observation preserves authentic NASA metadata (DOI, product, resolution, CRS, bounds).
2. Location queries return authentic grid coordinates and unadulterated precipitation measurements.
3. Coordinates outside coverage return LOCATION_OUTSIDE_DATASET_COVERAGE.
4. Historical observations are never labelled live NRT.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NASAIMERGGridExtent(BaseModel):
    """Geodetic spatial extent in WGS84 (EPSG:4326)."""
    min_latitude: float = Field(..., description="Minimum latitude (-90.0 to +90.0)")
    max_latitude: float = Field(..., description="Maximum latitude (-90.0 to +90.0)")
    min_longitude: float = Field(..., description="Minimum longitude (-180.0 to +180.0)")
    max_longitude: float = Field(..., description="Maximum longitude (-180.0 to +180.0)")


class NASAIMERGObservationSummary(BaseModel):
    """Summary record for an ingested authentic NASA IMERG observation."""
    granule_id: str
    product: str = "3IMERGHH_GIS"
    version: str = "V07B"
    filename: str
    file_path: str
    sha256: str
    observation_start: str
    observation_end: str
    download_time: Optional[str] = None
    bounds: NASAIMERGGridExtent
    resolution_deg: float = 0.1
    crs: str = "EPSG:4326"
    rows: int = 1800
    cols: int = 3600
    precipitation_variable: str = "total.rate"
    precipitation_units: str = "0.1(mm/hr)"
    scale_factor: float = 10.0
    valid_pixel_count: int
    missing_pixel_count: int
    min_precipitation_raw: float
    max_precipitation_raw: float
    mean_precipitation_raw: float
    max_precipitation_mm_per_hr: float
    provider: str = "NASA GPM / PPS"
    source: str = "NASA PPS"
    mode: str = "Historical / Verified NASA Observation"
    status: str = "INGESTED_VERIFIED"


class NASAIMERGLocationResponse(BaseModel):
    """Response model for point location rainfall query against authentic NASA IMERG observation."""
    requested_latitude: float
    requested_longitude: float
    grid_latitude: Optional[float] = None
    grid_longitude: Optional[float] = None
    precipitation: Optional[float] = Field(None, description="Precipitation rate in raw product units (0.1 mm/hr)")
    precipitation_mm_per_hr: Optional[float] = Field(None, description="Precipitation rate scaled to meteorological mm/hr")
    precipitation_accum_mm: Optional[float] = Field(None, description="Half-hour total accumulation in mm (if available)")
    unit: str = "0.1(mm/hr)"
    unit_scaled: str = "mm/hr"
    scale_factor: float = 10.0
    is_valid_measurement: bool = False
    is_missing_no_data: bool = False
    observation_start: Optional[str] = None
    observation_end: Optional[str] = None
    observation_time: Optional[str] = None
    granule_id: Optional[str] = None
    product: str = "3IMERGHH_GIS"
    version: str = "V07B"
    provider: str = "NASA GPM / PPS"
    source: str = "NASA PPS"
    mode: str = "Historical / Verified NASA Observation"
    coverage: bool = False
    status: str = "COVERED"
    message: str = "Observation retrieved from authentic NASA GPM/PPS IMERG raster."
    provenance: Optional[Dict[str, Any]] = None
    zero_fabrication_rule: str = (
        "Zero precipitation numbers, coordinates, or timestamps are fabricated. "
        "All values are derived directly from authentic NASA PPS GeoTIFF rasters."
    )


class NASAIMERGObservationsResponse(BaseModel):
    """List of all available locally ingested NASA IMERG observations."""
    total_observations: int
    observations: List[NASAIMERGObservationSummary]
    mode: str = "Historical / Verified NASA Observation"
    provider: str = "NASA GPM / PPS"
    zero_fabrication_rule: str = (
        "All catalogued observations are authenticated NASA PPS products verified with cryptographic checksums."
    )

