"""Normalized Pydantic Schemas for Cyclone-X Scientific Data Intelligence.

Adheres strictly to scientific integrity and zero-fabrication standards.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class DataMode(str, Enum):
    """Scientific Data Modes clearly demarcating provenance."""
    DEMO = "DEMO"
    HISTORICAL = "HISTORICAL"
    USER_UPLOAD = "USER_UPLOAD"
    LIVE_NRT = "LIVE_NRT"
    MODEL_PREDICTION = "MODEL_PREDICTION"
    OFFICIAL_INFO = "OFFICIAL_INFO"


class DataSourceStatus(str, Enum):
    """Operational status of external scientific data sources."""
    ACTIVE = "ACTIVE"
    AVAILABLE = "AVAILABLE"
    DEMO = "DEMO"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    UNAVAILABLE = "UNAVAILABLE"
    DEGRADED = "DEGRADED"


# ---------------------------------------------------------------------------
# Core Normalized Observations
# ---------------------------------------------------------------------------

class CycloneObservation(BaseModel):
    """Normalized observation record for tropical cyclone positions."""
    id: str = Field(..., description="Unique observation identifier")
    storm_id: str = Field(..., min_length=1, description="Official storm ID (e.g., BOB01-2020)")
    storm_name: str = Field(..., min_length=1, description="Official or operational storm name")
    timestamp: datetime = Field(..., description="Observation UTC timestamp")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Center latitude [-90, 90]")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Center longitude [-180, 180]")
    wind_speed: float = Field(..., ge=0.0, le=350.0, description="10-min sustained wind speed in knots")
    central_pressure: float = Field(..., ge=800.0, le=1050.0, description="Estimated central pressure in hPa")
    classification: str = Field(..., description="IMD or WMO category (e.g., Super Cyclonic Storm)")
    source: str = Field(..., description="Observational data source (e.g., IMD_BEST_TRACK, MOSDAC)")
    source_reference: Optional[str] = Field(None, description="DOI, bulletin reference, or catalog ID")
    data_mode: DataMode = Field(..., description="Provenance mode")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        if not (-90.0 <= v <= 90.0):
            raise ValueError(f"Latitude {v} out of physical bounds [-90, 90]")
        return round(v, 4)

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        if not (-180.0 <= v <= 180.0):
            raise ValueError(f"Longitude {v} out of physical bounds [-180, 180]")
        return round(v, 4)


class SatelliteObservation(BaseModel):
    """Normalized metadata for geostationary / polar satellite observations."""
    id: str = Field(..., description="Unique satellite granule ID")
    satellite: str = Field(..., description="Satellite platform (e.g. INSAT-3D, INSAT-3DR, INSAT-3DS)")
    sensor: str = Field(..., description="Sensor payload (e.g. IMAGER, SOUNDER)")
    product: str = Field(..., description="Product channel or composite (e.g. TIR1, VIS, WV, RGB)")
    timestamp: datetime = Field(..., description="Scan UTC acquisition time")
    file_path: str = Field(..., description="Relative storage path or safe URI")
    source_reference: Optional[str] = Field(None, description="Catalog reference or granule identifier")
    latitude_min: float = Field(..., ge=-90.0, le=90.0)
    latitude_max: float = Field(..., ge=-90.0, le=90.0)
    longitude_min: float = Field(..., ge=-180.0, le=180.0)
    longitude_max: float = Field(..., ge=-180.0, le=180.0)
    resolution: Optional[str] = Field("4km", description="Nominal sub-satellite resolution")
    processing_status: str = Field("RAW", description="L1B, L2, or NORMALIZED")
    data_mode: DataMode = Field(..., description="Provenance mode")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("latitude_max")
    @classmethod
    def validate_lat_bounds(cls, v: float, values: Any) -> float:
        if "latitude_min" in values.data and v < values.data["latitude_min"]:
            raise ValueError("latitude_max must be >= latitude_min")
        return v


class EnvironmentalObservation(BaseModel):
    """Atmospheric and oceanic environmental context observation."""
    id: str = Field(..., description="Observation ID")
    timestamp: datetime = Field(..., description="Observation UTC timestamp")
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    sea_surface_temperature: Optional[float] = Field(None, ge=-5.0, le=45.0, description="SST in deg Celsius")
    rainfall: Optional[float] = Field(None, ge=0.0, le=500.0, description="Precipitation rate in mm/hr")
    wind_speed: Optional[float] = Field(None, ge=0.0, le=350.0, description="Surface wind in knots")
    wind_direction: Optional[float] = Field(None, ge=0.0, le=360.0, description="Wind vector direction in degrees")
    pressure: Optional[float] = Field(None, ge=800.0, le=1080.0, description="Surface pressure in hPa")
    humidity: Optional[float] = Field(None, ge=0.0, le=100.0, description="Relative humidity %")
    source: str = Field(..., description="Source system (e.g., NASA_GPM_IMERG, ERA5, ARGO)")
    data_mode: DataMode = Field(..., description="Provenance mode")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DataProvenance(BaseModel):
    """Immutable audit trail of a scientific data ingestion or processing batch."""
    id: str = Field(..., description="Provenance record identifier")
    source: str = Field(..., description="Source name (e.g. MOSDAC, IMD, NASA_GPM)")
    provider: str = Field(..., description="Publishing agency (ISRO, IMD, NASA)")
    product: str = Field(..., description="Dataset product name")
    dataset_version: str = Field("1.0", description="Dataset release version")
    acquisition_time: Optional[datetime] = Field(None, description="Physical sensor acquisition UTC")
    processing_time: Optional[datetime] = Field(None, description="System ingestion/normalization UTC")
    source_reference: Optional[str] = Field(None, description="Official catalog ID, DOI or URL")
    license_notes: Optional[str] = Field("Open Government / Scientific Research License")
    data_mode: DataMode = Field(..., description="Mode tag")
    status: str = Field("AVAILABLE", description="Source status tag")


# ---------------------------------------------------------------------------
# API Reports & Responses
# ---------------------------------------------------------------------------

class ImportHistoricalReport(BaseModel):
    """Validation report returned after historical CSV/JSON ingestion."""
    records_received: int = Field(..., ge=0)
    records_imported: int = Field(..., ge=0)
    records_rejected: int = Field(..., ge=0)
    validation_errors: List[str] = Field(default_factory=list)
    imported_storm_id: Optional[str] = None
    data_mode: DataMode = DataMode.HISTORICAL
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UploadSatelliteResponse(BaseModel):
    """Response returned after satellite image upload."""
    id: str
    filename: str
    file_path: str
    file_type: str
    file_size_bytes: int
    dimensions: Optional[Dict[str, int]] = None
    timestamp: Optional[datetime] = None
    data_mode: DataMode = DataMode.USER_UPLOAD
    message: str = "Satellite image uploaded successfully and labeled as USER UPLOADED DATA."


class DataSourceInfo(BaseModel):
    """Metadata detailing status and capabilities of an external scientific source."""
    name: str
    provider: str
    description: str
    supported_products: List[str]
    status: DataSourceStatus
    data_mode: DataMode
    configured: bool
    requires_auth: bool
    last_checked: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str


class DataStatusResponse(BaseModel):
    """System-wide scientific data intelligence status."""
    active_data_mode: DataMode
    database_status: str
    is_database_connected: bool
    total_cyclone_observations: int
    total_satellite_observations: int
    total_environmental_observations: int
    total_provenance_records: int
    sources: List[DataSourceInfo]
    scientific_disclaimer: str = (
        "EXPERIMENTAL RESEARCH SYSTEM - NOT AN OFFICIAL WEATHER FORECAST. "
        "DOES NOT REPLACE OFFICIAL ADVISORIES FROM IMD, RSMC, OR NHC."
    )

