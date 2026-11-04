"""Normalized Pydantic schemas for Cyclone-X Scientific Dataset Integration (Phase 7).

Adheres strictly to scientific integrity and zero-fabrication standards:
- Clear DataMode markers (REAL_RESEARCH, DEMO / SCENARIO, USER PERTURBATION, UNAVAILABLE)
- Explicit DatasetStatus tracking (AVAILABLE, NOT_DOWNLOADED, CREDENTIALS_REQUIRED, etc.)
- Strict separation of observed and modeled values; missing fields remain None.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ScientificDataMode(str, Enum):
    """Mandatory data mode classifications."""
    REAL_RESEARCH = "REAL_RESEARCH"
    DEMO_SCENARIO = "DEMO / SCENARIO"
    USER_PERTURBATION = "USER PERTURBATION"
    UNAVAILABLE = "UNAVAILABLE"


class ScientificDatasetStatus(str, Enum):
    """Operational status of datasets in the central registry."""
    AVAILABLE = "AVAILABLE"
    NOT_DOWNLOADED = "NOT_DOWNLOADED"
    DOWNLOAD_REQUIRED = "DOWNLOAD_REQUIRED"
    CREDENTIALS_REQUIRED = "CREDENTIALS_REQUIRED"
    ACCESS_RESTRICTED = "ACCESS_RESTRICTED"
    PROCESSING = "PROCESSING"
    VALIDATED = "VALIDATED"
    INVALID = "INVALID"
    DEMO_ONLY = "DEMO_ONLY"


class QualityStatus(str, Enum):
    """Dataset quality control verdict."""
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class DatasetMetadata(BaseModel):
    """Metadata schema representing a registered scientific dataset."""
    dataset_id: str = Field(..., description="Unique dataset key, e.g. ibtracs_nio")
    name: str = Field(..., description="Human-readable dataset title")
    source: str = Field(..., description="Publishing body or archive, e.g. NOAA_NCEI, IMD, ISRO")
    provider: str = Field(..., description="Primary agency or institution")
    description: str = Field(..., description="Dataset scientific scope and purpose")
    variables: List[str] = Field(default_factory=list, description="Contained physical variables")
    units: Dict[str, str] = Field(default_factory=dict, description="Units for each variable")
    format: str = Field(..., description="File format, e.g. CSV, HDF5, GeoTIFF, NetCDF")
    coverage: str = Field(..., description="Geographical coverage description")
    temporal_resolution: Optional[str] = Field(None, description="e.g. 3-hourly, 30-min, daily")
    spatial_resolution: Optional[str] = Field(None, description="e.g. 0.1 deg, 4km, point")
    time_range: Optional[str] = Field(None, description="e.g. 2019-2023, 1980-present")
    coverage_start: Optional[str] = Field(None, description="Temporal coverage start ISO date or year")
    coverage_end: Optional[str] = Field(None, description="Temporal coverage end ISO date or year")
    license: str = Field(..., description="Access license or terms of use")
    access_type: str = Field("PUBLIC", description="PUBLIC, REGISTRATION_REQUIRED, RESTRICTED")
    status: ScientificDatasetStatus = Field(..., description="Current availability / connectivity status")
    local_path: Optional[str] = Field(None, description="Relative path if present on disk")
    file_size_bytes: Optional[int] = Field(None, description="Physical file size on disk in bytes")
    record_count: Optional[int] = Field(None, description="Number of verified observations or granules")
    source_url: str = Field(..., description="Official access or catalog URL")
    retrieved_at: Optional[datetime] = Field(None, description="Acquisition UTC timestamp")
    checksum: Optional[str] = Field(None, description="SHA-256 integrity checksum")
    data_mode: ScientificDataMode = Field(..., description="REAL_RESEARCH, DEMO / SCENARIO, etc.")
    provenance: Dict[str, Any] = Field(default_factory=dict, description="Cryptographic provenance payload")


class NormalizedTrackRecord(BaseModel):
    """Standardized cyclone best-track observation record."""
    storm_id: str = Field(..., description="Official storm ID (e.g. 2020137N10086 or BOB01-2020)")
    storm_name: str = Field(..., description="Storm name in uppercase")
    basin: str = Field(..., description="Ocean basin code, e.g. NI (North Indian Ocean), BB, AS")
    timestamp: datetime = Field(..., description="Observation UTC timestamp")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Center latitude [-90, 90]")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Center longitude [-180, 180]")
    max_wind_kt: Optional[float] = Field(None, ge=0.0, le=350.0, description="10-min or 1-min sustained wind in knots (None if unrecorded)")
    central_pressure_hpa: Optional[float] = Field(None, ge=800.0, le=1050.0, description="Estimated central pressure in hPa (None if unrecorded)")
    movement_direction_deg: Optional[float] = Field(None, ge=0.0, le=360.0, description="Forward direction (0-360 deg)")
    movement_speed_kmh: Optional[float] = Field(None, ge=0.0, le=120.0, description="Translation speed in km/h")
    source: str = Field(..., description="Source dataset, e.g. IBTRACS, IMD_BEST_TRACK")
    source_record_id: Optional[str] = Field(None, description="Original row ID or bulletin ID")
    data_mode: ScientificDataMode = Field(default=ScientificDataMode.REAL_RESEARCH)
    provenance: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("latitude")
    @classmethod
    def round_latitude(cls, v: float) -> float:
        return round(v, 4)

    @field_validator("longitude")
    @classmethod
    def round_longitude(cls, v: float) -> float:
        return round(v, 4)


class NormalizedSatelliteRecord(BaseModel):
    """Standardized satellite imagery observation metadata."""
    satellite: str = Field(..., description="Platform, e.g. INSAT-3D, INSAT-3DR, INSAT-3DS")
    sensor: str = Field(..., description="Payload, e.g. IMAGER, SOUNDER")
    channel: str = Field(..., description="Channel name, e.g. TIR1, VIS, WV")
    timestamp: datetime = Field(..., description="Scan UTC acquisition time")
    file_path: Optional[str] = Field(None, description="Relative path if downloaded")
    georeferenced: bool = Field(False, description="Whether coordinates are spatially anchored")
    resolution: Optional[str] = Field(None, description="Nominal resolution, e.g. 4km")
    processing_level: str = Field("L1B", description="L1B, L2, or L3")
    source: str = Field(..., description="MOSDAC_ISRO, NOAA, etc.")
    source_url: Optional[str] = Field(None)
    data_mode: ScientificDataMode = Field(default=ScientificDataMode.REAL_RESEARCH)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class NormalizedRainfallRecord(BaseModel):
    """Standardized precipitation observation record."""
    product: str = Field(..., description="e.g. GPM_IMERG_FINAL, IMD_AWS")
    timestamp: datetime = Field(...)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    precipitation_mmh: Optional[float] = Field(None, ge=0.0, description="Precipitation rate in mm/h")
    quality_flag: Optional[str] = Field(None, description="Reliability flag, e.g. HIGH, MEDIUM, QUESTIONABLE")
    data_mode: ScientificDataMode = Field(default=ScientificDataMode.REAL_RESEARCH)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class NormalizedEnvironmentalRecord(BaseModel):
    """Standardized atmospheric/oceanic reanalysis record."""
    timestamp: datetime = Field(...)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    sea_surface_temperature_c: Optional[float] = Field(None, ge=-5.0, le=45.0)
    vertical_wind_shear_kt: Optional[float] = Field(None, ge=0.0, le=150.0)
    mean_sea_level_pressure_hpa: Optional[float] = Field(None, ge=800.0, le=1050.0)
    data_mode: ScientificDataMode = Field(default=ScientificDataMode.REAL_RESEARCH)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class DataQualityReport(BaseModel):
    """Quality control verification report for a scientific dataset."""
    dataset_id: str
    dataset_name: str
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    missing_values: Dict[str, int] = Field(default_factory=dict)
    duplicate_timestamps: int = 0
    time_range: Optional[str] = None
    spatial_coverage: Optional[str] = None
    variables: List[str] = Field(default_factory=list)
    units: Dict[str, str] = Field(default_factory=dict)
    quality_status: QualityStatus = QualityStatus.PASS
    issues: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TrainingManifest(BaseModel):
    """Temporal-leakage-protected training dataset manifest."""
    manifest_id: str
    task: str = Field(..., description="trajectory, intensity, center_detection")
    split: str = Field(..., description="train, validation, test")
    record_count: int
    storm_ids: List[str]
    temporal_range: str
    split_strategy: str
    source_dataset_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    file_path: str

