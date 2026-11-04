"""Pydantic schemas for Cyclone-X Scientific End-to-End AI Pipeline (Phase 10)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class GeospatialMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    latitude_min: Optional[float] = Field(None, description="Southernmost latitude bound [-90, +90]")
    latitude_max: Optional[float] = Field(None, description="Northernmost latitude bound [-90, +90]")
    longitude_min: Optional[float] = Field(None, description="Westernmost longitude bound [-180, +180]")
    longitude_max: Optional[float] = Field(None, description="Easternmost longitude bound [-180, +180]")
    acquisition_time: Optional[str] = Field(None, description="ISO-8601 observation timestamp")
    satellite: Optional[str] = Field("INSAT-3D", description="Satellite platform name")
    sensor: Optional[str] = Field("IMAGER", description="Sensor instrument")
    channel: Optional[str] = Field("TIR1 (10.8 µm)", description="Spectral band")
    product_id: Optional[str] = Field(None, description="Official product catalog identifier")
    source: Optional[str] = Field("MOSDAC/ISRO", description="Data provider organization")
    spatial_reference: Optional[str] = Field("EPSG:4326 (WGS 84)", description="Coordinate reference system")
    georeferencing_available: bool = Field(False, description="True only if valid bounding coordinates are verified")


class CenterFix(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pixel_x: float = Field(..., description="Estimated circulation center X in pixel space")
    pixel_y: float = Field(..., description="Estimated circulation center Y in pixel space")
    bounding_box_pixels: List[float] = Field(..., description="[xmin, ymin, xmax, ymax] in pixel coordinates")
    symmetry_score: float = Field(..., description="Radial gradient symmetry score")
    localization_method: str = Field(..., description="Fast Radial Symmetry / Mass Centroid")
    latitude: Optional[float] = Field(None, description="Geodetic latitude (strictly None if georeferencing is missing)")
    longitude: Optional[float] = Field(None, description="Geodetic longitude (strictly None if georeferencing is missing)")
    coordinates_available: bool = Field(..., description="True only if calibrated geographic coordinates exist")
    spatial_note: str = Field(..., description="Zero-fabrication spatial provenance explanation")


class ForecastPoint(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    forecast_hour: int = Field(..., description="Lead time in hours (+6, +12, +24)")
    timestamp: str = Field(..., description="Valid projection timestamp (ISO-8601)")
    latitude: float = Field(..., description="Predicted geodetic latitude")
    longitude: float = Field(..., description="Predicted geodetic longitude")
    uncertainty_km: float = Field(..., description="Empirical 68% uncertainty radius in kilometers")
    uncertainty_radius_km_68: float = Field(..., description="Empirical 68% confidence radius")
    uncertainty_radius_km_95: float = Field(..., description="Empirical 95% confidence radius")


class TrajectoryForecast(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_name: str = Field("trajectory_lstm_v1", description="Active trajectory model")
    model_version: str = Field("1.0.0", description="Model version")
    origin_fix: Dict[str, Any] = Field(..., description="Initial fix coordinates and timestamp")
    predictions: List[ForecastPoint] = Field(..., description="Multi-horizon forecast points")
    uncertainty_method: str = Field(..., description="Empirical residual calculation methodology")
    uncertainty: Dict[str, float] = Field(..., description="Empirical radii by horizon in km")
    is_experimental: bool = Field(True, description="Strict research classification")


class PipelineRiskAssessment(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    risk_score: float = Field(..., description="Explainable composite risk index [0-100]")
    risk_level: str = Field(..., description="NONE, LOW, MODERATE, HIGH, or SEVERE")
    risk_factors: Dict[str, Any] = Field(..., description="Transparent factor breakdown with availability states")
    risk_confidence: float = Field(..., description="Statistical confidence in assessment [0-1]")
    data_completeness: float = Field(..., description="Ratio of observed factors vs total factors [0-1]")
    explanation: str = Field(..., description="Human-readable decision-support rationale")
    disclaimer: str = Field(..., description="MoES/IMD scientific research notice")


class PipelineProvenance(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Unified Scientific Audit Trail & Provenance Fields
    image_source: str = Field(..., description="Authentic originating provider or archive")
    image_dataset: str = Field(..., description="Specific dataset or catalog benchmark name")
    image_id: Optional[str] = Field(None, description="Image identifier or filename")
    image_sha256: Optional[str] = Field(None, description="Cryptographic SHA-256 digest")
    data_mode: str = Field(..., description="Operational ingestion mode")
    detection_model: str = Field(..., description="Active CNN detection model identifier")
    detection_model_version: str = Field(..., description="Active detection model version")
    detection_checkpoint: Optional[str] = Field(None, description="Active detection model checkpoint weights")
    trajectory_model: Optional[str] = Field(None, description="Active trajectory model identifier (null if unused)")
    trajectory_model_version: Optional[str] = Field(None, description="Trajectory model version (null if unused)")
    trajectory_checkpoint: Optional[str] = Field(None, description="Trajectory weights path (null if unused)")
    track_source: Optional[str] = Field(None, description="Authentic track data source (null if no track used)")
    track_id: Optional[str] = Field(None, description="Cyclone or storm track identifier")
    track_provenance: Optional[str] = Field(None, description="Track dataset provenance description")
    analysis_timestamp_utc: str = Field(..., description="ISO-8601 UTC timestamp of analysis execution")

    # Legacy Backwards Compatibility Fields
    source: str = "UNVERIFIED"
    acquisition_timestamp: str = "UNVERIFIED"
    satellite: str = "INSAT-3D"
    sensor: str = "IMAGER"
    channel: str = "TIR1 (10.8 µm)"
    product_id: Optional[str] = None
    georeferencing_available: bool = False
    preprocessing_version: str = "v2.0-kaggle-insat3d-rgb"
    detection_version: str = "2.0.0"
    inference_timestamp: Optional[str] = None
    original_filename: Optional[str] = None
    file_format: Optional[str] = None
    crs: Optional[str] = None
    bounds: Optional[Dict[str, float]] = None
    resolution: Optional[List[float]] = None
    geospatial_status: Optional[str] = None


class PipelineAnalyzeResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # 1. Detection Stage
    detected: bool
    label: str
    confidence: float
    class_probabilities: Dict[str, float]
    structural_type: str

    # 2. Center Localization & Geospatial Stage
    center_fix: Optional[CenterFix] = None
    geospatial_status: str
    raster_metadata: Optional[Dict[str, Any]] = None

    # 3. Track History & Trajectory Forecasting Stage
    track_history_length: int
    track_history_required: int = 4
    trajectory_status: str
    trajectory_forecast: Optional[TrajectoryForecast] = None

    # 4. Risk Engine Stage
    risk_assessment: PipelineRiskAssessment

    # 5. Intensity Estimation Stage (ResNet-18)
    intensity: Optional[Dict[str, Any]] = None

    # 6. Provenance & Scientific Disclaimers
    provenance: PipelineProvenance
    processing_status: str
    error_detail: Optional[str] = None
    disclaimer: str


class PipelineStatusResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pipeline_status: str
    phase9_detection: Dict[str, Any]
    phase8_trajectory: Dict[str, Any]
    intensity_estimation: Optional[Dict[str, Any]] = None
    risk_engine: Dict[str, Any]
    geospatial_capability: Dict[str, Any]
    supported_data_modes: List[str]
    live_nrt_configured: bool
    disclaimer: str

