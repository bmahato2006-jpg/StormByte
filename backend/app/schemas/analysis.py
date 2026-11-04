"""Pydantic Schemas for Satellite AI Image Analysis & Cyclone Detection.

Adheres strictly to Cyclone-X Zero-Fabrication guidelines:
- Truthful reporting of missing georeferencing and untrained intensity models.
- Explicit qualitative confidence categories with algorithmic rationales.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AnalysisMode(str, Enum):
    DEMO = "DEMO"
    RESEARCH = "RESEARCH"
    LIVE = "LIVE"


class ConfidenceTier(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNAVAILABLE = "UNAVAILABLE"


class CenterDetectionResult(BaseModel):
    """Estimated cyclone center in pixel space and geographic coordinates."""
    pixel_x: float = Field(..., description="Horizontal pixel coordinate from left")
    pixel_y: float = Field(..., description="Vertical pixel coordinate from top")
    estimated_radius_pixels: float = Field(..., description="Estimated convective eye or core radius in pixels")
    confidence: ConfidenceTier = Field(..., description="Qualitative confidence rating")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Algorithmic symmetry alignment metric [0, 1]")
    method: str = Field(..., description="Algorithm used (e.g., RADIAL_GRADIENT_SYMMETRY, CONVECTIVE_CENTROID)")
    georeferenced: bool = Field(..., description="True if coordinates were projected using authentic geographic bounds")
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0, description="Center latitude in degrees N")
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0, description="Center longitude in degrees E")
    status_message: str = Field(..., description="Human-readable geographic resolution status")


class ClassificationResult(BaseModel):
    """Structural cyclone classification derived from convective morphology."""
    category: str = Field(..., description="Classification category (e.g. Severe Tropical Storm)")
    confidence: ConfidenceTier = Field(..., description="Qualitative confidence tier")
    is_experimental: bool = Field(True, description="Always true for research baseline models")
    criteria_summary: str = Field(..., description="Structural rules or morphology criteria evaluated")
    disclaimer: str = Field(
        "AI-generated experimental analysis. Not an official IMD/RSMC classification.",
        description="Mandatory scientific disclaimer"
    )


class EarlyCyclogenesisResult(BaseModel):
    """Early cyclogenesis classification and 24-48h intensification prediction."""
    available: bool = Field(True, description="True when cyclogenesis classification head is active")
    probability: float = Field(..., ge=0.0, le=1.0, description="Cyclogenesis probability score [0, 1]")
    probability_percent: float = Field(..., ge=0.0, le=100.0, description="Cyclogenesis probability percentage")
    risk_tier: str = Field(..., description="LOW, MODERATE, HIGH, CRITICAL")
    intensification_24to48h: bool = Field(..., description="True if disturbance is predicted to intensify to >=34 kt in 24-48h")
    pre_cyclone_stage: str = Field(..., description="Pre-cyclonic disturbance stage")
    disturbance_radius_km: float = Field(..., description="Estimated radius of pre-cyclonic cloud cluster / circulation")
    circulation_center: Dict[str, float] = Field(..., description="Estimated circulation center coordinates")
    headline: str = Field(..., description="Early alert headline for dashboards")
    environmental_summary: Dict[str, Any] = Field(default_factory=dict, description="SST, vertical shear, vorticity, RH")


class IntensityResult(BaseModel):
    """Intensity estimation interface adhering to zero-fabrication of untrained models."""
    available: bool = Field(False, description="False until a validated trained model is deployed")
    wind_speed_knots: Optional[float] = Field(None, description="Estimated sustained wind speed in knots")
    central_pressure_hpa: Optional[float] = Field(None, description="Estimated central pressure in hPa")
    category: Optional[str] = Field(None, description="Intensity stage")
    uncertainty_range: Optional[str] = Field(None, description="Model uncertainty bounds")
    status_message: str = Field(
        "Intensity estimation unavailable — trained intensity model required.",
        description="Truthful capability statement"
    )


class ExplainabilityResult(BaseModel):
    """Explainable AI attribution, visual overlays, and feature reasoning."""
    explanation_type: str = Field(..., description="XAI methodology (e.g., Saliency Gradient, Contour Saliency)")
    explanation_text: str = Field(..., description="Scientific reasoning behind detection")
    key_visual_regions: List[Dict[str, Any]] = Field(default_factory=list, description="Coordinates of salient features")
    overlay_image_url: Optional[str] = Field(None, description="Static URL to transparent detection overlay")
    overlay_base64: Optional[str] = Field(None, description="Base64 encoded PNG overlay for instantaneous UI rendering")
    model_version: str = Field(..., description="Version of the model that generated the explanation")


class PreprocessingMetadata(BaseModel):
    """Summary of operations performed on the satellite image."""
    original_dimensions: Dict[str, int]
    processed_dimensions: Dict[str, int]
    channels: int
    quality_score: float = Field(..., ge=0.0, le=100.0, description="Composite image quality score (0-100)")
    sharpness: float = Field(..., description="Laplacian variance sharpness metric")
    dynamic_range: float = Field(..., description="Pixel value dynamic range [0, 1]")
    operations_applied: List[str] = Field(default_factory=list)


class SatelliteAnalysisResponse(BaseModel):
    """Comprehensive structured result of satellite image AI analysis."""
    analysis_id: str = Field(..., description="Unique analysis run identifier")
    source: str = Field(..., description="Originating satellite platform or user upload name")
    source_type: str = Field(..., description="Platform type (GEOSTATIONARY, POLAR, USER_UPLOAD)")
    timestamp: datetime = Field(..., description="Image scan UTC timestamp")
    processing_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    image_hash: str = Field(..., description="SHA-256 cryptographic hash of the input image")
    analysis_mode: AnalysisMode = Field(..., description="DEMO, RESEARCH, or LIVE")
    model_name: str = Field(..., description="Active detector model name")
    model_version: str = Field(..., description="Active detector model version")
    provenance_id: str = Field(..., description="Audit ledger reference ID in ProvenanceTracker")
    preprocessing: PreprocessingMetadata
    detection_detected: bool = Field(..., description="True if a cyclone candidate was identified")
    detection_confidence: ConfidenceTier = Field(..., description="Overall detection confidence tier")
    candidate_bounding_box: Optional[Dict[str, int]] = Field(None, description="Pixel bounding box [x_min, y_min, x_max, y_max]")
    center: CenterDetectionResult
    classification: ClassificationResult
    intensity: IntensityResult
    cyclogenesis: Optional[EarlyCyclogenesisResult] = Field(None, description="Early cyclogenesis risk evaluation and 24-48h alert")
    explainability: ExplainabilityResult
    scientific_disclaimer: str = Field(
        "AI-generated experimental analysis. Not an official IMD warning or operational forecast.",
        description="Permanent scientific integrity notice"
    )


class ModelInfo(BaseModel):
    """Metadata describing a registered computer vision or meteorological ML model."""
    name: str
    version: str
    framework: str
    input_requirements: str
    output_schema: str
    training_status: str  # ACTIVE or NOT_TRAINED
    validation_status: str
    description: str


class ModelRegistryResponse(BaseModel):
    """Catalog of registered machine learning architectures in CYCLONE-X."""
    active_detector: str
    models: List[ModelInfo]
    notice: str = "All deep learning models report NOT_TRAINED until verified against WMO benchmarks."

