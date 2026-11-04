"""Pydantic schemas for Cyclone-X AI Detection & Classification (Phase 9)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DetectionPredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_base64: Optional[str] = Field(None, description="Base64-encoded satellite image bytes")
    image_path: Optional[str] = Field(None, description="Local or relative path to satellite image file")
    data_mode: Optional[str] = Field("UPLOAD", description="DEMO, HISTORICAL, UPLOAD, or LIVE_NRT")
    threshold: Optional[float] = Field(0.50, ge=0.0, le=1.0, description="Detection threshold")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Georeferencing & sensor metadata")


class CenterLocalization(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pixel_center: Dict[str, float] = Field(..., description="Estimated circulation center in pixel space (x, y)")
    bounding_box_pixels: List[float] = Field(..., description="Vortex bounding box [xmin, ymin, xmax, ymax]")
    symmetry_score: float = Field(..., description="Radial gradient symmetry convergence score")
    localization_method: str = Field(..., description="Fast Radial Symmetry / Convective Centroid")
    latitude: Optional[float] = Field(None, description="Geodetic latitude (strictly None if not georeferenced)")
    longitude: Optional[float] = Field(None, description="Geodetic longitude (strictly None if not georeferenced)")
    coordinates_available: bool = Field(..., description="True only if verified georeferencing metadata exists")
    spatial_note: str = Field(..., description="Zero-fabrication spatial provenance description")


class DetectionPredictResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    detected: bool = Field(..., description="Whether a tropical cyclone vortex was detected")
    label: str = Field(..., description="CYCLONE or NO_CYCLONE")
    confidence: float = Field(..., description="Calibrated softmax detection probability [0.0, 1.0]")
    class_probabilities: Dict[str, float] = Field(..., description="Softmax probabilities for each class")
    structural_type: str = Field(..., description="Morphological structure category")
    model_name: str = Field(..., description="Active model identifier")
    model_version: str = Field(..., description="Model version")
    data_mode: str = Field(..., description="DEMO, HISTORICAL, UPLOAD, or LIVE_NRT")
    observation_timestamp: str = Field(..., description="Observation capture time")
    satellite_source: str = Field(..., description="Sensor or satellite source")
    sensor_channel: str = Field(..., description="Spectral band")
    center_localization: Optional[CenterLocalization] = Field(None, description="Vortex center localization")
    trajectory_forecast: Optional[Dict[str, Any]] = Field(None, description="Chained Phase 8 Trajectory LSTM forecast")
    trajectory_status: str = Field(..., description="Status of trajectory chaining")
    image_metadata: Dict[str, Any] = Field(default_factory=dict, description="Image dimensions and quality metrics")
    processing_status: str = Field(..., description="SUCCESS or error status")
    disclaimer: str = Field(..., description="Mandatory MoES/IMD scientific research notice")


class DetectionTrainRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    epochs: Optional[int] = Field(10, ge=1, le=50, description="Number of training epochs")
    batch_size: Optional[int] = Field(8, ge=2, le=64, description="Batch size")
    learning_rate: Optional[float] = Field(2e-4, ge=1e-5, le=1e-2, description="Learning rate")


class DetectionTrainResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_name: str
    status: str
    epochs_completed: int
    best_epoch: int
    final_train_loss: float
    best_val_loss: float
    best_val_acc: float
    checkpoint_path: str
    training_duration_seconds: float
    history: Dict[str, List[float]]
    trained_at: str


class ConfusionMatrix(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


class DetectionEvaluationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    evaluation_id: str
    evaluated_at: str
    model_info: Dict[str, Any]
    test_dataset: Dict[str, Any]
    metrics: Dict[str, Optional[float]]
    confusion_matrix: ConfusionMatrix
    sample_details: Optional[List[Dict[str, Any]]] = None
    disclaimer: str


class DetectionStatusResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_name: str
    is_trained: bool
    status: str
    checkpoint_path: str
    metrics_available: bool
    supported_modes: List[str]
    live_nrt_configured: bool
    disclaimer: str


class KaggleDatasetStatisticsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dataset_name: str
    kaggle_slug: str
    total_samples: int
    cyclone_samples: int
    no_cyclone_samples: int
    splits: Dict[str, int]
    intensity_distribution_knots: Dict[str, Any]
    image_specs: Dict[str, Any]
    sources: List[Dict[str, Any]]
    integrity_notice: str


class ModelComparisonResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    v1: Dict[str, Any]
    v2: Dict[str, Any]
    delta: Dict[str, Any]
    report_markdown: str
    comparison_summary: str

