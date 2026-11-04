"""Pydantic schemas for Cyclone-X ML Trajectory Forecasting (Phase 8)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class MLTrajectoryPredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    track_history: List[Dict[str, Any]] = Field(
        ...,
        description="Chronological historical fixes (at least 1, recommended 4) containing latitude, longitude, and optional timestamp",
    )
    forecast_horizons: Optional[List[int]] = Field(
        default=[6, 12, 24],
        description="Forecast lead times in hours (defaults to [6, 12, 24])",
    )
    model_type: Optional[str] = Field(
        default="lstm",
        description="Model architecture to use: 'transformer' or 'lstm' (defaults to 'lstm' for backward compatibility)",
    )


class MLTrajectoryPoint(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    forecast_lead_hours: int = Field(..., description="Lead time in hours (+6, +12, +24)")
    forecast_time_label: str = Field(..., description="Lead time label (+6h, +12h, +24h)")
    valid_time: str = Field(..., description="ISO 8601 UTC timestamp of the valid forecast")
    latitude: float = Field(..., description="Forecasted latitude in decimal degrees")
    longitude: float = Field(..., description="Forecasted longitude in decimal degrees")
    uncertainty_radius_km_68: float = Field(..., description="Empirical 68% confidence error radius in km")
    uncertainty_radius_km_95: float = Field(..., description="Empirical 95% confidence error radius in km")
    model_name: str = Field(default="trajectory_transformer_v1", description="Identifier of model")
    is_experimental: bool = Field(default=True, description="Strictly True for experimental ML models")


class MLTrajectoryPredictResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_name: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version")
    architecture: str = Field(..., description="Neural network architecture description")
    status: str = Field(..., description="Operational status: OPERATIONAL_EXPERIMENTAL")
    is_experimental: bool = Field(..., description="Always True for research forecasts")
    origin_fix: Dict[str, Any] = Field(..., description="Initial fix from which displacements were projected")
    predictions: List[MLTrajectoryPoint] = Field(..., description="Multi-horizon predictions")
    uncertainty_method: str = Field(..., description="Method used to calculate error bounds")
    checkpoint_metadata: Dict[str, Any] = Field(default_factory=dict, description="Checkpoint provenance")
    disclaimer: str = Field(..., description="Mandatory MoES/IMD scientific research disclaimer")


class MLTrajectoryTrainRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_type: Optional[str] = Field("transformer", description="Model architecture: 'transformer' or 'lstm'")
    epochs: Optional[int] = Field(12, ge=1, le=50, description="Training epochs (default 12)")
    batch_size: Optional[int] = Field(64, ge=16, le=256, description="Batch size (default 64)")
    learning_rate: Optional[float] = Field(1e-3, ge=1e-5, le=1e-1, description="AdamW learning rate")


class MLTrajectoryTrainResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_name: str = Field(..., description="Model name")
    status: str = Field(..., description="Status: TRAINED")
    epochs_completed: int = Field(..., description="Number of epochs executed")
    best_epoch: int = Field(..., description="Epoch with lowest validation loss")
    final_train_loss: float = Field(..., description="Final loss on training split")
    best_val_loss: float = Field(..., description="Lowest loss on validation split")
    checkpoint_path: str = Field(..., description="Local path to serialized model weights")
    scaler_path: str = Field(..., description="Local path to fitted feature scaler")
    training_duration_seconds: float = Field(..., description="Training runtime in seconds")
    trained_at: str = Field(..., description="ISO timestamp of training completion")


class HorizonComparisonMetric(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    horizon: str
    ml_mae_km: float
    ml_rmse_km: float
    ml_median_km: float
    ml_p90_km: float
    physics_mae_km: float
    physics_rmse_km: float
    physics_median_km: float
    physics_p90_km: float
    difference_km: float
    relative_diff_pct: float
    better_model: str
    # Optional 3-way comparisons
    transformer_mae_km: Optional[float] = None
    lstm_mae_km: Optional[float] = None
    transformer_rmse_km: Optional[float] = None
    lstm_rmse_km: Optional[float] = None


class StormEvaluationBreakdown(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    storm_id: str
    storm_name: str
    test_samples: int
    ml_mae_km: float
    physics_mae_km: float
    by_horizon: Dict[str, Dict[str, float]]
    transformer_mae_km: Optional[float] = None
    lstm_mae_km: Optional[float] = None


class MLTrajectoryEvaluationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    evaluation_id: str
    evaluated_at: str
    test_dataset: Dict[str, Any]
    models_evaluated: List[Dict[str, Any]]
    comparison_table: List[HorizonComparisonMetric]
    ml_model_metrics: Dict[str, Any]
    physics_model_metrics: Dict[str, Any]
    per_storm_breakdown: List[StormEvaluationBreakdown]
    sample_results: Optional[List[Dict[str, Any]]] = None
    disclaimer: str

