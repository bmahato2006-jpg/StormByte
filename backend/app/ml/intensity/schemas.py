"""Pydantic schemas for Cyclone-X AI Intensity Estimation (ResNet-18)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class IntensityPredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_base64: Optional[str] = Field(None, description="Base64-encoded satellite image bytes")
    image_path: Optional[str] = Field(None, description="Local or relative path to satellite image file")
    data_mode: Optional[str] = Field("UPLOAD", description="DEMO, HISTORICAL, UPLOAD, or LIVE_NRT")
    environmental_pressure: Optional[float] = Field(1010.0, description="Ambient environmental pressure (hPa)")


class IntensityPredictionData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    vmax_knots: float = Field(..., description="Estimated Maximum Sustained Surface Wind Speed (knots)")
    vmax_kmh: float = Field(..., description="Estimated Maximum Sustained Wind Speed (km/h)")
    vmax_uncertainty_lower_kt: float = Field(..., description="95% Confidence Interval lower bound (knots)")
    vmax_uncertainty_upper_kt: float = Field(..., description="95% Confidence Interval upper bound (knots)")
    central_pressure_hpa: float = Field(..., description="Derived central minimum sea-level pressure (hPa)")
    central_pressure_uncertainty_lower_hpa: float = Field(..., description="95% CI lower bound (hPa)")
    central_pressure_uncertainty_upper_hpa: float = Field(..., description="95% CI upper bound (hPa)")
    category_name: str = Field(..., description="Official IMD tropical cyclone classification tier")
    category_code: str = Field(..., description="Short category identifier code")
    confidence_score: float = Field(..., description="Statistical confidence score [0.0, 1.0]")
    available: bool = Field(True, description="Whether real intensity prediction was successfully computed")


class IntensityPredictResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    available: bool = Field(..., description="Whether intensity estimation is available")
    intensity: Optional[IntensityPredictionData] = Field(None, description="Intensity prediction payload")
    model_name: str = Field("cyclone_intensity_resnet18_v1", description="Model identifier")
    model_version: str = Field("1.0.0", description="Model version")
    data_mode: str = Field("UPLOAD", description="DEMO, HISTORICAL, UPLOAD, or LIVE_NRT")
    disclaimer: str = Field(
        "Scientific AI estimate. Not an official IMD/WMO warning. Consult national meteorological agencies.",
        description="Mandatory advisory"
    )
    message: Optional[str] = Field(None, description="Optional diagnostic or status message")


class IntensityStatusResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_name: str
    is_trained: bool
    status: str
    checkpoint_path: str
    metrics_available: bool
    test_metrics: Optional[Dict[str, Any]] = None
    supported_modes: List[str]
    disclaimer: str
