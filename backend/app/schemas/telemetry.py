"""Pydantic schemas for MOSDAC and NASA IMERG live telemetry and ingestion pipelines."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VortexDetails(BaseModel):
    """Details for a confirmed tropical cyclone vortex detected in satellite telemetry."""

    eye_pixel_x: float = Field(..., description="Eye center X in pixel space")
    eye_pixel_y: float = Field(..., description="Eye center Y in pixel space")
    eye_confidence: float = Field(..., description="Confidence score from CycloneCenterNet")
    wind_speed_knots: float = Field(..., description="Maximum sustained wind speed in knots")
    central_pressure_hpa: float = Field(..., description="Estimated central barometric pressure in hPa")
    imd_category: str = Field(..., description="Official IMD cyclone intensity category")
    model_name: str = Field("cyclone_intensity_resnet18_v1", description="Intensity estimation model")


class MOSDACTelemetryStatus(BaseModel):
    """Operational status and telemetry summary for the MOSDAC INSAT ingestion pipeline."""

    is_live: bool = Field(False, description="True if telemetry feed is active and receiving granules")
    last_scan_utc: Optional[str] = Field(None, description="ISO UTC timestamp of the most recently ingested scan")
    granules_available: int = Field(0, ge=0, description="Total count of ingested satellite granules available")
    vortex_detected: bool = Field(False, description="True if a tropical cyclonic vortex was confirmed by CNN V3")
    active_cyclone_id: Optional[str] = Field(None, description="Identifier for currently tracked active cyclonic vortex")
    latency_ms: Optional[float] = Field(None, description="Ingestion and inference latency in milliseconds")
    operational_health: str = Field("IDLE", description="Pipeline health: HEALTHY, DEGRADED, IDLE, or LOCKED")
    latest_granule_id: Optional[str] = Field(None, description="Identifier of the latest ingested granule")
    latest_vortex_details: Optional[Dict[str, Any]] = Field(None, description="Sub-pixel eye coordinates and intensity")
    last_poll_utc: Optional[str] = Field(None, description="Timestamp of the most recent catalog poll")
    polling_interval_minutes: int = Field(30, description="Scheduled polling interval in minutes")


class IMERGTelemetryStatus(BaseModel):
    """Operational status and telemetry summary for the NASA GPM IMERG precipitation pipeline."""

    is_live: bool = Field(False, description="True if precipitation telemetry feed is active and receiving granules")
    last_scan_utc: Optional[str] = Field(None, description="ISO UTC timestamp of the most recently ingested scan")
    granules_available: int = Field(0, ge=0, description="Total count of ingested precipitation granules available")
    max_rain_rate_mm_hr: Optional[float] = Field(None, description="Maximum observed rainfall rate in mm/hr across sector")
    mean_rain_rate_mm_hr: Optional[float] = Field(None, description="Mean observed rainfall rate in mm/hr across sector")
    heavy_rain_area_km2: Optional[float] = Field(None, description="Total area under intense precipitation (>= 15 mm/hr) in km²")
    latency_ms: Optional[float] = Field(None, description="Ingestion and tensorization latency in milliseconds")
    operational_health: str = Field("IDLE", description="Pipeline health: HEALTHY, DEGRADED, IDLE, or UNCONFIGURED")
    latest_granule_id: Optional[str] = Field(None, description="Identifier of the latest ingested granule")
    dataset_name: str = Field("GPM_3IMERGHHL", description="Active NASA GPM IMERG product name")
    tensor_shape: Optional[List[int]] = Field(None, description="Extracted PyTorch tensor shape [1, H, W]")
    last_poll_utc: Optional[str] = Field(None, description="Timestamp of the most recent catalog poll")


class EnvironmentTelemetryStatus(BaseModel):
    """Operational status and telemetry summary for Oceanic & Atmospheric Environmental monitoring."""

    is_live: bool = Field(True, description="True if live atmospheric and oceanic stream is active")
    last_scan_utc: Optional[str] = Field(None, description="ISO UTC timestamp of the latest environmental observation")
    sea_surface_temperature: Optional[float] = Field(None, description="Current basin Sea Surface Temperature in °C")
    vertical_wind_shear: Optional[float] = Field(None, description="200-850 hPa deep layer vertical wind shear in knots")
    mean_sea_level_pressure: Optional[float] = Field(None, description="Peripheral surface atmospheric pressure in hPa")
    relative_humidity: Optional[float] = Field(None, description="Mid-tropospheric 700 hPa relative humidity in %")
    ocean_heat_content: Optional[float] = Field(None, description="Estimated Tropical Cyclone Heat Potential in kJ/cm²")
    source: str = Field("NOAA OISST / INCOIS / Open-Meteo NRT", description="Authentic telemetry source agencies")
    data_mode: str = Field("LIVE_NRT / SATELLITE & MARINE", description="Operational data mode tag")
    operational_health: str = Field("HEALTHY", description="Pipeline health: HEALTHY, DEGRADED, or IDLE")
    granules_available: int = Field(24, ge=0, description="Available environmental hourly/daily raster frames")
    latency_ms: Optional[float] = Field(None, description="Telemetry fetch latency in milliseconds")

