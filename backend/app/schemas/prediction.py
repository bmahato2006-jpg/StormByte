from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional
from backend.app.schemas.provenance import DataProvenance


class PredictionPoint(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    forecastTime: str = Field(..., description="Lead time label (+6h, +12h, +24h) or ISO valid time", validation_alias="forecast_time")
    leadHours: Optional[int] = Field(None, description="Forecast horizon in hours (6, 12, 24)", validation_alias="lead_hours")
    latitude: float = Field(..., description="Forecasted latitude")
    longitude: float = Field(..., description="Forecasted longitude")
    uncertainty: float = Field(..., description="Mean error radius in km")
    semiMajorKm: Optional[float] = Field(None, description="Along-track error semi-major axis in km", validation_alias="semi_major_km")
    semiMinorKm: Optional[float] = Field(None, description="Cross-track error semi-minor axis in km", validation_alias="semi_minor_km")
    headingDeg: Optional[float] = Field(None, description="Track forward translation bearing in degrees (0-360)", validation_alias="heading_deg")
    predictedWindSpeed: Optional[float] = Field(None, description="Forecasted sustained wind speed in knots", validation_alias="predicted_wind_speed")
    predictedPressure: Optional[float] = Field(None, description="Forecasted central pressure in hPa", validation_alias="predicted_pressure")
    intensityCategory: Optional[str] = Field(None, description="Meteorological intensity stage", validation_alias="intensity_category")
    intensityTrend: Optional[str] = Field(None, description="Thermodynamic progression status or UNMODULATED (NO_ENVIRONMENTAL_OBS)", validation_alias="intensity_trend")
    modelVersion: str = Field(..., description="Model version tag or active engine ID", validation_alias="model_version")
    generatedAt: str = Field(..., description="Timestamp when prediction was generated", validation_alias="generated_at")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cycloneId: str = Field(..., description="Unique cyclone identifier", validation_alias="cyclone_id")
    predictions: List[PredictionPoint]
    uncertaintyPolygon: List[List[float]] = Field(
        default=[],
        description="GeoJSON-style closed polygon coordinates [ [lat, lon], ... ] representing uncertainty envelope",
        validation_alias="uncertainty_polygon"
    )
    modelVersion: str = Field(..., description="Model version tag or active engine ID", validation_alias="model_version")
    generatedAt: str = Field(..., description="Timestamp when prediction was generated", validation_alias="generated_at")
    provenance: DataProvenance
    environmentalObservationsAvailable: bool = Field(False, description="Whether real or scenario environmental conditions were present", validation_alias="environmental_observations_available")
    sstAvailable: bool = Field(False, description="Whether SST observation was provided", validation_alias="sst_available")
    windShearAvailable: bool = Field(False, description="Whether vertical wind shear observation was provided", validation_alias="wind_shear_available")
    environmentalSourceType: str = Field("NONE", description="Source type: MEASURED_OBSERVATION, DEMO_SCENARIO, USER_PERTURBATION, or NONE", validation_alias="environmental_source_type")


class HollandRadialSample(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    radius_km: float = Field(..., description="Distance from vortex circulation center in km")
    wind_speed_knots: float = Field(..., description="Calculated gradient wind speed in knots")


class QuadrantRadii(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    r34_km: float = Field(..., description="Gale-force wind radius (>= 34 kt) in km")
    r50_km: float = Field(..., description="Storm-force wind radius (>= 50 kt) in km")
    r64_km: float = Field(..., description="Hurricane-force wind radius (>= 64 kt) in km")


class HollandVortexProfile(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    storm_id: str
    central_pressure_hpa: float
    environmental_pressure_hpa: float
    central_pressure_deficit_hpa: float
    holland_b: float = Field(..., description="Holland dimensionless vortex shape parameter B [1.0, 2.5]")
    radius_max_wind_km: float = Field(..., description="Estimated Radius of Maximum Wind (RMW) in km")
    peak_gradient_wind_knots: float = Field(..., description="Peak gradient wind speed in knots at RMW")
    quadrant_radii: Dict[str, QuadrantRadii] = Field(..., description="NE, SE, SW, NW gale wind extents")
    radial_curve: List[HollandRadialSample] = Field(..., description="Continuous radial wind profile samples")
    scientific_note: str = Field(
        default="MATHEMATICAL VORTEX PROFILE — NOT AN OFFICIAL IMD ADVISORY. Computed via Holland (1980) Wind-Pressure Profile with translation asymmetry. R34/R50/R64 radii are model-derived analytic estimates.",
        description="Scientific methodology note"
    )


class TrajectoryForecastRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    storm_id: Optional[str] = Field("CUSTOM_RUN", description="Cyclone identifier")
    track_points: List[Dict[str, Any]] = Field(..., min_length=1, description="Sequential historical track observations")
    steering_u_knots: Optional[float] = Field(None, description="500 hPa zonal steering wind component (knots)")
    steering_v_knots: Optional[float] = Field(None, description="500 hPa meridional steering wind component (knots)")
    sst_c: Optional[float] = Field(None, description="Sea Surface Temperature in °C (None if unavailable)")
    shear_knots: Optional[float] = Field(None, description="850-200 hPa vertical wind shear in knots (None if unavailable)")
    horizons_hours: Optional[List[int]] = Field([6, 12, 24], description="Target forecast horizons in hours")
    environmental_source_type: Optional[str] = Field(None, description="Source type: MEASURED_OBSERVATION, DEMO_SCENARIO, USER_PERTURBATION, or NONE")


class OceanRiskRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees (-90 to +90)")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees (-180 to +180)")
    radius_km: Optional[float] = Field(100.0, ge=10.0, le=1000.0, description="Evaluation radius around point in km (default 100 km)")
    evaluation_date: Optional[str] = Field(None, description="Optional observation date in YYYY-MM-DD format")


class EvidenceItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    factor: str = Field(..., description="Factor name, e.g. 'Satellite Vortex / Convection'")
    status: str = Field(..., description="Status: 'PRESENT', 'ABSENT', or 'UNAVAILABLE'")
    value: Optional[Any] = Field(None, description="Observed scientific value or None")
    threshold: str = Field(..., description="Favorable threshold or criteria, e.g. 'SST >= 26.5°C'")
    favorable: bool = Field(False, description="Whether factor is favorable for cyclone genesis")
    details: str = Field(..., description="Scientific explanation or diagnostic context")


class TemporalPersistence(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    window_hours: int = Field(24, description="Lookback persistence analysis window in hours")
    trend: str = Field(..., description="Persistence trend: 'STABLE', 'INTENSIFYING', 'DISSIPATING', 'INSUFFICIENT_HISTORY'")
    observations_evaluated: int = Field(..., description="Number of sequential observations evaluated")
    details: str = Field(..., description="Detailed breakdown of temporal evolution")


class OceanRiskLocation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")
    radius_km: float = Field(..., description="Evaluation radius in km")
    basin: str = Field(..., description="Ocean basin or sector name")
    is_ocean: bool = Field(..., description="Whether point is verified open ocean")
    is_equatorial_restricted: bool = Field(False, description="Whether location is within 5° of equator (insufficient Coriolis)")


class OceanRiskResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    location: OceanRiskLocation
    risk_level: str = Field(..., description="Risk tier: 'VERY LOW', 'LOW', 'MODERATE', 'HIGH', or 'VERY HIGH'")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Normalized risk index from 0 to 100")
    model_estimated_likelihood: str = Field(..., description="Standardized likelihood wording e.g. 'Model-estimated likelihood: MODERATE (48/100)'")
    probability_calibrated: bool = Field(False, description="Strict zero-fabrication indicator: false indicates experimental uncalibrated index")
    evidence_checklist: List[EvidenceItem] = Field(..., description="Itemized multi-source scientific evidence checklist")
    temporal_persistence: TemporalPersistence = Field(..., description="24-hour temporal persistence assessment")
    key_drivers: List[str] = Field(..., description="Primary atmospheric/oceanic factors driving assessment")
    limitations: List[str] = Field(..., description="Explicit scientific caveats, latencies, and data gaps")
    reasoning: str = Field(..., description="Physical mechanism synthesis explaining the assessment")
    experimental_notice: str = Field(
        "Experimental model-generated cyclone-development risk assessment based only on available validated data and models.",
        description="Scientific non-certainty notice"
    )
    disclaimer: str = Field(
        "SCIENTIFIC NOTICE: Experimental prediction — Not an official IMD warning or evacuation bulletin. Tropical cyclogenesis is inherently stochastic. Consult RSMC New Delhi (IMD) for official advisories.",
        description="Official safety and operational disclaimer"
    )
    provenance: Dict[str, Any] = Field(default_factory=dict, description="Cryptographic provenance and audit record")


class OceanRiskStatusResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    service_status: str = Field("OPERATIONAL", description="Operational health of prediction service")
    supported_basins: List[str] = Field(..., description="Covered basins")
    data_sources: Dict[str, Dict[str, Any]] = Field(..., description="Status and availability of data layers")
    scientific_governance: Dict[str, Any] = Field(..., description="Rules enforced: Zero-Fabrication, Non-Certainty, Coriolis")

