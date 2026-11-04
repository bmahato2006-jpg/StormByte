from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from backend.app.schemas.provenance import DataProvenance


class AffectedDistrictDetail(BaseModel):
    districtName: str = Field(..., description="District and locality designation")
    state: str = Field(..., description="Indian State (e.g. Odisha, West Bengal, Andhra Pradesh, Gujarat)")
    latitude: float
    longitude: float
    population: int = Field(..., description="Total Census of India population")
    exposedPopulation: int = Field(..., description="Estimated population directly exposed to hazard envelope")
    alertTier: str = Field(..., description="RED, ORANGE, YELLOW, GREEN")
    closestApproachKm: float = Field(..., description="Distance to closest point along the AI predicted trajectory")
    arrivalEta: str = Field(..., description="Estimated horizon of closest approach or landfall (e.g. '+12h (Landfall Sector)')")
    maxPredictedWindKt: float = Field(..., description="Maximum predicted sustained wind at district center in knots")
    maxPredictedGustKmh: float = Field(..., description="Peak projected wind gust in km/h")
    stormSurgeMeters: float = Field(..., description="Projected coastal surge inundation height in meters")
    rainfallHazardMm: float = Field(..., description="Projected 24h precipitation hazard in mm")
    inUncertaintyCone: bool = Field(..., description="Whether district centroid falls inside the AI 70% uncertainty cone")
    criticalInfrastructureCount: int = Field(0, description="Count of major ports, power plants, hospitals in district")
    evacuationAdvisory: str = Field(..., description="Actionable disaster mitigation & civil protection directive")


class AIHazardZones(BaseModel):
    redAlertPolygon: List[List[float]] = Field(default_factory=list, description="GeoJSON coordinates [[lat, lon], ...] of core gale/landfall danger zone")
    orangeAlertPolygon: List[List[float]] = Field(default_factory=list, description="GeoJSON coordinates [[lat, lon], ...] of high wind/surge impact zone")
    yellowAlertPolygon: List[List[float]] = Field(default_factory=list, description="GeoJSON coordinates [[lat, lon], ...] of outer advisory alert zone")


class AIModelMetadata(BaseModel):
    detectionModel: str = Field("CycloneCenterNet-V3 (Modified Gaussian Focal Loss)", description="Satellite storm detection and eye localization model")
    trajectoryModel: str = Field("Physics-Guided Advection & Trajectory Transformer V1", description="Trajectory sequence predictor (+6h, +12h, +24h)")
    intensityModel: str = Field("RapidIntensificationNet (PyTorch 24h RI Classifier)", description="Deep learning rapid intensification predictor")
    vortexModel: str = Field("Holland (1980) Wind-Pressure Profile with Boundary Layer Friction", description="Radial wind decay and surge amplification model")
    nrtSatelliteSource: str = Field("ISRO INSAT-3D TIR1 & NASA GIBS MODIS/VIIRS", description="Active near-real-time satellite observation provider")
    nrtSatelliteTimestamp: str = Field(..., description="Timestamp of the ingested NRT satellite scene")
    isLiveNRT: bool = Field(True, description="True if telemetry is connected directly to live NRT satellite feed")


class RiskAssessment(BaseModel):
    cycloneId: str
    riskLevel: str = Field(..., description="Overall risk rating: LOW, MODERATE, HIGH, EXTREME")
    riskIndex: Optional[float] = Field(None, description="Composite risk score 0-100 from impact engine (real NRT, not hardcoded)")
    windRisk: str = Field(..., description="Wind hazard severity rating")
    rainfallRisk: str = Field(..., description="Inundation/precipitation risk rating")
    coastalRisk: str = Field(..., description="Storm surge and coastal impact rating")
    exposureRisk: str = Field(..., description="Vulnerable population and infrastructure rating")
    surgeEstimateMeters: Optional[float] = Field(None, description="Physical storm surge estimation in meters")
    estimatedPopulationAtRisk: Optional[int] = Field(None, description="Total exposed population in hazard zone")
    primaryLandfallDistrict: Optional[str] = Field(None, description="Primary coastal landfall target district identified by AI")
    affectedDistricts: Optional[List[AffectedDistrictDetail]] = Field(default_factory=list, description="List of coastal districts impacted along AI trajectory")
    hazardZones: Optional[AIHazardZones] = Field(default_factory=AIHazardZones, description="AI-generated spatial polygons for RED, ORANGE, and YELLOW alert zones")
    aiModelMetadata: Optional[AIModelMetadata] = Field(None, description="Active AI models and NRT satellite provenance details")
    generatedAt: str = Field(..., description="Assessment timestamp")
    provenance: DataProvenance



