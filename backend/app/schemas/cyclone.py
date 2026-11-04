from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from backend.app.schemas.provenance import DataProvenance

class EarlyGenesisAlert(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0, description="Cyclogenesis probability score [0, 1]")
    probabilityPercent: float = Field(..., ge=0.0, le=100.0, description="Probability percentage")
    riskTier: str = Field(..., description="LOW, MODERATE, HIGH, CRITICAL")
    intensification24to48h: bool = Field(..., description="True if disturbance is predicted to intensify to >=34 kt in 24-48h")
    preCycloneStage: str = Field(..., description="Current disturbance stage, e.g. TROPICAL_CLOUD_CLUSTER, DEPRESSION")
    disturbanceRadiusKm: float = Field(..., description="Radius of cloud cluster / circulation envelope")
    circulationCenter: Dict[str, float] = Field(..., description="Latitude and Longitude of circulation center")
    environmentalFactors: Dict[str, Any] = Field(default_factory=dict, description="SST, vertical shear, vorticity, RH")
    headline: str = Field(..., description="Summary headline for alert badges")

class CycloneBase(BaseModel):
    id: str = Field(..., description="Unique storm identifier (e.g. BOB-01-2020)")
    name: str = Field(..., description="Official designated storm name (e.g. AMPHAN)")
    basin: str = Field("North Indian Ocean", description="Oceanic basin of origin")
    latitude: float = Field(..., description="Latitude coordinate in decimal degrees")
    longitude: float = Field(..., description="Longitude coordinate in decimal degrees")
    timestamp: str = Field(..., description="ISO timestamp of the current observation")
    intensity: str = Field(..., description="Cyclone intensity classification")
    windSpeed: float = Field(..., description="Maximum sustained wind speed in knots")
    pressure: float = Field(..., description="Central pressure in hPa / millibars")
    movementDirection: str = Field(..., description="Translation vector heading (e.g. NNE, NE)")
    movementSpeed: float = Field(..., description="Forward translation speed in km/h")
    dataSource: str = Field(..., description="Originating sensor or catalog")
    dataTimestamp: str = Field(..., description="Timestamp recorded by data provider")
    provenance: DataProvenance
    earlyGenesis: Optional[EarlyGenesisAlert] = None

class CycloneSummary(BaseModel):
    id: str
    name: str
    basin: str
    latitude: float
    longitude: float
    intensity: str
    windSpeed: float
    pressure: float
    dataType: str
    timestamp: str
    earlyGenesis: Optional[EarlyGenesisAlert] = None

class TrackPoint(BaseModel):
    timestamp: str
    latitude: float
    longitude: float
    intensity: str
    windSpeed: float
    pressure: float

class CycloneTrackResponse(BaseModel):
    cycloneId: str
    name: str
    track: List[TrackPoint]
    provenance: DataProvenance

