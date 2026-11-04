from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from backend.app.schemas.provenance import DataProvenance

class QuadrantValues(BaseModel):
    NE: float = Field(..., description="Northeast quadrant radius in km")
    SE: float = Field(..., description="Southeast quadrant radius in km")
    SW: float = Field(..., description="Southwest quadrant radius in km")
    NW: float = Field(..., description="Northwest quadrant radius in km")

class QuadrantRadii(BaseModel):
    R34: QuadrantValues = Field(..., description="34-knot gale wind radius")
    R50: QuadrantValues = Field(..., description="50-knot storm wind radius")
    R64: QuadrantValues = Field(..., description="64-knot hurricane wind radius")

class RadialWindPoint(BaseModel):
    radiusKm: float
    windSpeedKnots: float

class DigitalTwinResponse(BaseModel):
    cycloneId: str
    stormName: str
    radiusMaxWindKm: float = Field(..., description="Radius of Maximum Wind (RMW) in km")
    centralPressureHpa: float
    maxWindKnots: float
    hollandB: float = Field(..., description="Holland vortex shape parameter B (1.0 - 2.5)")
    coriolisF: float = Field(..., description="Coriolis parameter at storm latitude")
    quadrantRadii: QuadrantRadii
    radialProfile: List[RadialWindPoint] = Field(default=[], description="Analytical Holland wind profile curve")
    rightFrontQuadrantRisk: str = Field(..., description="Quadrant with maximum destructive wind & surge risk")
    provenance: DataProvenance

