"""Pydantic Schemas for AI Impact & Disaster Risk Intelligence (Phase 6).

Follows strict zero-fabrication guardrails:
- Component availability is explicit (available=False when no real feed is connected).
- Risk index is a deterministic analytical metric, NOT a fabricated probability.
- Provenance chaining preserves parent observation lineage.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, AliasChoices
from backend.app.schemas.provenance import DataProvenance


class ComponentRisk(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    available: bool = Field(..., description="Whether input data was available for assessment")
    risk_level: str = Field(
        ...,
        description="Risk category: LOW, MODERATE, HIGH, EXTREME, UNAVAILABLE",
        serialization_alias="riskLevel",
        validation_alias=AliasChoices("risk_level", "riskLevel"),
    )
    status: str = Field("ACTIVE_ESTIMATE", description="Operational status code")
    source: str = Field(..., description="Originating scientific model, sensor array, or scenario baseline")
    source_type: str = Field(
        ...,
        description="Classification: MEASURED OBSERVATION, MODEL-DERIVED, DEMO / SCENARIO, USER PERTURBATION, UNAVAILABLE, EXPERIMENTAL",
        serialization_alias="sourceType",
        validation_alias=AliasChoices("source_type", "sourceType"),
    )
    method: str = Field(..., description="Mathematical or physical methodology applied")
    explanation: str = Field(..., description="Transparent rationale explaining the rating")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Numerical diagnostics")
    disclaimer: Optional[str] = Field(None, description="Mandatory non-operational advisory")


class SettlementAtRisk(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    state: str
    latitude: float
    longitude: float
    estimated_population: int = Field(
        ...,
        serialization_alias="estimatedPopulation",
        validation_alias=AliasChoices("estimated_population", "estimatedPopulation"),
    )
    distance_to_core_km: float = Field(
        ...,
        serialization_alias="distanceToCoreKm",
        validation_alias=AliasChoices("distance_to_core_km", "distanceToCoreKm"),
    )
    hazard_level: str = Field(
        ...,
        serialization_alias="hazardLevel",
        validation_alias=AliasChoices("hazard_level", "hazardLevel"),
    )


class PopulationExposure(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    available: bool
    source: str
    source_type: str = Field(
        ...,
        serialization_alias="sourceType",
        validation_alias=AliasChoices("source_type", "sourceType"),
    )
    total_exposed_population: Optional[int] = Field(
        None,
        serialization_alias="totalExposedPopulation",
        validation_alias=AliasChoices("total_exposed_population", "totalExposedPopulation"),
    )
    explanation: str
    settlements: List[SettlementAtRisk] = Field(default_factory=list)
    disclaimer: Optional[str] = None


class InfrastructureItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    type: str = Field(..., description="hospital, port, airport, power_grid, school, bridge")
    latitude: float
    longitude: float
    risk_level: str = Field(
        ...,
        serialization_alias="riskLevel",
        validation_alias=AliasChoices("risk_level", "riskLevel"),
    )
    distance_km: float = Field(
        ...,
        serialization_alias="distanceKm",
        validation_alias=AliasChoices("distance_km", "distanceKm"),
    )
    source: str
    source_type: str = Field(
        ...,
        serialization_alias="sourceType",
        validation_alias=AliasChoices("source_type", "sourceType"),
    )


class InfrastructureExposure(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    available: bool
    source: str
    source_type: str = Field(
        ...,
        serialization_alias="sourceType",
        validation_alias=AliasChoices("source_type", "sourceType"),
    )
    critical_count: int = Field(
        0,
        serialization_alias="criticalCount",
        validation_alias=AliasChoices("critical_count", "criticalCount"),
    )
    items: List[InfrastructureItem] = Field(default_factory=list)
    explanation: str
    disclaimer: Optional[str] = None


class WhyExplanation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    headline: str
    factor_breakdown: List[str] = Field(
        ...,
        serialization_alias="factorBreakdown",
        validation_alias=AliasChoices("factor_breakdown", "factorBreakdown"),
    )
    primary_drivers: List[str] = Field(
        ...,
        serialization_alias="primaryDrivers",
        validation_alias=AliasChoices("primary_drivers", "primaryDrivers"),
    )
    engine_type: str = Field(
        "RULE-BASED EXPLANATION ENGINE",
        serialization_alias="engineType",
        validation_alias=AliasChoices("engine_type", "engineType"),
    )
    limitations: List[str] = Field(default_factory=list)


class CompositeRiskAssessment(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    storm_id: str = Field(
        ...,
        serialization_alias="stormId",
        validation_alias=AliasChoices("storm_id", "stormId"),
    )
    storm_name: Optional[str] = Field(
        None,
        serialization_alias="stormName",
        validation_alias=AliasChoices("storm_name", "stormName"),
    )
    risk_level: str = Field(
        ...,
        description="LOW, MODERATE, HIGH, EXTREME, UNKNOWN / INSUFFICIENT DATA",
        serialization_alias="riskLevel",
        validation_alias=AliasChoices("risk_level", "riskLevel"),
    )
    risk_index: Optional[float] = Field(
        None,
        description="Deterministic 0-100 MODEL-DERIVED RISK INDEX (None if insufficient data)",
        serialization_alias="riskIndex",
        validation_alias=AliasChoices("risk_index", "riskIndex"),
    )
    available_components: List[str] = Field(
        default_factory=list,
        serialization_alias="availableComponents",
        validation_alias=AliasChoices("available_components", "availableComponents"),
    )
    unavailable_components: List[str] = Field(
        default_factory=list,
        serialization_alias="unavailableComponents",
        validation_alias=AliasChoices("unavailable_components", "unavailableComponents"),
    )
    wind_risk: ComponentRisk = Field(
        ...,
        serialization_alias="windRisk",
        validation_alias=AliasChoices("wind_risk", "windRisk"),
    )
    rainfall_risk: ComponentRisk = Field(
        ...,
        serialization_alias="rainfallRisk",
        validation_alias=AliasChoices("rainfall_risk", "rainfallRisk"),
    )
    flood_risk: ComponentRisk = Field(
        ...,
        serialization_alias="floodRisk",
        validation_alias=AliasChoices("flood_risk", "floodRisk"),
    )
    coastal_risk: ComponentRisk = Field(
        ...,
        serialization_alias="coastalRisk",
        validation_alias=AliasChoices("coastal_risk", "coastalRisk"),
    )
    population_exposure: PopulationExposure = Field(
        ...,
        serialization_alias="populationExposure",
        validation_alias=AliasChoices("population_exposure", "populationExposure"),
    )
    infrastructure_exposure: InfrastructureExposure = Field(
        ...,
        serialization_alias="infrastructureExposure",
        validation_alias=AliasChoices("infrastructure_exposure", "infrastructureExposure"),
    )
    why_explanation: WhyExplanation = Field(
        ...,
        serialization_alias="whyExplanation",
        validation_alias=AliasChoices("why_explanation", "whyExplanation"),
    )
    method: str
    limitations: str
    provenance: DataProvenance
    is_scenario_output: bool = Field(
        False,
        serialization_alias="isScenarioOutput",
        validation_alias=AliasChoices("is_scenario_output", "isScenarioOutput"),
    )
    scenario_perturbations: Optional[Dict[str, Any]] = Field(
        None,
        serialization_alias="scenarioPerturbations",
        validation_alias=AliasChoices("scenario_perturbations", "scenarioPerturbations"),
    )
    generated_at: str = Field(
        ...,
        serialization_alias="generatedAt",
        validation_alias=AliasChoices("generated_at", "generatedAt"),
    )


class ScenarioRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    storm_id: Optional[str] = Field("BOB-01-2020", validation_alias=AliasChoices("storm_id", "stormId"))
    wind_speed_override_knots: Optional[float] = Field(None, validation_alias=AliasChoices("wind_speed_override_knots", "windSpeedOverrideKnots"))
    track_offset_lat: Optional[float] = Field(None, validation_alias=AliasChoices("track_offset_lat", "trackOffsetLat"))
    track_offset_lon: Optional[float] = Field(None, validation_alias=AliasChoices("track_offset_lon", "trackOffsetLon"))
    sst_c: Optional[float] = Field(None, validation_alias=AliasChoices("sst_c", "sstC"))
    shear_knots: Optional[float] = Field(None, validation_alias=AliasChoices("shear_knots", "shearKnots"))
    rainfall_scenario_mm: Optional[float] = Field(None, validation_alias=AliasChoices("rainfall_scenario_mm", "rainfallScenarioMm"))
    coastal_tide_surge_scenario_m: Optional[float] = Field(None, validation_alias=AliasChoices("coastal_tide_surge_scenario_m", "coastalTideSurgeScenarioM"))
    include_demo_exposure: bool = Field(True, validation_alias=AliasChoices("include_demo_exposure", "includeDemoExposure"))


class RiskMapGeoJSON(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: str = "FeatureCollection"
    features: List[Dict[str, Any]] = Field(default_factory=list)

