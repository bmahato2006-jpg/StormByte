from typing import List
from fastapi import APIRouter, Path
from backend.app.schemas.cyclone import CycloneBase, CycloneSummary, CycloneTrackResponse
from backend.app.schemas.prediction import PredictionResponse
from backend.app.schemas.risk import RiskAssessment, AffectedDistrictDetail
from backend.app.schemas.environment import EnvironmentalConditions
from backend.app.services.cyclone_service import cyclone_service

router = APIRouter(prefix="/cyclones", tags=["Cyclones"])

@router.get("", response_model=List[CycloneSummary], summary="List available active and benchmark cyclones")
def list_cyclones():
    """
    Returns a catalog of registered tropical cyclones.
    In Phase 1, loads curated historical benchmarks (Amphan, Biparjoy) strictly labeled DEMO DATA.
    """
    return cyclone_service.list_cyclones()

@router.post("/live/sync-satellite", summary="Ingest latest NRT satellite telemetry & re-run AI inference")
def sync_live_satellite():
    """
    Forces immediate ingestion of latest ISRO/NASA Near-Real-Time satellite scenes,
    re-runs AI CenterNet detection, Trajectory Transformer, and recalculates
    hazard alert zones (Red, Orange, Yellow) and exposed coastal districts.
    """
    return cyclone_service.sync_live_satellite_telemetry()

@router.get("/{id}", response_model=CycloneBase, summary="Get current observation and state for a cyclone")
def get_cyclone(
    id: str = Path(..., description="Cyclone identifier or name, e.g. BOB-01-2020 or amphan")
):
    """
    Returns the latest recorded observation, coordinates, central pressure, wind speed,
    and provenance metadata.
    """
    return cyclone_service.get_cyclone(id)

@router.get("/{id}/track", response_model=CycloneTrackResponse, summary="Get historical track trajectory")
def get_cyclone_track(
    id: str = Path(..., description="Cyclone identifier or name")
):
    """
    Returns the temporal sequence of historical best-track observations.
    """
    return cyclone_service.get_cyclone_track(id)

@router.get("/{id}/prediction", response_model=PredictionResponse, summary="Get trajectory forecasts and uncertainty cone")
def get_cyclone_prediction(
    id: str = Path(..., description="Cyclone identifier or name")
):
    """
    Returns forward projected coordinates (+6h, +12h, +24h) and uncertainty envelope.
    In Phase 1, baseline demo projection is served; AI model is explicitly marked not available.
    """
    return cyclone_service.get_cyclone_prediction(id)

@router.get("/{id}/risk", response_model=RiskAssessment, summary="Get multi-hazard disaster risk assessment")
def get_cyclone_risk(
    id: str = Path(..., description="Cyclone identifier or name")
):
    """
    Returns vulnerability ratings for wind, rainfall, coastal surge, and exposed population.
    """
    return cyclone_service.get_cyclone_risk(id)

@router.get("/{id}/affected-districts", response_model=List[AffectedDistrictDetail], summary="Get coastal districts impacted along AI trajectory and uncertainty cone")
def get_cyclone_affected_districts(
    id: str = Path(..., description="Cyclone identifier or name")
):
    """
    Returns coastal districts projected to be impacted by gale winds, gusts, storm surge,
    and rainfall along the forward AI-predicted trajectory and uncertainty cone.
    """
    return cyclone_service.get_cyclone_affected_districts(id)

@router.get("/{id}/environment", response_model=EnvironmentalConditions, summary="Get atmospheric and oceanic metrics")
def get_cyclone_environment(
    id: str = Path(..., description="Cyclone identifier or name")
):
    """
    Returns Sea Surface Temperature (SST), 850-200 hPa vertical wind shear, and pressure metrics.
    """
    return cyclone_service.get_cyclone_environment(id)

@router.get("/{id}/digital-twin", summary="Get Cyclone Digital Twin vortex structure and quadrant radii")
def get_cyclone_digital_twin(
    id: str = Path(..., description="Cyclone identifier or name")
):
    """
    Returns Holland vortex state, Radius of Maximum Wind (RMW), quadrant gale radii (R34/R50/R64),
    and sector hazard risk profile.
    """
    return cyclone_service.get_cyclone_digital_twin(id)


@router.get("/{id}/rapid-intensification", summary="Get 24h Rapid Intensification (RI) machine learning prediction")
def get_rapid_intensification(
    id: str = Path(..., description="Cyclone identifier or name")
):
    """
    Returns 24h Rapid Intensification (RI) probability (Delta V >= 30 kt in 24h)
    evaluated using the trained rapid_intensification_xgb machine learning classifier.
    """
    return cyclone_service.get_rapid_intensification_risk(id)


