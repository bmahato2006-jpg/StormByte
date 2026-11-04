"""FastAPI Router for AI Impact & Disaster Risk Intelligence (Phase 6).

Endpoints:
- POST /api/impact/analyze: Generic cyclone observation impact evaluation
- GET  /api/impact/{storm_id}: Standard multi-hazard impact assessment
- GET  /api/impact/{storm_id}/risk-map: GeoJSON hazard boundaries and exposure points
- GET  /api/impact/{storm_id}/exposure: Population and critical infrastructure exposure
- GET  /api/impact/{storm_id}/provenance: Complete provenance lineage and hash audit
- POST /api/impact/scenario: What-If simulation with user parameter perturbations
- GET  /api/impact/models: Impact and exposure model status catalog
"""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
from backend.app.impact.impact_engine import impact_engine
from backend.app.impact.impact_models import (
    CompositeRiskAssessment,
    RiskMapGeoJSON,
    ScenarioRequest,
)
from backend.app.ml.model_registry import model_registry
from backend.app.schemas.provenance import DataProvenance
from backend.app.services.cyclone_service import cyclone_service

router = APIRouter(prefix="/impact", tags=["AI Impact & Disaster Risk Intelligence"])


@router.post(
    "/analyze",
    response_model=CompositeRiskAssessment,
    summary="Execute multi-hazard disaster impact assessment on custom observation data",
)
def analyze_custom_impact(payload: Dict[str, Any]):
    """Analyzes wind, rainfall, coastal, population, and infrastructure hazards on custom cyclone inputs."""
    try:
        assessment = impact_engine.evaluate_impact(cyclone_data=payload)
        return assessment
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Impact assessment error: {str(e)}",
        )


@router.get(
    "/models",
    summary="List active and experimental disaster impact assessment models",
)
def get_impact_models():
    """Returns catalog of rule engines, physical surrogates, and baseline impact models."""
    all_models = model_registry.list_models()
    impact_model_names = {
        "impact_rule_engine",
        "impact_neural_engine",
        "wind_risk_analytic",
        "flood_risk_baseline",
        "flood_risk_net",
        "coastal_risk_baseline",
        "exposure_intersection_engine",
        "surge_deep_surrogate",
        "wavewatch_deep_surrogate",
    }
    impact_models = [m for m in all_models if m.name in impact_model_names]
    return {
        "active_composite_engine": model_registry.active_impact_model_name,
        "active_wind_model": "wind_risk_analytic",
        "models": impact_models,
    }


@router.get(
    "/{storm_id}",
    response_model=CompositeRiskAssessment,
    summary="Retrieve complete disaster impact intelligence for a registered storm",
)
def get_storm_impact(storm_id: str):
    """Computes integrated multi-hazard disaster risk index and component breakdowns."""
    try:
        raw_cyclone = cyclone_service._get_raw(storm_id)
        prediction_resp = None
        try:
            prediction_resp = cyclone_service.get_cyclone_prediction(storm_id).model_dump()
        except Exception:
            pass

        parent_hash = raw_cyclone.get("provenance", {}).get("dataHash")
        assessment = impact_engine.evaluate_impact(
            cyclone_data=raw_cyclone,
            prediction_data=prediction_resp,
            parent_hash=parent_hash,
        )
        return assessment
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate storm impact: {str(e)}",
        )


@router.get(
    "/{storm_id}/risk-map",
    response_model=RiskMapGeoJSON,
    summary="Retrieve GeoJSON hazard envelopes, surge zones, and exposure points for map rendering",
)
def get_storm_risk_map(storm_id: str):
    """Generates GeoJSON FeatureCollection for interactive Leaflet risk layers."""
    try:
        raw_cyclone = cyclone_service._get_raw(storm_id)
        prediction_resp = None
        try:
            prediction_resp = cyclone_service.get_cyclone_prediction(storm_id).model_dump()
        except Exception:
            pass

        parent_hash = raw_cyclone.get("provenance", {}).get("dataHash")
        assessment = impact_engine.evaluate_impact(
            cyclone_data=raw_cyclone,
            prediction_data=prediction_resp,
            parent_hash=parent_hash,
        )
        geojson = impact_engine.generate_risk_map_geojson(
            assessment=assessment,
            cyclone_data=raw_cyclone,
            prediction_data=prediction_resp,
        )
        return geojson
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate risk map GeoJSON: {str(e)}",
        )


@router.get(
    "/{storm_id}/exposure",
    summary="Retrieve population and critical infrastructure exposure details",
)
def get_storm_exposure(storm_id: str):
    """Returns exposed settlements and critical facilities within hazard zones."""
    try:
        raw_cyclone = cyclone_service._get_raw(storm_id)
        assessment = impact_engine.evaluate_impact(cyclone_data=raw_cyclone)
        return {
            "storm_id": storm_id,
            "population_exposure": assessment.population_exposure,
            "infrastructure_exposure": assessment.infrastructure_exposure,
            "provenance": assessment.provenance,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve exposure records: {str(e)}",
        )


@router.get(
    "/{storm_id}/provenance",
    response_model=DataProvenance,
    summary="Inspect cryptographic provenance and parent observation hash lineage",
)
def get_storm_impact_provenance(storm_id: str):
    """Returns DataProvenance chaining record for impact intelligence."""
    try:
        raw_cyclone = cyclone_service._get_raw(storm_id)
        parent_hash = raw_cyclone.get("provenance", {}).get("dataHash")
        assessment = impact_engine.evaluate_impact(
            cyclone_data=raw_cyclone,
            parent_hash=parent_hash,
        )
        return assessment.provenance
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve provenance record: {str(e)}",
        )


@router.post(
    "/scenario",
    response_model=CompositeRiskAssessment,
    summary="Execute What-If scenario simulation with user parameter perturbations",
)
def run_impact_scenario(request: ScenarioRequest):
    """Computes counterfactual scenario impact assessment without modifying underlying observations."""
    try:
        storm_id = request.storm_id or "BOB-01-2020"
        raw_cyclone = cyclone_service._get_raw(storm_id)

        assessment = impact_engine.evaluate_impact(
            cyclone_data=raw_cyclone,
            scenario_params=request,
            parent_hash=raw_cyclone.get("provenance", {}).get("dataHash"),
        )
        return assessment
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Scenario evaluation error: {str(e)}",
        )

