"""Prediction & Physical Vortex API Endpoints for Cyclone-X.

Provides:
- POST /api/prediction/forecast: Execute physics-guided forward trajectory prediction and uncertainty cone generation.
- GET  /api/prediction/vortex/{storm_id}: Retrieve Holland (1980) physical vortex profile, RMW, and quadrant radii.
- GET  /api/prediction/models: Catalog of trajectory and physical vortex models.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Path, status

from backend.app.schemas.prediction import (
    HollandRadialSample,
    HollandVortexProfile,
    PredictionPoint,
    PredictionResponse,
    QuadrantRadii,
    TrajectoryForecastRequest,
    OceanRiskRequest,
    OceanRiskResponse,
    OceanRiskStatusResponse,
)
from backend.app.core.provenance import create_provenance
from backend.app.ml.trajectory import trajectory_forecaster
from backend.app.ml.uncertainty import uncertainty_quantifier
from backend.app.ml.physics_vortex import holland_vortex_engine
from backend.app.ml.model_registry import model_registry
from backend.app.services.cyclone_service import cyclone_service
from backend.app.services.ocean_cyclone_risk_service import ocean_cyclone_risk_service


router = APIRouter(prefix="/prediction", tags=["Trajectory & Physics Prediction"])


@router.post(
    "/forecast",
    response_model=PredictionResponse,
    summary="Execute physics-guided trajectory prediction & uncertainty cone generation"
)
def run_trajectory_forecast(request: TrajectoryForecastRequest):
    """Executes physics-guided advection (+6h, +12h, +24h) and builds the bivariate Gaussian uncertainty cone."""
    try:
        steering = None
        if request.steering_u_knots is not None and request.steering_v_knots is not None:
            steering = (request.steering_u_knots, request.steering_v_knots)

        sst_val = request.sst_c
        shear_val = request.shear_knots
        sst_avail = sst_val is not None
        shear_avail = shear_val is not None
        env_avail = sst_avail or shear_avail
        env_source = request.environmental_source_type or ("USER_PERTURBATION" if env_avail else "NONE")

        raw_preds = trajectory_forecaster.forecast(
            track_history=request.track_points,
            steering_vector_knots=steering,
            environmental_sst_c=sst_val,
            vertical_wind_shear_knots=shear_val,
            forecast_horizons_hours=request.horizons_hours or [6, 12, 24],
        )

        origin_pt = request.track_points[-1]
        origin_lat = float(origin_pt["latitude"])
        origin_lon = float(origin_pt["longitude"])

        prediction_points: List[PredictionPoint] = []
        for p in raw_preds:
            lead_h = p["forecast_lead_hours"]
            s_maj, s_min, r_km = uncertainty_quantifier.compute_ellipse_axes(lead_h)

            prediction_points.append(
                PredictionPoint(
                    forecastTime=p["forecast_time_label"],
                    leadHours=lead_h,
                    latitude=p["latitude"],
                    longitude=p["longitude"],
                    uncertainty=r_km,
                    semiMajorKm=s_maj,
                    semiMinorKm=s_min,
                    headingDeg=p.get("heading_deg"),
                    predictedWindSpeed=p["predicted_wind_speed_knots"],
                    predictedPressure=p["predicted_pressure_hpa"],
                    intensityCategory=p.get("intensity_category"),
                    intensityTrend=p.get("intensity_trend"),
                    modelVersion="physics_guided_advection-v0.5.0",
                    generatedAt=datetime.now(timezone.utc).isoformat(),
                )
            )

        # Build dynamic continuous cone polygon
        polygon = uncertainty_quantifier.build_uncertainty_cone_polygon(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            forecast_points=raw_preds,
        )

        import hashlib
        parent_hash = hashlib.sha256(str(request.track_points[-1]).encode("utf-8")).hexdigest()[:16]

        from backend.app.schemas.provenance import DataProvenance
        prov_dict = create_provenance(
            source="PHYSICS_ADVECTION_ENGINE",
            processing_version="v0.5.0-phase5",
            model_version="physics_guided_advection-v0.5.0",
            data_type="AI_PREDICTION",
            parent_hash=parent_hash,
        )
        prov = DataProvenance(**prov_dict)

        return PredictionResponse(
            cycloneId=request.storm_id or "CUSTOM_RUN",
            predictions=prediction_points,
            uncertaintyPolygon=polygon,
            modelVersion="physics_guided_advection-v0.5.0",
            generatedAt=datetime.now(timezone.utc).isoformat(),
            provenance=prov,
            environmentalObservationsAvailable=env_avail,
            sstAvailable=sst_avail,
            windShearAvailable=shear_avail,
            environmentalSourceType=env_source,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Trajectory forecast error: {str(e)}",
        )


@router.get(
    "/vortex/{storm_id}",
    response_model=HollandVortexProfile,
    summary="Compute Holland (1980) physical vortex wind profile & quadrant radii"
)
def get_vortex_profile(
    storm_id: str = Path(..., description="Cyclone identifier or name, e.g. BOB-01-2020 or amphan")
):
    """Calculates continuous radial wind profile V(r) and quadrant gale radii from Holland (1980) model."""
    cyclone = cyclone_service.get_cyclone(storm_id)
    if not cyclone:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cyclone '{storm_id}' not found.",
        )

    central_pressure = float(cyclone.pressure)
    lat = float(cyclone.latitude)
    env_pressure = 1012.0
    trans_speed = float(cyclone.movementSpeed) / 1.852 if cyclone.movementSpeed else 8.0  # km/h to knots

    # Translation bearing
    heading_deg = 315.0
    if cyclone.movementDirection == "NW":
        heading_deg = 315.0
    elif cyclone.movementDirection == "N":
        heading_deg = 360.0
    elif cyclone.movementDirection == "NE":
        heading_deg = 45.0

    profile = holland_vortex_engine.generate_radial_profile(
        central_pressure=central_pressure,
        environmental_pressure=env_pressure,
        latitude=lat,
    )

    quad_radii = holland_vortex_engine.compute_quadrant_gale_radii(
        central_pressure=central_pressure,
        environmental_pressure=env_pressure,
        latitude=lat,
        translation_speed_knots=trans_speed,
        translation_direction_deg=heading_deg,
    )

    quad_models: Dict[str, QuadrantRadii] = {
        quad: QuadrantRadii(
            r34_km=v["r34_km"],
            r50_km=v["r50_km"],
            r64_km=v["r64_km"],
        )
        for quad, v in quad_radii.items()
    }

    curve_samples = [
        HollandRadialSample(
            radius_km=pt["radius_km"],
            wind_speed_knots=pt["wind_speed_knots"],
        )
        for pt in profile["curve"]
    ]

    return HollandVortexProfile(
        storm_id=cyclone.id,
        central_pressure_hpa=profile["central_pressure_hpa"],
        environmental_pressure_hpa=profile["environmental_pressure_hpa"],
        central_pressure_deficit_hpa=profile["central_pressure_deficit_hpa"],
        holland_b=profile["holland_b"],
        radius_max_wind_km=profile["radius_max_wind_km"],
        peak_gradient_wind_knots=profile["peak_gradient_wind_knots"],
        quadrant_radii=quad_models,
        radial_curve=curve_samples,
    )


@router.get(
    "/models",
    summary="Query registered trajectory and physical vortex models"
)
def get_prediction_models():
    """Returns active and cataloged trajectory models and physical simulation engines."""
    all_models = model_registry.list_models()
    relevant = [
        m for m in all_models
        if "trajectory" in m.name or "vortex" in m.name or "advection" in m.name
    ]
    return {
        "active_trajectory_model": model_registry.active_trajectory_model_name,
        "active_vortex_model": model_registry.active_vortex_model_name,
        "models": relevant,
    }


@router.post(
    "/ocean-risk",
    response_model=OceanRiskResponse,
    summary="Evaluate experimental cyclone development risk for an ocean area"
)
def evaluate_ocean_cyclone_risk(request: OceanRiskRequest):
    """Evaluates multi-source satellite, precipitation, environmental, and track evidence to compute an explainable cyclone development risk assessment."""
    try:
        return ocean_cyclone_risk_service.analyze_ocean_risk(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocean cyclone risk analysis error: {str(e)}"
        )


@router.get(
    "/ocean-risk/status",
    response_model=OceanRiskStatusResponse,
    summary="Get operational status and layer availability for ocean cyclone risk prediction"
)
def get_ocean_risk_status():
    """Returns the operational availability of NRT satellite, NASA IMERG, ECMWF ERA5, and best track datasets."""
    try:
        return ocean_cyclone_risk_service.get_service_status()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch ocean risk status: {str(e)}"
        )

