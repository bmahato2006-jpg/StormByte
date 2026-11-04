from fastapi import APIRouter
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.cyclones import router as cyclones_router
from backend.app.api.v1.data import router as data_router
from backend.app.api.v1.observations import router as observations_router
from backend.app.api.v1.analysis import router as analysis_router
from backend.app.api.v1.prediction import router as prediction_router
from backend.app.api.v1.impact import router as impact_router
from backend.app.api.v1.datasets import router as datasets_router
from backend.app.api.v1.ml_trajectory import router as ml_trajectory_router
from backend.app.api.v1.ml_detection import router as ml_detection_router
from backend.app.api.v1.ml_pipeline import router as ml_pipeline_router
from backend.app.api.v1.ml_intensity import router as ml_intensity_router
from backend.app.api.v1.satellite import router as satellite_router
from backend.app.api.v1.best_track import router as best_track_router
from backend.app.api.v1.live_monitor import router as live_monitor_router
from backend.app.api.v1.performance import router as performance_router
from backend.app.api.v1.simulation import router as simulation_router
from backend.app.api.v1.analyst import router as analyst_router
from backend.app.api.telemetry import router as telemetry_router
from backend.app.ml.placeholders import get_ml_system_status

api_router = APIRouter()

# Include health router (/health -> when mounted under /api, becomes /api/health)
api_router.include_router(health_router)
api_router.include_router(cyclones_router)
api_router.include_router(data_router)
api_router.include_router(observations_router)
api_router.include_router(analysis_router)
api_router.include_router(prediction_router)
api_router.include_router(impact_router)
api_router.include_router(datasets_router)
api_router.include_router(ml_trajectory_router)
api_router.include_router(ml_detection_router)
api_router.include_router(ml_intensity_router)
api_router.include_router(ml_pipeline_router)
api_router.include_router(satellite_router)
api_router.include_router(best_track_router)
api_router.include_router(live_monitor_router)
api_router.include_router(performance_router)
api_router.include_router(simulation_router)
api_router.include_router(analyst_router)
api_router.include_router(telemetry_router)


@api_router.get("/ml/status", tags=["AI/ML Status"], summary="Inspect status of ML models")
def get_ml_status():
    """
    Transparently reports whether ML models are loaded or awaiting training in future phases.
    """
    return get_ml_system_status()


