from fastapi import APIRouter
from backend.app.schemas.health import HealthResponse
from backend.app.core.config import settings

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse, summary="System health check")
def get_health():
    return HealthResponse(
        status="healthy",
        service="cyclone-x",
        version=settings.VERSION
    )

