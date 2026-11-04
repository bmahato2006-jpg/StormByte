from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.exceptions import global_exception_handler
from backend.app.api.v1.router import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.PROJECT_NAME} API v{settings.VERSION} [{settings.ENVIRONMENT}]")
    logger.info(f"Demo Mode active: {settings.DEMO_MODE}")
    try:
        from backend.app.services.satellite_pipeline_service import satellite_pipeline_service
        satellite_pipeline_service.start_background_loop()
    except Exception as e:
        logger.warning(f"Could not start satellite pipeline background task: {e}")
    yield
    try:
        from backend.app.services.satellite_pipeline_service import satellite_pipeline_service
        satellite_pipeline_service.stop_background_loop()
    except Exception:
        pass
    logger.info(f"Shutting down {settings.PROJECT_NAME} API")

app = FastAPI(
    title="CYCLONE-X Intelligence Platform",
    description="AI-Powered Tropical Cyclone Intelligence, Digital Twin and Disaster Decision-Support Platform",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
app.add_exception_handler(Exception, global_exception_handler)

# Mount aggregated API routes under /api
app.include_router(api_router, prefix=settings.API_PREFIX)

from backend.app.api.telemetry import router as telemetry_router
app.include_router(telemetry_router)

@app.get("/", tags=["Root"])
def root():
    return {
        "title": "CYCLONE-X Intelligence Platform API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/api/health",
        "mode": "DEMO_MODE" if settings.DEMO_MODE else "LIVE_RESEARCH"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)

