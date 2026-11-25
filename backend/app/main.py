import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.core.config import settings
from backend.app.core.exceptions import (
    global_exception_handler,
    http_exception_handler,
    rate_limit_exception_handler,
    validation_exception_handler,
)
from backend.app.core.limiter import limiter
from backend.app.core.logging import logger, request_id_ctx_var
from backend.app.api.v1.router import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.PROJECT_NAME} API v{settings.VERSION} [{settings.ENVIRONMENT}]")
    logger.info(f"Demo Mode active: {settings.DEMO_MODE}")
    if not settings.TESTING and settings.ENVIRONMENT != "testing":
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

# Attach rate limiter to app state
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

from backend.app.schemas.health import HealthResponse

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Correlation ID and Security Response Headers Middleware
@app.middleware("http")
async def request_context_and_security_headers_middleware(request: Request, call_next):
    req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = req_id
    token = request_id_ctx_var.set(req_id)
    start_time = time.perf_counter()
    try:
        response: Response = await call_next(request)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time-Ms"] = str(duration_ms)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
    finally:
        request_id_ctx_var.reset(token)

# Register RFC-7807 compliant exception handlers
app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

# Health endpoint (direct /health for cloud orchestrator probes & /api/health via api_router)
@app.get("/health", response_model=HealthResponse, tags=["Health"], summary="System health check")
def health():
    return HealthResponse(
        status="healthy",
        service="cyclone-x",
        version=settings.VERSION
    )

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
        "health": "/health",
        "api_health": "/api/health",
        "mode": "DEMO_MODE" if settings.DEMO_MODE else "LIVE_RESEARCH"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)

