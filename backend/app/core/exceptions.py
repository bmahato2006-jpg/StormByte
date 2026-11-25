import uuid
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.core.config import settings
from backend.app.core.logging import logger, request_id_ctx_var


def _get_request_id(request: Request) -> str:
    req_id = getattr(request.state, "request_id", None)
    if not req_id:
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    return req_id


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    req_id = _get_request_id(request)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": exc.status_code,
            "error": exc.__class__.__name__,
            "message": exc.detail,
            "path": request.url.path,
            "request_id": req_id,
        },
        headers=getattr(exc, "headers", None),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    req_id = _get_request_id(request)
    errors = []
    for err in exc.errors():
        field = " -> ".join([str(loc) for loc in err.get("loc", []) if loc != "body"])
        errors.append({"field": field or "body", "issue": err.get("msg", "Invalid value")})

    logger.warning(f"Request validation failure on {request.url.path} [{req_id}]: {errors}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": 422,
            "error": "ValidationError",
            "message": "Input validation failed. Please review the parameter formats.",
            "details": errors,
            "path": request.url.path,
            "request_id": req_id,
        },
    )


async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    req_id = _get_request_id(request)
    logger.warning(f"Rate limit exceeded on {request.url.path} by {request.client.host if request.client else 'unknown'}")
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "status": 429,
            "error": "RateLimitExceeded",
            "message": "Too many requests. Please slow down and try again shortly.",
            "detail": str(exc.detail),
            "request_id": req_id,
        },
        headers={"Retry-After": "60"},
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    req_id = _get_request_id(request)
    logger.error(f"Unhandled exception on {request.url.path} [{req_id}]: {str(exc)}", exc_info=True)

    # In production, never leak internal traceback details or local paths
    client_msg = (
        str(exc)
        if settings.DEBUG and settings.ENVIRONMENT != "production"
        else "An unexpected internal error occurred. Please contact system support with your request ID."
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": 500,
            "error": "InternalServerError",
            "message": client_msg,
            "path": request.url.path,
            "request_id": req_id,
        },
    )
