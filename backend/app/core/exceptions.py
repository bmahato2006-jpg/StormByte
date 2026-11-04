from fastapi import Request
from fastapi.responses import JSONResponse
from backend.app.core.logging import logger

async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred in the CYCLONE-X service.",
            "path": request.url.path,
        }
    )

