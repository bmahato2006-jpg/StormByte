"""Satellite AI Analysis Endpoints for Cyclone-X.

Provides:
- POST /api/analysis/satellite: Computer-vision cyclone candidate and center detection
- GET  /api/analysis/{analysis_id}: Retrieve analysis record
- GET  /api/analysis/{analysis_id}/explanation: Retrieve explainability reasoning
- GET  /api/analysis/{analysis_id}/overlay: Retrieve rendered detection overlay PNG
- GET  /api/analysis/models: Query ModelRegistry catalog and statuses
"""

from datetime import datetime, timezone
import io
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, Response

from backend.app.schemas.analysis import (
    AnalysisMode,
    ExplainabilityResult,
    ModelRegistryResponse,
    SatelliteAnalysisResponse,
)
from backend.app.ml.inference import inference_pipeline
from backend.app.ml.model_registry import model_registry

router = APIRouter(prefix="/analysis", tags=["Satellite AI Analysis"])


@router.post(
    "/satellite",
    response_model=SatelliteAnalysisResponse,
    summary="Execute AI analysis on satellite imagery"
)
async def analyze_satellite_image(
    file: UploadFile = File(..., description="Satellite image (PNG, JPG, JPEG, TIFF)"),
    cyclone_id: Optional[str] = Form(None, description="Associated storm ID (e.g. BOB-01-2020)"),
    scan_timestamp: Optional[str] = Form(None, description="Acquisition UTC timestamp in ISO format"),
    satellite_name: Optional[str] = Form(None, description="Satellite sensor/source name"),
    analysis_mode: AnalysisMode = Form(AnalysisMode.RESEARCH, description="DEMO, RESEARCH, or LIVE"),
    lat_min: Optional[float] = Form(None, ge=-90.0, le=90.0),
    lat_max: Optional[float] = Form(None, ge=-90.0, le=90.0),
    lon_min: Optional[float] = Form(None, ge=-180.0, le=180.0),
    lon_max: Optional[float] = Form(None, ge=-180.0, le=180.0),
):
    """Executes the Phase 4 computer-vision pipeline on uploaded satellite imagery.

    Performs preprocessing, candidate detection, symmetry-based center localization,
    morphological classification, explainable attribution overlay generation, and provenance tracking.
    """
    try:
        content = await file.read()

        # Parse timestamp if supplied
        parsed_time = None
        if scan_timestamp:
            try:
                parsed_time = datetime.fromisoformat(scan_timestamp.replace("Z", "+00:00"))
            except Exception:
                parsed_time = None

        # Assemble georeferencing bounds if all 4 corners supplied
        geo_bounds = None
        if (
            lat_min is not None
            and lat_max is not None
            and lon_min is not None
            and lon_max is not None
        ):
            if lat_max < lat_min:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="lat_max must be greater than or equal to lat_min.",
                )
            if lon_max < lon_min:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="lon_max must be greater than or equal to lon_min.",
                )
            geo_bounds = {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            }

        result = inference_pipeline.run_analysis(
            image_bytes=content,
            filename=file.filename or "satellite_granule.png",
            cyclone_id=cyclone_id,
            scan_timestamp=parsed_time,
            source_name=satellite_name,
            analysis_mode=analysis_mode,
            geo_bounds=geo_bounds,
        )
        return result

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {str(e)}",
        )


@router.get(
    "/models",
    response_model=ModelRegistryResponse,
    summary="Query registered machine learning architectures"
)
def get_registered_models():
    """Lists all computer vision, classification, and intensity models.

    Untrained models report training_status = 'NOT_TRAINED' per scientific integrity rules.
    """
    return model_registry.get_registry_response()


@router.get(
    "/{analysis_id}",
    response_model=SatelliteAnalysisResponse,
    summary="Get structured analysis result by ID"
)
def get_analysis_by_id(analysis_id: str):
    """Retrieve full analysis report by analysis ID."""
    result = inference_pipeline.get_analysis(analysis_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis ID '{analysis_id}' not found.",
        )
    return result


@router.get(
    "/{analysis_id}/explanation",
    response_model=ExplainabilityResult,
    summary="Get explainable AI reasoning and feature attribution"
)
def get_analysis_explanation(analysis_id: str):
    """Retrieve explainability attribution and feature breakdown."""
    result = inference_pipeline.get_analysis(analysis_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis ID '{analysis_id}' not found.",
        )
    return result.explainability


@router.get(
    "/{analysis_id}/overlay",
    summary="Download or stream transparent detection overlay PNG"
)
def get_analysis_overlay_image(analysis_id: str):
    """Returns the transparent annotated PNG overlay showing detected storm bounds and center."""
    result = inference_pipeline.get_analysis(analysis_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis ID '{analysis_id}' not found.",
        )

    # Check filesystem overlay
    base_upload_dir = Path(__file__).resolve().parent.parent.parent.parent / "uploads" / "overlays"
    file_path = base_upload_dir / f"overlay_{analysis_id}.png"

    if file_path.exists():
        return FileResponse(file_path, media_type="image/png")

    # If static file was evicted, decode from base64 representation
    if result.explainability.overlay_base64:
        raw_b64 = result.explainability.overlay_base64.split(",")[-1]
        import base64
        data = base64.b64decode(raw_b64)
        return Response(content=data, media_type="image/png")

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Overlay image not found for this analysis.",
    )

