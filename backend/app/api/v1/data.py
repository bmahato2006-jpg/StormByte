"""Data intelligence and provenance endpoints for Cyclone-X.

Follows strict scientific integrity and zero-fabrication guidelines.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel

from backend.app.data.schemas.observations import (
    DataMode,
    DataProvenance,
    DataSourceInfo,
    DataStatusResponse,
    ImportHistoricalReport,
    UploadSatelliteResponse,
)
from backend.app.data.services.data_service import data_service
from backend.app.data.services.upload_service import upload_service
from backend.app.data.provenance.tracker import provenance_tracker

router = APIRouter(tags=["Data Intelligence & Provenance"])


class HistoricalImportPayload(BaseModel):
    """Payload for direct JSON historical track ingestion."""
    format: str = "json"
    content: Optional[str] = None
    records: Optional[List[Dict[str, Any]]] = None


@router.get(
    "/data/status",
    response_model=DataStatusResponse,
    summary="System-wide scientific data status"
)
def get_data_status():
    """Returns database connection status, active scientific data mode,

    observation record counts, and external adapter availability.
    """
    return data_service.get_data_status()


@router.get(
    "/data/sources",
    response_model=List[DataSourceInfo],
    summary="Registered scientific data sources and credentials status"
)
def get_data_sources():
    """Returns catalog of external meteorological sources (MOSDAC, IMD, NASA GPM).

    Truthfully displays whether authentic credentials are configured.
    """
    return data_service.get_sources()


@router.get(
    "/provenance",
    response_model=List[DataProvenance],
    summary="Scientific data provenance and audit ledger"
)
def get_provenance_records(
    limit: int = Query(50, ge=1, le=200),
    mode: Optional[DataMode] = Query(None, description="Filter by data mode")
):
    """Returns immutable provenance records for all ingested datasets,

    including acquisition timestamps, providers, and licenses.
    """
    if mode:
        return provenance_tracker.get_by_mode(mode)[:limit]
    return provenance_tracker.get_all(limit=limit)


@router.get(
    "/data/provenance",
    response_model=List[DataProvenance],
    include_in_schema=False
)
def get_data_provenance_alias(limit: int = Query(50, ge=1, le=200)):
    """Alias for /provenance."""
    return provenance_tracker.get_all(limit=limit)


@router.post(
    "/data/upload",
    response_model=UploadSatelliteResponse,
    summary="Upload satellite imagery securely"
)
async def upload_satellite_image(
    file: UploadFile = File(..., description="Satellite image (PNG, JPG, JPEG, TIFF)"),
    satellite_name: Optional[str] = Form(None),
    sensor_type: Optional[str] = Form(None),
    product_type: Optional[str] = Form(None),
):
    """Uploads satellite raster granules with safety checks.

    Strictly labeled as USER UPLOADED DATA per scientific integrity guidelines.
    """
    try:
        content = await file.read()
        response, observation = upload_service.process_upload(
            filename=file.filename or "unknown.png",
            content=content,
            content_type=file.content_type,
            satellite_name=satellite_name,
            sensor_type=sensor_type,
            product_type=product_type,
        )
        data_service.add_satellite_observation(observation)
        return response
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing image: {str(e)}",
        )


@router.post(
    "/data/import/historical",
    response_model=ImportHistoricalReport,
    summary="Import historical cyclone tracks (CSV or JSON)"
)
async def import_historical_track(request: Request):
    """Ingests historical cyclone track points with coordinate and parameter validation.

    Accepts multipart file upload (.csv, .json) or JSON body payload.
    Returns a comprehensive validation report detailing accepted and rejected records.
    """
    import json
    content_type = request.headers.get("content-type", "")
    content = ""
    file_format = "json"

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file")
        if uploaded_file and hasattr(uploaded_file, "read"):
            file_bytes = await uploaded_file.read()
            content = file_bytes.decode("utf-8", errors="replace")
            fname = (getattr(uploaded_file, "filename", "") or "").lower()
            if fname.endswith(".csv"):
                file_format = "csv"
            elif fname.endswith(".json"):
                file_format = "json"
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file format '{fname}'. Only .csv and .json are supported.",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file attached under 'file' in form-data.",
            )
    else:
        # JSON body payload
        try:
            body_json = await request.json()
            if isinstance(body_json, list):
                content = json.dumps(body_json)
                file_format = "json"
            elif isinstance(body_json, dict):
                file_format = str(body_json.get("format", "json")).lower()
                if "records" in body_json and body_json["records"] is not None:
                    content = json.dumps(body_json["records"])
                elif "content" in body_json and body_json["content"]:
                    content = body_json["content"]
                elif "storm_id" in body_json:
                    content = json.dumps([body_json])
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="JSON payload must contain 'records', 'content', or be an observation record/array.",
                    )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse JSON body: {str(e)}",
            )

    if not content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty historical track content received.",
        )

    report = data_service.import_historical_data(content=content, file_format=file_format)
    return report


