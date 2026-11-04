"""FastAPI Router for Cyclone-X Real Historical Best Track Data.

Endpoints:
- GET  /api/best-track/storms                : List and filter historical storms
- GET  /api/best-track/storms/{storm_id}     : Get storm summary metadata
- GET  /api/best-track/storms/{storm_id}/track: Get ordered genuine fixes & GeoJSON
- GET  /api/best-track/search                : Quick search by storm name/year
- GET  /api/best-track/stats                 : Master catalog counts and distributions
- POST /api/best-track/import/imd            : Trigger IMD import
- POST /api/best-track/import/ibtracs        : Trigger IBTrACS import
- POST /api/best-track/import/all            : Trigger full Best Track import
- POST /api/best-track/match-image           : Spatiotemporal satellite-to-track association
- POST /api/best-track/evaluate-forecast     : Trajectory LSTM offline backtest evaluation
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.app.data.best_track.cli import main as run_cli_import
from backend.app.data.best_track.schemas import (
    BestTrackStats,
    ForecastEvaluationResult,
    TrackAssociationResult,
)
from backend.app.data.best_track.service import BestTrackService

router = APIRouter(prefix="/best-track", tags=["Best Track Historical Database"])
service = BestTrackService()


class MatchImageRequest(BaseModel):
    storm_id: str
    image_timestamp_utc: Optional[str] = None
    image_latitude: Optional[float] = None
    image_longitude: Optional[float] = None


class EvaluateForecastRequest(BaseModel):
    storm_id: str
    lead_steps: int = Field(4, ge=4, description="Number of initial fixes to use as history (must be >= 4)")


class ImportRequest(BaseModel):
    year: Optional[int] = None
    dry_run: bool = False


@router.get("/storms", summary="List historical storms with filtering")
def list_storms(
    source: Optional[str] = Query(None, description="Filter by source: IMD_RSMC_NEW_DELHI or NOAA_IBTRACS"),
    year: Optional[int] = Query(None, description="Filter by season/year (1842-2026)"),
    basin: Optional[str] = Query(None, description="Filter by basin: BOB, AS, or NI"),
    search: Optional[str] = Query(None, description="Search by storm name or ID"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Lists indexed historical cyclones matching the query filters."""
    return service.list_storms(
        source=source, year=year, basin=basin, query=search, limit=limit, offset=offset
    )


@router.get("/search", summary="Search historical storms")
def search_storms(
    q: str = Query(..., min_length=1, description="Storm name or year search term"),
    limit: int = Query(20, ge=1, le=100),
):
    """Searches indexed historical cyclones by name or partial identifier."""
    results, total = service.repo.get_storms(search=q, limit=limit)
    return {"total": total, "query": q, "storms": results}


@router.get("/stats", response_model=BestTrackStats, summary="Get master catalog statistics")
def get_stats():
    """Returns total storms, total authentic fixes, year range, and source/basin counts."""
    return service.get_stats()


@router.get("/storms/{storm_id}", summary="Get historical storm summary")
def get_storm(storm_id: str):
    """Returns summary metadata and observation window for a specific storm."""
    storm = service.get_storm(storm_id)
    if not storm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Storm '{storm_id}' not found in Best Track database.",
        )
    return storm


@router.get("/storms/{storm_id}/track", summary="Get genuine track fixes and GeoJSON")
def get_storm_track(storm_id: str):
    """Returns all genuine sequential fixes and a GeoJSON LineString feature for map rendering."""
    result = service.get_storm_track(storm_id)
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result["error"],
        )
    return result


@router.post("/match-image", response_model=TrackAssociationResult, summary="Evaluate satellite image association")
def match_image(request: MatchImageRequest):
    """Evaluates whether a satellite image belongs to a cyclone track based on verified coordinates and time."""
    return service.match_satellite_image_to_track(
        storm_id=request.storm_id,
        image_timestamp_utc=request.image_timestamp_utc,
        image_latitude=request.image_latitude,
        image_longitude=request.image_longitude,
    )


@router.post("/evaluate-forecast", response_model=ForecastEvaluationResult, summary="Trajectory LSTM historical backtest")
def evaluate_forecast(request: EvaluateForecastRequest):
    """Benchmarks Trajectory LSTM against subsequent verified fixes with strict zero-leakage."""
    try:
        return service.evaluate_forecast(
            storm_id=request.storm_id,
            lead_steps=request.lead_steps,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


def _run_bg_import(source: str, year: Optional[int] = None, dry_run: bool = False):
    """Helper for executing background import jobs."""
    import sys
    sys_argv_backup = sys.argv
    try:
        args = ["cli.py", "--source", source]
        if year:
            args.extend(["--year", str(year)])
        if dry_run:
            args.append("--dry-run")
        sys.argv = args
        run_cli_import()
    finally:
        sys.argv = sys_argv_backup


@router.post("/import/imd", summary="Trigger IMD Best Track import")
def import_imd(request: Optional[ImportRequest] = None, bg_tasks: BackgroundTasks = BackgroundTasks()):
    """Triggers import of IMD RSMC New Delhi historical best track records."""
    req = request or ImportRequest()
    bg_tasks.add_task(_run_bg_import, source="imd", year=req.year, dry_run=req.dry_run)
    return {
        "status": "IMPORT_JOB_QUEUED",
        "source": "IMD_RSMC_NEW_DELHI",
        "year": req.year or "ALL",
        "dry_run": req.dry_run,
        "message": "IMD Best Track import scheduled as a background task.",
    }


@router.post("/import/ibtracs", summary="Trigger NOAA IBTrACS import")
def import_ibtracs(request: Optional[ImportRequest] = None, bg_tasks: BackgroundTasks = BackgroundTasks()):
    """Triggers import of NOAA IBTrACS North Indian Ocean records."""
    req = request or ImportRequest()
    bg_tasks.add_task(_run_bg_import, source="ibtracs", year=req.year, dry_run=req.dry_run)
    return {
        "status": "IMPORT_JOB_QUEUED",
        "source": "NOAA_IBTRACS",
        "year": req.year or "ALL",
        "dry_run": req.dry_run,
        "message": "NOAA IBTrACS import scheduled as a background task.",
    }


@router.post("/import/all", summary="Trigger complete Best Track import")
def import_all(request: Optional[ImportRequest] = None, bg_tasks: BackgroundTasks = BackgroundTasks()):
    """Triggers import of both IMD and IBTrACS datasets with deduplication and validation."""
    req = request or ImportRequest()
    bg_tasks.add_task(_run_bg_import, source="all", year=req.year, dry_run=req.dry_run)
    return {
        "status": "IMPORT_JOB_QUEUED",
        "source": "ALL",
        "year": req.year or "ALL",
        "dry_run": req.dry_run,
        "message": "Full Best Track pipeline scheduled as a background task.",
    }
