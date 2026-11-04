"""Observation query endpoints for Cyclone-X.

Provides normalized historical, demo, and user-uploaded observations.
"""

from typing import List, Optional
from fastapi import APIRouter, Path, Query
from backend.app.data.schemas.observations import CycloneObservation, DataMode
from backend.app.data.services.data_service import data_service

router = APIRouter(tags=["Observations"])


@router.get(
    "/observations",
    response_model=List[CycloneObservation],
    summary="Query normalized cyclone observations"
)
def get_observations(
    storm_id: Optional[str] = Query(None, description="Filter by storm ID (e.g. BOB-01-2020)"),
    data_mode: Optional[DataMode] = Query(None, description="Filter by data mode (DEMO, HISTORICAL, etc.)"),
    source: Optional[str] = Query(None, description="Filter by observation source"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Retrieve normalized physical cyclone observations.

    All records contain verified coordinates, wind, central pressure, and provenance flags.
    """
    return data_service.get_cyclone_observations(
        storm_id=storm_id,
        data_mode=data_mode,
        source=source,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/cyclones/{storm_id}/observations",
    response_model=List[CycloneObservation],
    summary="Get observations for a specific cyclone"
)
def get_cyclone_observations(
    storm_id: str = Path(..., description="Cyclone identifier, e.g. BOB-01-2020 or amphan"),
    limit: int = Query(200, ge=1, le=1000),
):
    """Retrieve chronologically sorted observations for a specific cyclone."""
    return data_service.get_cyclone_observations(
        storm_id=storm_id,
        limit=limit,
    )

