"""Scientific Dataset REST API Endpoints for Cyclone-X (Phase 7).

Provides transparent access to registered meteorological datasets, quality control reports,
cryptographic provenance chains, and normalized track/satellite/rainfall/environmental records.

Follows Cyclone-X Scientific Integrity Rules:
- Real research data is clearly labeled REAL_RESEARCH.
- When datasets require credentials, reports CREDENTIALS_REQUIRED or ACCESS_RESTRICTED.
- Missing meteorological values remain None/null.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from backend.app.data.adapters.era5_adapter import Era5Adapter
from backend.app.data.adapters.gpm_adapter import GpmAdapter
from backend.app.data.adapters.ibtracs_adapter import IbtracsAdapter
from backend.app.data.adapters.imd_adapter import ImdAdapter
from backend.app.data.adapters.mosdac_adapter import MosdacAdapter
from backend.app.data.adapters.exposure_adapter import ExposureAdapter
from backend.app.data.registry import dataset_registry
from backend.app.data.schemas import (
    DataQualityReport,
    DatasetMetadata,
    NormalizedEnvironmentalRecord,
    NormalizedRainfallRecord,
    NormalizedSatelliteRecord,
    NormalizedTrackRecord,
    ScientificDataMode,
    ScientificDatasetStatus,
)

router = APIRouter(prefix="/datasets", tags=["Scientific Datasets & Data Hub"])


@router.get(
    "",
    response_model=List[DatasetMetadata],
    summary="List all registered scientific datasets"
)
def list_datasets(
    status_filter: Optional[ScientificDatasetStatus] = Query(None, description="Filter by availability status"),
    data_mode: Optional[ScientificDataMode] = Query(None, description="Filter by data mode")
):
    """Returns catalog of all 8 integrated scientific datasets with operational status,

    physical variables, spatial coverage, license, and SHA-256 checksums.
    """
    datasets = dataset_registry.get_all()
    if status_filter:
        datasets = [d for d in datasets if d.status == status_filter]
    if data_mode:
        datasets = [d for d in datasets if d.data_mode == data_mode]
    return datasets


@router.get(
    "/tracks",
    response_model=List[NormalizedTrackRecord],
    summary="Query normalized historical best-track observations"
)
def get_tracks(
    storm_name: Optional[str] = Query(None, description="Filter by storm name, e.g. AMPHAN, BIPARJOY, FANI"),
    storm_id: Optional[str] = Query(None, description="Filter by official storm ID"),
    source: Optional[str] = Query(None, description="Filter by source: IBTRACS_NOAA or IMD_RSMC_NEW_DELHI"),
    limit: int = Query(100, ge=1, le=500)
):
    """Retrieve normalized historical cyclone track records from IBTrACS and IMD archives."""
    all_tracks: List[NormalizedTrackRecord] = []

    ibtracs = IbtracsAdapter()
    if ibtracs.is_available():
        all_tracks.extend(ibtracs.normalize())

    imd = ImdAdapter()
    if imd.is_available():
        all_tracks.extend(imd.normalize())

    if storm_name:
        target = storm_name.strip().upper()
        all_tracks = [t for t in all_tracks if target in t.storm_name.upper()]

    if storm_id:
        target_id = storm_id.strip().upper()
        all_tracks = [t for t in all_tracks if target_id in t.storm_id.upper()]

    if source:
        all_tracks = [t for t in all_tracks if source.upper() in t.source.upper()]

    # Sort chronologically
    all_tracks.sort(key=lambda x: x.timestamp)
    return all_tracks[:limit]


@router.get(
    "/satellite",
    response_model=List[NormalizedSatelliteRecord],
    summary="Query normalized satellite granules and sensor metadata"
)
def get_satellite_records(
    satellite: Optional[str] = Query(None, description="Filter by satellite: INSAT-3D, INSAT-3DR, INSAT-3DS"),
    channel: Optional[str] = Query(None, description="Filter by channel: TIR1, VIS, WV"),
    limit: int = Query(50, ge=1, le=200)
):
    """Retrieve normalized satellite observation metadata from MOSDAC / ISRO catalog."""
    mosdac = MosdacAdapter()
    records = mosdac.normalize()

    if satellite:
        target_sat = satellite.strip().upper()
        records = [r for r in records if target_sat in r.satellite.upper()]

    if channel:
        target_chan = channel.strip().upper()
        records = [r for r in records if target_chan in r.channel.upper()]

    return records[:limit]


@router.get(
    "/rainfall",
    summary="Query normalized precipitation estimates"
)
def get_rainfall_records(limit: int = Query(50, ge=1, le=100)):
    """Retrieve registered precipitation products from NASA GPM IMERG."""
    gpm = GpmAdapter()
    catalog = gpm.parse_catalog()
    return {
        "dataset": "NASA GPM IMERG (V07B)",
        "status": gpm.get_status().value,
        "is_configured": gpm.is_configured(),
        "records_count": len(catalog),
        "products": catalog[:limit],
        "disclaimer": "Live half-hourly HDF5 downloads require NASA Earthdata Login credentials."
    }


@router.get(
    "/environmental",
    summary="Query normalized atmospheric and oceanic reanalysis metadata"
)
def get_environmental_records(limit: int = Query(50, ge=1, le=100)):
    """Retrieve registered environmental variables from ECMWF ERA5."""
    era5 = Era5Adapter()
    catalog = era5.parse_catalog()
    return {
        "dataset": "ECMWF ERA5 Reanalysis",
        "status": era5.get_status().value,
        "is_configured": era5.is_configured(),
        "records_count": len(catalog),
        "products": catalog[:limit],
        "disclaimer": "Live ERA5 atmospheric GRIB/NetCDF downloads require ECMWF CDS API Key."
    }


@router.get(
    "/{dataset_id}",
    response_model=DatasetMetadata,
    summary="Get detailed metadata for a specific dataset"
)
def get_dataset(dataset_id: str):
    """Retrieve full technical and provenance metadata for a dataset."""
    ds = dataset_registry.get_by_id(dataset_id)
    if not ds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found in registry."
        )
    return ds


@router.get(
    "/{dataset_id}/status",
    summary="Inspect operational connection and physical file status"
)
def get_dataset_status(dataset_id: str):
    """Inspect file presence, credential state, and operational mode."""
    ds = dataset_registry.get_by_id(dataset_id)
    if not ds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found in registry."
        )
    return {
        "dataset_id": ds.dataset_id,
        "name": ds.name,
        "status": ds.status.value,
        "data_mode": ds.data_mode.value,
        "is_on_disk": bool(ds.local_path),
        "local_path": ds.local_path,
        "checksum": ds.checksum,
        "access_type": ds.access_type,
        "source_url": ds.source_url
    }


@router.get(
    "/{dataset_id}/quality",
    response_model=DataQualityReport,
    summary="Run quality control audit on dataset"
)
def get_dataset_quality(dataset_id: str):
    """Executes automated scientific quality control and returns audit report."""
    if dataset_id == "ibtracs_nio":
        adapter = IbtracsAdapter()
        return adapter.run_quality_control()
    elif dataset_id == "imd_best_track":
        adapter = ImdAdapter()
        return adapter.run_quality_control()
    elif dataset_id == "mosdac_insat":
        adapter = MosdacAdapter()
        return adapter.run_quality_control()
    elif dataset_id == "nasa_gpm_imerg":
        adapter = GpmAdapter()
        return adapter.run_quality_control()
    elif dataset_id == "era5_reanalysis":
        adapter = Era5Adapter()
        return adapter.run_quality_control()
    elif dataset_id in ("worldpop_density", "osm_infrastructure", "srtm_elevation"):
        adapter = ExposureAdapter()
        return adapter.run_quality_control()
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' does not have a configured quality auditor."
        )


@router.get(
    "/{dataset_id}/provenance",
    summary="Retrieve cryptographic provenance record"
)
def get_dataset_provenance(dataset_id: str):
    """Returns immutable SHA-256 provenance metadata chain for the dataset."""
    ds = dataset_registry.get_by_id(dataset_id)
    if not ds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found in registry."
        )
    return {
        "dataset_id": ds.dataset_id,
        "source": ds.source,
        "provider": ds.provider,
        "checksum_sha256": ds.checksum,
        "provenance_chain": ds.provenance
    }

