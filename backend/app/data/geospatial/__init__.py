"""Cyclone-X Scientific Geospatial Processing Module (Phase 11).

Provides:
- Unified reading of scientific rasters (GeoTIFF, NetCDF, HDF5, PNG, JPEG)
- Strict bounds, CRS, and coordinate validation
- Automatic WGS84 georeferencing
- Zero coordinate fabrication governance
- Near-real-time satellite ingestion adapter interfaces
"""

from backend.app.data.geospatial.schemas import (
    GeographicCoordinate,
    GeoreferencingResult,
    GeospatialValidationResult,
    LocationAwareNRTObservation,
    NRTAdapterStatus,
    NRTLatestGranuleResponse,
    NRTLatLonSearchResponse,
    NRTSearchObservation,
    PixelCoordinate,
    RasterMetadata,
    RequestedLocation,
    SatelliteRaster,
)
from backend.app.data.geospatial.validator import GeospatialValidator
from backend.app.data.geospatial.metadata import RasterMetadataExtractor
from backend.app.data.geospatial.georeferencer import Georeferencer
from backend.app.data.geospatial.reader import read_raster
from backend.app.data.geospatial.live_adapters import (
    LIVE_ADAPTERS,
    MOSDACInsatAdapter,
    MOSDACStatus,
    NASAEarthdataAdapter,
    NASAGPMImergAdapter,
    get_all_adapters_status,
    mosdac_adapter,
    nasa_earthdata_adapter,
    nasa_gpm_adapter,
)
from backend.app.data.geospatial.nrt_service import (
    NRTObservationService,
    nrt_observation_service,
    haversine_distance_km,
    is_point_in_bounds,
    calculate_bounds_from_transform,
    calculate_bounds_from_grid,
)

__all__ = [
    "read_raster",
    "SatelliteRaster",
    "RasterMetadata",
    "PixelCoordinate",
    "GeographicCoordinate",
    "GeoreferencingResult",
    "GeospatialValidationResult",
    "NRTAdapterStatus",
    "NRTLatestGranuleResponse",
    "LocationAwareNRTObservation",
    "RequestedLocation",
    "NRTSearchObservation",
    "NRTLatLonSearchResponse",
    "GeospatialValidator",
    "RasterMetadataExtractor",
    "Georeferencer",
    "MOSDACInsatAdapter",
    "MOSDACStatus",
    "NASAEarthdataAdapter",
    "NASAGPMImergAdapter",
    "mosdac_adapter",
    "nasa_earthdata_adapter",
    "nasa_gpm_adapter",
    "LIVE_ADAPTERS",
    "get_all_adapters_status",
    "NRTObservationService",
    "nrt_observation_service",
    "haversine_distance_km",
    "is_point_in_bounds",
    "calculate_bounds_from_transform",
    "calculate_bounds_from_grid",
]
