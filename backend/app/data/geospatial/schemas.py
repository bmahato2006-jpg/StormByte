"""Pydantic schemas and dataclasses for Cyclone-X Scientific Geospatial Processing (Phase 11).

Provides normalized representations for:
- Raster metadata (GeoTIFF, NetCDF, HDF5, standard images)
- Georeferencing transformations and coordinate fixes
- Strict validation results
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class PixelCoordinate(BaseModel):
    """Pixel space coordinates (x=column, y=row) in raster image."""
    model_config = ConfigDict(populate_by_name=True)

    x: float = Field(..., description="Pixel column coordinate [0, width]")
    y: float = Field(..., description="Pixel row coordinate [0, height]")


class GeographicCoordinate(BaseModel):
    """Geodetic coordinates in WGS84 (EPSG:4326)."""
    model_config = ConfigDict(populate_by_name=True)

    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees [-90, +90]")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees [-180, +180]")


class RasterMetadata(BaseModel):
    """Normalized metadata extracted from scientific raster datasets."""
    model_config = ConfigDict(populate_by_name=True)

    source_format: str = Field(..., description="Raster format: GEOTIFF, NETCDF, HDF5, PNG, JPEG")
    width: int = Field(..., gt=0, description="Raster width in pixels")
    height: int = Field(..., gt=0, description="Raster height in pixels")
    channels: int = Field(1, ge=1, description="Number of spectral bands / channels")
    dtype: str = Field("float32", description="Underlying NumPy data type string")
    crs: Optional[str] = Field(None, description="Coordinate Reference System (e.g. EPSG:4326, PROJ string, WKT)")
    transform: Optional[List[float]] = Field(None, description="Affine transformation coefficients")
    bounds: Optional[Dict[str, float]] = Field(
        None, description="Bounding extents (latitude_min, latitude_max, longitude_min, longitude_max)"
    )
    resolution: Optional[List[float]] = Field(None, description="Pixel resolution [res_x, res_y]")
    acquisition_time: Optional[str] = Field(None, description="ISO-8601 observation timestamp")
    satellite: Optional[str] = Field(None, description="Satellite platform (e.g. INSAT-3D, INSAT-3DR)")
    sensor: Optional[str] = Field(None, description="Instrument/sensor name (e.g. IMAGER, SOUNDER)")
    channel_name: Optional[str] = Field(None, description="Spectral channel (e.g. TIR1, TIR2, VIS, WV)")
    product_id: Optional[str] = Field(None, description="Official scientific product identifier")
    source: Optional[str] = Field(None, description="Data origin (e.g. MOSDAC/ISRO, NASA Earthdata)")
    variable_names: List[str] = Field(default_factory=list, description="Extracted variable/dataset names")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Raw global/file attributes")
    georeferencing_available: bool = Field(False, description="True only if calibrated geographic reference exists")


class GeoreferencingResult(BaseModel):
    """Structured result of transforming pixel coordinates to geodetic positions."""
    model_config = ConfigDict(populate_by_name=True)

    latitude: Optional[float] = Field(None, description="Geodetic latitude in WGS84")
    longitude: Optional[float] = Field(None, description="Geodetic longitude in WGS84")
    coordinates_available: bool = Field(False, description="True only if genuine geographic coordinates resolved")
    crs: Optional[str] = Field(None, description="Source Coordinate Reference System")
    method: str = Field("UNREFERENCED", description="Georeferencing methodology")
    status: str = Field("GEOSPATIAL_METADATA_REQUIRED", description="Diagnostic status code")
    spatial_note: str = Field(..., description="Scientific provenance / explanation")
    bounds: Optional[Dict[str, float]] = Field(None, description="Validated bounding extents")


class GeospatialValidationResult(BaseModel):
    """Result of geospatial metadata and bounds validation."""
    model_config = ConfigDict(populate_by_name=True)

    is_valid: bool = Field(..., description="Whether validation passed")
    error_code: Optional[str] = Field(None, description="Standardized error code if invalid")
    error_message: Optional[str] = Field(None, description="Human-readable explanation of failure")


class SatelliteRaster:
    """In-memory scientific satellite raster container with array data and metadata."""

    def __init__(
        self,
        array: np.ndarray,
        metadata: RasterMetadata,
        lat_array: Optional[np.ndarray] = None,
        lon_array: Optional[np.ndarray] = None,
    ):
        self.array = array
        self.metadata = metadata
        self.lat_array = lat_array
        self.lon_array = lon_array

    @property
    def width(self) -> int:
        return self.metadata.width

    @property
    def height(self) -> int:
        return self.metadata.height

    @property
    def georeferencing_available(self) -> bool:
        return self.metadata.georeferencing_available

    def to_rgb_image(self) -> Image.Image:
        """Converts the scientific raster array into a normalized 8-bit 3-channel RGB PIL Image.

        Designed for input to the Phase 9 CycloneDetectorCNN V2 model:
        - Replaces NaNs and Infinities with minimum finite values.
        - Handles 2D single-band thermal IR / brightness temperature.
        - Handles 3D multi-band or RGB arrays.
        - Applies linear min-max stretch to [0, 255].
        """
        arr = self.array.copy()

        # Handle NaNs and Infs safely
        finite_mask = np.isfinite(arr)
        if not np.any(finite_mask):
            # Complete void - return black image
            return Image.new("RGB", (self.width, self.height), (0, 0, 0))

        valid_vals = arr[finite_mask]
        min_v = float(np.min(valid_vals))
        max_v = float(np.max(valid_vals))

        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            # Channels-first (C, H, W) -> transpose to (H, W, C)
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 3:
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
            elif arr.shape[2] >= 3:
                # 3 or more channels - use first 3 channels
                arr = arr[:, :, :3]

        if arr.ndim == 2:
            # Single band (e.g. IR 10.8µm)
            if max_v > min_v:
                norm = np.clip((arr - min_v) / (max_v - min_v), 0.0, 1.0)
            else:
                norm = np.zeros_like(arr)
            norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
            u8 = (norm * 255.0).astype(np.uint8)
            img_gray = Image.fromarray(u8, mode="L")
            return img_gray.convert("RGB")
        else:
            # Multi-channel
            if max_v > min_v:
                norm = np.clip((arr - min_v) / (max_v - min_v), 0.0, 1.0)
            else:
                norm = np.zeros_like(arr)
            norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
            u8 = (norm * 255.0).astype(np.uint8)
            return Image.fromarray(u8, mode="RGB")


class NRTAdapterStatus(BaseModel):
    """Operational and configuration status of a live satellite NRT adapter."""
    model_config = ConfigDict(populate_by_name=True)

    status: str = Field(
        ...,
        description="Operational status: CREDENTIALS_REQUIRED, AUTHENTICATION_FAILED, ACCESS_NOT_GRANTED, CONNECTED, NO_DATA_AVAILABLE, CONFIGURED, or ERROR"
    )
    adapter: str = Field(..., description="Adapter identifier (e.g. mosdac_insat3d)")
    provider: str = Field(..., description="Satellite data provider name")
    configured: bool = Field(..., description="Whether valid authentication credentials are configured")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of status check")
    satellites: List[str] = Field(default_factory=list, description="Supported satellite platforms")
    products: List[str] = Field(default_factory=list, description="Supported imagery and sensor products")
    target_sector: str = Field("NORTH_INDIAN_OCEAN", description="Monitored geographic domain")
    dataset_id: Optional[str] = Field(None, description="Active MOSDAC DatasetId")
    environment_variables: Dict[str, bool] = Field(default_factory=dict, description="Configuration status of required environment variables")
    connectivity: str = Field(..., description="AUTHENTICATION_REQUIRED, CREDENTIALS_PRESENT_UNVERIFIED, CONNECTED, AUTHENTICATION_FAILED, ACCESS_NOT_GRANTED, or ERROR")
    session_authenticated: Optional[bool] = Field(False, description="True only if active authenticated session has succeeded")
    message: str = Field(..., description="Human-readable status and guidance message")
    official_documentation: Optional[str] = Field(None, description="Link to official download API documentation")
    zero_fabrication_rule: str = Field(..., description="Scientific integrity governance rule")
    registered_adapters: Optional[Dict[str, Any]] = Field(None, description="Status summary across all registered live adapters")


class NRTLatestGranuleResponse(BaseModel):
    """Result of requesting the latest calibrated satellite raster granule."""
    model_config = ConfigDict(populate_by_name=True)

    status: str = Field(
        ...,
        description="Status: CREDENTIALS_REQUIRED, AUTHENTICATION_FAILED, ACCESS_NOT_GRANTED, CONNECTED, NO_DATA_AVAILABLE, or ERROR"
    )
    adapter: str = Field(..., description="Adapter identifier")
    provider: str = Field(..., description="Satellite provider name")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of request")
    granule_available: bool = Field(False, description="True only if an authentic live satellite raster granule was retrieved")
    granule: Optional[Dict[str, Any]] = Field(None, description="Granule metadata if available, strictly null if unauthenticated")
    provenance: Optional[Dict[str, Any]] = Field(None, description="Lineage and cryptographic provenance if data retrieved")
    message: str = Field(..., description="Truthful operational status message")
    zero_fabrication_rule: str = Field(
        "Zero satellite observations, coordinates, or imagery are fabricated.",
        description="Scientific integrity governance note"
    )


class LocationAwareNRTObservation(BaseModel):
    """Normalized schema for location-aware NRT satellite observations.

    Guarantees that real latitude, longitude, and spatial bounds are preserved.
    Rejects files lacking geospatial metadata with status GEOSPATIAL_METADATA_REQUIRED.
    """
    model_config = ConfigDict(populate_by_name=True)

    latitude: float = Field(..., ge=-90.0, le=90.0, description="Center or observation geodetic latitude [-90, +90]")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Center or observation geodetic longitude [-180, +180]")
    min_latitude: float = Field(..., ge=-90.0, le=90.0, description="Minimum bounding latitude")
    max_latitude: float = Field(..., ge=-90.0, le=90.0, description="Maximum bounding latitude")
    min_longitude: float = Field(..., ge=-180.0, le=180.0, description="Minimum bounding longitude")
    max_longitude: float = Field(..., ge=-180.0, le=180.0, description="Maximum bounding longitude")
    crs: str = Field(..., description="Coordinate Reference System / projection (e.g. EPSG:4326, PROJCS)")
    acquisition_time: str = Field(..., description="ISO-8601 observation timestamp")
    satellite: str = Field(..., description="Satellite platform name (e.g. INSAT-3D, INSAT-3DR, Aqua, Terra)")
    sensor: str = Field(..., description="Sensor / instrument name (e.g. IMAGER, MODIS, VIIRS)")
    dataset_id: str = Field(..., description="Official dataset identifier (e.g. 3DIMG_L1B_STD, GIBS)")
    granule_id: str = Field(..., description="Unique granule identifier")
    provider: str = Field(default="MOSDAC/ISRO", description="Authoritative satellite data provider")
    resolution: Optional[str] = Field(None, description="Spatial pixel resolution (e.g. 1km, 4km)")
    channel: Optional[str] = Field(None, description="Spectral band name (e.g. TIR1, VIS)")
    grid_dimensions: Optional[List[int]] = Field(None, description="Original grid dimensions [height, width]")
    file_path: Optional[str] = Field(None, description="Path to authentic satellite image file on disk if cached")
    file_format: Optional[str] = Field(None, description="Underlying file format: GEOTIFF, NETCDF, HDF5, PNG, JPEG")
    sha256: Optional[str] = Field(None, description="Cryptographic SHA-256 hash of the granule")
    is_location_ready: bool = Field(True, description="True only if authentic geographic coordinates are validated")


class RequestedLocation(BaseModel):
    """User-requested geographic target location."""
    model_config = ConfigDict(populate_by_name=True)

    latitude: float = Field(..., ge=-90.0, le=90.0, description="Requested latitude in decimal degrees [-90, +90]")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Requested longitude in decimal degrees [-180, +180]")


class NRTSearchObservation(BaseModel):
    """Observation detail returned in location search."""
    model_config = ConfigDict(populate_by_name=True)

    latitude: float = Field(..., description="Observation center geodetic latitude")
    longitude: float = Field(..., description="Observation center geodetic longitude")
    min_latitude: float = Field(..., description="Minimum bounding latitude")
    max_latitude: float = Field(..., description="Maximum bounding latitude")
    min_longitude: float = Field(..., description="Minimum bounding longitude")
    max_longitude: float = Field(..., description="Maximum bounding longitude")
    bounds: Dict[str, float] = Field(..., description="Geographic bounding box dict")
    acquisition_time: str = Field(..., description="Observation timestamp")
    granule_id: str = Field(..., description="Granule identifier")
    satellite: Optional[str] = Field(None, description="Satellite name")
    sensor: Optional[str] = Field(None, description="Sensor name")
    dataset_id: Optional[str] = Field(None, description="Dataset identifier")
    crs: Optional[str] = Field(None, description="Coordinate Reference System")
    provider: Optional[str] = Field(None, description="Provider")
    channel: Optional[str] = Field(None, description="Channel / band")
    resolution: Optional[str] = Field(None, description="Resolution")
    grid_dimensions: Optional[List[int]] = Field(None, description="Grid dimensions [H, W]")
    sha256: Optional[str] = Field(None, description="SHA-256 hash")


class NRTLatLonSearchResponse(BaseModel):
    """Response returned by GET /api/satellite/nrt/search."""
    model_config = ConfigDict(populate_by_name=True)

    requested_location: RequestedLocation
    observation: Optional[NRTSearchObservation] = Field(None, description="Nearest valid covering observation, null if outside coverage")
    coverage: bool = Field(..., description="True if requested point is inside actual granule bounds")
    distance_km: Optional[float] = Field(None, description="Distance from requested location to observation center in km")
    source: str = Field("MOSDAC/ISRO", description="Data provider source")
    status: str = Field(..., description="COVERED or LOCATION_OUTSIDE_GRANULE_COVERAGE")
    message: Optional[str] = Field(None, description="Human-readable explanation")
    detection: Optional[Dict[str, Any]] = Field(None, description="Cyclone detection result on the observation")
    provenance: Optional[Dict[str, Any]] = Field(None, description="Provenance audit trail")
    zero_fabrication_rule: str = Field(
        "Zero coordinates, boundaries, or detections are fabricated. Observations require verified geospatial metadata.",
        description="Scientific integrity governance note"
    )


