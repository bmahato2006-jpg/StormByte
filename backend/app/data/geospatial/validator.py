"""Geospatial validation for scientific satellite rasters (Phase 11).

Strict Scientific Governance:
1. Zero-Fabrication Rule: Coordinates are never invented, interpolated from fake extents, or assigned defaults.
2. Validates that:
   - -90.0 <= latitude_min < latitude_max <= 90.0
   - -180.0 <= longitude_min < longitude_max <= 180.0
   - image_width > 0 and image_height > 0
   - coordinate arrays contain finite numbers (no NaN or Inf)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from backend.app.data.geospatial.schemas import (
    GeospatialValidationResult,
    RasterMetadata,
)


class GeospatialValidator:
    """Validates spatial metadata, extents, and coordinate arrays for scientific integrity."""

    @classmethod
    def validate_bounds(
        cls,
        lat_min: Optional[float],
        lat_max: Optional[float],
        lon_min: Optional[float],
        lon_max: Optional[float],
    ) -> GeospatialValidationResult:
        """Strictly validates latitude and longitude bounding extents."""
        if any(v is None for v in (lat_min, lat_max, lon_min, lon_max)):
            return GeospatialValidationResult(
                is_valid=False,
                error_code="GEOSPATIAL_METADATA_REQUIRED",
                error_message="Bounding coordinates are missing or incomplete.",
            )

        try:
            lat_min_f = float(lat_min)  # type: ignore
            lat_max_f = float(lat_max)  # type: ignore
            lon_min_f = float(lon_min)  # type: ignore
            lon_max_f = float(lon_max)  # type: ignore
        except (ValueError, TypeError):
            return GeospatialValidationResult(
                is_valid=False,
                error_code="GEOSPATIAL_BOUNDS_PARSE_ERROR",
                error_message="Bounding coordinates must be numeric.",
            )

        if not np.isfinite([lat_min_f, lat_max_f, lon_min_f, lon_max_f]).all():
            return GeospatialValidationResult(
                is_valid=False,
                error_code="NON_FINITE_BOUNDS",
                error_message="Bounding coordinates must be finite numbers (no NaN or Inf).",
            )

        # Latitude range check [-90, +90]
        if not (-90.0 <= lat_min_f <= 90.0) or not (-90.0 <= lat_max_f <= 90.0):
            return GeospatialValidationResult(
                is_valid=False,
                error_code="INVALID_LATITUDE_RANGE",
                error_message=f"Latitude values must be between -90.0 and +90.0 (received min={lat_min_f}, max={lat_max_f}).",
            )

        # Longitude range check [-180, +180]
        if not (-180.0 <= lon_min_f <= 180.0) or not (-180.0 <= lon_max_f <= 180.0):
            return GeospatialValidationResult(
                is_valid=False,
                error_code="INVALID_LONGITUDE_RANGE",
                error_message=f"Longitude values must be between -180.0 and +180.0 (received min={lon_min_f}, max={lon_max_f}).",
            )

        # Ordering check
        if lat_min_f >= lat_max_f:
            return GeospatialValidationResult(
                is_valid=False,
                error_code="INVALID_BOUNDS_ORDER",
                error_message=f"latitude_min ({lat_min_f}) must be strictly less than latitude_max ({lat_max_f}).",
            )

        if lon_min_f >= lon_max_f:
            return GeospatialValidationResult(
                is_valid=False,
                error_code="INVALID_BOUNDS_ORDER",
                error_message=f"longitude_min ({lon_min_f}) must be strictly less than longitude_max ({lon_max_f}).",
            )

        return GeospatialValidationResult(is_valid=True)

    @classmethod
    def validate_coordinate_array(
        cls,
        coords: np.ndarray,
        name: str = "coordinate",
        val_min: float = -180.0,
        val_max: float = 180.0,
    ) -> GeospatialValidationResult:
        """Validates that a 1D or 2D coordinate array contains finite, bounded values."""
        if coords is None or coords.size == 0:
            return GeospatialValidationResult(
                is_valid=False,
                error_code="EMPTY_COORDINATE_ARRAY",
                error_message=f"{name} array is empty or None.",
            )

        if not np.all(np.isfinite(coords)):
            return GeospatialValidationResult(
                is_valid=False,
                error_code="NON_FINITE_COORDINATES",
                error_message=f"{name} array contains non-finite values (NaN or Inf).",
            )

        c_min = float(np.min(coords))
        c_max = float(np.max(coords))

        if c_min < val_min or c_max > val_max:
            return GeospatialValidationResult(
                is_valid=False,
                error_code=f"INVALID_{name.upper()}_RANGE",
                error_message=f"{name} values outside valid range [{val_min}, {val_max}]: range [{c_min:.4f}, {c_max:.4f}].",
            )

        return GeospatialValidationResult(is_valid=True)

    @classmethod
    def validate_dimensions(cls, width: int, height: int) -> GeospatialValidationResult:
        """Validates raster pixel dimensions."""
        if width <= 0 or height <= 0:
            return GeospatialValidationResult(
                is_valid=False,
                error_code="INVALID_IMAGE_DIMENSIONS",
                error_message=f"Raster dimensions must be positive (received width={width}, height={height}).",
            )
        return GeospatialValidationResult(is_valid=True)

