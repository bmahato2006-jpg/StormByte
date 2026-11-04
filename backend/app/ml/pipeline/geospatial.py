"""Geospatial validation, raster georeferencing, and coordinate transformation for Cyclone-X.

Strict Scientific Governance:
1. Zero-Fabrication Rule: Coordinates are never invented or filled with default values.
2. If georeferencing metadata is missing: returns None, coordinates_available=False,
   and diagnostic status "GEOSPATIAL_METADATA_REQUIRED".
3. Validates that:
   - -90.0 <= latitude_min < latitude_max <= 90.0
   - -180.0 <= longitude_min < longitude_max <= 180.0
   If invalid, returns None and an explicit validation error code.
"""

from typing import Any, Dict, List, Optional, Tuple


class GeospatialValidator:
    """Validates raster spatial reference metadata and converts pixel coordinates to geodetic positions."""

    @classmethod
    def validate_bounds(
        cls,
        lat_min: Optional[float],
        lat_max: Optional[float],
        lon_min: Optional[float],
        lon_max: Optional[float],
    ) -> Tuple[bool, Optional[str]]:
        """Strictly validates latitude and longitude bounding extents.

        Returns:
            (is_valid: bool, error_message: Optional[str])
        """
        if any(v is None for v in (lat_min, lat_max, lon_min, lon_max)):
            return False, "GEOSPATIAL_METADATA_REQUIRED"

        try:
            lat_min_f = float(lat_min)  # type: ignore
            lat_max_f = float(lat_max)  # type: ignore
            lon_min_f = float(lon_min)  # type: ignore
            lon_max_f = float(lon_max)  # type: ignore
        except (ValueError, TypeError):
            return False, "GEOSPATIAL_BOUNDS_PARSE_ERROR: Bounding coordinates must be numeric."

        # Latitude range check [-90, +90]
        if not (-90.0 <= lat_min_f <= 90.0) or not (-90.0 <= lat_max_f <= 90.0):
            return (
                False,
                f"INVALID_LATITUDE_RANGE: Latitude values must be between -90.0 and +90.0 (received min={lat_min_f}, max={lat_max_f}).",
            )

        # Longitude range check [-180, +180]
        if not (-180.0 <= lon_min_f <= 180.0) or not (-180.0 <= lon_max_f <= 180.0):
            return (
                False,
                f"INVALID_LONGITUDE_RANGE: Longitude values must be between -180.0 and +180.0 (received min={lon_min_f}, max={lon_max_f}).",
            )

        # Invariant: min must be strictly less than max
        if lat_min_f >= lat_max_f:
            return (
                False,
                f"INVALID_BOUNDS_ORDER: latitude_min ({lat_min_f}) must be strictly less than latitude_max ({lat_max_f}).",
            )

        if lon_min_f >= lon_max_f:
            return (
                False,
                f"INVALID_BOUNDS_ORDER: longitude_min ({lon_min_f}) must be strictly less than longitude_max ({lon_max_f}).",
            )

        return True, None

    @classmethod
    def resolve_center_coordinates(
        cls,
        pixel_x: float,
        pixel_y: float,
        image_width: int,
        image_height: int,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Converts pixel circulation center to geodetic latitude and longitude if verified bounds exist.

        Returns structured dictionary with coordinates, availability flag, and diagnostic notes.
        """
        # Extract bounds supporting both canonical naming conventions
        lat_min = metadata.get("latitude_min") if metadata.get("latitude_min") is not None else metadata.get("lat_min")
        lat_max = metadata.get("latitude_max") if metadata.get("latitude_max") is not None else metadata.get("lat_max")
        lon_min = metadata.get("longitude_min") if metadata.get("longitude_min") is not None else metadata.get("lon_min")
        lon_max = metadata.get("longitude_max") if metadata.get("longitude_max") is not None else metadata.get("lon_max")

        is_valid, err = cls.validate_bounds(lat_min, lat_max, lon_min, lon_max)

        if not is_valid:
            return {
                "latitude": None,
                "longitude": None,
                "coordinates_available": False,
                "status": err or "GEOSPATIAL_METADATA_REQUIRED",
                "spatial_note": (
                    "Geographic position unavailable: This satellite image does not contain "
                    "calibrated geospatial metadata. Forecast track generation requires a verified geographic position."
                    if err == "GEOSPATIAL_METADATA_REQUIRED"
                    else f"Geospatial bounds validation failed: {err}"
                ),
                "bounds": None,
            }

        lat_min_f = float(lat_min)  # type: ignore
        lat_max_f = float(lat_max)  # type: ignore
        lon_min_f = float(lon_min)  # type: ignore
        lon_max_f = float(lon_max)  # type: ignore

        if image_width <= 0 or image_height <= 0:
            return {
                "latitude": None,
                "longitude": None,
                "coordinates_available": False,
                "status": "INVALID_IMAGE_DIMENSIONS",
                "spatial_note": "Image dimensions must be greater than zero for georeferencing.",
                "bounds": None,
            }

        # Linear affine mapping from raster pixel coordinate to geodetic coordinate
        # (Pixel 0,0 is at North-West / Top-Left corner)
        norm_y = max(0.0, min(1.0, pixel_y / float(image_height)))
        norm_x = max(0.0, min(1.0, pixel_x / float(image_width)))

        geo_lat = round(lat_max_f - norm_y * (lat_max_f - lat_min_f), 4)
        geo_lon = round(lon_min_f + norm_x * (lon_max_f - lon_min_f), 4)

        return {
            "latitude": geo_lat,
            "longitude": geo_lon,
            "coordinates_available": True,
            "status": "CALIBRATED_GEOREFERENCE",
            "spatial_note": "Resolved from calibrated satellite raster bounding box.",
            "bounds": {
                "latitude_min": lat_min_f,
                "latitude_max": lat_max_f,
                "longitude_min": lon_min_f,
                "longitude_max": lon_max_f,
            },
        }

