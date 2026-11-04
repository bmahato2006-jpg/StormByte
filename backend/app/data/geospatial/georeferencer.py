"""Automatic georeferencing service for scientific satellite rasters (Phase 11).

Converts:
  Pixel Coordinate (x, y)
          ↓
  Raster Transform / Grid Lookup
          ↓
  Source CRS Projection
          ↓
  WGS84 Geodetic Position (lat, lon)

Strict Scientific Governance:
1. Zero Coordinate Fabrication: If georeferencing metadata or coordinate grids are absent,
   returns latitude=None, longitude=None, coordinates_available=False,
   and status="GEOSPATIAL_METADATA_REQUIRED".
2. Coordinates are NEVER inferred from filename, dataset name, image centers, or assumed Indian Ocean bounds.
3. Reprojection is performed using pyproj / rasterio.warp for non-WGS84 coordinate reference systems.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from backend.app.data.geospatial.schemas import (
    GeoreferencingResult,
    PixelCoordinate,
    RasterMetadata,
    SatelliteRaster,
)
from backend.app.data.geospatial.validator import GeospatialValidator


class Georeferencer:
    """Performs rigorous coordinate transformation from raster pixel space to WGS84."""

    @classmethod
    def georeference_pixel(
        cls,
        pixel_x: float,
        pixel_y: float,
        raster: Optional[SatelliteRaster] = None,
        metadata: Optional[Union[RasterMetadata, Dict[str, Any]]] = None,
    ) -> GeoreferencingResult:
        """Resolves pixel position (x=column, y=row) into geographic WGS84 (lat, lon)."""
        # 1. Normalize metadata
        meta_dict: Dict[str, Any] = {}
        if raster is not None:
            meta_dict = raster.metadata.model_dump()
        elif isinstance(metadata, RasterMetadata):
            meta_dict = metadata.model_dump()
        elif isinstance(metadata, dict):
            meta_dict = dict(metadata)

        width = meta_dict.get("width") or (raster.width if raster else 0)
        height = meta_dict.get("height") or (raster.height if raster else 0)

        # Dimension validation
        dim_val = GeospatialValidator.validate_dimensions(width, height)
        if not dim_val.is_valid:
            return GeoreferencingResult(
                latitude=None,
                longitude=None,
                coordinates_available=False,
                method="INVALID_DIMENSIONS",
                status=dim_val.error_code or "INVALID_IMAGE_DIMENSIONS",
                spatial_note="Raster dimensions must be greater than zero for georeferencing.",
            )

        # Clamp pixel coordinates within raster boundaries
        px = max(0.0, min(float(width), float(pixel_x)))
        py = max(0.0, min(float(height), float(pixel_y)))

        # ---------------------------------------------------------------------
        # Case 1: Coordinate Arrays (NetCDF / HDF5 1D or 2D grids)
        # ---------------------------------------------------------------------
        if raster is not None and raster.lat_array is not None and raster.lon_array is not None:
            lat_arr = raster.lat_array
            lon_arr = raster.lon_array

            # 1D coordinate vectors: lat_arr has length H, lon_arr has length W
            if lat_arr.ndim == 1 and lon_arr.ndim == 1:
                idx_y = max(0, min(len(lat_arr) - 1, int(round(py))))
                idx_x = max(0, min(len(lon_arr) - 1, int(round(px))))
                res_lat = float(lat_arr[idx_y])
                res_lon = float(lon_arr[idx_x])

                if np.isfinite(res_lat) and np.isfinite(res_lon) and -90.0 <= res_lat <= 90.0 and -180.0 <= res_lon <= 180.0:
                    return GeoreferencingResult(
                        latitude=round(res_lat, 4),
                        longitude=round(res_lon, 4),
                        coordinates_available=True,
                        crs="EPSG:4326",
                        method="COORDINATE_ARRAY_1D_LOOKUP",
                        status="CALIBRATED_GEOREFERENCE",
                        spatial_note="Resolved from calibrated 1D latitude and longitude coordinate variables.",
                        bounds=meta_dict.get("bounds"),
                    )

            # 2D coordinate grids: lat_arr has shape (H, W), lon_arr has shape (H, W)
            elif lat_arr.ndim == 2 and lon_arr.ndim == 2:
                idx_y = max(0, min(lat_arr.shape[0] - 1, int(round(py))))
                idx_x = max(0, min(lat_arr.shape[1] - 1, int(round(px))))
                res_lat = float(lat_arr[idx_y, idx_x])
                res_lon = float(lon_arr[idx_y, idx_x])

                if np.isfinite(res_lat) and np.isfinite(res_lon) and -90.0 <= res_lat <= 90.0 and -180.0 <= res_lon <= 180.0:
                    return GeoreferencingResult(
                        latitude=round(res_lat, 4),
                        longitude=round(res_lon, 4),
                        coordinates_available=True,
                        crs="EPSG:4326",
                        method="COORDINATE_GRID_2D_LOOKUP",
                        status="CALIBRATED_GEOREFERENCE",
                        spatial_note="Resolved from calibrated 2D curvilinear latitude and longitude grid.",
                        bounds=meta_dict.get("bounds"),
                    )

        # ---------------------------------------------------------------------
        # Case 2: GeoTIFF Affine Transform & CRS Projection
        # ---------------------------------------------------------------------
        tf_list = meta_dict.get("transform")
        crs_str = meta_dict.get("crs")

        if tf_list and len(tf_list) >= 6 and crs_str:
            try:
                # Affine transform: x_geo = a*px + b*py + c, y_geo = d*px + e*py + f
                a, b, c, d, e, f = tf_list[:6]
                geo_x = a * px + b * py + c
                geo_y = d * px + e * py + f

                # Reproject to WGS84 if needed
                is_wgs84 = "4326" in crs_str or "WGS 84" in crs_str or "WGS84" in crs_str

                if is_wgs84:
                    lat_val = float(geo_y)
                    lon_val = float(geo_x)
                else:
                    import pyproj
                    transformer = pyproj.Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)
                    lon_val, lat_val = transformer.transform(geo_x, geo_y)

                if np.isfinite(lat_val) and np.isfinite(lon_val) and -90.0 <= lat_val <= 90.0 and -180.0 <= lon_val <= 180.0:
                    return GeoreferencingResult(
                        latitude=round(float(lat_val), 4),
                        longitude=round(float(lon_val), 4),
                        coordinates_available=True,
                        crs=crs_str,
                        method="AFFINE_TRANSFORM_REPROJECTION",
                        status="CALIBRATED_GEOREFERENCE",
                        spatial_note=f"Resolved via raster affine transform and reprojected from {crs_str} to WGS84.",
                        bounds=meta_dict.get("bounds"),
                    )
            except Exception as ex:
                pass  # Fall through to bounds check

        # ---------------------------------------------------------------------
        # Case 3: Explicit Calibrated Bounding Extents
        # ---------------------------------------------------------------------
        b_dict = meta_dict.get("bounds") or {}
        lat_min = b_dict.get("latitude_min") if b_dict.get("latitude_min") is not None else meta_dict.get("latitude_min")
        lat_max = b_dict.get("latitude_max") if b_dict.get("latitude_max") is not None else meta_dict.get("latitude_max")
        lon_min = b_dict.get("longitude_min") if b_dict.get("longitude_min") is not None else meta_dict.get("longitude_min")
        lon_max = b_dict.get("longitude_max") if b_dict.get("longitude_max") is not None else meta_dict.get("longitude_max")

        b_val = GeospatialValidator.validate_bounds(lat_min, lat_max, lon_min, lon_max)
        if b_val.is_valid:
            lat_min_f = float(lat_min)  # type: ignore
            lat_max_f = float(lat_max)  # type: ignore
            lon_min_f = float(lon_min)  # type: ignore
            lon_max_f = float(lon_max)  # type: ignore

            # Standard linear mapping: top edge is lat_max, bottom edge is lat_min
            norm_y = py / float(height)
            norm_x = px / float(width)

            geo_lat = lat_max_f - norm_y * (lat_max_f - lat_min_f)
            geo_lon = lon_min_f + norm_x * (lon_max_f - lon_min_f)

            return GeoreferencingResult(
                latitude=round(geo_lat, 4),
                longitude=round(geo_lon, 4),
                coordinates_available=True,
                crs="EPSG:4326",
                method="CALIBRATED_BOUNDS_INTERPOLATION",
                status="CALIBRATED_GEOREFERENCE",
                spatial_note="Resolved from verified raster bounding extents.",
                bounds={
                    "latitude_min": lat_min_f,
                    "latitude_max": lat_max_f,
                    "longitude_min": lon_min_f,
                    "longitude_max": lon_max_f,
                },
            )

        # ---------------------------------------------------------------------
        # Case 4: Zero Coordinate Fabrication Rule (Unreferenced Raster)
        # ---------------------------------------------------------------------
        return GeoreferencingResult(
            latitude=None,
            longitude=None,
            coordinates_available=False,
            method="UNREFERENCED",
            status="GEOSPATIAL_METADATA_REQUIRED",
            spatial_note=(
                "Geographic position unavailable: This satellite image does not contain "
                "calibrated geospatial metadata. Forecast track generation requires a verified geographic position."
            ),
            bounds=None,
        )

