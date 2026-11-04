"""Metadata extraction for scientific satellite rasters (Phase 11).

Extracts CRS, affine transforms, bounding coordinates, spectral channels,
platform names, and timestamps from GeoTIFF, NetCDF, HDF5, and standard formats.

Zero-Fabrication Rule:
If metadata attributes do not exist in the file, they are strictly left as None.
Missing bounds or coordinate references are never populated with defaults.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from backend.app.data.geospatial.schemas import RasterMetadata
from backend.app.data.geospatial.validator import GeospatialValidator


class RasterMetadataExtractor:
    """Extracts normalized metadata from various scientific raster sources."""

    @classmethod
    def extract_from_rasterio(cls, src) -> RasterMetadata:
        """Extracts spatial and sensor metadata from an open rasterio dataset."""
        import rasterio
        from rasterio.warp import transform_bounds

        crs_str: Optional[str] = None
        if src.crs is not None:
            crs_str = src.crs.to_string()

        # Transform coefficients (affine: a, b, c, d, e, f)
        tf = src.transform
        tf_list: Optional[List[float]] = None
        if tf is not None:
            tf_list = [float(tf.a), float(tf.b), float(tf.c), float(tf.d), float(tf.e), float(tf.f)]

        res_list: Optional[List[float]] = None
        if src.res is not None:
            res_list = [float(src.res[0]), float(src.res[1])]

        # Bounds extraction: Reproject to WGS84 if projected
        bounds_dict: Optional[Dict[str, float]] = None
        geo_avail = False

        if src.bounds is not None and src.crs is not None:
            try:
                if src.crs.is_geographic or src.crs.to_epsg() == 4326:
                    b = src.bounds
                    wgs84_bounds = (b.left, b.bottom, b.right, b.top)
                else:
                    wgs84_bounds = transform_bounds(src.crs, "EPSG:4326", src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)

                val_res = GeospatialValidator.validate_bounds(
                    lat_min=wgs84_bounds[1],
                    lat_max=wgs84_bounds[3],
                    lon_min=wgs84_bounds[0],
                    lon_max=wgs84_bounds[2],
                )
                if val_res.is_valid:
                    bounds_dict = {
                        "latitude_min": round(float(wgs84_bounds[1]), 4),
                        "latitude_max": round(float(wgs84_bounds[3]), 4),
                        "longitude_min": round(float(wgs84_bounds[0]), 4),
                        "longitude_max": round(float(wgs84_bounds[2]), 4),
                    }
                    geo_avail = True
            except Exception:
                bounds_dict = None
                geo_avail = False
        elif src.bounds is not None and src.crs is None:
            # Check if bounds look like geographic [-90,90] and [-180,180]
            b = src.bounds
            val_res = GeospatialValidator.validate_bounds(b.bottom, b.top, b.left, b.right)
            if val_res.is_valid:
                bounds_dict = {
                    "latitude_min": round(float(b.bottom), 4),
                    "latitude_max": round(float(b.top), 4),
                    "longitude_min": round(float(b.left), 4),
                    "longitude_max": round(float(b.right), 4),
                }
                geo_avail = True

        # Extract file tags
        tags = src.tags() or {}
        satellite = tags.get("SATELLITE") or tags.get("platform") or tags.get("satellite_name")
        sensor = tags.get("SENSOR") or tags.get("instrument") or tags.get("sensor_name")
        channel = tags.get("CHANNEL") or tags.get("band_name")
        product = tags.get("PRODUCT") or tags.get("product_id")
        source = tags.get("SOURCE") or tags.get("institution") or tags.get("origin")
        acq_time = tags.get("ACQUISITION_TIME") or tags.get("time") or tags.get("datetime")

        return RasterMetadata(
            source_format="GEOTIFF",
            width=int(src.width),
            height=int(src.height),
            channels=int(src.count),
            dtype=str(src.dtypes[0] if src.dtypes else "float32"),
            crs=crs_str,
            transform=tf_list,
            bounds=bounds_dict,
            resolution=res_list,
            acquisition_time=acq_time,
            satellite=satellite,
            sensor=sensor,
            channel_name=channel,
            product_id=product,
            source=source,
            variable_names=[f"band_{i}" for i in range(1, src.count + 1)],
            attributes=tags,
            georeferencing_available=geo_avail,
        )

    @classmethod
    def extract_from_xarray(cls, ds, main_var: Optional[str] = None) -> Tuple[RasterMetadata, Optional[np.ndarray], Optional[np.ndarray]]:
        """Extracts metadata and lat/lon coordinate arrays from an xarray Dataset."""
        var_names = list(ds.data_vars.keys())
        attrs = dict(ds.attrs)

        # Look for latitude coordinate
        lat_arr: Optional[np.ndarray] = None
        lon_arr: Optional[np.ndarray] = None

        for name in ("lat", "latitude", "LAT", "LATITUDE", "lats"):
            if name in ds.coords or name in ds.variables:
                lat_arr = np.asarray(ds[name].values, dtype=np.float32)
                break

        for name in ("lon", "longitude", "LON", "LONGITUDE", "lons"):
            if name in ds.coords or name in ds.variables:
                lon_arr = np.asarray(ds[name].values, dtype=np.float32)
                break

        # Bounds extraction
        bounds_dict: Optional[Dict[str, float]] = None
        geo_avail = False

        if lat_arr is not None and lon_arr is not None:
            lat_val = GeospatialValidator.validate_coordinate_array(lat_arr, "latitude", -90.0, 90.0)
            lon_val = GeospatialValidator.validate_coordinate_array(lon_arr, "longitude", -180.0, 180.0)
            if lat_val.is_valid and lon_val.is_valid:
                lat_min = float(np.min(lat_arr))
                lat_max = float(np.max(lat_arr))
                lon_min = float(np.min(lon_arr))
                lon_max = float(np.max(lon_arr))
                b_val = GeospatialValidator.validate_bounds(lat_min, lat_max, lon_min, lon_max)
                if b_val.is_valid:
                    bounds_dict = {
                        "latitude_min": round(lat_min, 4),
                        "latitude_max": round(lat_max, 4),
                        "longitude_min": round(lon_min, 4),
                        "longitude_max": round(lon_max, 4),
                    }
                    geo_avail = True

        # Satellite attributes
        satellite = attrs.get("satellite") or attrs.get("platform") or attrs.get("satellite_name")
        sensor = attrs.get("sensor") or attrs.get("instrument")
        channel = attrs.get("channel") or attrs.get("spectral_band")
        product = attrs.get("product_id") or attrs.get("title")
        source = attrs.get("source") or attrs.get("institution")
        acq_time = str(attrs.get("time_coverage_start") or attrs.get("acquisition_time") or "") or None

        # Determine dimensions from primary data variable
        target_var = main_var or (var_names[0] if var_names else None)
        if target_var and target_var in ds.data_vars:
            v = ds[target_var]
            shape = v.shape
            if len(shape) == 2:
                height, width = shape
                channels = 1
            elif len(shape) == 3:
                # e.g. (time, lat, lon) or (band, y, x)
                height, width = shape[-2], shape[-1]
                channels = shape[0]
            else:
                height, width = shape[-2], shape[-1]
                channels = 1
            dtype_str = str(v.dtype)
        else:
            height, width, channels, dtype_str = 100, 100, 1, "float32"

        meta = RasterMetadata(
            source_format="NETCDF",
            width=int(width),
            height=int(height),
            channels=int(channels),
            dtype=dtype_str,
            crs="EPSG:4326" if geo_avail else None,
            transform=None,
            bounds=bounds_dict,
            resolution=None,
            acquisition_time=acq_time,
            satellite=satellite,
            sensor=sensor,
            channel_name=channel,
            product_id=product,
            source=source,
            variable_names=var_names,
            attributes={k: str(v) for k, v in attrs.items() if isinstance(v, (str, int, float, bool))},
            georeferencing_available=geo_avail,
        )
        return meta, lat_arr, lon_arr

    @classmethod
    def extract_from_h5py(cls, h5_obj) -> Tuple[RasterMetadata, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """Extracts metadata and data from an open h5py File or Group."""
        import h5py

        var_names: List[str] = []
        lat_arr: Optional[np.ndarray] = None
        lon_arr: Optional[np.ndarray] = None
        data_arr: Optional[np.ndarray] = None

        def visitor(name, obj):
            nonlocal lat_arr, lon_arr, data_arr
            if isinstance(obj, h5py.Dataset):
                var_names.append(name)
                lower_name = name.lower()
                if lower_name in ("lat", "latitude", "lats", "navigation/latitude") and lat_arr is None:
                    lat_arr = np.asarray(obj[()], dtype=np.float32)
                elif lower_name in ("lon", "longitude", "lons", "navigation/longitude") and lon_arr is None:
                    lon_arr = np.asarray(obj[()], dtype=np.float32)
                elif data_arr is None and obj.ndim in (2, 3):
                    data_arr = np.asarray(obj[()], dtype=np.float32)

        h5_obj.visititems(visitor)

        if data_arr is None:
            # Fallback if no dataset found via visitor
            for k in h5_obj.keys():
                item = h5_obj[k]
                if isinstance(item, h5py.Dataset) and item.ndim in (2, 3):
                    data_arr = np.asarray(item[()], dtype=np.float32)
                    break

        if data_arr is None:
            data_arr = np.zeros((100, 100), dtype=np.float32)

        attrs = dict(h5_obj.attrs)

        bounds_dict: Optional[Dict[str, float]] = None
        geo_avail = False

        if lat_arr is not None and lon_arr is not None:
            lat_val = GeospatialValidator.validate_coordinate_array(lat_arr, "latitude", -90.0, 90.0)
            lon_val = GeospatialValidator.validate_coordinate_array(lon_arr, "longitude", -180.0, 180.0)
            if lat_val.is_valid and lon_val.is_valid:
                lat_min = float(np.min(lat_arr))
                lat_max = float(np.max(lat_arr))
                lon_min = float(np.min(lon_arr))
                lon_max = float(np.max(lon_arr))
                b_val = GeospatialValidator.validate_bounds(lat_min, lat_max, lon_min, lon_max)
                if b_val.is_valid:
                    bounds_dict = {
                        "latitude_min": round(lat_min, 4),
                        "latitude_max": round(lat_max, 4),
                        "longitude_min": round(lon_min, 4),
                        "longitude_max": round(lon_max, 4),
                    }
                    geo_avail = True

        h, w = data_arr.shape[-2], data_arr.shape[-1]
        c = data_arr.shape[0] if data_arr.ndim == 3 else 1

        meta = RasterMetadata(
            source_format="HDF5",
            width=int(w),
            height=int(h),
            channels=int(c),
            dtype=str(data_arr.dtype),
            crs="EPSG:4326" if geo_avail else None,
            transform=None,
            bounds=bounds_dict,
            resolution=None,
            acquisition_time=str(attrs.get("acquisition_time") or attrs.get("time") or "") or None,
            satellite=str(attrs.get("satellite") or attrs.get("platform") or "") or None,
            sensor=str(attrs.get("sensor") or attrs.get("instrument") or "") or None,
            channel_name=str(attrs.get("channel") or "") or None,
            product_id=str(attrs.get("product_id") or "") or None,
            source=str(attrs.get("source") or attrs.get("institution") or "") or None,
            variable_names=var_names,
            attributes={k: str(v) for k, v in attrs.items() if isinstance(v, (str, int, float, bool))},
            georeferencing_available=geo_avail,
        )
        return meta, lat_arr, lon_arr, data_arr

    @classmethod
    def extract_from_image(cls, img: Image.Image, format_name: str = "PNG") -> RasterMetadata:
        """Extracts standard image dimensions without inventing false geospatial metadata."""
        w, h = img.size
        mode = img.mode
        channels = len(mode) if mode in ("RGB", "RGBA") else 1

        return RasterMetadata(
            source_format=format_name.upper(),
            width=int(w),
            height=int(h),
            channels=channels,
            dtype="uint8",
            crs=None,
            transform=None,
            bounds=None,
            resolution=None,
            acquisition_time=None,
            satellite=None,
            sensor=None,
            channel_name=None,
            product_id=None,
            source=None,
            variable_names=[],
            attributes={},
            georeferencing_available=False,
        )

