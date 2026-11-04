"""Unified scientific raster reader for Cyclone-X (Phase 11).

Supports:
1. GeoTIFF (via rasterio)
2. NetCDF (via xarray / netCDF4)
3. HDF5 (via h5py)
4. PNG / JPEG (via Pillow)

Safe handling:
- Reads from filesystem paths or in-memory byte buffers.
- Normalizes output to SatelliteRaster.
- Does not assume metadata exists; unavailable fields remain None.
"""

import io
from pathlib import Path
import tempfile
from typing import Optional, Union
import numpy as np
from PIL import Image

from backend.app.data.geospatial.metadata import RasterMetadataExtractor
from backend.app.data.geospatial.schemas import RasterMetadata, SatelliteRaster


def read_raster(
    source: Union[str, Path, bytes],
    filename: Optional[str] = None,
) -> SatelliteRaster:
    """Reads a scientific or standard raster dataset and returns a normalized SatelliteRaster."""
    fname = ""
    if filename:
        fname = filename.lower()
    elif isinstance(source, (str, Path)):
        fname = Path(source).name.lower()

    # Detect bytes vs Path
    raw_bytes: Optional[bytes] = None
    file_path: Optional[Path] = None

    if isinstance(source, bytes):
        raw_bytes = source
    else:
        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"Satellite raster file not found: {file_path}")

    # Inspect magic bytes if buffer available
    magic = raw_bytes[:16] if raw_bytes is not None else b""
    if not magic and file_path is not None:
        try:
            with open(file_path, "rb") as f:
                magic = f.read(16)
        except Exception:
            magic = b""

    # Determine format
    is_geotiff = fname.endswith((".tif", ".tiff")) or magic.startswith((b"II*\x00", b"MM\x00*"))
    is_netcdf = fname.endswith((".nc", ".netcdf")) or magic.startswith((b"CDF\x01", b"CDF\x02"))
    is_hdf5 = fname.endswith((".h5", ".hdf5")) or magic.startswith(b"\x89HDF\r\n\x1a\n")
    is_png = fname.endswith(".png") or magic.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = fname.endswith((".jpg", ".jpeg")) or magic.startswith(b"\xff\xd8\xff")

    # -------------------------------------------------------------------------
    # 1. GeoTIFF Handler (rasterio)
    # -------------------------------------------------------------------------
    if is_geotiff:
        import rasterio
        from rasterio.io import MemoryFile

        if raw_bytes is not None:
            with MemoryFile(raw_bytes) as memfile:
                with memfile.open() as src:
                    meta = RasterMetadataExtractor.extract_from_rasterio(src)
                    arr = src.read()
                    if arr.shape[0] == 1:
                        arr = arr[0]
                    return SatelliteRaster(array=arr.astype(np.float32), metadata=meta)
        else:
            assert file_path is not None
            with rasterio.open(file_path) as src:
                meta = RasterMetadataExtractor.extract_from_rasterio(src)
                arr = src.read()
                if arr.shape[0] == 1:
                    arr = arr[0]
                return SatelliteRaster(array=arr.astype(np.float32), metadata=meta)

    # -------------------------------------------------------------------------
    # 2. NetCDF Handler (xarray)
    # -------------------------------------------------------------------------
    if is_netcdf:
        import xarray as xr

        # xarray / netCDF4 backend typically requires a seekable file path
        if raw_bytes is not None:
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp.write(raw_bytes)
                tmp_path = Path(tmp.name)
            try:
                with xr.open_dataset(tmp_path) as ds:
                    meta, lat_arr, lon_arr = RasterMetadataExtractor.extract_from_xarray(ds)
                    # Extract 2D/3D array
                    target_var = meta.variable_names[0] if meta.variable_names else None
                    if target_var:
                        data = ds[target_var].values.astype(np.float32)
                        if data.ndim == 3 and data.shape[0] == 1:
                            data = data[0]
                    else:
                        data = np.zeros((meta.height, meta.width), dtype=np.float32)
                    return SatelliteRaster(array=data, metadata=meta, lat_array=lat_arr, lon_array=lon_arr)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
        else:
            assert file_path is not None
            with xr.open_dataset(file_path) as ds:
                meta, lat_arr, lon_arr = RasterMetadataExtractor.extract_from_xarray(ds)
                target_var = meta.variable_names[0] if meta.variable_names else None
                if target_var:
                    data = ds[target_var].values.astype(np.float32)
                    if data.ndim == 3 and data.shape[0] == 1:
                        data = data[0]
                else:
                    data = np.zeros((meta.height, meta.width), dtype=np.float32)
                return SatelliteRaster(array=data, metadata=meta, lat_array=lat_arr, lon_array=lon_arr)

    # -------------------------------------------------------------------------
    # 3. HDF5 Handler (h5py)
    # -------------------------------------------------------------------------
    if is_hdf5:
        import h5py

        if raw_bytes is not None:
            bio = io.BytesIO(raw_bytes)
            with h5py.File(bio, "r") as h5_file:
                meta, lat_arr, lon_arr, data_arr = RasterMetadataExtractor.extract_from_h5py(h5_file)
                return SatelliteRaster(array=data_arr, metadata=meta, lat_array=lat_arr, lon_array=lon_arr)
        else:
            assert file_path is not None
            with h5py.File(file_path, "r") as h5_file:
                meta, lat_arr, lon_arr, data_arr = RasterMetadataExtractor.extract_from_h5py(h5_file)
                return SatelliteRaster(array=data_arr, metadata=meta, lat_array=lat_arr, lon_array=lon_arr)

    # -------------------------------------------------------------------------
    # 4. Standard Image Formats (PNG, JPEG via Pillow)
    # -------------------------------------------------------------------------
    try:
        if raw_bytes is not None:
            img = Image.open(io.BytesIO(raw_bytes))
        else:
            assert file_path is not None
            img = Image.open(file_path)

        fmt = img.format or ("PNG" if is_png else "JPEG" if is_jpeg else "IMAGE")
        meta = RasterMetadataExtractor.extract_from_image(img, fmt)
        arr = np.array(img, dtype=np.float32)

        return SatelliteRaster(array=arr, metadata=meta)
    except Exception as e:
        raise ValueError(f"Unsupported or corrupted raster file format: {str(e)}")

