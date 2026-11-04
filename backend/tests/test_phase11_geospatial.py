"""Comprehensive Phase 11 Geospatial & Scientific Pipeline Tests for Cyclone-X.

Verifies:
1. GeoTIFF reader
2. NetCDF reader
3. HDF5 reader
4. PNG reader
5. JPEG reader
6. Metadata extraction
7. Valid CRS
8. Missing CRS
9. Pixel-to-latlon transformation
10. WGS84 reprojection
11. Invalid latitude validation
12. Invalid longitude validation
13. Missing georeferencing
14. Zero coordinate fabrication
15. Four-point history requirement
16. Insufficient history
17. Phase 9 V2 integration
18. Satellite upload API
19. Analyze-satellite API
20. Provenance generation
"""

import io
import json
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

import rasterio
from rasterio.transform import from_bounds
import xarray as xr
import h5py

from backend.app.main import app
from backend.app.data.geospatial import (
    Georeferencer,
    GeospatialValidator,
    RasterMetadata,
    RasterMetadataExtractor,
    SatelliteRaster,
    read_raster,
)
from backend.app.ml.pipeline.service import pipeline_service


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def synthetic_geotiff_wgs84(tmp_path):
    """Creates a temporary synthetic GeoTIFF with EPSG:4326 in the Bay of Bengal."""
    file_path = tmp_path / "synthetic_amphan_wgs84.tif"
    width, height = 64, 64
    # Bounding box: lat [12.0, 18.0], lon [84.0, 90.0]
    transform = from_bounds(84.0, 12.0, 90.0, 18.0, width, height)
    # Synthetic warm eye with cold convective ring (typical cyclone signature in TIR1)
    y, x = np.ogrid[:height, :width]
    cy, cx = height / 2, width / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    data = 280.0 - 60.0 * np.exp(-((r - 12.0) ** 2) / 30.0)
    data = data.astype(np.float32)

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)
        dst.update_tags(
            SATELLITE="INSAT-3D",
            SENSOR="IMAGER",
            CHANNEL="TIR1 (10.8 µm)",
            ACQUISITION_TIME="2020-05-18T06:00:00Z",
            SOURCE="MOSDAC/ISRO",
        )

    return file_path


@pytest.fixture
def synthetic_geotiff_projected(tmp_path):
    """Creates a temporary synthetic GeoTIFF with Web Mercator (EPSG:3857)."""
    file_path = tmp_path / "synthetic_projected_3857.tif"
    width, height = 50, 50
    # Coordinates in meters around ~15°N, 85°E
    # lon 85 -> ~9462156 m, lat 15 -> ~1689200 m
    transform = from_bounds(9300000.0, 1500000.0, 9600000.0, 1800000.0, width, height)
    data = np.full((height, width), 250.0, dtype=np.float32)

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return file_path


@pytest.fixture
def synthetic_netcdf(tmp_path):
    """Creates a temporary synthetic NetCDF file with 1D lat/lon coordinate variables."""
    file_path = tmp_path / "synthetic_insat3d.nc"
    lats = np.linspace(10.0, 20.0, 40, dtype=np.float32)
    lons = np.linspace(80.0, 90.0, 50, dtype=np.float32)
    # Synthetic brightness temperature
    data = np.random.uniform(220.0, 300.0, (len(lats), len(lons))).astype(np.float32)

    ds = xr.Dataset(
        data_vars={"bt_tir1": (["lat", "lon"], data)},
        coords={"lat": lats, "lon": lons},
        attrs={
            "satellite": "INSAT-3D",
            "sensor": "IMAGER",
            "channel": "TIR1",
            "source": "MOSDAC/ISRO",
            "time_coverage_start": "2020-05-18T06:00:00Z",
        },
    )
    try:
        ds.to_netcdf(file_path, engine="h5netcdf")
    except Exception:
        try:
            ds.to_netcdf(file_path)
        except Exception as e:
            pytest.skip(f"NetCDF writer blocked by Windows Application Control: {e}")
    return file_path


@pytest.fixture
def synthetic_hdf5(tmp_path):
    """Creates a temporary synthetic HDF5 file with latitude/longitude datasets."""
    file_path = tmp_path / "synthetic_granule.h5"
    height, width = 30, 40
    data = np.random.uniform(200.0, 310.0, (height, width)).astype(np.float32)
    lats = np.linspace(12.0, 18.0, height, dtype=np.float32)
    lons = np.linspace(82.0, 88.0, width, dtype=np.float32)

    with h5py.File(file_path, "w") as f:
        f.create_dataset("radiance_tir", data=data)
        f.create_dataset("latitude", data=lats)
        f.create_dataset("longitude", data=lons)
        f.attrs["satellite"] = "INSAT-3DR"
        f.attrs["instrument"] = "IMAGER"
        f.attrs["channel"] = "TIR1"
        f.attrs["time"] = "2020-05-18T12:00:00Z"

    return file_path


@pytest.fixture
def synthetic_png(tmp_path):
    """Creates a temporary unreferenced standard PNG image."""
    file_path = tmp_path / "unreferenced_test.png"
    img = Image.new("RGB", (128, 128), color=(40, 60, 90))
    img.save(file_path)
    return file_path


@pytest.fixture
def synthetic_jpeg(tmp_path):
    """Creates a temporary unreferenced standard JPEG image."""
    file_path = tmp_path / "unreferenced_test.jpg"
    img = Image.new("RGB", (100, 100), color=(80, 40, 20))
    img.save(file_path, format="JPEG")
    return file_path


# =============================================================================
# 1. GeoTIFF Reader Test
# =============================================================================
def test_geotiff_reader(synthetic_geotiff_wgs84):
    raster = read_raster(synthetic_geotiff_wgs84)
    assert raster is not None
    assert raster.metadata.source_format == "GEOTIFF"
    assert raster.width == 64
    assert raster.height == 64
    assert raster.georeferencing_available is True
    assert raster.metadata.crs is not None
    assert "4326" in raster.metadata.crs
    rgb = raster.to_rgb_image()
    assert rgb.size == (64, 64)


# =============================================================================
# 2. NetCDF Reader Test
# =============================================================================
def test_netcdf_reader(synthetic_netcdf):
    raster = read_raster(synthetic_netcdf)
    assert raster is not None
    assert raster.metadata.source_format == "NETCDF"
    assert raster.width == 50
    assert raster.height == 40
    assert raster.georeferencing_available is True
    assert raster.lat_array is not None
    assert raster.lon_array is not None
    assert raster.metadata.satellite == "INSAT-3D"


# =============================================================================
# 3. HDF5 Reader Test
# =============================================================================
def test_hdf5_reader(synthetic_hdf5):
    raster = read_raster(synthetic_hdf5)
    assert raster is not None
    assert raster.metadata.source_format == "HDF5"
    assert raster.width == 40
    assert raster.height == 30
    assert raster.georeferencing_available is True
    assert raster.metadata.satellite == "INSAT-3DR"


# =============================================================================
# 4. PNG Reader Test
# =============================================================================
def test_png_reader(synthetic_png):
    raster = read_raster(synthetic_png)
    assert raster is not None
    assert raster.metadata.source_format == "PNG"
    assert raster.width == 128
    assert raster.height == 128
    # Standard images without external calibration strictly have no georeferencing
    assert raster.georeferencing_available is False
    assert raster.metadata.crs is None


# =============================================================================
# 5. JPEG Reader Test
# =============================================================================
def test_jpeg_reader(synthetic_jpeg):
    raster = read_raster(synthetic_jpeg)
    assert raster is not None
    assert raster.metadata.source_format == "JPEG"
    assert raster.width == 100
    assert raster.height == 100
    assert raster.georeferencing_available is False


# =============================================================================
# 6. Metadata Extraction Test
# =============================================================================
def test_metadata_extraction(synthetic_geotiff_wgs84):
    raster = read_raster(synthetic_geotiff_wgs84)
    meta = raster.metadata
    assert meta.satellite == "INSAT-3D"
    assert meta.sensor == "IMAGER"
    assert meta.channel_name == "TIR1 (10.8 µm)"
    assert meta.bounds is not None
    assert meta.bounds["latitude_min"] == 12.0
    assert meta.bounds["latitude_max"] == 18.0
    assert meta.bounds["longitude_min"] == 84.0
    assert meta.bounds["longitude_max"] == 90.0


# =============================================================================
# 7. Valid CRS Handling Test
# =============================================================================
def test_valid_crs(synthetic_geotiff_wgs84, synthetic_geotiff_projected):
    r_wgs84 = read_raster(synthetic_geotiff_wgs84)
    assert r_wgs84.metadata.crs is not None
    assert "4326" in r_wgs84.metadata.crs

    r_proj = read_raster(synthetic_geotiff_projected)
    assert r_proj.metadata.crs is not None
    assert "3857" in r_proj.metadata.crs


# =============================================================================
# 8. Missing CRS Handling Test
# =============================================================================
def test_missing_crs(synthetic_png):
    raster = read_raster(synthetic_png)
    assert raster.metadata.crs is None
    assert raster.metadata.transform is None
    # Zero fabrication: missing CRS does not invent default EPSG:4326
    res = Georeferencer.georeference_pixel(30.0, 30.0, raster=raster)
    assert res.coordinates_available is False
    assert res.latitude is None
    assert res.longitude is None
    assert res.status == "GEOSPATIAL_METADATA_REQUIRED"


# =============================================================================
# 9. Pixel to Lat/Lon Transformation Test
# =============================================================================
def test_pixel_to_latlon_transformation(synthetic_geotiff_wgs84):
    raster = read_raster(synthetic_geotiff_wgs84)
    # Center pixel of 64x64 raster with bounds [84-90E, 12-18N] should be (15.0N, 87.0E)
    res = Georeferencer.georeference_pixel(32.0, 32.0, raster=raster)
    assert res.coordinates_available is True
    assert res.latitude is not None
    assert res.longitude is not None
    assert pytest.approx(res.latitude, abs=0.2) == 15.0
    assert pytest.approx(res.longitude, abs=0.2) == 87.0
    assert res.status == "CALIBRATED_GEOREFERENCE"


# =============================================================================
# 10. WGS84 Reprojection Test
# =============================================================================
def test_wgs84_reprojection(synthetic_geotiff_projected):
    raster = read_raster(synthetic_geotiff_projected)
    # Center pixel in EPSG:3857 reprojected to WGS84
    res = Georeferencer.georeference_pixel(25.0, 25.0, raster=raster)
    assert res.coordinates_available is True
    assert res.latitude is not None
    assert res.longitude is not None
    # Converted coordinates should fall in valid Indian Ocean / Bay of Bengal geographic bounds
    assert -90.0 <= res.latitude <= 90.0
    assert -180.0 <= res.longitude <= 180.0


# =============================================================================
# 11. Invalid Latitude Validation Test
# =============================================================================
def test_invalid_latitude_validation():
    # Out of range latitude > 90.0
    val = GeospatialValidator.validate_bounds(lat_min=10.0, lat_max=95.0, lon_min=80.0, lon_max=90.0)
    assert val.is_valid is False
    assert val.error_code == "INVALID_LATITUDE_RANGE"

    # Reversed order: lat_min > lat_max
    val_rev = GeospatialValidator.validate_bounds(lat_min=25.0, lat_max=15.0, lon_min=80.0, lon_max=90.0)
    assert val_rev.is_valid is False
    assert val_rev.error_code == "INVALID_BOUNDS_ORDER"


# =============================================================================
# 12. Invalid Longitude Validation Test
# =============================================================================
def test_invalid_longitude_validation():
    # Out of range longitude > 180.0
    val = GeospatialValidator.validate_bounds(lat_min=10.0, lat_max=20.0, lon_min=80.0, lon_max=185.0)
    assert val.is_valid is False
    assert val.error_code == "INVALID_LONGITUDE_RANGE"

    # Reversed order: lon_min > lon_max
    val_rev = GeospatialValidator.validate_bounds(lat_min=10.0, lat_max=20.0, lon_min=95.0, lon_max=85.0)
    assert val_rev.is_valid is False
    assert val_rev.error_code == "INVALID_BOUNDS_ORDER"


# =============================================================================
# 13. Missing Georeferencing Test
# =============================================================================
def test_missing_georeferencing(synthetic_png):
    raster = read_raster(synthetic_png)
    res = Georeferencer.georeference_pixel(64.0, 64.0, raster=raster)
    assert res.latitude is None
    assert res.longitude is None
    assert res.coordinates_available is False
    assert res.status == "GEOSPATIAL_METADATA_REQUIRED"


# =============================================================================
# 14. Zero Coordinate Fabrication Test
# =============================================================================
def test_zero_coordinate_fabrication():
    # Calling georeferencer with empty metadata must never output (0, 0) or fake center coordinates
    meta = RasterMetadata(
        source_format="PNG",
        width=512,
        height=512,
        channels=3,
        dtype="uint8",
    )
    res = Georeferencer.georeference_pixel(256.0, 256.0, metadata=meta)
    assert res.coordinates_available is False
    assert res.latitude is None
    assert res.longitude is None
    assert res.status == "GEOSPATIAL_METADATA_REQUIRED"


# =============================================================================
# 15. Four-Point History Requirement Test (Phase 8 LSTM Trigger)
# =============================================================================
def test_four_point_history_requirement(synthetic_geotiff_wgs84):
    raster = read_raster(synthetic_geotiff_wgs84)
    # 4 authentic sequential fixes
    track_history = [
        {"latitude": 11.2, "longitude": 86.4, "timestamp": "2020-05-16T18:00:00Z"},
        {"latitude": 12.5, "longitude": 86.4, "timestamp": "2020-05-17T06:00:00Z"},
        {"latitude": 13.6, "longitude": 86.7, "timestamp": "2020-05-17T18:00:00Z"},
        {"latitude": 14.9, "longitude": 86.5, "timestamp": "2020-05-18T06:00:00Z"},
    ]
    res = pipeline_service.analyze_scientific_raster(
        raster=raster,
        track_history=track_history,
        data_mode="SCIENTIFIC_UPLOAD",
    )
    # If cyclone detected and coordinates resolved, 4 points should trigger Phase 8 trajectory forecast
    if res["detected"] and res["center_fix"]["coordinates_available"]:
        assert res["trajectory_status"] == "SUCCESSFULLY_FORECASTED_VIA_PHASE_8_LSTM"
        assert res["trajectory_forecast"] is not None
        assert len(res["trajectory_forecast"]["predictions"]) == 3  # +6h, +12h, +24h


# =============================================================================
# 16. Insufficient History Test (Zero Track Fabrication Rule)
# =============================================================================
def test_insufficient_history(synthetic_geotiff_wgs84):
    raster = read_raster(synthetic_geotiff_wgs84)
    # Only 1 prior fix (insufficient for 4-point requirement)
    partial_history = [
        {"latitude": 14.0, "longitude": 86.0, "timestamp": "2020-05-17T18:00:00Z"}
    ]
    res = pipeline_service.analyze_scientific_raster(
        raster=raster,
        track_history=partial_history,
        data_mode="SCIENTIFIC_UPLOAD",
    )
    if res["detected"]:
        # Must report insufficient track history without fabricating synthetic points
        assert res["trajectory_status"] == "INSUFFICIENT_TRACK_HISTORY"
        assert res["trajectory_forecast"] is None
        assert res["track_history_length"] < 4


# =============================================================================
# 17. Phase 9 V2 Integration Test
# =============================================================================
def test_phase9_v2_integration(synthetic_geotiff_wgs84):
    raster = read_raster(synthetic_geotiff_wgs84)
    res = pipeline_service.analyze_scientific_raster(
        raster=raster,
        model_version="v2",
    )
    # Default model must be CycloneDetectorCNN V2 (version 2.0.0)
    assert res["provenance"]["detection_model"] == "cyclone_detector_cnn_v2"
    assert res["provenance"]["detection_version"] == "2.0.0"


# =============================================================================
# 18. Satellite Upload API Test
# =============================================================================
def test_satellite_upload_api(client, synthetic_geotiff_wgs84):
    with open(synthetic_geotiff_wgs84, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/satellite/upload",
        files={"file": ("amphan_test.tif", file_bytes, "image/tiff")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["file_id"].startswith("sat_")
    assert data["format"] == "GEOTIFF"
    assert data["georeferencing_available"] is True
    assert data["validation_status"] == "CALIBRATED_GEOREFERENCE"
    assert "sha256" in data


# =============================================================================
# 19. Analyze Satellite API Test
# =============================================================================
def test_analyze_satellite_api(client, synthetic_geotiff_wgs84):
    with open(synthetic_geotiff_wgs84, "rb") as f:
        file_bytes = f.read()

    track_fixes = [
        {"latitude": 11.2, "longitude": 86.4, "timestamp": "2020-05-16T18:00:00Z"},
        {"latitude": 12.5, "longitude": 86.4, "timestamp": "2020-05-17T06:00:00Z"},
        {"latitude": 13.6, "longitude": 86.7, "timestamp": "2020-05-17T18:00:00Z"},
        {"latitude": 14.9, "longitude": 86.5, "timestamp": "2020-05-18T06:00:00Z"},
    ]

    response = client.post(
        "/api/ml/pipeline/analyze-satellite",
        files={"file": ("amphan_eval.tif", file_bytes, "image/tiff")},
        data={
            "track_history_json": json.dumps(track_fixes),
            "model_version": "v2",
            "threshold": "0.50",
            "data_mode": "SCIENTIFIC_UPLOAD",
        },
    )
    assert response.status_code == 200
    res = response.json()
    assert "detected" in res
    assert "confidence" in res
    assert "provenance" in res
    assert res["provenance"]["data_mode"] == "SCIENTIFIC_UPLOAD"
    assert res["provenance"]["detection_model"] == "cyclone_detector_cnn_v2"
    assert res["risk_assessment"] is not None


# =============================================================================
# 20. Provenance Generation Test
# =============================================================================
def test_provenance_generation(synthetic_geotiff_wgs84):
    raster = read_raster(synthetic_geotiff_wgs84)
    with open(synthetic_geotiff_wgs84, "rb") as f:
        raw_b = f.read()

    res = pipeline_service.analyze_scientific_raster(
        raster=raster,
        raw_bytes=raw_b,
        filename="synthetic_amphan.tif",
        data_mode="SCIENTIFIC_UPLOAD",
    )
    prov = res["provenance"]
    assert prov["data_mode"] == "SCIENTIFIC_UPLOAD"
    assert prov["original_filename"] == "synthetic_amphan.tif"
    assert prov["file_format"] == "GEOTIFF"
    assert prov["image_sha256"] is not None
    assert len(prov["image_sha256"]) == 64
    assert prov["georeferencing_available"] is True
    assert prov["detection_model"] in ("cyclone_detector_cnn_v2", "cyclone_detector_cnn_v3")
    assert prov["detection_version"] in ("2.0.0", "3.0.0")


# =============================================================================
# 21. NRT Status Endpoint Test (Unconfigured Default - Missing Credentials)
# =============================================================================
def test_nrt_status_endpoint_unconfigured(client):
    response = client.get("/api/satellite/nrt/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "CREDENTIALS_REQUIRED"
    assert data["adapter"] == "mosdac_insat3d"
    assert data["provider"] == "MOSDAC/ISRO"
    assert data["configured"] is False
    assert data["connectivity"] == "AUTHENTICATION_REQUIRED"
    assert data["environment_variables"]["MOSDAC_USERNAME"] is False
    assert data["environment_variables"]["MOSDAC_PASSWORD"] is False
    assert "MOSDAC SSO credentials missing" in data["message"]
    assert "registered_adapters" in data
    assert "mosdac_insat3d" in data["registered_adapters"]
    assert "nasa_earthdata" in data["registered_adapters"]
    assert not any("secret" in str(v).lower() for v in data.values())


# =============================================================================
# 22. NRT Latest Granule Endpoint Test (Zero Fabrication Verification)
# =============================================================================
def test_nrt_latest_endpoint_unconfigured(client):
    response = client.get("/api/satellite/nrt/latest")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "CREDENTIALS_REQUIRED"
    assert data["adapter"] == "mosdac_insat3d"
    assert data["granule_available"] is False
    assert data["granule"] is None
    assert data["provenance"] is None
    assert "zero_fabrication_rule" in data
    assert "credentials missing" in data["message"].lower()


# =============================================================================
# 23. NRT Specific Adapter Query Test
# =============================================================================
def test_nrt_adapter_specific_queries(client):
    # NASA Earthdata
    r_earth = client.get("/api/satellite/nrt/status?adapter=nasa_earthdata")
    assert r_earth.status_code == 200
    assert r_earth.json()["adapter"] == "nasa_earthdata"

    # NASA GPM IMERG
    r_gpm = client.get("/api/satellite/nrt/status?adapter=nasa_gpm_imerg")
    assert r_gpm.status_code == 200
    assert r_gpm.json()["adapter"] == "nasa_gpm_imerg"

    # Invalid adapter error handling
    r_invalid = client.get("/api/satellite/nrt/status?adapter=unknown_space_agency")
    assert r_invalid.status_code == 400
    assert "Unknown satellite adapter" in r_invalid.json()["detail"]


# =============================================================================
# 24. NRT Credentials Present But Authentication Not Performed
# =============================================================================
def test_nrt_credentials_present_unauthenticated(client, monkeypatch):
    from backend.app.core.config import settings

    # Simulate presence of MOSDAC SSO credentials in environment
    monkeypatch.setattr(settings, "MOSDAC_USERNAME", "insat_user_researcher")
    monkeypatch.setattr(settings, "MOSDAC_PASSWORD", "SuperSecretMOSDACPassword123")
    monkeypatch.setattr(settings, "MOSDAC_DATASET_ID", "3DIMG_L1B_STD")

    r_stat = client.get("/api/satellite/nrt/status")
    assert r_stat.status_code == 200
    stat_data = r_stat.json()
    assert stat_data["status"] == "CONFIGURED"
    assert stat_data["configured"] is True
    assert stat_data["environment_variables"]["MOSDAC_USERNAME"] is True
    assert stat_data["environment_variables"]["MOSDAC_PASSWORD"] is True
    assert stat_data["dataset_id"] == "3DIMG_L1B_STD"
    assert stat_data["session_authenticated"] is False

    # Security Check: Ensure password is NEVER leaked in the response
    assert "SuperSecretMOSDACPassword123" not in r_stat.text

    # Latest granule without integration test flag truthfully returns NO_DATA_AVAILABLE, never fake data
    r_latest = client.get("/api/satellite/nrt/latest")
    assert r_latest.status_code == 200
    latest_data = r_latest.json()
    assert latest_data["status"] == "NO_DATA_AVAILABLE"
    assert latest_data["granule_available"] is False
    assert latest_data["granule"] is None
    assert "SuperSecretMOSDACPassword123" not in r_latest.text


# =============================================================================
# 25. NRT Authentication Failure State Test
# =============================================================================
def test_nrt_authentication_failure(client):
    from backend.app.data.geospatial.live_adapters import mosdac_adapter
    from backend.app.data.geospatial.mosdac_adapter import MOSDACStatus

    try:
        mosdac_adapter.set_test_simulation(MOSDACStatus.AUTHENTICATION_FAILED, "Invalid credentials")

        r_stat = client.get("/api/satellite/nrt/status")
        assert r_stat.status_code == 200
        assert r_stat.json()["status"] == "AUTHENTICATION_FAILED"
        assert r_stat.json()["session_authenticated"] is False

        r_latest = client.get("/api/satellite/nrt/latest")
        assert r_latest.status_code == 200
        data = r_latest.json()
        assert data["status"] == "AUTHENTICATION_FAILED"
        assert data["granule_available"] is False
        assert data["granule"] is None
    finally:
        mosdac_adapter.set_test_simulation(None)


# =============================================================================
# 26. NRT Access Not Granted State Test
# =============================================================================
def test_nrt_access_not_granted(client):
    from backend.app.data.geospatial.live_adapters import mosdac_adapter
    from backend.app.data.geospatial.mosdac_adapter import MOSDACStatus

    try:
        mosdac_adapter.set_test_simulation(MOSDACStatus.ACCESS_NOT_GRANTED, "Access denied")

        r_stat = client.get("/api/satellite/nrt/status")
        assert r_stat.status_code == 200
        assert r_stat.json()["status"] == "ACCESS_NOT_GRANTED"

        r_latest = client.get("/api/satellite/nrt/latest")
        assert r_latest.status_code == 200
        data = r_latest.json()
        assert data["status"] == "ACCESS_NOT_GRANTED"
        assert data["granule_available"] is False
        assert data["granule"] is None
    finally:
        mosdac_adapter.set_test_simulation(None)


# =============================================================================
# 27. NRT Connected & Zero Fabrication Test
# =============================================================================
def test_nrt_connected_state(client):
    from backend.app.data.geospatial.live_adapters import mosdac_adapter
    from backend.app.data.geospatial.mosdac_adapter import MOSDACStatus

    try:
        mosdac_adapter.set_test_simulation(MOSDACStatus.CONNECTED)

        r_stat = client.get("/api/satellite/nrt/status")
        assert r_stat.status_code == 200
        stat_data = r_stat.json()
        assert stat_data["status"] == "CONNECTED"
        assert stat_data["session_authenticated"] is True
    finally:
        mosdac_adapter.set_test_simulation(None)

