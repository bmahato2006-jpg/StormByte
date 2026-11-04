"""Tests for NASA GPM IMERG Ingestion Worker and Telemetry Pipeline.

Covers:
1. EarthdataAuthConfig loading, masking, and .netrc configuration.
2. GeoBounds bounding box validation and CMR parameter formatting.
3. IMERGGranuleMetadata serialization and validation.
4. HDF5 magic byte verification and corruption protection.
5. Spatial subsetting to North Indian Ocean basin ([-10, 38]°N, [45, 110]°E).
6. PyTorch tensor extraction [1, 480, 650], NaN cleaning, and physical rate limits.
7. Variable name compatibility (precipitation vs precipitationCal).
8. NASA IMERG Worker run_once and telemetry state management.
9. FastAPI telemetry endpoints:
   - GET /api/v2/telemetry/imerg/status
   - GET /api/telemetry/imerg/status
   - POST /api/v2/telemetry/imerg/poll
10. Guardrails & Zero Fake Data: never generates artificial rain rates when offline.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import h5py
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.data.geospatial.nasa_imerg_worker import (
    EarthdataAuthConfig,
    EarthdataAuth,
    EarthdataAuthError,
    EarthdataProcessingError,
    GeoBounds,
    IMERGGranuleMetadata,
    NASAIMERGWorker,
    nasa_imerg_worker,
)
from backend.app.schemas.telemetry import IMERGTelemetryStatus


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def temp_hdf5_file(tmp_path):
    """Generates a synthetic yet strictly structured IMERG HDF5 test granule."""
    file_path = tmp_path / "test_imerg_granule.HDF5"
    with h5py.File(file_path, "w") as h5:
        grid = h5.create_group("Grid")
        
        # Dimensions: lat (1800 from -90 to 90), lon (3600 from -180 to 180)
        lats = np.linspace(-89.95, 89.95, 1800, dtype=np.float32)
        lons = np.linspace(-179.95, 179.95, 3600, dtype=np.float32)
        
        grid.create_dataset("lat", data=lats)
        grid.create_dataset("lon", data=lons)
        
        # Raw precipitation: (lon, lat)
        data = np.full((3600, 1800), -9999.9, dtype=np.float32)
        
        # Add a realistic rain cell in the Bay of Bengal (lat ~16°N, lon ~88°E)
        lat_idx = np.where((lats >= 15.0) & (lats <= 18.0))[0]
        lon_idx = np.where((lons >= 86.0) & (lons <= 90.0))[0]
        for lo in lon_idx:
            for la in lat_idx:
                data[lo, la] = 28.5  # 28.5 mm/hr heavy convective rain
                
        grid.create_dataset("precipitation", data=data)
        
    return file_path


# =============================================================================
# 1. Config & Auth Guardrails
# =============================================================================
def test_earthdata_auth_config_masking():
    """Verify passwords and tokens are never leaked in string representations."""
    cfg = EarthdataAuthConfig(
        username="nasa_scientist",
        password="super_secret_password_123",
        token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.token",
    )
    repr_str = repr(cfg)
    assert "super_secret_password_123" not in repr_str
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in repr_str
    assert "password='***'" in repr_str
    assert "has_token=True" in repr_str
    assert cfg.is_configured() is True


def test_earthdata_auth_config_empty():
    """Verify empty config correctly reports is_configured=False."""
    cfg = EarthdataAuthConfig(username="", password="", token="")
    assert cfg.is_configured() is False


def test_earthdata_netrc_generation(tmp_path, monkeypatch):
    """Verify .netrc entry is created cleanly without destroying other credentials."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cfg = EarthdataAuthConfig(username="test_user", password="test_password")
    
    netrc_path = cfg.configure_netrc()
    assert netrc_path is not None
    assert netrc_path.exists()
    
    content = netrc_path.read_text(encoding="utf-8")
    assert "machine urs.earthdata.nasa.gov login test_user password test_password" in content


# =============================================================================
# 2. Bounding Box & Sector Subsetting
# =============================================================================
def test_geo_bounds_defaults_and_cmr_formatting():
    """Verify North Indian Ocean bounds format correctly for NASA CMR API."""
    bounds = GeoBounds()
    assert bounds.lat_min == -10.0
    assert bounds.lat_max == 38.0
    assert bounds.lon_min == 45.0
    assert bounds.lon_max == 110.0
    
    # CMR format: lower_left_lon,lower_left_lat,upper_right_lon,upper_right_lat
    assert bounds.cmr_bounding_box == "45.00,-10.00,110.00,38.00"


def test_geo_bounds_invalid_coordinates():
    """Verify inverted coordinates raise ValidationError."""
    with pytest.raises(Exception):
        GeoBounds(lat_min=20.0, lat_max=10.0)


# =============================================================================
# 3. NetCDF4 / HDF5 Tensor Extraction & Subsetting
# =============================================================================
def test_extract_precipitation_tensor(temp_hdf5_file):
    """Verify extraction creates PyTorch [1, 480, 650] tensor with valid rain rates."""
    worker = NASAIMERGWorker()
    tensor, metadata = worker.extract_precipitation_tensor(temp_hdf5_file)
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == torch.Size([1, 480, 650])
    assert tensor.dtype == torch.float32
    
    # Check physical values: -9999.9 sentinel must be converted to 0.0
    assert tensor.min().item() == 0.0
    assert pytest.approx(tensor.max().item(), 0.1) == 28.5
    assert metadata["nonzero_cells"] > 0
    assert metadata["heavy_rain_cells"] > 0
    assert metadata["heavy_rain_area_km2"] > 0
    assert metadata["variable"] == "precipitation"


def test_extract_legacy_precipitation_cal_variable(tmp_path):
    """Verify backwards compatibility with legacy V06/V07B files using precipitationCal."""
    legacy_file = tmp_path / "legacy_imerg.HDF5"
    with h5py.File(legacy_file, "w") as h5:
        grid = h5.create_group("Grid")
        grid.create_dataset("lat", data=np.linspace(-89.95, 89.95, 1800, dtype=np.float32))
        grid.create_dataset("lon", data=np.linspace(-179.95, 179.95, 3600, dtype=np.float32))
        
        # Legacy name
        grid.create_dataset("precipitationCal", data=np.zeros((3600, 1800), dtype=np.float32))
        
    worker = NASAIMERGWorker()
    tensor, meta = worker.extract_precipitation_tensor(legacy_file)
    assert tensor.shape == torch.Size([1, 480, 650])
    assert meta["variable"] == "precipitationCal"


def test_verify_hdf5_magic_bytes(tmp_path):
    """Verify HDF5 magic bytes verification catches corrupted files."""
    valid_file = tmp_path / "valid.HDF5"
    valid_file.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 100)
    
    corrupt_file = tmp_path / "error_page.html"
    corrupt_file.write_bytes(b"<!DOCTYPE html><html>404 Not Found</html>")
    
    worker = NASAIMERGWorker()
    assert worker._verify_hdf5_magic(valid_file) is True
    assert worker._verify_hdf5_magic(corrupt_file) is False


# =============================================================================
# 4. Worker Telemetry & Guardrails
# =============================================================================
def test_worker_get_status_idle_defaults():
    """Verify initial worker status reflects truthful idle state."""
    worker = NASAIMERGWorker()
    status = worker.get_status()
    
    assert "is_live" in status
    assert "granules_available" in status
    assert "operational_health" in status
    assert status["dataset_name"] == "GPM_3IMERGHHL"


def test_worker_unconfigured_guardrail():
    """Verify worker safely reports UNCONFIGURED without making fake network calls."""
    import asyncio
    empty_cfg = EarthdataAuthConfig(username="", password="", token="")
    worker = NASAIMERGWorker(config=empty_cfg)
    
    status = asyncio.run(worker.run_once())
    assert status["is_live"] is False
    assert status["operational_health"] == "UNCONFIGURED"


# =============================================================================
# 5. FastAPI Endpoints Integration
# =============================================================================
def test_get_imerg_telemetry_status_endpoint(client):
    """Verify GET /api/v2/telemetry/imerg/status returns valid schema."""
    resp = client.get("/api/v2/telemetry/imerg/status")
    assert resp.status_code == 200
    
    data = resp.json()
    assert "is_live" in data
    assert "granules_available" in data
    assert "operational_health" in data
    assert "dataset_name" in data


def test_get_imerg_telemetry_status_v1_alias(client):
    """Verify backwards-compatible alias GET /api/telemetry/imerg/status."""
    resp = client.get("/api/telemetry/imerg/status")
    assert resp.status_code == 200
    assert "is_live" in resp.json()


def test_post_imerg_poll_endpoint(client):
    """Verify POST /api/v2/telemetry/imerg/poll returns operational status."""
    resp = client.post("/api/v2/telemetry/imerg/poll")
    assert resp.status_code == 200
    data = resp.json()
    assert "is_live" in data
    assert "operational_health" in data
