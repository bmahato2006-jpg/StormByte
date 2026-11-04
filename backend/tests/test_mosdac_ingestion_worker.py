"""Comprehensive unit and integration test suite for the ISRO/MOSDAC Ingestion Worker."""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import pytest
import numpy as np
import h5py
import rasterio
from rasterio.transform import from_origin
import torch
import httpx

from backend.app.data.geospatial.mosdac_ingestion_worker import (
    MOSDACAuthConfig,
    MOSDACLockoutManager,
    MOSDACClient,
    MOSDACIngestionWorker,
    GranuleMetadata,
    GeoBounds,
    fetch_live_insat_tiles,
    download_satellite_granule,
    extract_and_normalize_tir1,
    MOSDACAuthError,
    MOSDACAccountLockedError,
    MOSDACCorruptFileError,
    MOSDACDownloadError,
)


# =============================================================================
# 1. Authentication & Configuration Tests
# =============================================================================

def test_mosdac_auth_config_and_masking(monkeypatch):
    """Verify MOSDACAuthConfig securely reads environment and masks credentials."""
    monkeypatch.setenv("MOSDAC_USERNAME", "isro_analyst")
    monkeypatch.setenv("MOSDAC_PASSWORD", "super_secret_password_123")
    monkeypatch.setenv("MOSDAC_DATASET_ID", "3DIMG_L1B_STD")

    config = MOSDACAuthConfig.from_environment()
    assert config.username == "isro_analyst"
    assert config.password == "super_secret_password_123"
    assert config.is_configured() is True

    # Critical Security Test: Passwords must NEVER be leaked in repr or str
    repr_str = repr(config)
    assert "super_secret_password_123" not in repr_str
    assert "***" in repr_str
    assert "is***" in repr_str or "***" in repr_str


def test_mosdac_auth_config_unconfigured(monkeypatch):
    """Verify is_configured() is False when credentials are missing."""
    monkeypatch.delenv("MOSDAC_USERNAME", raising=False)
    monkeypatch.delenv("MOSDAC_PASSWORD", raising=False)

    config = MOSDACAuthConfig(username="", password="", api_key="")
    assert config.is_configured() is False


@pytest.mark.anyio
async def test_mosdac_auth_config_with_api_key(monkeypatch):
    """Verify MOSDACAuthConfig supports direct MOSDAC_API_KEY access token."""
    monkeypatch.delenv("MOSDAC_USERNAME", raising=False)
    monkeypatch.delenv("MOSDAC_PASSWORD", raising=False)
    monkeypatch.setenv("MOSDAC_API_KEY", "mosdac_secret_access_key_999")

    config = MOSDACAuthConfig.from_environment()
    assert config.is_configured() is True
    assert config.api_key == "mosdac_secret_access_key_999"

    # Must not leak key in repr
    assert "mosdac_secret_access_key_999" not in repr(config)

    # Direct authentication without HTTP handshake
    client = MOSDACClient(config)
    token = await client.authenticate()
    assert token == "mosdac_secret_access_key_999"


# =============================================================================
# 2. Lockout Guardrail Tests (ISRO 3-Failure Rule)
# =============================================================================

@pytest.mark.anyio
async def test_mosdac_lockout_manager_guardrail():
    """Verify that >= 3 failed logins triggers a 1-hour account lockout."""
    manager = MOSDACLockoutManager(max_failures=3, lockout_seconds=3600)

    # 1st failure: allowed
    await manager.record_failure("401 Unauthorized attempt 1")
    assert manager.consecutive_failures == 1
    assert manager.locked_until is None
    await manager.check_lockout()  # Should not raise

    # 2nd failure: allowed
    await manager.record_failure("401 Unauthorized attempt 2")
    assert manager.consecutive_failures == 2
    assert manager.locked_until is None
    await manager.check_lockout()  # Should not raise

    # 3rd failure: TRIGGER 1-HOUR LOCKOUT
    await manager.record_failure("401 Unauthorized attempt 3")
    assert manager.consecutive_failures == 3
    assert manager.locked_until is not None

    # Subsequent check must raise MOSDACAccountLockedError
    with pytest.raises(MOSDACAccountLockedError) as exc_info:
        await manager.check_lockout()
    assert "account locked due to 3 consecutive authentication failures" in str(exc_info.value).lower()

    # Successful login resets the counter
    await manager.record_success()
    assert manager.consecutive_failures == 0
    assert manager.locked_until is None
    await manager.check_lockout()  # Should not raise


# =============================================================================
# 3. Query Parameter Construction & Live Search Tests
# =============================================================================

def test_geo_bounds_validation():
    """Verify GeoBounds enforces valid North Indian Ocean coordinates."""
    bounds = GeoBounds(lat_min=-10.0, lat_max=38.0, lon_min=45.0, lon_max=110.0)
    assert bounds.bbox_string == "45.00,-10.00,110.00,38.00"

    with pytest.raises(ValueError):
        # lat_max < lat_min
        GeoBounds(lat_min=20.0, lat_max=10.0, lon_min=50.0, lon_max=80.0)


@pytest.mark.anyio
async def test_fetch_live_insat_tiles_mock_response():
    """Verify query payload parameters and parsing with mock HTTPX transport."""
    granule_data = {
        "granules": [
            {
                "granuleId": "3DIMG_16SEP2026_0300_L1B_STD.h5",
                "datasetId": "3DIMG_L1B_STD",
                "acquisitionTime": "2026-09-16T03:00:00Z",
                "downloadUrl": "https://mosdac.gov.in/downloadapi/file/3DIMG_16SEP2026_0300_L1B_STD.h5",
                "fileSize": 45000000,
                "md5": "d41d8cd98f00b204e9800998ecf8427e",
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["datasetId"] == "3DIMG_L1B_STD"
        assert "TIR1" in body["channels"]
        assert "startTime" in body and "endTime" in body
        return httpx.Response(200, json=granule_data)

    transport = httpx.MockTransport(handler)
    mock_http = httpx.AsyncClient(transport=transport)

    config = MOSDACAuthConfig(username="test", password="pwd", api_url="https://mosdac.gov.in")
    client = MOSDACClient(config=config)
    client._http_client = mock_http

    tiles = await fetch_live_insat_tiles(
        lat_min=-10.0,
        lat_max=38.0,
        lon_min=45.0,
        lon_max=110.0,
        dataset_id="3DIMG_L1B_STD",
        time_window_minutes=60,
        client=client,
    )

    assert len(tiles) == 1
    g = tiles[0]
    assert g.granule_id == "3DIMG_16SEP2026_0300_L1B_STD.h5"
    assert g.satellite == "INSAT-3D"
    assert g.channel == "TIR1"
    assert g.file_size_bytes == 45000000


@pytest.mark.anyio
async def test_fetch_live_insat_tiles_no_data_zero_fabrication():
    """Verify zero fabrication: returns empty list when remote server has no granules."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"granules": []})

    transport = httpx.MockTransport(handler)
    mock_http = httpx.AsyncClient(transport=transport)

    client = MOSDACClient(MOSDACAuthConfig(username="u", password="p"))
    client._http_client = mock_http

    tiles = await fetch_live_insat_tiles(client=client)
    assert tiles == []


# =============================================================================
# 4. HDF5 Extraction & Normalization Tests
# =============================================================================

def test_extract_and_normalize_tir1_hdf5(tmp_path):
    """Verify HDF5 reading and PyTorch normalization to [1, H, W] in [0, 1]."""
    h5_file = tmp_path / "3DIMG_TEST_L1B_STD.h5"

    # Create synthetic calibrated HDF5 file
    H, W = 100, 100
    with h5py.File(str(h5_file), "w") as f:
        # Create TIR1 dataset with realistic Kelvin temperatures (190K cloud top to 300K warm ocean)
        data = np.linspace(190.0, 310.0, H * W, dtype=np.float32).reshape(H, W)
        # Add a cold cyclone convective core in the center (185 K)
        data[40:60, 40:60] = 185.0
        # Add space background / fill value (-999)
        data[0:5, 0:5] = -999.0

        ds = f.create_dataset("IMG_TIR1", data=data)
        ds.attrs["slope"] = 1.0
        ds.attrs["intercept"] = 0.0
        ds.attrs["FillValue"] = -999.0

    tensor = extract_and_normalize_tir1(h5_file, target_shape=(512, 512))

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == torch.Size([1, 512, 512])
    assert tensor.dtype == torch.float32

    # Bounds assertion
    assert (tensor >= 0.0).all()
    assert (tensor <= 1.0).all()
    assert not torch.isnan(tensor).any()

    # Meteorological Inversion Assertion:
    # Cold cyclone core (185K) must map to high intensity (~1.0)
    # Warm ocean (310K) must map to low intensity (~0.0)
    center_val = float(tensor[0, 256, 256].item())
    corner_val = float(tensor[0, 500, 500].item())
    assert center_val > corner_val, f"Convective core ({center_val:.2f}) should be brighter than warm sea ({corner_val:.2f})"


def test_extract_and_normalize_tir1_geotiff(tmp_path):
    """Verify GeoTIFF reading and PyTorch normalization."""
    tif_file = tmp_path / "insat_tir1.tif"

    H, W = 64, 64
    data = np.random.uniform(200.0, 300.0, (H, W)).astype(np.float32)
    transform = from_origin(80.0, 20.0, 0.04, 0.04)

    with rasterio.open(
        str(tif_file),
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype=data.dtype,
        crs="+proj=latlong",
        transform=transform,
        nodata=-999.0,
    ) as dst:
        dst.write(data, 1)

    tensor = extract_and_normalize_tir1(tif_file, target_shape=(256, 256))
    assert tensor.shape == torch.Size([1, 256, 256])
    assert (tensor >= 0.0).all() and (tensor <= 1.0).all()


def test_extract_and_normalize_corrupted_file(tmp_path):
    """Verify corrupt files raise MOSDACCorruptFileError."""
    bad_file = tmp_path / "corrupted.bin"
    bad_file.write_bytes(b"NOT_A_VALID_HDF5_OR_GEOTIFF")

    with pytest.raises(MOSDACCorruptFileError):
        extract_and_normalize_tir1(bad_file)


# =============================================================================
# 5. Worker End-to-End Run Once
# =============================================================================

@pytest.mark.anyio
async def test_worker_run_once_with_mocked_pipeline(tmp_path):
    """Verify MOSDACIngestionWorker executes run_once cycle end-to-end."""
    # 1. Create temporary mock HDF5 granule
    granule_path = tmp_path / "3DIMG_2026_TEST.h5"
    with h5py.File(str(granule_path), "w") as f:
        data = np.full((50, 50), 250.0, dtype=np.float32)
        f.create_dataset("TIR1", data=data)

    granule_bytes = granule_path.read_bytes()

    # 2. Mock MOSDAC HTTP server
    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "/internal/uops" in url_str:
            return httpx.Response(200, json={"token": "MOCK_MOSDAC_TOKEN_123"})
        elif "/downloadapi/search" in url_str:
            return httpx.Response(200, json={
                "granules": [{
                    "granuleId": "3DIMG_2026_TEST.h5",
                    "datasetId": "3DIMG_L1B_STD",
                    "acquisitionTime": datetime.now(timezone.utc).isoformat(),
                    "downloadUrl": "https://mosdac.gov.in/downloadapi/file/3DIMG_2026_TEST.h5",
                    "fileSize": len(granule_bytes),
                }]
            })
        elif "3DIMG_2026_TEST.h5" in url_str:
            return httpx.Response(200, content=granule_bytes)
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    config = MOSDACAuthConfig(
        username="test_user",
        password="test_password",
        download_dir=tmp_path / "satellite_downloads",
    )

    worker = MOSDACIngestionWorker(config=config, target_shape=(128, 128))
    worker.client._http_client = mock_client

    result = await worker.run_once()
    assert result is not None
    assert result["granule_id"] == "3DIMG_2026_TEST.h5"
    assert result["satellite"] == "INSAT-3D"
    assert result["tensor_shape"] == [1, 128, 128]
    assert "pytorch_tensor" in result
    assert isinstance(result["pytorch_tensor"], torch.Tensor)
