"""Comprehensive unit and integration tests for Satellite Pipeline Service and Telemetry API."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
from starlette.testclient import TestClient
import torch

from backend.app.data.geospatial.mosdac_ingestion_worker import (
    MOSDACAccountLockedError,
    MOSDACAuthConfig,
)
from backend.app.main import app
from backend.app.ml.inference import run_detection_pipeline
from backend.app.schemas.telemetry import MOSDACTelemetryStatus
from backend.app.services.satellite_pipeline_service import (
    SatellitePipelineService,
    satellite_pipeline_service,
)


def test_run_detection_pipeline_tensor_input():
    """Verifies that run_detection_pipeline processes [1, H, W] tensors and returns CNN V3 results."""
    # Create normalized [1, 256, 256] tensor in [0, 1]
    tensor = torch.full((1, 256, 256), 0.2, dtype=torch.float32)
    res = run_detection_pipeline(tensor)

    assert "detected" in res
    assert "confidence" in res
    assert "predicted_label" in res
    assert res["model_name"] in ["cyclone_detector_cnn_v3", "cyclone_detector_cnn_v2"]
    assert isinstance(res["detected"], bool)
    assert 0.0 <= res["confidence"] <= 1.0


def test_run_detection_pipeline_numpy_input():
    """Verifies that run_detection_pipeline processes 2D numpy arrays."""
    arr = np.full((128, 128), 0.5, dtype=np.float32)
    res = run_detection_pipeline(arr)
    assert "detected" in res
    assert isinstance(res["confidence"], float)


@pytest.mark.anyio
async def test_pipeline_service_unconfigured_credentials():
    """Verifies that unconfigured MOSDAC credentials truthfully report UNCONFIGURED with zero fake data."""
    unconfigured_config = MOSDACAuthConfig(username="", password="", api_key="")
    service = SatellitePipelineService(auth_config=unconfigured_config, allow_open_fallback=False)

    status = await service.poll_once()
    assert status.is_live is False
    assert status.granules_available == 0
    assert status.operational_health == "UNCONFIGURED"
    assert status.vortex_detected is False
    assert status.active_cyclone_id is None


@pytest.mark.anyio
async def test_pipeline_service_autonomous_open_stream():
    """Verifies that SatellitePipelineService autonomously ingests authentic ISRO/IMD INSAT-3D stream."""
    unconfigured_config = MOSDACAuthConfig(username="", password="", api_key="")
    service = SatellitePipelineService(auth_config=unconfigured_config, allow_open_fallback=True)

    status = await service.poll_once()
    assert status.is_live is True
    assert status.operational_health == "HEALTHY"
    assert status.granules_available >= 1
    assert status.last_scan_utc is not None


@pytest.mark.anyio
async def test_pipeline_service_poll_no_granules():
    """Verifies truthful idle reporting when catalog query returns no new granules."""
    mock_worker = MagicMock()
    mock_worker.run_once = AsyncMock(return_value=None)
    config = MOSDACAuthConfig(username="test_user", password="test_password")

    service = SatellitePipelineService(worker=mock_worker, auth_config=config)
    status = await service.poll_once()

    assert status.is_live is False
    assert status.granules_available == 0
    assert status.operational_health == "HEALTHY"
    assert status.vortex_detected is False


@pytest.mark.anyio
async def test_pipeline_service_poll_successful_granule_no_cyclone():
    """Verifies ingestion and inference flow when a granule is returned without a cyclonic storm."""
    # Synthetic flat tensor (no cyclone)
    flat_tensor = torch.zeros((1, 128, 128), dtype=torch.float32)
    granule_payload = {
        "granule_id": "3DIMG_16SEP2026_0400_L1B_STD",
        "satellite": "INSAT-3D",
        "dataset_id": "3DIMG_L1B_STD",
        "timestamp_utc": "2026-09-16T04:00:00Z",
        "channel": "TIR1",
        "file_path": "data/raw/satellite/test.h5",
        "pytorch_tensor": flat_tensor,
    }

    mock_worker = MagicMock()
    mock_worker.run_once = AsyncMock(return_value=granule_payload)
    config = MOSDACAuthConfig(username="test_user", password="test_password")

    service = SatellitePipelineService(worker=mock_worker, auth_config=config)
    status = await service.poll_once()

    assert status.is_live is True
    assert status.granules_available == 1
    assert status.last_scan_utc == "2026-09-16T04:00:00Z"
    assert status.vortex_detected is False
    assert status.active_cyclone_id is None
    assert status.latency_ms is not None
    assert status.latency_ms > 0


@pytest.mark.anyio
async def test_pipeline_service_poll_with_confirmed_cyclone():
    """Verifies that when CNN V3 detects a storm, CenterNet and Intensity ResNet-18 are invoked."""
    cyclone_tensor = torch.rand((1, 256, 256), dtype=torch.float32)
    granule_payload = {
        "granule_id": "3DIMG_16SEP2026_0430_L1B_STD",
        "satellite": "INSAT-3DR",
        "dataset_id": "3RIMG_L1B_STD",
        "timestamp_utc": "2026-09-16T04:30:00Z",
        "channel": "TIR1",
        "file_path": "data/raw/satellite/storm.h5",
        "pytorch_tensor": cyclone_tensor,
    }

    mock_worker = MagicMock()
    mock_worker.run_once = AsyncMock(return_value=granule_payload)
    config = MOSDACAuthConfig(username="test_user", password="test_password")

    service = SatellitePipelineService(worker=mock_worker, auth_config=config)

    # Mock run_detection_pipeline to simulate confirmed cyclone
    with patch(
        "backend.app.services.satellite_pipeline_service.run_detection_pipeline",
        return_value={
            "detected": True,
            "confidence": 0.94,
            "predicted_label": "CYCLONE",
            "model_name": "cyclone_detector_cnn_v3",
            "model_version": "3.0.0",
        },
    ):
        status = await service.poll_once()

    assert status.is_live is True
    assert status.granules_available == 1
    assert status.vortex_detected is True
    assert status.active_cyclone_id is not None
    assert "CYC-NIO" in status.active_cyclone_id
    assert status.latest_vortex_details is not None
    assert "eye_pixel_x" in status.latest_vortex_details
    assert "eye_pixel_y" in status.latest_vortex_details
    assert "wind_speed_knots" in status.latest_vortex_details
    assert "central_pressure_hpa" in status.latest_vortex_details
    assert status.latest_vortex_details["model_name"] == "cyclone_intensity_resnet18_v1"


@pytest.mark.anyio
async def test_pipeline_service_lockout_handling():
    """Verifies that MOSDACAccountLockedError sets status to LOCKED and halts requests."""
    mock_worker = MagicMock()
    mock_worker.run_once = AsyncMock(
        side_effect=MOSDACAccountLockedError("Account locked for 1 hour")
    )
    config = MOSDACAuthConfig(username="test_user", password="test_password")

    service = SatellitePipelineService(worker=mock_worker, auth_config=config)
    status = await service.poll_once()

    assert status.is_live is False
    assert status.operational_health == "LOCKED"


def test_telemetry_status_endpoint_contract():
    """Tests that GET /api/v2/telemetry/mosdac/status matches exact specification."""
    client = TestClient(app)
    response = client.get("/api/v2/telemetry/mosdac/status")

    assert response.status_code == 200
    data = response.json()

    # Exact fields required by user prompt
    assert "is_live" in data
    assert "last_scan_utc" in data
    assert "granules_available" in data
    assert "vortex_detected" in data
    assert "active_cyclone_id" in data

    assert isinstance(data["is_live"], bool)
    assert isinstance(data["granules_available"], int)
    assert isinstance(data["vortex_detected"], bool)


def test_telemetry_status_v1_alias():
    """Tests that GET /telemetry/mosdac/status and /api/telemetry/mosdac/status resolve cleanly."""
    client = TestClient(app)
    r1 = client.get("/telemetry/mosdac/status")
    assert r1.status_code == 200

    r2 = client.get("/api/telemetry/mosdac/status")
    assert r2.status_code == 200
    assert r1.json()["granules_available"] == r2.json()["granules_available"]
