"""
Unit and Integration Tests for NASA Earthdata Live Satellite Stream
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.data.geospatial.nasa_earthdata_live import (
    get_token_info,
    SECTOR_BOUNDS,
    SUPPORTED_LAYERS,
)

client = TestClient(app)


def test_token_info_decodes_correctly():
    """Valid token should report configured and parse user ID."""
    info = get_token_info()
    assert info["configured"] is True
    assert info["status"] == "CONFIGURED"
    assert info.get("username") is not None
    assert info.get("expires_at") is not None


def test_sector_bounds_geometry():
    """Verify sector boundary values are meteorologically valid."""
    bob = SECTOR_BOUNDS["BAY_OF_BENGAL"]
    assert bob["lat_min"] < bob["lat_max"]
    assert bob["lon_min"] < bob["lon_max"]
    assert 0.0 <= bob["lat_min"] <= 30.0


def test_nasa_token_status_endpoint():
    """GET /api/satellite/nasa/token-status returns 200 with structured info."""
    resp = client.get("/api/satellite/nasa/token-status")
    assert resp.status_code == 200
    data = resp.json()
    assert "configured" in data
    assert data["configured"] is True
    assert "username" in data


def test_nasa_fetch_live_endpoint():
    """POST /api/satellite/nasa/fetch-live should fetch live scene and run pipeline."""
    resp = client.post("/api/satellite/nasa/fetch-live", json={
        "sector": "BAY_OF_BENGAL",
        "layer": "MODIS_TIR1",
        "threshold": 0.50
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "detected" in data
    assert "center_fix" in data
    assert "provenance" in data
    assert "NASA" in data["provenance"]["satellite"]
