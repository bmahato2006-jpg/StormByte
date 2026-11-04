"""
Tests for IMD INSAT-3D Live Satellite Service and Endpoints
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.data.geospatial.imd_insat_live import (
    fetch_live_imd_insat_scene,
    IMD_LIVE_URLS,
)
from backend.app.data.geospatial.live_adapters import LIVE_ADAPTERS, get_all_adapters_status


@pytest.fixture
def client():
    return TestClient(app)


def test_adapters_status_includes_imd_and_nasa():
    statuses = get_all_adapters_status()
    assert "imd_insat" in statuses
    assert "nasa_earthdata" in statuses
    assert statuses["imd_insat"]["configured"] is True
    assert statuses["imd_insat"]["status"] == "CONFIGURED"


def test_live_feed_status_endpoint(client):
    res = client.get("/api/satellite/live/feed-status")
    assert res.status_code == 200
    data = res.json()
    assert "live_satellite_streams" in data
    assert "nasa_eosdis" in data["live_satellite_streams"]
    assert "imd_isro" in data["live_satellite_streams"]
    assert data["live_satellite_streams"]["imd_isro"]["status"] == "CONNECTED"


def test_imd_fetch_live_mocked():
    from PIL import Image
    import io

    # Create dummy 900x900 image
    img = Image.new("RGB", (900, 900), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    fake_bytes = buf.getvalue()

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = fake_bytes
    mock_resp.headers = {"Last-Modified": "Sun, 13 Sep 2026 11:45:00 GMT"}
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        meta = fetch_live_imd_insat_scene(crop_to_nio=True)
        assert meta["satellite"] == "ISRO INSAT-3D / INSAT-3DR"
        assert meta["channel"] == "TIR1 (10.8 µm Thermal Infrared)"
        assert meta["crs"] == "EPSG:4326 (WGS 84)"
        assert meta["verified_live"] is True
        assert len(meta["sha256"]) == 64


def test_imd_fetch_live_endpoint_mocked(client):
    from PIL import Image
    import io

    img = Image.new("RGB", (900, 900), color=(80, 120, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    fake_bytes = buf.getvalue()

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = fake_bytes
    mock_resp.headers = {"Last-Modified": "Sun, 13 Sep 2026 12:00:00 GMT"}
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = client.post("/api/satellite/imd/fetch-live", json={"product": "INSAT3D_TIR1", "threshold": 0.5})
        assert res.status_code == 200
        data = res.json()
        assert "provenance" in data
        assert "center_fix" in data
        assert data["provenance"]["crs"] == "EPSG:4326"
