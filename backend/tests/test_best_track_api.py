"""Tests for Best Track FastAPI Endpoints."""

from fastapi.testclient import TestClient
import pytest
from backend.app.main import app

client = TestClient(app)


def test_api_stats():
    resp = client.get("/api/best-track/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_storms"] >= 2000
    assert data["total_fixes"] >= 60000
    assert "IMD_RSMC_NEW_DELHI" in data["source_counts"]
    assert "NOAA_IBTRACS" in data["source_counts"]


def test_api_list_storms():
    resp = client.get("/api/best-track/storms?year=2020&search=AMPHAN")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    names = [s["storm_name"] for s in data["storms"]]
    assert "AMPHAN" in names


def test_api_get_storm_track():
    resp = client.get("/api/best-track/storms/IMD-2020-AMPHAN/track")
    assert resp.status_code == 200
    data = resp.json()
    assert data["fixes_count"] >= 50
    assert data["geojson"]["type"] == "Feature"
    assert data["geojson"]["geometry"]["type"] == "LineString"


def test_api_match_image_unverified():
    # Calling match without coordinates returns UNVERIFIED (zero coordinate fabrication)
    resp = client.post("/api/best-track/match-image", json={"storm_id": "IMD-2020-AMPHAN"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "UNVERIFIED"
    assert "UNVERIFIED" in data["details"]
