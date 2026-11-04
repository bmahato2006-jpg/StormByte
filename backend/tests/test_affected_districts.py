import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.impact.exposure import ExposureAnalyzer

client = TestClient(app)


def test_affected_districts_api_endpoint():
    """Verify GET /api/cyclones/{id}/affected-districts endpoint returns structured district hazard assessments."""
    response = client.get("/api/cyclones/IMD-2020-AMPHAN/affected-districts")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

    first = data[0]
    assert "districtName" in first
    assert "state" in first
    assert "alertTier" in first
    assert first["alertTier"] in ["RED", "ORANGE", "YELLOW"]
    assert "closestApproachKm" in first
    assert "arrivalEta" in first
    assert "maxPredictedGustKmh" in first
    assert "stormSurgeMeters" in first
    assert "exposedPopulation" in first
    assert "evacuationAdvisory" in first


def test_risk_endpoint_includes_affected_districts_and_hazard_zones():
    """Verify GET /api/cyclones/{id}/risk schema includes primaryLandfallDistrict, affectedDistricts, hazardZones, and aiModelMetadata."""
    response = client.get("/api/cyclones/IMD-2020-AMPHAN/risk")
    assert response.status_code == 200
    data = response.json()
    assert "primaryLandfallDistrict" in data
    assert data["primaryLandfallDistrict"] is not None
    assert "affectedDistricts" in data
    assert isinstance(data["affectedDistricts"], list)
    assert len(data["affectedDistricts"]) > 0

    # AI Hazard Zones
    assert "hazardZones" in data
    assert "redAlertPolygon" in data["hazardZones"]
    assert "orangeAlertPolygon" in data["hazardZones"]
    assert "yellowAlertPolygon" in data["hazardZones"]
    assert len(data["hazardZones"]["redAlertPolygon"]) > 3
    assert len(data["hazardZones"]["orangeAlertPolygon"]) > 3
    assert len(data["hazardZones"]["yellowAlertPolygon"]) > 3

    # AI Model Provenance Metadata
    assert "aiModelMetadata" in data
    assert data["aiModelMetadata"]["detectionModel"].startswith("CycloneCenterNet")
    assert "nrtSatelliteSource" in data["aiModelMetadata"]
    assert "nrtSatelliteTimestamp" in data["aiModelMetadata"]


def test_exposure_analyzer_zero_fabrication():
    """Verify that when a storm is non-cyclonic (<34 kt), zero districts and empty hazard zones are returned."""
    landfall, districts, zones = ExposureAnalyzer.evaluate_ai_trajectory_district_impact(
        cyclone_lat=15.0,
        cyclone_lon=88.0,
        current_wind_knots=20.0,  # Ambient breeze, non-cyclonic
        current_pressure_hpa=1010.0,
        prediction_points=[],
    )
    assert landfall is None
    assert districts == []
    assert zones.redAlertPolygon == []
    assert zones.orangeAlertPolygon == []
    assert zones.yellowAlertPolygon == []


def test_sync_live_satellite_endpoint():
    """Verify POST /api/cyclones/live/sync-satellite forces NRT ingestion and returns synchronized state."""
    response = client.post("/api/cyclones/live/sync-satellite")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "synchronized"
    assert "timestamp" in data
    assert "risk" in data
    assert "hazardZones" in data["risk"]
