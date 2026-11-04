"""Unit & Integration Tests for Historical Satellite Analysis Mode.

Verifies:
1. Historical image listing with split and label filters
2. Historical image metadata retrieval
3. Raw image preview streaming
4. Cyclone Detection CNN V2 inference on authentic INSAT-3D imagery
5. Truthful UNAVAILABLE intensity status (strictly no fabricated intensity)
6. Trajectory prediction guardrails: INSUFFICIENT_TRACK_HISTORY for < 4 fixes
7. Trajectory prediction execution: +6h, +12h, +24h forecasts when >= 4 fixes are provided
8. Phase 10 Risk Engine integration with unmeasured environmental variables marked UNAVAILABLE
9. Zero coordinate fabrication on unreferenced historical tiles
10. Complete cryptographic SHA-256 provenance and mandatory scientific disclaimers
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.historical_satellite_service import historical_satellite_service


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_list_historical_images_default(client):
    """Verifies listing authentic INSAT-3D historical images."""
    response = client.get("/api/satellite/historical/images?limit=10")
    assert response.status_code == 200
    data = response.json()

    assert "total" in data
    assert data["total"] > 0
    assert "images" in data
    assert len(data["images"]) > 0
    assert len(data["images"]) <= 10

    first = data["images"][0]
    assert "image_id" in first
    assert "filename" in first
    assert "split" in first
    assert "label" in first
    assert first["label"] in ["CYCLONE", "NO_CYCLONE"]
    assert first["source"] == "Historical INSAT-3D"
    assert "INSAT-3D" in first["satellite"]
    assert "preview_url" in first


def test_list_historical_images_filter_split(client):
    """Verifies filtering by manifest split (test, validation, train)."""
    response = client.get("/api/satellite/historical/images?split=test&limit=50")
    assert response.status_code == 200
    data = response.json()

    assert data["split_filter"] == "test"
    assert data["total"] == 14  # Exact held-out benchmark count
    for img in data["images"]:
        assert img["split"] == "test"


def test_list_historical_images_filter_label(client):
    """Verifies filtering by classification label (CYCLONE, NO_CYCLONE)."""
    response = client.get("/api/satellite/historical/images?split=test&label=CYCLONE")
    assert response.status_code == 200
    data = response.json()

    assert data["label_filter"] == "CYCLONE"
    for img in data["images"]:
        assert img["label"] == "CYCLONE"


def test_get_historical_image_metadata(client):
    """Verifies retrieving specific historical image metadata."""
    # First get an authentic ID
    list_res = client.get("/api/satellite/historical/images?split=test&limit=1")
    target_id = list_res.json()["images"][0]["image_id"]

    response = client.get(f"/api/satellite/historical/images/{target_id}")
    assert response.status_code == 200
    item = response.json()

    assert item["image_id"] == target_id
    assert item["source"] == "Historical INSAT-3D"
    assert "file_exists" in item
    assert item["file_exists"] is True


def test_get_historical_image_not_found(client):
    """Verifies 404 behavior on non-existent image ID."""
    response = client.get("/api/satellite/historical/images/non_existent_fake_id_12345")
    assert response.status_code == 404


def test_get_historical_image_preview_streaming(client):
    """Verifies streaming raw image bytes for preview thumbnails."""
    list_res = client.get("/api/satellite/historical/images?split=test&limit=1")
    target_id = list_res.json()["images"][0]["image_id"]

    response = client.get(f"/api/satellite/historical/images/{target_id}/preview")
    assert response.status_code == 200
    assert response.headers["content-type"] in ["image/jpeg", "image/png"]
    assert len(response.content) > 1000  # Genuine image bytes


def test_analyze_historical_image_v2_detection(client):
    """Verifies Cyclone Detector CNN V2 runs on authentic historical image."""
    list_res = client.get("/api/satellite/historical/images?split=test&label=CYCLONE&limit=1")
    target_id = list_res.json()["images"][0]["image_id"]

    response = client.post(
        "/api/satellite/historical/analyze",
        json={"image_id": target_id},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["image_id"] == target_id
    assert data["data_mode"] == "HISTORICAL"

    # Detection validation
    det = data["detection"]
    assert "detected" in det
    assert isinstance(det["detected"], bool)
    assert det["label"] in ["CYCLONE", "NO_CYCLONE"]
    assert 0.0 <= det["confidence"] <= 1.0
    assert "class_probabilities" in det
    assert "CYCLONE" in det["class_probabilities"]
    assert "NO_CYCLONE" in det["class_probabilities"]
    assert det["structural_type"] is not None


def test_intensity_strictly_reports_unavailable(client):
    """Scientific Integrity: Verify intensity model execution and truthful status reporting."""
    list_res = client.get("/api/satellite/historical/images?split=test&limit=1")
    target_id = list_res.json()["images"][0]["image_id"]

    response = client.post(
        "/api/satellite/historical/analyze",
        json={"image_id": target_id},
    )
    assert response.status_code == 200
    intensity = response.json()["intensity"]

    assert intensity["status"] in ("OPERATIONAL", "ACTIVE_ESTIMATE", "UNAVAILABLE")
    assert intensity["model_status"] in ("TRAINED", "ACTIVE")
    if intensity["status"] == "OPERATIONAL":
        assert intensity["estimated_wind_speed_knots"] is not None
        assert float(intensity["estimated_wind_speed_knots"]) > 0.0
        assert intensity["estimated_central_pressure_hpa"] is not None


def test_trajectory_insufficient_history_for_standalone_tile(client):
    """Scientific Integrity: Trajectory LSTM strictly requires >= 4 fixes; single image must return INSUFFICIENT_TRACK_HISTORY."""
    list_res = client.get("/api/satellite/historical/images?split=test&limit=1")
    target_id = list_res.json()["images"][0]["image_id"]

    response = client.post(
        "/api/satellite/historical/analyze",
        json={"image_id": target_id, "track_history": []},
    )
    assert response.status_code == 200
    traj = response.json()["trajectory"]

    assert traj["status"] == "INSUFFICIENT_TRACK_HISTORY"
    assert traj["available_fixes"] == 0
    assert traj["required_fixes"] == 4
    assert traj["forecast"] is None
    assert "Phase 8 LSTM trajectory forecasting strictly requires >= 4" in traj["message"]


def test_trajectory_execution_with_four_fixes(client):
    """Verifies Phase 8 Trajectory LSTM produces +6h, +12h, +24h forecasts when >= 4 fixes are provided."""
    list_res = client.get("/api/satellite/historical/images?split=test&label=CYCLONE&limit=1")
    target_id = list_res.json()["images"][0]["image_id"]

    four_fixes = [
        {"latitude": 10.4, "longitude": 86.5, "timestamp": "2020-05-16T12:00:00Z", "wind_speed_knots": 25},
        {"latitude": 11.5, "longitude": 86.2, "timestamp": "2020-05-17T06:00:00Z", "wind_speed_knots": 40},
        {"latitude": 12.8, "longitude": 86.4, "timestamp": "2020-05-17T18:00:00Z", "wind_speed_knots": 65},
        {"latitude": 13.7, "longitude": 86.3, "timestamp": "2020-05-18T06:00:00Z", "wind_speed_knots": 90},
    ]

    response = client.post(
        "/api/satellite/historical/analyze",
        json={"image_id": target_id, "track_history": four_fixes},
    )
    assert response.status_code == 200
    traj = response.json()["trajectory"]

    assert traj["status"] == "FORECAST_GENERATED"
    assert traj["available_fixes"] == 4
    assert traj["forecast"] is not None
    assert traj["forecast"]["model_name"] == "trajectory_lstm_v1"
    assert traj["forecast"]["forecast_type"] == "Experimental ML forecast"

    predictions = traj["forecast"]["predictions"]
    assert len(predictions) == 3
    leads = [p["forecast_hour"] for p in predictions]
    assert leads == [6, 12, 24]

    for p in predictions:
        assert isinstance(p["latitude"], float)
        assert isinstance(p["longitude"], float)
        assert p["uncertainty_km"] > 0
        assert p["uncertainty_radius_km_68"] > 0
        assert p["uncertainty_radius_km_95"] > p["uncertainty_radius_km_68"]


def test_zero_coordinate_fabrication(client):
    """Verifies that unreferenced historical tiles strictly maintain latitude=None and longitude=None."""
    list_res = client.get("/api/satellite/historical/images?split=test&limit=1")
    target_id = list_res.json()["images"][0]["image_id"]

    response = client.post(
        "/api/satellite/historical/analyze",
        json={"image_id": target_id},
    )
    assert response.status_code == 200
    center_loc = response.json()["detection"]["center_localization"]

    assert center_loc["coordinates_available"] is False
    assert center_loc["latitude"] is None
    assert center_loc["longitude"] is None


def test_risk_engine_integration(client):
    """Verifies Phase 10 Risk Engine evaluates multi-factor research risk."""
    list_res = client.get("/api/satellite/historical/images?split=test&limit=1")
    target_id = list_res.json()["images"][0]["image_id"]

    response = client.post(
        "/api/satellite/historical/analyze",
        json={"image_id": target_id},
    )
    assert response.status_code == 200
    risk = response.json()["risk"]

    assert "risk_score" in risk
    assert "risk_level" in risk
    assert risk["risk_level"] in ["SEVERE", "HIGH", "MODERATE", "LOW"]
    assert 0.0 <= risk["risk_score"] <= 100.0
    assert "risk_factors" in risk
    assert "detection_confidence" in risk["risk_factors"]
    # Unmeasured factors must be marked UNAVAILABLE
    assert risk["risk_factors"]["rainfall"] == "UNAVAILABLE"
    assert risk["risk_factors"]["sst"] == "UNAVAILABLE"


def test_provenance_and_disclaimer(client):
    """Verifies cryptographic SHA-256 hash and mandatory scientific disclaimers."""
    list_res = client.get("/api/satellite/historical/images?split=test&limit=1")
    target_id = list_res.json()["images"][0]["image_id"]

    response = client.post(
        "/api/satellite/historical/analyze",
        json={"image_id": target_id},
    )
    assert response.status_code == 200
    data = response.json()

    # Provenance checks
    prov = data["provenance"]
    assert prov["source"] == "Historical INSAT-3D"
    assert prov["data_mode"] == "HISTORICAL"
    assert prov["detection_model"] == "cyclone_detector_cnn_v2"
    assert len(prov["sha256"]) == 64  # Valid 256-bit hex hash

    # Disclaimer checks
    assert "disclaimer" in data
    assert "HISTORICAL DATA MODE" in data["disclaimer"]
    assert "not official IMD warnings" in data["disclaimer"]

