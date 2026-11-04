"""Comprehensive Scientific Test Suite for Phase 10 End-to-End AI Pipeline.

Verifies:
1. Valid georeferencing
2. Missing georeferencing (coordinates=None, GEOSPATIAL_METADATA_REQUIRED)
3. Invalid latitude bounds
4. Invalid longitude bounds
5. Cyclone detected
6. No cyclone (clear background)
7. Insufficient track history (< 4 fixes -> INSUFFICIENT_TRACK_HISTORY)
8. Sufficient track history (>= 4 fixes -> forecast generated)
9. Phase 8 Trajectory LSTM chaining (+6h, +12h, +24h with empirical uncertainty)
10. Phase 8 unavailable handling
11. Explainable research risk calculation
12. Missing environmental data (marked UNAVAILABLE)
13. LIVE_NRT credentials missing (CREDENTIALS_REQUIRED)
14. Cryptographic provenance & metadata tracking
15. Complete pipeline REST APIs
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.ml.detection.config import (
    CHECKPOINT_V1_PATH,
    CHECKPOINT_V2_PATH,
    DATA_DIR,
)
from backend.app.ml.pipeline.geospatial import GeospatialValidator
from backend.app.ml.pipeline.risk_engine import PipelineRiskEngine
from backend.app.ml.pipeline.service import pipeline_service


client = TestClient(app)

CYCLONE_TILE_PATH = (
    DATA_DIR / "processed" / "satellite" / "detection" / "images" / "cyclone" / "kaggle_insat3d_ir_029_42.jpg"
)
if not CYCLONE_TILE_PATH.exists():
    CYCLONE_TILE_PATH = (
        DATA_DIR / "processed" / "satellite" / "detection" / "images" / "cyclone" / "insat3d_amphan_tile_026.png"
    )
CLEAR_TILE_PATH = DATA_DIR / "processed" / "satellite" / "detection" / "images" / "no_cyclone" / "insat3d_amphan_tile_031.png"


# 1. Test Valid Georeferencing
def test_valid_georeferencing():
    meta = {
        "latitude_min": 15.0,
        "latitude_max": 22.0,
        "longitude_min": 84.0,
        "longitude_max": 91.0,
    }
    res = GeospatialValidator.resolve_center_coordinates(
        pixel_x=128.0,
        pixel_y=128.0,
        image_width=256,
        image_height=256,
        metadata=meta,
    )
    assert res["coordinates_available"] is True
    assert res["latitude"] == 18.5
    assert res["longitude"] == 87.5
    assert res["status"] == "CALIBRATED_GEOREFERENCE"


# 2. Test Missing Georeferencing (Zero Coordinate Fabrication)
def test_missing_georeferencing():
    meta = {}
    res = GeospatialValidator.resolve_center_coordinates(
        pixel_x=128.0,
        pixel_y=128.0,
        image_width=256,
        image_height=256,
        metadata=meta,
    )
    assert res["coordinates_available"] is False
    assert res["latitude"] is None
    assert res["longitude"] is None
    assert res["status"] == "GEOSPATIAL_METADATA_REQUIRED"


# 3. Test Invalid Latitude Bounds
def test_invalid_latitude_bounds():
    # Min greater than max
    meta = {"lat_min": 25.0, "lat_max": 10.0, "lon_min": 80.0, "lon_max": 90.0}
    is_valid, err = GeospatialValidator.validate_bounds(25.0, 10.0, 80.0, 90.0)
    assert is_valid is False
    assert "INVALID_BOUNDS_ORDER" in str(err)

    # Out of range [-90, 90]
    is_valid_range, err_range = GeospatialValidator.validate_bounds(-95.0, 10.0, 80.0, 90.0)
    assert is_valid_range is False
    assert "INVALID_LATITUDE_RANGE" in str(err_range)


# 4. Test Invalid Longitude Bounds
def test_invalid_longitude_bounds():
    # Min greater than max
    is_valid, err = GeospatialValidator.validate_bounds(10.0, 20.0, 95.0, 80.0)
    assert is_valid is False
    assert "INVALID_BOUNDS_ORDER" in str(err)

    # Out of range [-180, 180]
    is_valid_range, err_range = GeospatialValidator.validate_bounds(10.0, 20.0, 80.0, 195.0)
    assert is_valid_range is False
    assert "INVALID_LONGITUDE_RANGE" in str(err_range)


# 5. Test Cyclone Detected
def test_cyclone_detected():
    assert CYCLONE_TILE_PATH.exists()
    res = pipeline_service.analyze(CYCLONE_TILE_PATH, metadata={}, data_mode="UPLOAD")
    assert res["detected"] is True
    assert res["label"] == "CYCLONE"
    assert res["confidence"] > 0.50
    assert res["center_fix"] is not None
    assert res["center_fix"]["pixel_x"] > 0


# 6. Test No Cyclone (Clear Background)
def test_no_cyclone():
    assert CLEAR_TILE_PATH.exists()
    res = pipeline_service.analyze(CLEAR_TILE_PATH, metadata={}, data_mode="UPLOAD")
    assert res["detected"] is False
    assert res["label"] == "NO_CYCLONE"
    assert res["risk_assessment"]["risk_level"] == "NONE"


# 7. Test Insufficient Track History (< 4 fixes)
def test_insufficient_track_history():
    meta = {
        "latitude_min": 15.0,
        "latitude_max": 22.0,
        "longitude_min": 84.0,
        "longitude_max": 91.0,
        # Only 2 historical fixes provided
        "track_history": [
            {"latitude": 13.0, "longitude": 85.0, "timestamp": "2020-05-17T00:00:00Z"},
            {"latitude": 14.0, "longitude": 85.5, "timestamp": "2020-05-17T06:00:00Z"},
        ],
    }
    res = pipeline_service.analyze(CYCLONE_TILE_PATH, metadata=meta, data_mode="HISTORICAL")
    assert res["detected"] is True
    assert res["center_fix"]["latitude"] is not None
    assert res["track_history_length"] < 4
    assert res["trajectory_status"] == "INSUFFICIENT_TRACK_HISTORY"
    assert res["trajectory_forecast"] is None


# 8. Test Sufficient Track History (>= 4 fixes)
def test_sufficient_track_history():
    meta = {
        "latitude_min": 15.0,
        "latitude_max": 22.0,
        "longitude_min": 84.0,
        "longitude_max": 91.0,
        "track_history": [
            {"latitude": 11.2, "longitude": 86.4, "timestamp": "2020-05-16T18:00:00Z"},
            {"latitude": 12.5, "longitude": 86.4, "timestamp": "2020-05-17T06:00:00Z"},
            {"latitude": 13.6, "longitude": 86.7, "timestamp": "2020-05-17T18:00:00Z"},
            {"latitude": 14.9, "longitude": 86.5, "timestamp": "2020-05-18T06:00:00Z"},
        ],
    }
    res = pipeline_service.analyze(CYCLONE_TILE_PATH, metadata=meta, data_mode="HISTORICAL")
    assert res["detected"] is True
    assert res["trajectory_status"] == "SUCCESSFULLY_FORECASTED_VIA_PHASE_8_LSTM"
    assert res["trajectory_forecast"] is not None
    assert len(res["trajectory_forecast"]["predictions"]) == 3


# 9. Test Phase 8 Chaining with Empirical Uncertainty
def test_phase8_chaining_uncertainty():
    meta = {
        "latitude_min": 15.0,
        "latitude_max": 22.0,
        "longitude_min": 84.0,
        "longitude_max": 91.0,
        "track_history": [
            {"latitude": 11.2, "longitude": 86.4, "timestamp": "2020-05-16T18:00:00Z"},
            {"latitude": 12.5, "longitude": 86.4, "timestamp": "2020-05-17T06:00:00Z"},
            {"latitude": 13.6, "longitude": 86.7, "timestamp": "2020-05-17T18:00:00Z"},
            {"latitude": 14.9, "longitude": 86.5, "timestamp": "2020-05-18T06:00:00Z"},
        ],
    }
    res = pipeline_service.analyze(CYCLONE_TILE_PATH, metadata=meta, data_mode="HISTORICAL")
    preds = res["trajectory_forecast"]["predictions"]
    horizons = [p["forecast_hour"] for p in preds]
    assert horizons == [6, 12, 24]

    p6 = preds[0]
    p12 = preds[1]
    p24 = preds[2]

    # Uncertainty grows with horizon
    assert p6["uncertainty_radius_km_68"] < p12["uncertainty_radius_km_68"] < p24["uncertainty_radius_km_68"]
    assert p6["uncertainty_radius_km_95"] >= p6["uncertainty_radius_km_68"]


# 10. Test Phase 8 Trajectory Unavailable Handling
def test_phase8_unavailable_handling(monkeypatch):
    monkeypatch.setattr(pipeline_service.trajectory_service, "is_available", lambda: False)
    meta = {
        "latitude_min": 15.0,
        "latitude_max": 22.0,
        "longitude_min": 84.0,
        "longitude_max": 91.0,
        "track_history": [
            {"latitude": 11.0, "longitude": 85.0, "timestamp": "2020-05-16T00:00:00Z"},
            {"latitude": 12.0, "longitude": 85.5, "timestamp": "2020-05-16T06:00:00Z"},
            {"latitude": 13.0, "longitude": 86.0, "timestamp": "2020-05-16T12:00:00Z"},
            {"latitude": 14.0, "longitude": 86.5, "timestamp": "2020-05-16T18:00:00Z"},
        ],
    }
    res = pipeline_service.analyze(CYCLONE_TILE_PATH, metadata=meta, data_mode="HISTORICAL")
    assert res["trajectory_status"] == "TRAJECTORY_MODEL_AWAITING_TRAINING"
    assert res["trajectory_forecast"] is None


# 11. Test Risk Calculation
def test_risk_calculation():
    risk = PipelineRiskEngine.evaluate(
        detected=True,
        detection_confidence=0.85,
        latitude=18.5,
        longitude=87.5,
        forecast_predictions=[
            {"forecast_hour": 6, "latitude": 19.2, "longitude": 87.8, "uncertainty_radius_km_68": 35.0},
            {"forecast_hour": 12, "latitude": 20.0, "longitude": 88.0, "uncertainty_radius_km_68": 68.5},
            {"forecast_hour": 24, "latitude": 21.5, "longitude": 88.2, "uncertainty_radius_km_68": 129.6},
        ],
        environmental_obs={"wind_speed_knots": 65.0, "sst_celsius": 30.0},
    )
    assert risk["risk_score"] > 0
    assert risk["risk_level"] in ["HIGH", "SEVERE"]
    assert risk["risk_factors"]["detection_confidence"]["status"] == "AVAILABLE"
    assert risk["risk_factors"]["land_proximity"]["status"] == "AVAILABLE"
    assert risk["risk_factors"]["wind"]["status"] == "AVAILABLE"


# 12. Test Missing Environmental Data (Marked UNAVAILABLE)
def test_missing_environmental_data():
    risk = PipelineRiskEngine.evaluate(
        detected=True,
        detection_confidence=0.60,
        latitude=15.0,
        longitude=85.0,
        environmental_obs={},  # No wind, SST, or rainfall
    )
    assert risk["risk_factors"]["wind"] == "UNAVAILABLE"
    assert risk["risk_factors"]["rainfall"] == "UNAVAILABLE"
    assert risk["risk_factors"]["sst"] == "UNAVAILABLE"
    assert risk["data_completeness"] < 0.60


# 13. Test LIVE_NRT Credentials Missing (Zero Fabrication)
def test_live_nrt_credentials_missing(monkeypatch):
    monkeypatch.setattr(settings, "MOSDAC_API_KEY", None)
    res = pipeline_service.analyze(CYCLONE_TILE_PATH, metadata={}, data_mode="LIVE_NRT")
    assert res["processing_status"] == "CREDENTIALS_REQUIRED"
    assert res["trajectory_status"] == "CREDENTIALS_REQUIRED"
    assert "MOSDAC_API_KEY" in res["error_detail"]


# 14. Test Cryptographic Provenance Record
def test_pipeline_provenance():
    res = pipeline_service.analyze(CYCLONE_TILE_PATH, metadata={}, data_mode="UPLOAD")
    prov = res["provenance"]
    assert prov["detection_model"] == "cyclone_detector_cnn_v2"
    assert prov["detection_version"] == "2.0.0"
    assert prov["image_sha256"] is not None
    assert len(prov["image_sha256"]) == 64
    assert "disclaimer" in res


# 15. Test Complete Pipeline REST APIs
def test_complete_pipeline_api():
    # Test GET /api/ml/pipeline/status
    r_status = client.get("/api/ml/pipeline/status")
    assert r_status.status_code == 200
    st_json = r_status.json()
    assert st_json["pipeline_status"] == "OPERATIONAL_EXPERIMENTAL"
    assert st_json["phase9_detection"]["is_trained"] is True
    assert st_json["phase8_trajectory"]["is_trained"] is True

    # Test POST /api/ml/pipeline/analyze with demo sample
    r_analyze = client.post(
        "/api/ml/pipeline/analyze",
        data={"data_mode": "DEMO", "demo_sample_id": "amphan_georeferenced_with_history"},
    )
    assert r_analyze.status_code == 200
    an_json = r_analyze.json()
    assert an_json["detected"] is True
    assert an_json["center_fix"]["latitude"] is not None
    assert an_json["trajectory_forecast"] is not None
    assert len(an_json["trajectory_forecast"]["predictions"]) == 3
    assert an_json["risk_assessment"]["risk_score"] > 0


# 16. Test Phase 10 V2 Model Integration & Provenance
def test_phase10_default_detection_model_is_v2():
    """Verify that Phase 10 default detection model is cyclone_detector_cnn_v2 or v3 with valid checkpoint."""
    assert CHECKPOINT_V2_PATH.exists(), f"V2 Checkpoint missing at {CHECKPOINT_V2_PATH}"
    assert pipeline_service.detection_service.model_name in ("cyclone_detector_cnn_v2", "cyclone_detector_cnn_v3")
    assert pipeline_service.detection_service.model_version in ("2.0.0", "3.0.0")
    assert pipeline_service.detection_service.is_available() is True


def test_phase10_invokes_v2_inference():
    """Verify that Phase 10 actually executes inference using active weights."""
    res = pipeline_service.analyze(CYCLONE_TILE_PATH, metadata={}, data_mode="UPLOAD")
    assert res["detected"] is True
    assert res["label"] == "CYCLONE"
    assert res["confidence"] > 0.50
    assert res["provenance"]["detection_model"] in ("cyclone_detector_cnn_v2", "cyclone_detector_cnn_v3")
    assert res["provenance"]["detection_version"] in ("2.0.0", "3.0.0")


def test_phase10_api_status_reports_v2():
    """Verify that GET /api/ml/pipeline/status reports V2 as active model and V1 as baseline."""
    r = client.get("/api/ml/pipeline/status")
    assert r.status_code == 200
    data = r.json()
    p9 = data["phase9_detection"]
    assert p9["model_name"] == "cyclone_detector_cnn_v2"
    assert p9["version"] == "2.0.0"
    assert p9["status"] == "TRAINED"
    assert "cyclone_detector_cnn_v2.pt" in p9["checkpoint"]
    assert p9["metrics"]["accuracy_percentage"] == 100.0

    # Verify V1 baseline is reported and preserved
    assert "baseline_model" in p9
    base = p9["baseline_model"]
    assert base["model_name"] == "cyclone_detector_cnn_v1"
    assert base["version"] == "1.0.0"
    assert base["is_trained"] is True
    assert "cyclone_detector_cnn_v1.pt" in base["checkpoint"]


def test_phase10_api_analyze_reports_v2():
    """Verify that POST /api/ml/pipeline/analyze returns active model in provenance."""
    r = client.post(
        "/api/ml/pipeline/analyze",
        data={"data_mode": "DEMO", "demo_sample_id": "amphan_georeferenced_with_history"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["provenance"]["detection_model"] in ("cyclone_detector_cnn_v2", "cyclone_detector_cnn_v3")
    assert data["provenance"]["detection_version"] in ("2.0.0", "3.0.0")


def test_phase10_v1_baseline_availability():
    """Verify that V1 baseline model remains callable when requested."""
    assert CHECKPOINT_V1_PATH.exists(), f"V1 Checkpoint missing at {CHECKPOINT_V1_PATH}"
    assert pipeline_service.detection_service_v1.is_available() is True
    assert pipeline_service.detection_service_v1.model_name == "cyclone_detector_cnn_v1"

    # Analyze using explicit v1 via service
    res_v1 = pipeline_service.analyze(CYCLONE_TILE_PATH, metadata={}, model_version="v1")
    assert res_v1["provenance"]["detection_model"] == "cyclone_detector_cnn_v1"
    assert res_v1["provenance"]["detection_version"] == "1.0.0"

    # Analyze using explicit v1 via API
    r_v1 = client.post(
        "/api/ml/pipeline/analyze",
        data={
            "data_mode": "DEMO",
            "demo_sample_id": "amphan_vortex_core",
            "model_version": "v1",
        },
    )
    assert r_v1.status_code == 200
    data_v1 = r_v1.json()
    assert data_v1["provenance"]["detection_model"] == "cyclone_detector_cnn_v1"
    assert data_v1["provenance"]["detection_version"] == "1.0.0"


# 17. Test Unified Provenance Object & V3 Audit Trail
def test_unified_provenance_structure_and_v3():
    """Verify unified provenance structure, V3 execution, and zero-fabrication rules."""
    r_v3 = client.post(
        "/api/ml/pipeline/analyze",
        data={
            "data_mode": "DEMO",
            "demo_sample_id": "amphan_georeferenced_with_history",
            "model_version": "v3",
        },
    )
    assert r_v3.status_code == 200
    data_v3 = r_v3.json()
    prov = data_v3["provenance"]

    # Check all required unified provenance keys
    required_keys = [
        "image_source",
        "image_dataset",
        "image_id",
        "image_sha256",
        "data_mode",
        "detection_model",
        "detection_model_version",
        "detection_checkpoint",
        "trajectory_model",
        "trajectory_model_version",
        "trajectory_checkpoint",
        "track_source",
        "track_id",
        "track_provenance",
        "analysis_timestamp_utc",
    ]
    for k in required_keys:
        assert k in prov, f"Missing unified provenance key: {k}"

    # V3 model verification
    assert prov["detection_model"] == "cyclone_detector_cnn_v3"
    assert prov["detection_model_version"] == "3.0.0"
    assert prov["detection_checkpoint"] == "cyclone_detector_cnn_v3.pt"

    # Trajectory model verification: run produced a forecast, so it MUST show trajectory_lstm_v1
    assert prov["trajectory_model"] == "trajectory_lstm_v1"
    assert prov["trajectory_model_version"] == "1.0.0"
    assert prov["trajectory_checkpoint"] == "trajectory_lstm_v1.pt"
    assert prov["track_source"] == "IBTrACS v04r01 (NOAA/WMO)"
    assert "IMD-2020-AMPHAN" in prov["track_id"]

    # Data source verification: Kaggle INSAT-3D benchmark, NOT fake MOSDAC
    assert prov["image_source"] == "Kaggle INSAT-3D Benchmark"
    assert "Kaggle" in prov["image_dataset"]
    assert prov["acquisition_timestamp"] == "UNVERIFIED"


def test_unified_provenance_no_trajectory():
    """Verify that when no trajectory is produced, trajectory_model is strictly None (NONE / NOT USED)."""
    r_no_traj = client.post(
        "/api/ml/pipeline/analyze",
        data={
            "data_mode": "DEMO",
            "demo_sample_id": "amphan_vortex_core",
            "model_version": "v3",
        },
    )
    assert r_no_traj.status_code == 200
    data = r_no_traj.json()
    prov = data["provenance"]

    assert data["trajectory_forecast"] is None
    assert prov["trajectory_model"] is None
    assert prov["trajectory_model_version"] is None
    assert prov["trajectory_checkpoint"] is None
    assert prov["track_source"] is None
    assert prov["track_id"] is None


def test_stage2_and_stage3_calibrated_pipeline():
    """Verify that amphan_georeferenced_with_history resolves Stage 2 and forecasts Stage 3."""
    r = client.post(
        "/api/ml/pipeline/analyze",
        data={
            "data_mode": "DEMO",
            "demo_sample_id": "amphan_georeferenced_with_history",
            "model_version": "v3",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["detected"] is True
    # Stage 2: Center Fix & Geodesy
    assert data["center_fix"]["coordinates_available"] is True
    assert 15.0 <= data["center_fix"]["latitude"] <= 22.0
    assert 84.0 <= data["center_fix"]["longitude"] <= 91.0
    # Stage 3: Phase 8 Trajectory LSTM
    assert data["trajectory_status"] == "SUCCESSFULLY_FORECASTED_VIA_PHASE_8_LSTM"
    assert data["trajectory_forecast"] is not None
    assert len(data["trajectory_forecast"]["predictions"]) == 3
    assert data["provenance"]["trajectory_model"] == "trajectory_lstm_v1"


def test_stage2_and_stage3_upload_with_auto_calibrate():
    """Verify that uploading a cyclone image with auto_calibrate resolves Stage 2 and Stage 3."""
    import json
    with open(CYCLONE_TILE_PATH, "rb") as f:
        r = client.post(
            "/api/ml/pipeline/analyze",
            files={"file": ("cyclone_tile.jpg", f, "image/jpeg")},
            data={
                "data_mode": "UPLOAD",
                "model_version": "v3",
                "metadata_json": json.dumps({"auto_calibrate": True, "auto_anchor_history": True}),
            },
        )
    assert r.status_code == 200
    data = r.json()
    assert data["detected"] is True
    assert data["center_fix"]["coordinates_available"] is True
    assert data["center_fix"]["latitude"] is not None
    assert data["center_fix"]["longitude"] is not None
    assert data["trajectory_status"] == "SUCCESSFULLY_FORECASTED_VIA_PHASE_8_LSTM"
    assert data["trajectory_forecast"] is not None
    assert len(data["trajectory_forecast"]["predictions"]) == 3


def test_stage2_and_stage3_auto_calibrate_metadata():
    """Verify that auto_calibrate metadata triggers Stage 2 geodesy and Stage 3 trajectory."""
    import json
    r = client.post(
        "/api/ml/pipeline/analyze",
        data={
            "data_mode": "DEMO",
            "demo_sample_id": "amphan_vortex_core",
            "model_version": "v3",
            "metadata_json": json.dumps({"auto_calibrate": True, "auto_anchor_history": True}),
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["detected"] is True
    assert data["center_fix"]["coordinates_available"] is True
    assert data["trajectory_forecast"] is not None
    assert len(data["trajectory_forecast"]["predictions"]) == 3



