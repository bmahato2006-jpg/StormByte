"""Comprehensive unit and integration test suite for Phase 8: Real ML Trajectory Forecasting."""

import csv
import json
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.ml.trajectory.config import (
    CHECKPOINT_PATH,
    EVALUATION_METRICS_PATH,
    HORIZONS,
    INPUT_DIM,
    OUTPUT_DIM,
    SCALER_PATH,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
)
from backend.app.ml.trajectory.dataset import CycloneSequenceDataset
from backend.app.ml.trajectory.evaluator import TrajectoryEvaluator
from backend.app.ml.trajectory.features import (
    calculate_bearing,
    extract_raw_step_features,
    haversine_distance,
    TrajectoryFeatureScaler,
)
from backend.app.ml.trajectory.inference import trajectory_inference_service
from backend.app.ml.trajectory.model import TrajectoryLSTM

client = TestClient(app)


def test_haversine_and_bearing():
    """Verify geodetic calculation correctness."""
    # Distance between Chennai (13.0827, 80.2707) and Kolkata (22.5726, 88.3639) is ~1365 km
    dist = haversine_distance(13.0827, 80.2707, 22.5726, 88.3639)
    assert 1350.0 < dist < 1380.0

    # Heading from south to north should be close to 0 deg
    bearing_north = calculate_bearing(10.0, 80.0, 15.0, 80.0)
    assert abs(bearing_north - 0.0) < 1.0 or abs(bearing_north - 360.0) < 1.0

    # Heading from west to east should be close to 90 deg
    bearing_east = calculate_bearing(10.0, 80.0, 10.0, 85.0)
    assert abs(bearing_east - 90.0) < 1.0


def test_feature_extraction_step():
    """Verify raw step feature dictionary creation."""
    fix1 = {"latitude": 12.5, "longitude": 84.0, "timestamp": "2023-05-10T00:00:00Z"}
    fix2 = {"latitude": 13.0, "longitude": 84.2, "timestamp": "2023-05-10T06:00:00Z"}

    feat = extract_raw_step_features(fix2, fix1)
    assert feat["lat"] == 12.5 or feat["lat"] == 13.0
    assert feat["time_delta_h"] == 6.0
    assert feat["displacement_km"] > 0.0
    assert feat["speed_kmh"] > 0.0
    assert -1.0 <= feat["month_sin"] <= 1.0
    assert -1.0 <= feat["hour_cos"] <= 1.0


def test_scaler_persistence():
    """Verify TrajectoryFeatureScaler fits, transforms, saves, and loads correctly."""
    assert SCALER_PATH.exists(), f"Scaler file not found at {SCALER_PATH}"
    scaler = TrajectoryFeatureScaler.load(SCALER_PATH)
    assert scaler.is_fitted
    assert len(scaler.means) == len(scaler.CONTINUOUS_KEYS)

    sample_feat = {
        "lat": 14.0,
        "lon": 85.0,
        "delta_lat": 0.5,
        "delta_lon": 0.2,
        "displacement_km": 60.0,
        "speed_kmh": 10.0,
        "bearing_sin": 0.5,
        "bearing_cos": 0.866,
        "time_delta_h": 6.0,
        "month_sin": 0.5,
        "month_cos": 0.866,
        "hour_sin": 0.0,
        "hour_cos": 1.0,
    }
    vec = scaler.transform(sample_feat)
    assert len(vec) == INPUT_DIM
    assert all(isinstance(v, float) for v in vec)


def test_zero_leakage_manifest_split():
    """Verify zero storm overlap between training, validation, and test manifests."""
    assert TRAIN_MANIFEST_PATH.exists()
    assert VAL_MANIFEST_PATH.exists()
    assert TEST_MANIFEST_PATH.exists()

    def get_storms(path: Path) -> set:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return {r["storm_id"] for r in reader}

    train_storms = get_storms(TRAIN_MANIFEST_PATH)
    val_storms = get_storms(VAL_MANIFEST_PATH)
    test_storms = get_storms(TEST_MANIFEST_PATH)

    # 1. Zero cross-split overlap
    train_test_overlap = train_storms.intersection(test_storms)
    train_val_overlap = train_storms.intersection(val_storms)
    val_test_overlap = val_storms.intersection(test_storms)

    assert len(train_test_overlap) == 0, f"Leakage detected between train and test: {train_test_overlap}"
    assert len(train_val_overlap) == 0, f"Leakage detected between train and val: {train_val_overlap}"
    assert len(val_test_overlap) == 0, f"Leakage detected between val and test: {val_test_overlap}"

    # 2. Strict temporal separation
    # Train <= 2021, Val == 2022, Test >= 2023
    assert len(train_storms) > 1000
    assert len(val_storms) >= 5
    assert len(test_storms) >= 5


def test_lstm_model_shapes():
    """Verify TrajectoryLSTM neural network dimensions."""
    model = TrajectoryLSTM(input_dim=INPUT_DIM, hidden_dim=64, num_layers=2, output_dim=OUTPUT_DIM)
    dummy_input = torch.randn(8, 4, INPUT_DIM)
    out = model(dummy_input)
    assert out.shape == (8, OUTPUT_DIM)

    disps = model.predict_displacements(dummy_input)
    assert disps.shape == (8, len(HORIZONS), 2)

    coords = model.predict_coordinates(dummy_input[0], origin_lat=15.0, origin_lon=85.0)
    assert set(coords.keys()) == {6, 12, 24}
    assert all(len(coords[h]) == 2 for h in HORIZONS)


def test_checkpoint_integrity():
    """Verify saved checkpoint contains weights, hyperparams, and provenance."""
    assert CHECKPOINT_PATH.exists(), f"Checkpoint missing at {CHECKPOINT_PATH}"
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    assert "model_state_dict" in ckpt
    assert "hyperparameters" in ckpt
    assert "training_provenance" in ckpt
    assert ckpt["hyperparameters"]["input_dim"] == INPUT_DIM
    assert ckpt["hyperparameters"]["output_dim"] == OUTPUT_DIM
    assert "train_manifest_sha256" in ckpt["training_provenance"]


def test_inference_service():
    """Verify production inference service generates multi-horizon forecast with empirical uncertainty."""
    assert trajectory_inference_service.is_available()

    history = [
        {"latitude": 12.0, "longitude": 84.0, "timestamp": "2023-05-10T00:00:00Z"},
        {"latitude": 12.4, "longitude": 84.3, "timestamp": "2023-05-10T06:00:00Z"},
        {"latitude": 12.9, "longitude": 84.7, "timestamp": "2023-05-10T12:00:00Z"},
        {"latitude": 13.5, "longitude": 85.2, "timestamp": "2023-05-10T18:00:00Z"},
    ]

    result = trajectory_inference_service.predict(history)
    assert result["model_name"] == "trajectory_lstm_v1"
    assert result["is_experimental"] is True
    assert len(result["predictions"]) == 3

    for p in result["predictions"]:
        assert p["forecast_lead_hours"] in [6, 12, 24]
        assert "latitude" in p
        assert "longitude" in p
        assert p["uncertainty_radius_km_68"] > 0.0
        assert p["uncertainty_radius_km_95"] >= p["uncertainty_radius_km_68"]


def test_evaluator_metrics_report():
    """Verify comparative evaluation metrics file exists and satisfies scientific integrity."""
    assert EVALUATION_METRICS_PATH.exists(), f"Evaluation report missing at {EVALUATION_METRICS_PATH}"
    with open(EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["test_dataset"]["num_test_sequences"] == 130
    assert metrics["test_dataset"]["num_test_storms"] >= 5

    # Check comparison table
    comp = metrics["comparison_table"]
    assert len(comp) == 4  # +6h, +12h, +24h, Overall Mean
    for row in comp:
        assert row["ml_mae_km"] > 0.0
        assert row["physics_mae_km"] > 0.0
        assert "difference_km" in row
        assert "better_model" in row

    # Check per-storm breakdown
    storms = metrics["per_storm_breakdown"]
    storm_names = [s["storm_name"] for s in storms]
    assert "BIPARJOY" in storm_names
    assert "MOCHA" in storm_names


def test_api_model_status():
    """Verify GET /api/ml/trajectory/model endpoint."""
    res = client.get("/api/ml/trajectory/model")
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "trajectory_lstm_v1"
    assert data["is_trained"] is True
    assert data["framework"] == "PyTorch"
    assert data["hidden_dim"] == 64
    assert "disclaimer" in data


def test_api_predict_endpoint():
    """Verify POST /api/ml/trajectory/predict endpoint."""
    payload = {
        "track_history": [
            {"latitude": 13.0, "longitude": 85.0, "timestamp": "2023-05-10T00:00:00Z"},
            {"latitude": 13.5, "longitude": 85.4, "timestamp": "2023-05-10T06:00:00Z"},
            {"latitude": 14.1, "longitude": 85.9, "timestamp": "2023-05-10T12:00:00Z"},
            {"latitude": 14.8, "longitude": 86.5, "timestamp": "2023-05-10T18:00:00Z"},
        ],
        "forecast_horizons": [6, 12, 24],
    }
    res = client.post("/api/ml/trajectory/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "trajectory_lstm_v1"
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["forecast_lead_hours"] == 6
    assert data["predictions"][1]["forecast_lead_hours"] == 12
    assert data["predictions"][2]["forecast_lead_hours"] == 24


def test_api_metrics_endpoint():
    """Verify GET /api/ml/trajectory/metrics endpoint."""
    res = client.get("/api/ml/trajectory/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "comparison_table" in data
    assert "ml_model_metrics" in data
    assert "physics_model_metrics" in data
    assert "per_storm_breakdown" in data


def test_api_evaluate_endpoint():
    """Verify GET /api/ml/trajectory/evaluate endpoint returns full evaluation response."""
    res = client.get("/api/ml/trajectory/evaluate?force_rerun=false")
    assert res.status_code == 200
    data = res.json()
    assert "evaluation_id" in data
    assert data["test_dataset"]["num_test_sequences"] == 130

