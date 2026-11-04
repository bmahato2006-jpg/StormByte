"""Comprehensive unit and integration test suite for Phase 13: Spatio-Temporal Trajectory Transformer."""

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
    SEQ_LENGTH,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    TRANSFORMER_CHECKPOINT_PATH,
    TRANSFORMER_EVALUATION_METRICS_PATH,
    TRANSFORMER_METADATA_PATH,
    VAL_MANIFEST_PATH,
)
from backend.app.ml.trajectory.dataset import CycloneSequenceDataset
from backend.app.ml.trajectory.features import TrajectoryFeatureScaler, haversine_distance
from backend.app.ml.trajectory.inference import trajectory_inference_service
from backend.app.ml.trajectory.transformer_model import TrajectoryTransformer

client = TestClient(app)


def test_transformer_architecture_forward():
    """Verify TrajectoryTransformer forward pass shapes, positional encodings, and coordinate outputs."""
    model = TrajectoryTransformer(
        input_dim=INPUT_DIM,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        output_dim=OUTPUT_DIM,
        dropout=0.1,
        seq_len=SEQ_LENGTH,
    )
    model.eval()

    batch_size = 8
    dummy_input = torch.randn(batch_size, SEQ_LENGTH, INPUT_DIM)

    # 1. Forward pass shape (batch, 6)
    out = model(dummy_input)
    assert out.shape == (batch_size, OUTPUT_DIM), f"Expected ({batch_size}, {OUTPUT_DIM}), got {out.shape}"

    # 2. Displacement shape (batch, 3, 2)
    disp = model.predict_displacements(dummy_input)
    assert disp.shape == (batch_size, len(HORIZONS), 2), f"Expected ({batch_size}, 3, 2), got {disp.shape}"

    # 3. Coordinate prediction dictionary
    coords = model.predict_coordinates(dummy_input[0], origin_lat=12.0, origin_lon=84.0)
    assert set(coords.keys()) == {6, 12, 24}
    for h in [6, 12, 24]:
        lat, lon = coords[h]
        assert isinstance(lat, float)
        assert isinstance(lon, float)
        assert -90.0 <= lat <= 90.0
        assert -180.0 <= lon <= 180.0


def test_transformer_checkpoint_and_metadata_exist():
    """Verify that the trained Transformer checkpoint and metadata files exist and are valid."""
    assert TRANSFORMER_CHECKPOINT_PATH.exists(), f"Missing checkpoint at {TRANSFORMER_CHECKPOINT_PATH}"
    assert TRANSFORMER_METADATA_PATH.exists(), f"Missing metadata at {TRANSFORMER_METADATA_PATH}"

    # Check torch checkpoint payload
    ckpt = torch.load(TRANSFORMER_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "hyperparameters" in ckpt
    assert "training_provenance" in ckpt
    assert ckpt["model_name"] == "trajectory_transformer_v1"
    assert ckpt["epoch"] >= 1
    assert ckpt["best_val_loss"] > 0

    # Check metadata JSON
    with open(TRANSFORMER_METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["model_name"] == "trajectory_transformer_v1"
    assert "Pre-LN" in meta["architecture"]
    assert meta["best_val_loss"] == ckpt["best_val_loss"]


def test_transformer_inference_service():
    """Verify TrajectoryInferenceService produces multi-horizon forecasts with TrajectoryTransformer."""
    assert trajectory_inference_service.is_available("transformer") is True

    test_history = [
        {"latitude": 11.0, "longitude": 84.0, "timestamp": "2023-05-10T00:00:00Z"},
        {"latitude": 11.5, "longitude": 84.3, "timestamp": "2023-05-10T06:00:00Z"},
        {"latitude": 12.0, "longitude": 84.7, "timestamp": "2023-05-10T12:00:00Z"},
        {"latitude": 12.5, "longitude": 85.0, "timestamp": "2023-05-10T18:00:00Z"},
    ]

    pred = trajectory_inference_service.predict(test_history, model_type="transformer")
    assert pred["model_name"] == "trajectory_transformer_v1"
    assert "Transformer" in pred["architecture"]
    assert len(pred["predictions"]) == 3

    for p in pred["predictions"]:
        assert p["forecast_lead_hours"] in [6, 12, 24]
        assert p["uncertainty_radius_km_68"] > 0
        assert p["uncertainty_radius_km_95"] > p["uncertainty_radius_km_68"]
        assert p["latitude"] > 0
        assert p["longitude"] > 0


def test_comparative_evaluation_report():
    """Verify that comparative evaluation report accurately benchmarks Transformer, LSTM, and Physics."""
    assert TRANSFORMER_EVALUATION_METRICS_PATH.exists(), "Evaluation report does not exist"

    with open(TRANSFORMER_EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["test_dataset"]["num_test_sequences"] == 130
    assert report["test_dataset"]["num_test_storms"] == 6

    # Master comparison table checks
    comp_table = report["comparison_table"]
    assert len(comp_table) == 4  # +6h, +12h, +24h, Overall Mean

    overall_row = next(r for r in comp_table if "Overall" in r["horizon"])
    assert overall_row["transformer_mae_km"] < 75.0, f"Unexpected high transformer MAE: {overall_row['transformer_mae_km']}"
    assert overall_row["lstm_mae_km"] < 70.0, f"Unexpected high LSTM MAE: {overall_row['lstm_mae_km']}"
    assert overall_row["physics_mae_km"] > 70.0

    # Check per-storm breakdown
    storms = report["per_storm_breakdown"]
    assert len(storms) == 6
    storm_names = [s["storm_name"] for s in storms]
    for expected in ["MOCHA", "BIPARJOY"]:
        assert any(expected in s.upper() for s in storm_names)


def test_api_models_endpoint():
    """Verify GET /api/ml/trajectory/models returns both Transformer and LSTM metadata."""
    res = client.get("/api/ml/trajectory/models")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert len(data["models"]) >= 2

    model_ids = [m["model_id"] for m in data["models"]]
    assert "trajectory_transformer_v1" in model_ids
    assert "trajectory_lstm_v1" in model_ids

    transformer_entry = next(m for m in data["models"] if m["model_id"] == "trajectory_transformer_v1")
    assert transformer_entry["is_trained"] is True
    assert transformer_entry["is_primary"] is True


def test_api_predict_transformer_and_lstm():
    """Verify POST /api/ml/trajectory/predict works for both transformer and lstm."""
    test_history = [
        {"latitude": 13.0, "longitude": 84.0, "timestamp": "2023-05-10T00:00:00Z"},
        {"latitude": 13.5, "longitude": 84.3, "timestamp": "2023-05-10T06:00:00Z"},
        {"latitude": 14.0, "longitude": 84.7, "timestamp": "2023-05-10T12:00:00Z"},
        {"latitude": 14.5, "longitude": 85.0, "timestamp": "2023-05-10T18:00:00Z"},
    ]

    # Test transformer
    res_t = client.post("/api/ml/trajectory/predict", json={"track_history": test_history, "model_type": "transformer"})
    assert res_t.status_code == 200
    data_t = res_t.json()
    assert data_t["model_name"] == "trajectory_transformer_v1"
    assert len(data_t["predictions"]) == 3

    # Test lstm
    res_l = client.post("/api/ml/trajectory/predict", json={"track_history": test_history, "model_type": "lstm"})
    assert res_l.status_code == 200
    data_l = res_l.json()
    assert data_l["model_name"] == "trajectory_lstm_v1"
    assert len(data_l["predictions"]) == 3
