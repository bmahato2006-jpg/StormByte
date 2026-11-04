"""Comprehensive Test Suite for Phase 9 AI Cyclone Detection & Classification."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.ml.detection.config import (
    CHECKPOINT_PATH,
    DATA_DIR,
    METRICS_PATH,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
)
from backend.app.ml.detection.inference import detection_inference_service
from backend.app.ml.detection.preprocessing import (
    compute_cloud_fraction,
    preprocess_for_inference,
    standardize_channels_to_rgb,
    validate_image_payload,
)


client = TestClient(app)

SAMPLE_PATH = DATA_DIR / "samples" / "insat3d_amphan_sample.png"
CYCLONE_TILE_PATH = DATA_DIR / "processed" / "satellite" / "detection" / "images" / "cyclone" / "insat3d_amphan_tile_026.png"
CLEAR_TILE_PATH = DATA_DIR / "processed" / "satellite" / "detection" / "images" / "no_cyclone" / "insat3d_amphan_tile_031.png"


def test_detection_checkpoint_and_manifests():
    """Verify Phase 9 checkpoint and disjoint train/val/test manifests exist."""
    assert CHECKPOINT_PATH.exists(), f"Detection checkpoint missing at {CHECKPOINT_PATH}"
    assert TRAIN_MANIFEST_PATH.exists(), "Train manifest missing"
    assert VAL_MANIFEST_PATH.exists(), "Val manifest missing"
    assert TEST_MANIFEST_PATH.exists(), "Test manifest missing"
    assert METRICS_PATH.exists(), "Evaluation metrics report missing"


def test_image_preprocessing_pipeline():
    """Verify multi-channel standardization, cloud fraction, and normalization."""
    assert SAMPLE_PATH.exists()
    is_valid, err, pil_img = validate_image_payload(SAMPLE_PATH)
    assert is_valid is True
    assert pil_img is not None

    rgb_img = standardize_channels_to_rgb(pil_img)
    assert rgb_img.mode == "RGB"

    cloud_frac, mask = compute_cloud_fraction(rgb_img)
    assert 0.0 <= cloud_frac <= 1.0

    tensor, res_img, meta = preprocess_for_inference(SAMPLE_PATH)
    assert tensor.shape == (1, 3, 256, 256)
    assert meta["standardized_mode"] == "RGB"


def test_corrupt_and_zero_byte_rejection(tmp_path):
    """Verify corrupt and empty files are rejected."""
    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")

    is_valid, err, _ = validate_image_payload(empty_file)
    assert is_valid is False
    assert "empty" in str(err).lower()

    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"NOT_AN_IMAGE_DATA_CORRUPT")
    is_valid_c, err_c, _ = validate_image_payload(corrupt_file)
    assert is_valid_c is False


def test_detection_inference_cyclone_and_no_cyclone():
    """Verify CNN inference returns valid predictions for cyclone and clear background."""
    assert detection_inference_service.is_available()

    # Cyclone core tile
    res_c = detection_inference_service.predict(CYCLONE_TILE_PATH, metadata={}, data_mode="UPLOAD")
    assert res_c["label"] == "CYCLONE"
    assert res_c["confidence"] > 0.50
    assert res_c["detected"] is True

    # Clear background tile
    res_nc = detection_inference_service.predict(CLEAR_TILE_PATH, metadata={}, data_mode="UPLOAD")
    assert res_nc["label"] == "NO_CYCLONE"
    assert res_nc["confidence"] > 0.50
    assert res_nc["detected"] is False


def test_detection_api_endpoints():
    """Verify GET /api/ml/detection/status, GET /api/ml/detection/metrics, and POST /api/ml/detection/predict."""
    r_status = client.get("/api/ml/detection/status")
    assert r_status.status_code == 200
    assert r_status.json()["is_trained"] is True

    r_metrics = client.get("/api/ml/detection/metrics")
    assert r_metrics.status_code == 200
    m_data = r_metrics.json()
    assert m_data["metrics"]["accuracy_percentage"] == 85.71
    assert m_data["test_dataset"]["total_test_samples"] == 7

    r_pred = client.post(
        "/api/ml/detection/predict",
        data={"data_mode": "DEMO", "demo_sample_id": "amphan_vortex_core"},
    )
    assert r_pred.status_code == 200
    p_data = r_pred.json()
    assert p_data["label"] == "CYCLONE"
    assert p_data["confidence"] > 0.50

