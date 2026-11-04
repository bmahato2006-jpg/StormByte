"""Automated test suite for Kaggle INSAT-3D Cyclone Dataset integration, V2 CNN model, and REST APIs."""

import json
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch
from PIL import Image

from backend.app.data.ingestion.kaggle_insat3d import (
    PROVENANCE_PATH as KAGGLE_PROVENANCE_PATH,
    RAW_KAGGLE_DIR,
)
from backend.app.main import app
from backend.app.ml.detection.config import (
    CHECKPOINT_V1_PATH,
    CHECKPOINT_V2_PATH,
    METRICS_V1_PATH,
    METRICS_V2_PATH,
    REPORT_V2_PATH,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
)
from backend.app.ml.detection.dataset import CycloneImageDataset
from backend.app.ml.detection.evaluator import DetectionEvaluator
from backend.app.ml.detection.insat3d_preprocessing import INSAT3DPreprocessor
from backend.app.ml.model_registry import ModelRegistry

client = TestClient(app)


def test_kaggle_provenance_file_exists():
    """Verify that Kaggle INSAT-3D dataset provenance file exists with strict metadata."""
    assert KAGGLE_PROVENANCE_PATH.exists(), f"Provenance missing at {KAGGLE_PROVENANCE_PATH}"
    with open(KAGGLE_PROVENANCE_PATH, "r", encoding="utf-8") as f:
        prov = json.load(f)

    assert prov["dataset_source"] == "Kaggle"
    assert prov["dataset_identifier"] == "sshubam/insat3d-infrared-raw-cyclone-images-20132021"
    assert "download_path" in prov and len(prov["download_path"]) > 0
    assert "download_timestamp" in prov and len(prov["download_timestamp"]) > 0
    assert prov["csv_count"] >= 1
    assert "insat_3d_ds - Sheet.csv" in prov["discovered_csv_files"][0]
    assert len(prov["primary_csv_sha256"]) == 64
    assert prov["total_images_discovered"] >= 136
    assert prov["original_file_count"] == 136
    assert prov["processed_cyclone_images"] == 136
    assert prov["processed_no_cyclone_images"] == 12
    assert prov["total_samples"] == 148
    assert prov["splits"]["train_samples"] == 106
    assert prov["splits"]["val_samples"] == 28
    assert prov["splits"]["test_samples"] == 14
    assert len(prov["manifest_hashes"]["test_manifest_sha256"]) == 64


def test_kaggle_dynamic_discovery():
    """Verify that inspect_dataset() dynamically discovers CSVs and image directories without hardcoding."""
    from backend.app.data.ingestion.kaggle_insat3d import kaggle_ingestion_pipeline

    stats = kaggle_ingestion_pipeline.inspect_dataset()
    assert stats["csv_rows_count"] == 136
    assert stats["primary_csv_path"] is not None
    assert Path(stats["primary_csv_path"]).exists()
    assert stats["best_image_dir"] is not None
    assert Path(stats["best_image_dir"]).exists()
    assert stats["best_image_dir_matched_count"] == 136
    assert stats["total_images_discovered"] >= 136
    assert len(stats["image_directories_found"]) >= 1
    assert len(stats["discovered_csv_files"]) >= 1



def test_detection_manifests_integrity():
    """Verify manifest CSVs exist, contain required headers, and target real files on disk."""
    for manifest_path, expected_count in [
        (TRAIN_MANIFEST_PATH, 106),
        (VAL_MANIFEST_PATH, 28),
        (TEST_MANIFEST_PATH, 14),
    ]:
        assert manifest_path.exists(), f"Manifest {manifest_path} missing"
        dataset = CycloneImageDataset(csv_path=manifest_path, is_training=False)
        assert len(dataset) == expected_count, (
            f"Expected {expected_count} samples in {manifest_path.name}, got {len(dataset)}"
        )

        # Verify first sample loads successfully
        sample = dataset[0]
        assert "image" in sample
        assert "label" in sample
        assert sample["image"].shape == (3, 256, 256)
        assert sample["label"].item() in (0, 1)


def test_insat3d_preprocessing():
    """Verify INSAT3DPreprocessor handles image loading, normalization, and tensor shapes."""
    preprocessor = INSAT3DPreprocessor(target_size=(256, 256))
    dataset = CycloneImageDataset(csv_path=TEST_MANIFEST_PATH, is_training=False)
    first_path = dataset.samples[0]["file_path"]

    tensor, pil_img, meta = preprocessor.preprocess_single(first_path)
    assert tensor.shape in ((1, 3, 256, 256), (3, 256, 256))
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert pil_img.size == (256, 256)
    assert "cloud_fraction" in meta
    assert 0.0 <= meta["cloud_fraction"] <= 1.0


def test_model_registry_v2():
    """Verify cyclone_detector_cnn_v2 is registered with status TRAINED."""
    registry = ModelRegistry()
    v2_info = registry.get_model("cyclone_detector_cnn_v2")
    assert v2_info is not None
    assert v2_info.version == "2.0.0"
    assert v2_info.training_status == "TRAINED"
    assert "INSAT-3D" in v2_info.description

    # Ensure v1 is also preserved
    v1_info = registry.get_model("cyclone_detector_cnn_v1")
    assert v1_info is not None
    assert v1_info.version == "1.0.0"
    assert v1_info.training_status == "TRAINED"


def test_v2_evaluation_artifacts():
    """Verify that V2 checkpoint, metrics JSON, and comparative Markdown report exist."""
    assert CHECKPOINT_V2_PATH.exists(), f"Missing checkpoint: {CHECKPOINT_V2_PATH}"
    assert METRICS_V2_PATH.exists(), f"Missing metrics: {METRICS_V2_PATH}"
    assert REPORT_V2_PATH.exists(), f"Missing report: {REPORT_V2_PATH}"

    with open(METRICS_V2_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["model_info"]["name"] == "cyclone_detector_cnn_v2"
    assert metrics["test_dataset"]["total_test_samples"] == 14
    assert metrics["metrics"]["accuracy"] == 1.0
    assert metrics["metrics"]["precision"] == 1.0
    assert metrics["metrics"]["recall"] == 1.0
    assert metrics["metrics"]["f1_score"] == 1.0
    assert metrics["confusion_matrix"]["true_positives"] == 13
    assert metrics["confusion_matrix"]["true_negatives"] == 1
    assert metrics["confusion_matrix"]["false_positives"] == 0
    assert metrics["confusion_matrix"]["false_negatives"] == 0


def test_api_dataset_statistics():
    """Test GET /api/ml/detection/dataset/statistics."""
    res = client.get("/api/ml/detection/dataset/statistics")
    assert res.status_code == 200
    data = res.json()
    assert data["total_samples"] == 148
    assert data["cyclone_samples"] == 136
    assert data["no_cyclone_samples"] == 12
    assert data["splits"]["train_samples"] == 106
    assert data["splits"]["val_samples"] == 28
    assert data["splits"]["test_samples"] == 14
    assert "sshubam" in data["kaggle_slug"]


def test_api_dataset_manifest_query():
    """Test GET /api/ml/detection/dataset."""
    res = client.get("/api/ml/detection/dataset?split=test&limit=5")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 14
    assert len(data["samples"]) == 5
    assert data["offset"] == 0
    assert data["limit"] == 5
    assert data["samples"][0]["split"] == "test"


def test_api_model_comparison():
    """Test GET /api/ml/detection/comparison."""
    res = client.get("/api/ml/detection/comparison")
    assert res.status_code == 200
    data = res.json()
    assert "v1" in data
    assert "v2" in data
    assert "delta" in data
    assert data["v1"]["model_info"]["name"] == "cyclone_detector_cnn_v1"
    assert data["v2"]["model_info"]["name"] == "cyclone_detector_cnn_v2"
    assert data["delta"]["accuracy_delta"] == 14.29
    assert data["delta"]["recall_delta"] == 0.25
    assert "CycloneDetectorCNN V2 achieved 100.00% accuracy" in data["comparison_summary"]
    assert "# CYCLONE-X: Tropical Cyclone Detection Benchmark" in data["report_markdown"]


def test_api_metrics_v2_version_switch():
    """Test GET /api/ml/detection/metrics?model_version=v2 vs default."""
    # Default (v1) preserves existing Phase 9 assertion
    r_v1 = client.get("/api/ml/detection/metrics")
    assert r_v1.status_code == 200
    assert r_v1.json()["metrics"]["accuracy_percentage"] == 85.71
    assert r_v1.json()["test_dataset"]["total_test_samples"] == 7

    # Explicit v2 query
    r_v2 = client.get("/api/ml/detection/metrics?model_version=v2")
    assert r_v2.status_code == 200
    assert r_v2.json()["metrics"]["accuracy_percentage"] == 100.0
    assert r_v2.json()["test_dataset"]["total_test_samples"] == 14
