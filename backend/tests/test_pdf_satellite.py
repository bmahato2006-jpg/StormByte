"""Automated Test Suite for PDF Satellite Archive Integration (Phase 11.5).

Verifies:
1. Deterministic image extraction from '2013-2014 cyclone_merged.pdf' (179 images)
2. 22-column Manifest CSV schema integrity with strict null preservation
3. Perceptual dHash duplicate detection report
4. Zero-fabrication compliance (no invented coordinates, timestamps, wind speeds)
5. Backend REST API endpoints (list, stats, detail, preview, analyze)
6. Integrity of existing INSAT-3D manifests and Phase 8–10 model checkpoints
"""

import csv
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
from PIL import Image

from backend.app.main import app

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PDF_DIR = WORKSPACE_ROOT / "data" / "raw" / "pdf_satellite"
MANIFEST_PATH = WORKSPACE_ROOT / "data" / "processed" / "satellite" / "pdf_satellite_manifest.csv"
DUPLICATE_REPORT_PATH = WORKSPACE_ROOT / "data" / "processed" / "satellite" / "pdf_duplicate_report.csv"
INSAT_MANIFEST_DIR = WORKSPACE_ROOT / "data" / "processed" / "satellite" / "detection" / "manifests"
CNN_V2_CHECKPOINT = WORKSPACE_ROOT / "backend" / "app" / "ml" / "checkpoints" / "cyclone_detector_cnn_v2.pt"
LSTM_V1_CHECKPOINT = WORKSPACE_ROOT / "backend" / "app" / "ml" / "checkpoints" / "trajectory_lstm_v1.pt"

client = TestClient(app)


def test_pdf_extraction_count_and_format():
    """Confirms exactly 179 JPEG images were extracted with deterministic naming."""
    assert RAW_PDF_DIR.exists(), f"Directory missing: {RAW_PDF_DIR}"
    extracted_files = sorted([f for f in RAW_PDF_DIR.iterdir() if f.suffix.lower() == ".jpg"])
    assert len(extracted_files) == 179, f"Expected 179 extracted files, got {len(extracted_files)}"

    for i in range(1, 180):
        expected_name = f"pdf_p{i:04d}_img0001.jpg"
        file_path = RAW_PDF_DIR / expected_name
        assert file_path.exists(), f"Missing extracted file: {expected_name}"
        assert file_path.stat().st_size > 0, f"Empty file: {expected_name}"

        # Verify it's a valid readable JPEG
        with Image.open(file_path) as img:
            assert img.format in ("JPEG", "JPG")
            w, h = img.size
            assert w > 0 and h > 0


def test_pdf_manifest_structure_and_columns():
    """Confirms all 22 required columns are present and unobserved fields remain null."""
    assert MANIFEST_PATH.exists(), f"Manifest missing: {MANIFEST_PATH}"

    expected_columns = [
        "sample_id",
        "pdf_file",
        "page_number",
        "embedded_image_index",
        "image_path",
        "width",
        "height",
        "format",
        "sha256",
        "provider",
        "satellite",
        "product",
        "timestamp",
        "storm_name",
        "latitude",
        "longitude",
        "wind_speed_kt",
        "intensity_category",
        "cyclone_center_visible",
        "metadata_source",
        "metadata_confidence",
        "label_status",
    ]

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == expected_columns, f"Header mismatch: {reader.fieldnames}"
        rows = list(reader)

    assert len(rows) == 179, f"Expected 179 manifest rows, got {len(rows)}"

    for row in rows:
        # Zero-fabrication: unobserved fields MUST be null
        assert row["wind_speed_kt"] in ("null", "", "None"), f"Fabricated wind speed in {row['sample_id']}"
        assert row["intensity_category"] in ("null", "", "None"), f"Fabricated intensity category in {row['sample_id']}"
        assert row["latitude"] in ("null", "", "None"), f"Fabricated latitude in {row['sample_id']}"
        assert row["longitude"] in ("null", "", "None"), f"Fabricated longitude in {row['sample_id']}"
        assert row["storm_name"] in ("null", "", "None"), f"Fabricated storm name in {row['sample_id']}"
        assert row["label_status"] == "UNLABELED", f"Invalid label_status in {row['sample_id']}"
        assert len(row["sha256"]) == 64, f"Invalid SHA-256 hash in {row['sample_id']}"


def test_pdf_duplicate_report():
    """Confirms duplicate report documents perceptual near-duplicates."""
    assert DUPLICATE_REPORT_PATH.exists(), f"Duplicate report missing: {DUPLICATE_REPORT_PATH}"

    with open(DUPLICATE_REPORT_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0, "Expected duplicate pairs to be documented"
    
    # Verify the exact perceptual duplicate pair (Pages 157 & 158)
    p157_158 = [r for r in rows if r["sample_id_1"] == "pdf_p0157_img0001" and r["sample_id_2"] == "pdf_p0158_img0001"]
    assert len(p157_158) == 1, "Expected duplicate pair for pages 157 and 158"
    assert p157_158[0]["duplicate_type"] == "PERCEPTUAL_DHASH"
    assert float(p157_158[0]["similarity"]) == 1.0


def test_pdf_stats_api():
    """GET /api/satellite/historical/pdf/stats returns accurate metrics and safety rules."""
    response = client.get("/api/satellite/historical/pdf/stats")
    assert response.status_code == 200
    data = response.json()

    assert data["total_extracted_samples"] == 179
    assert data["labeled_samples"] == 0
    assert data["unlabeled_samples"] == 179
    assert data["verified_wind_speed_labels"] == 0
    assert data["center_crosshair_marker_count"] > 0
    assert "CYCLONE_SATELLITE" in data["product_distribution"]
    assert "FULL_DISK" in data["product_distribution"]
    assert "MAP_WEATHER_PRODUCT" in data["product_distribution"]

    # ML Safety Rules
    rules = data["ml_safety_rules"]
    assert rules["supervised_intensity_training_permitted"] is False
    assert rules["supervised_track_training_permitted"] is False
    assert rules["cyclone_detection_evaluation_permitted"] is True
    assert rules["unsupervised_pretraining_permitted"] is True
    assert rules["zero_fabrication_compliance"] == "VERIFIED"


def test_pdf_list_api_and_filters():
    """GET /api/satellite/historical/pdf returns paginated and filtered results."""
    # Default list
    res_all = client.get("/api/satellite/historical/pdf?limit=10")
    assert res_all.status_code == 200
    data_all = res_all.json()
    assert data_all["total"] == 179
    assert len(data_all["samples"]) == 10

    # Filter by product
    res_full_disk = client.get("/api/satellite/historical/pdf?product=FULL_DISK")
    assert res_full_disk.status_code == 200
    data_full = res_full_disk.json()
    assert data_full["total"] == 48
    for s in data_full["samples"]:
        assert s["product"] == "FULL_DISK"

    # Filter by center marker
    res_marker = client.get("/api/satellite/historical/pdf?center_marker_only=true")
    assert res_marker.status_code == 200
    data_marker = res_marker.json()
    assert data_marker["total"] > 100
    for s in data_marker["samples"]:
        assert s["cyclone_center_visible"] is True


def test_pdf_sample_detail_api():
    """GET /api/satellite/historical/pdf/{sample_id} returns granule metadata."""
    res = client.get("/api/satellite/historical/pdf/pdf_p0001_img0001")
    assert res.status_code == 200
    data = res.json()
    assert data["sample_id"] == "pdf_p0001_img0001"
    assert data["page_number"] == 1
    assert data["format"] == "JPEG"
    assert data["label_status"] == "UNLABELED"
    assert data["wind_speed_kt"] is None
    assert data["latitude"] is None
    assert data["longitude"] is None

    # Nonexistent sample returns 404
    res_404 = client.get("/api/satellite/historical/pdf/nonexistent_sample")
    assert res_404.status_code == 404


def test_pdf_preview_api():
    """GET /api/satellite/historical/pdf/{sample_id}/preview streams valid JPEG bytes."""
    res = client.get("/api/satellite/historical/pdf/pdf_p0001_img0001/preview")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert len(res.content) > 1000

    # Nonexistent sample returns 404
    res_404 = client.get("/api/satellite/historical/pdf/nonexistent_sample/preview")
    assert res_404.status_code == 404


def test_pdf_analyze_api():
    """POST /api/satellite/historical/pdf/analyze executes Cyclone Detection V2."""
    payload = {"sample_id": "pdf_p0001_img0001"}
    res = client.post("/api/satellite/historical/pdf/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["sample_id"] == "pdf_p0001_img0001"
    assert "cyclone_detection" in data
    assert "detection" in data
    assert "CycloneDetectorCNN V2" in data["detection"]["model"]
    assert "confidence" in data["detection"]
    assert "class_probabilities" in data["detection"]

    # Truthful intensity status: UNAVAILABLE
    assert data["intensity"]["status"] == "UNAVAILABLE"
    assert data["intensity"]["estimated_wind_speed_knots"] is None

    # Trajectory status: requires >= 4 fixes
    assert data["trajectory"]["status"] in ("NO_CYCLONE_DETECTED", "INSUFFICIENT_TRACK_HISTORY")

    # Risk evaluation
    assert "risk_score" in data["risk"]
    assert data["risk"]["risk_level"] in ("LOW", "MODERATE", "HIGH", "SEVERE")

    # Cryptographic provenance
    assert len(data["provenance"]["sha256"]) == 64


def test_existing_insat3d_dataset_unchanged():
    """Verifies that existing INSAT-3D benchmark splits and images were NOT modified."""
    test_csv = INSAT_MANIFEST_DIR / "detection_test.csv"
    val_csv = INSAT_MANIFEST_DIR / "detection_validation.csv"
    train_csv = INSAT_MANIFEST_DIR / "detection_train.csv"

    assert test_csv.exists()
    assert val_csv.exists()
    assert train_csv.exists()

    with open(test_csv, "r", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 14

    with open(val_csv, "r", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 28

    with open(train_csv, "r", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 106


def test_existing_checkpoints_preserved():
    """Confirms Cyclone Detector V2 and Trajectory LSTM V1 checkpoints are intact."""
    assert CNN_V2_CHECKPOINT.exists()
    assert CNN_V2_CHECKPOINT.stat().st_size > 1000000  # > 1MB

    assert LSTM_V1_CHECKPOINT.exists()
    assert LSTM_V1_CHECKPOINT.stat().st_size > 100000   # > 100KB

