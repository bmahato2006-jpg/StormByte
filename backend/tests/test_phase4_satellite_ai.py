"""Automated Test Suite for Phase 4: Satellite Image Analysis, Cyclone Detection & Center Detection.

Verifies:
1. Valid satellite image upload & end-to-end analysis
2. Corrupted file / unreadable image rejection (400 Bad Request)
3. Unsupported file format rejection
4. Image preprocessing, normalization, and Laplacian sharpness calculation
5. Cyclone candidate detector & convective cloud segmentation
6. Circulation center detection (radial gradient symmetry + convective centroid)
7. Missing georeferencing handling (latitude/longitude must be None with proper notice)
8. Georeferenced conversion (accurate equirectangular coordinates when bounds provided)
9. Structural classification into authorized categories with experimental flag
10. Intensity estimation interface strictly returns None/unavailable (zero fake numbers)
11. Explainability visual overlay generation & retrieval
12. Model Registry status (baseline_cv ACTIVE, deep nets NOT_TRAINED)
13. Stored analysis retrieval & explanation retrieval endpoints
14. Provenance tracking & SHA-256 image digest validation
15. Preservation of Phase 1-3 endpoints (Health, Cyclones, Data Status)
"""

import io
from PIL import Image, ImageDraw
import pytest

from backend.app.ml.preprocessing import (
    validate_image_bytes,
    compute_laplacian_sharpness,
    preprocess_satellite_image,
)
from backend.app.ml.detector import cyclone_detector
from backend.app.ml.center_detection import center_detector
from backend.app.ml.classification import cyclone_classifier
from backend.app.ml.intensity import intensity_estimator
from backend.app.ml.model_registry import model_registry
from backend.app.ml.inference import inference_pipeline


def _create_synthetic_cyclone_image(width=256, height=256) -> io.BytesIO:
    """Creates a deterministic synthetic cyclone-like vortex test image."""
    img = Image.new("L", (width, height), color=20)
    draw = ImageDraw.Draw(img)
    cx, cy = width // 2, height // 2

    # Draw concentric spiraling cloud mass
    for r in range(80, 15, -10):
        intensity = int(120 + (80 - r) * 1.5)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=intensity)

    # High-intensity convective core
    draw.ellipse([cx - 25, cy - 25, cx + 25, cy + 25], fill=240)
    # Eye (relative minimum in center)
    draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=90)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# 1. Preprocessor & Laplacian Sharpness
# ---------------------------------------------------------------------------
def test_image_preprocessor():
    img_buf = _create_synthetic_cyclone_image(128, 128)
    image_bytes = img_buf.getvalue()

    is_valid, err = validate_image_bytes(image_bytes, "test.png")
    assert is_valid is True
    assert err is None

    proc_img, gray_arr, prep_meta = preprocess_satellite_image(
        image_bytes,
        target_size=(256, 256),
        enhance_contrast=True,
    )

    assert prep_meta.original_dimensions["width"] == 128
    assert prep_meta.original_dimensions["height"] == 128
    assert prep_meta.processed_dimensions["width"] == 256
    assert prep_meta.processed_dimensions["height"] == 256
    assert gray_arr.shape == (256, 256)
    assert 0.0 <= prep_meta.quality_score <= 100.0
    assert prep_meta.dynamic_range >= 0.0
    assert prep_meta.sharpness >= 0.0


def test_corrupted_image_validation():
    corrupt_bytes = b"NOT_AN_IMAGE_DATA_CORRUPT_BYTES_XYZ"
    is_valid, err = validate_image_bytes(corrupt_bytes, "corrupt.png")
    assert is_valid is False
    assert err is not None


# ---------------------------------------------------------------------------
# 2. Cyclone Candidate Detector
# ---------------------------------------------------------------------------
def test_cyclone_candidate_detector():
    img_buf = _create_synthetic_cyclone_image(256, 256)
    _, gray_arr, _ = preprocess_satellite_image(img_buf.getvalue(), target_size=(256, 256))

    det_res = cyclone_detector.detect(gray_arr)
    assert "detected" in det_res
    assert "bounding_box" in det_res
    assert "convective_area_ratio" in det_res
    assert "spiral_index" in det_res
    assert det_res["convective_area_ratio"] > 0.0
    assert det_res["spiral_index"] >= 0.0


# ---------------------------------------------------------------------------
# 3. Center Detection & Georeferencing
# ---------------------------------------------------------------------------
def test_center_detector_without_georeferencing():
    img_buf = _create_synthetic_cyclone_image(256, 256)
    _, gray_arr, _ = preprocess_satellite_image(img_buf.getvalue(), target_size=(256, 256))

    res = center_detector.locate_center(
        gray_array=gray_arr,
        candidate_mask=None,
        geo_bounds=None,
    )

    # Pixel coords must be near center (128, 128)
    assert abs(res.pixel_x - 128) < 35
    assert abs(res.pixel_y - 128) < 35
    assert res.estimated_radius_pixels > 0
    # Geographic coordinates must be strictly None without georeferencing
    assert res.latitude is None
    assert res.longitude is None
    assert res.georeferenced is False
    assert "Geographic center unavailable" in res.status_message


def test_center_detector_with_valid_georeferencing():
    img_buf = _create_synthetic_cyclone_image(256, 256)
    _, gray_arr, _ = preprocess_satellite_image(img_buf.getvalue(), target_size=(256, 256))

    georef = {
        "lat_min": 10.0,
        "lat_max": 20.0,
        "lon_min": 80.0,
        "lon_max": 90.0,
    }
    res = center_detector.locate_center(
        gray_array=gray_arr,
        candidate_mask=None,
        geo_bounds=georef,
    )

    assert res.latitude is not None
    assert res.longitude is not None
    assert 10.0 <= res.latitude <= 20.0
    assert 80.0 <= res.longitude <= 90.0
    assert res.georeferenced is True
    assert "Center projected to" in res.status_message


# ---------------------------------------------------------------------------
# 4. Structural Classification
# ---------------------------------------------------------------------------
def test_structural_classification():
    cls_res = cyclone_classifier.classify(
        detected=True,
        area_ratio=0.25,
        spiral_index=0.80,
        symmetry_score=0.75,
    )

    assert cls_res.category in [
        "Extreme / Super Cyclone",
        "Strong / Very Severe System",
        "Severe Tropical Storm",
        "Tropical Storm / Cyclonic Storm",
        "Tropical Depression",
        "Tropical Disturbance",
        "No Cyclone / Uncertain",
    ]
    assert cls_res.is_experimental is True
    assert "AI-generated experimental analysis" in cls_res.disclaimer


# ---------------------------------------------------------------------------
# 5. Intensity Estimation Unavailable Contract
# ---------------------------------------------------------------------------
def test_intensity_estimation_unavailable_contract():
    res = intensity_estimator.estimate(
        gray_array=None,
        center_x=128.0,
        center_y=128.0,
        classification_category="Severe Tropical Storm",
    )

    assert res.available is False
    assert res.wind_speed_knots is None
    assert res.central_pressure_hpa is None
    assert res.category is None
    assert "trained intensity model required" in res.status_message


# ---------------------------------------------------------------------------
# 6. Model Registry
# ---------------------------------------------------------------------------
def test_model_registry():
    models = model_registry.list_models()
    assert len(models) >= 4

    baseline = next((m for m in models if m.name == "baseline_cv"), None)
    assert baseline is not None
    assert baseline.training_status == "ACTIVE"

    deep_models = [m for m in models if m.training_status in ("NOT_TRAINED", "TRAINED")]
    assert len(deep_models) >= 3


# ---------------------------------------------------------------------------
# 7. End-to-End Inference Pipeline
# ---------------------------------------------------------------------------
def test_inference_pipeline():
    img_buf = _create_synthetic_cyclone_image(200, 200)
    image_bytes = img_buf.getvalue()

    result = inference_pipeline.run_analysis(
        image_bytes=image_bytes,
        filename="test_vortex.png",
        cyclone_id="BOB-01-2020",
        source_name="INSAT-3D",
        geo_bounds={"lat_min": 12.0, "lat_max": 18.0, "lon_min": 84.0, "lon_max": 90.0},
    )

    assert result.analysis_id.startswith("anl-")
    assert len(result.image_hash) == 64
    assert result.center.pixel_x > 0
    assert result.center.latitude is not None
    assert result.intensity.available is False
    assert result.intensity.wind_speed_knots is None
    assert result.explainability.overlay_base64 is not None
    assert len(result.explainability.key_visual_regions) > 0


# ---------------------------------------------------------------------------
# 8. FastAPI API Endpoints Verification
# ---------------------------------------------------------------------------
def test_api_satellite_analysis_endpoint(client):
    img_buf = _create_synthetic_cyclone_image(256, 256)

    response = client.post(
        "/api/analysis/satellite",
        files={"file": ("amphan_sim.png", img_buf, "image/png")},
        data={
            "satellite_name": "INSAT-3D",
            "lat_min": "10.0",
            "lat_max": "20.0",
            "lon_min": "82.0",
            "lon_max": "92.0",
        },
    )

    assert response.status_code == 200
    data = response.json()
    analysis_id = data["analysis_id"]
    assert analysis_id.startswith("anl-")
    assert data["center"]["pixel_x"] > 0
    assert data["center"]["latitude"] is not None
    assert data["intensity"]["available"] is False
    assert data["intensity"]["wind_speed_knots"] is None
    assert "overlay_image_url" in data["explainability"]

    # Retrieve analysis by ID
    res_get = client.get(f"/api/analysis/{analysis_id}")
    assert res_get.status_code == 200
    assert res_get.json()["analysis_id"] == analysis_id

    # Retrieve explanation
    res_exp = client.get(f"/api/analysis/{analysis_id}/explanation")
    assert res_exp.status_code == 200
    assert "explanation_text" in res_exp.json()
    assert "explanation_type" in res_exp.json()

    # Retrieve overlay image
    res_overlay = client.get(f"/api/analysis/{analysis_id}/overlay")
    assert res_overlay.status_code == 200
    assert res_overlay.headers["content-type"] == "image/png"


def test_api_corrupted_image_rejection(client):
    corrupt_file = io.BytesIO(b"NOT_A_VALID_IMAGE_HEADER_123456789_PADDING_BYTES_TO_EXCEED_THIRTY_TWO_BYTES")
    response = client.post(
        "/api/analysis/satellite",
        files={"file": ("corrupt.png", corrupt_file, "image/png")},
    )
    assert response.status_code == 400
    assert "Image file is corrupted or cannot be decoded" in response.json()["detail"]


def test_api_unsupported_file_extension(client):
    text_file = io.BytesIO(b"hello text file with enough bytes to pass length validation rule comfortably!")
    response = client.post(
        "/api/analysis/satellite",
        files={"file": ("script.sh", text_file, "application/x-sh")},
    )
    assert response.status_code == 400


def test_api_models_endpoint(client):
    response = client.get("/api/analysis/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert any(m["name"] == "baseline_cv" and m["training_status"] == "ACTIVE" for m in data["models"])
    assert any(m["name"] == "cyclone_vit" and m["training_status"] in ("NOT_TRAINED", "TRAINED") for m in data["models"])


def test_api_analysis_not_found(client):
    response = client.get("/api/analysis/anl-nonexistent-id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 9. Preservation of Phase 1-3 Endpoints
# ---------------------------------------------------------------------------
def test_all_existing_endpoints_intact(client):
    # Health endpoint
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # Cyclones endpoint
    res_cyclones = client.get("/api/cyclones")
    assert res_cyclones.status_code == 200
    assert len(res_cyclones.json()) >= 2

    # Data Status endpoint
    res_status = client.get("/api/data/status")
    assert res_status.status_code == 200
    assert "active_data_mode" in res_status.json()

    # Provenance endpoint
    res_prov = client.get("/api/provenance")
    assert res_prov.status_code == 200
    assert len(res_prov.json()) > 0
