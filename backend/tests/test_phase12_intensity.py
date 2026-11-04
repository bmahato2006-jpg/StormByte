"""Test Suite for Phase 12: Pure PyTorch ResNet-18 Tropical Cyclone Intensity Estimation.

Verifies:
1. CycloneIntensityResNet18 tensor forward pass and layer dimensionalities.
2. Atkinson-Holliday wind-to-pressure physical calibration for North Indian Ocean cyclones.
3. Checkpoint persistence and metadata loading on disk.
4. Inference service prediction on verified INSAT-3D thermal IR imagery.
5. Zero-fabrication behavior on non-cyclonic images or when inputs are missing.
6. FastAPI endpoints: GET /status, GET /metrics, POST /predict.
7. Pipeline orchestration integration: analyze() includes intensity payload.
"""

import io
from pathlib import Path
from PIL import Image
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.ml.intensity.config import (
    CHECKPOINT_PATH,
    METADATA_PATH,
    knots_to_central_pressure,
    knots_to_imd_category,
)
from backend.app.ml.intensity.inference import intensity_inference_service
from backend.app.ml.intensity.model import CycloneIntensityResNet18
from backend.app.ml.pipeline.service import pipeline_service

client = TestClient(app)

SAMPLE_CYCLONE_IMAGE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "kaggle"
    / "insat3d"
    / "insat3d_ir_cyclone_ds"
    / "CYCLONE_DATASET_INFRARED"
    / "64(1).jpg"
)


# 1. Model Architecture Tests
def test_resnet18_architecture_forward():
    model = CycloneIntensityResNet18()
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()


def test_resnet18_parameter_count():
    model = CycloneIntensityResNet18()
    total_params = sum(p.numel() for p in model.parameters())
    assert 11_000_000 < total_params < 12_000_000  # Standard ResNet-18 ~11.24M params


# 2. Meteorological Physical Calibration Tests
def test_atkinson_holliday_pressure_calibration():
    # 30 kt Depression -> ~1000 hPa
    p_dep = knots_to_central_pressure(30.0)
    assert 995.0 <= p_dep <= 1005.0

    # 64 kt Severe / Very Severe Cyclone -> ~975-985 hPa
    p_scs = knots_to_central_pressure(64.0)
    assert 970.0 <= p_scs <= 985.0

    # 120 kt Super Cyclone -> ~915-930 hPa
    p_supr = knots_to_central_pressure(120.0)
    assert 910.0 <= p_supr <= 930.0

    # Monotonicity: higher wind -> lower central pressure
    assert knots_to_central_pressure(120.0) < knots_to_central_pressure(64.0) < knots_to_central_pressure(30.0)


def test_imd_category_mapping():
    assert knots_to_imd_category(25.0) == "Depression"
    assert knots_to_imd_category(30.0) == "Deep Depression"
    assert knots_to_imd_category(40.0) == "Cyclonic Storm"
    assert knots_to_imd_category(55.0) == "Severe Cyclonic Storm"
    assert knots_to_imd_category(75.0) == "Very Severe Cyclonic Storm"
    assert knots_to_imd_category(100.0) == "Extremely Severe Cyclonic Storm"
    assert knots_to_imd_category(130.0) == "Super Cyclonic Storm"


# 3. Checkpoint & Metadata Verification
def test_intensity_checkpoint_exists():
    assert CHECKPOINT_PATH.exists(), f"Missing intensity checkpoint at {CHECKPOINT_PATH}"
    assert METADATA_PATH.exists(), f"Missing intensity metadata at {METADATA_PATH}"
    assert intensity_inference_service.is_available() is True


# 4. Inference on Real Historical INSAT-3D Image
@pytest.mark.skipif(not SAMPLE_CYCLONE_IMAGE.exists(), reason="Sample INSAT-3D image not available")
def test_intensity_inference_real_image():
    res = intensity_inference_service.predict(SAMPLE_CYCLONE_IMAGE)
    assert res["available"] is True
    assert 20.0 <= res["wind_speed_knots"] <= 120.0
    assert 900.0 <= res["central_pressure_hpa"] <= 1010.0
    assert res["category"] in (
        "Depression", "Deep Depression", "Cyclonic Storm",
        "Severe Cyclonic Storm", "Very Severe Cyclonic Storm",
        "Extremely Severe Cyclonic Storm", "Super Cyclonic Storm",
    )
    unc = res["uncertainty_range"]
    assert "confidence_interval_95" in unc
    assert unc["confidence_interval_95"][0] <= res["wind_speed_knots"] <= unc["confidence_interval_95"][1]


# 5. Zero Fabrication Contract
def test_zero_fabrication_non_cyclonic():
    non_cyc_img = Image.new("RGB", (256, 256), color=(30, 30, 30))
    res = pipeline_service.analyze(image_input=non_cyc_img, data_mode="UPLOAD")
    assert res["detected"] is False
    assert res["intensity"]["available"] is False
    assert "inactive" in res["intensity"]["status_message"].lower()


# 6. FastAPI Endpoints Verification
def test_api_intensity_status():
    r = client.get("/api/ml/intensity/status")
    assert r.status_code == 200
    data = r.json()
    assert data["model_name"] == "cyclone_intensity_resnet18_v1"
    assert data["is_trained"] is True
    assert data["status"] == "OPERATIONAL_TRAINED"
    assert data["metrics_available"] is True
    assert data["test_metrics"]["mae_knots"] > 0


def test_api_intensity_metrics():
    r = client.get("/api/ml/intensity/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "test_metrics" in data
    assert "mae_knots" in data["test_metrics"]
    assert "rmse_knots" in data["test_metrics"]
    assert "slice_breakdown_by_category" in data["test_metrics"]


@pytest.mark.skipif(not SAMPLE_CYCLONE_IMAGE.exists(), reason="Sample INSAT-3D image not available")
def test_api_intensity_predict_upload():
    with open(SAMPLE_CYCLONE_IMAGE, "rb") as f:
        files = {"file": ("cyclone.jpg", f.read(), "image/jpeg")}
    r = client.post("/api/ml/intensity/predict", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["available"] is True
    assert data["intensity"] is not None
    assert data["intensity"]["vmax_knots"] > 0
    assert data["intensity"]["central_pressure_hpa"] < 1010.0
    assert data["intensity"]["confidence_score"] > 0.0


# 7. End-to-End Pipeline Integration
@pytest.mark.skipif(not SAMPLE_CYCLONE_IMAGE.exists(), reason="Sample INSAT-3D image not available")
def test_pipeline_analyze_includes_intensity():
    with open(SAMPLE_CYCLONE_IMAGE, "rb") as f:
        img_bytes = f.read()
    res = pipeline_service.analyze(image_input=img_bytes, data_mode="HISTORICAL")
    assert res["detected"] is True
    assert "intensity" in res
    assert res["intensity"]["available"] is True
    assert res["intensity"]["vmax_knots"] > 0
    assert res["intensity"]["central_pressure_hpa"] > 0
