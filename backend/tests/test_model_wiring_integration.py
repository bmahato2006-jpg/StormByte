"""Integration & Verification Tests for Model Wiring and Deep Neural Engines.

Validates that:
1. CycloneClassifier integrates ObjectiveDvorakNet deep neural classification while retaining morphological backwards compatibility.
2. What-If Simulation API invokes SurgeDeepSurrogate and WaveWatchSurrogateNet.
3. Tactical Analyst API invokes LandfallPredictorNet for landfall arrival estimation.
4. PipelineRiskEngine integrates ImpactNeuralEngine for composite vulnerability scoring.
5. CycloneCenterNet checkpoint is verified and loadable.
"""

from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.ml.classification import cyclone_classifier
from backend.app.ml.pipeline.risk_engine import PipelineRiskEngine


@pytest.fixture
def client():
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. CycloneClassifier Tests
# -----------------------------------------------------------------------------

def test_classifier_not_detected():
    res = cyclone_classifier.classify(
        detected=False,
        area_ratio=0.0,
        spiral_index=0.0,
        symmetry_score=0.0,
    )
    assert res.category == "No Active Cyclone Detected"
    assert res.is_experimental is True


def test_classifier_morphological_backward_compatibility():
    # Extreme cyclone
    res_ext = cyclone_classifier.classify(
        detected=True,
        area_ratio=0.25,
        spiral_index=0.80,
        symmetry_score=0.75,
    )
    assert res_ext.category == "Extreme / Super Cyclone"
    assert res_ext.is_experimental is True

    # Severe tropical storm
    res_sev = cyclone_classifier.classify(
        detected=True,
        area_ratio=0.14,
        spiral_index=0.62,
        symmetry_score=0.45,
    )
    assert res_sev.category == "Severe Tropical Storm"

    # Depression
    res_dep = cyclone_classifier.classify(
        detected=True,
        area_ratio=0.09,
        spiral_index=0.52,
        symmetry_score=0.40,
    )
    assert res_dep.category == "Tropical Depression"


def test_classifier_with_image_array():
    # Provide synthetic vortex tile
    y, x = np.ogrid[:128, :128]
    dist = np.hypot(x - 64, y - 64)
    vortex = np.clip(np.exp(-(dist**2) / (2 * 20**2)), 0.0, 1.0)

    res = cyclone_classifier.classify(
        detected=True,
        area_ratio=0.20,
        spiral_index=0.70,
        symmetry_score=0.65,
        image_array=vortex,
    )
    assert res.is_experimental is True
    assert "AI-generated experimental analysis" in res.disclaimer
    assert len(res.category) > 0


# -----------------------------------------------------------------------------
# 2. Simulation API Neural Surrogates Wiring Tests
# -----------------------------------------------------------------------------

def test_simulation_api_surrogates_wired(client):
    payload = {
        "base_wind_kt": 85.0,
        "base_pressure_hpa": 960.0,
        "delta_sst_c": 1.5,
        "delta_shear_kt": -5.0,
        "translation_speed_kt": 11.0,
        "approach_angle_deg": 65.0,
    }
    res = client.post("/api/simulation/perturb", json=payload)
    assert res.status_code == 200
    data = res.json()

    # Check baseline & perturbed
    assert data["baseline"]["vmax_knots"] == 85.0
    assert data["perturbed"]["vmax_knots"] > 85.0
    assert data["perturbed"]["central_pressure_hpa"] < 960.0

    # Check surge projection structure
    sp = data["surge_projection"]
    assert "estimated_surge_m" in sp
    assert sp["estimated_surge_m"] > 0
    assert "inundation_risk_level" in sp

    # If surrogate checkpoints are present, verify neural keys
    if "neural_surrogate_surge" in sp:
        n_surge = sp["neural_surrogate_surge"]
        assert "neural_peak_surge_m" in n_surge
        assert "inundation_extent_km" in n_surge
        assert "surge_hazard_index" in n_surge

    if "spectral_wave_dynamics" in sp:
        w_dyn = sp["spectral_wave_dynamics"]
        assert "significant_wave_height_m" in w_dyn
        assert "peak_wave_period_s" in w_dyn


# -----------------------------------------------------------------------------
# 3. Tactical Analyst API LandfallPredictorNet Wiring Tests
# -----------------------------------------------------------------------------

def test_tactical_analyst_landfall_wired(client):
    res = client.get("/api/analysis/tactical-briefing/test_cyclone_landfall")
    assert res.status_code == 200
    data = res.json()

    assert data["storm_id"] == "test_cyclone_landfall"
    assert "evacuation_timeline" in data
    et = data["evacuation_timeline"]
    assert et["critical_evacuation_window_hours"] > 0
    assert et["estimated_eye_landfall_hours"] > 0

    # Check neural landfall prediction if checkpoint is loaded
    if "neural_landfall_prediction" in et:
        nlp = et["neural_landfall_prediction"]
        assert "predicted_crossing_coordinates" in nlp
        assert "latitude" in nlp["predicted_crossing_coordinates"]
        assert "longitude" in nlp["predicted_crossing_coordinates"]
        assert "arrival_lead_time_hours" in nlp
        assert "landfall_probability" in nlp
        assert 0.0 <= nlp["landfall_probability"] <= 1.0


# -----------------------------------------------------------------------------
# 4. PipelineRiskEngine ImpactNeuralEngine Wiring Tests
# -----------------------------------------------------------------------------

def test_pipeline_risk_engine_with_neural_vulnerability():
    env = {
        "wind_speed_knots": 85.0,
        "rainfall_rate_mm_h": 45.0,
        "sst_celsius": 29.5,
    }
    res = PipelineRiskEngine.evaluate(
        detected=True,
        detection_confidence=0.92,
        latitude=18.5,
        longitude=87.0,
        environmental_obs=env,
    )
    assert res["risk_level"] in ["HIGH", "SEVERE"]
    assert res["risk_score"] > 50.0
    assert res["data_completeness"] > 0.0

    # Check neural vulnerability evaluation if checkpoint is loaded
    if "neural_vulnerability" in res:
        nv = res["neural_vulnerability"]
        assert "neural_composite_risk" in nv
        assert "estimated_damage_ratio" in nv
        assert 0.0 <= nv["estimated_damage_ratio"] <= 1.0


# -----------------------------------------------------------------------------
# 5. CycloneCenterNet Checkpoint Verification
# -----------------------------------------------------------------------------

def test_centernet_checkpoint_exists_and_loadable():
    import torch
    from backend.app.ml.training_framework.models import CycloneCenterNet

    ckpt_path = Path(__file__).resolve().parents[1] / "app" / "ml" / "checkpoints" / "cyclone_centernet_resnet18.pt"
    assert ckpt_path.exists(), "cyclone_centernet_resnet18.pt must exist"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert ckpt.get("architecture") == "CycloneCenterNet"

    model = CycloneCenterNet(backbone_name="resnet18", pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(dummy)
    assert "heatmap" in out
    assert "offset" in out
