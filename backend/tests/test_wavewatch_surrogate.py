"""Unit and integration test suite for WaveWatchSurrogateNet (Coupled WAVEWATCH III / SWAN ocean wave surrogate)."""

import json
from pathlib import Path
import pytest
import torch

from backend.app.ml.training_framework.extended_models import WaveWatchSurrogateNet
from backend.app.ml.model_registry import model_registry
from backend.app.impact.coastal_risk import coastal_risk_analyzer


CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "app" / "ml" / "checkpoints"


def test_wavewatch_surrogate_model_init_and_forward():
    """Verify WaveWatchSurrogateNet architecture instantiation and output shapes."""
    model = WaveWatchSurrogateNet(input_dim=7, hidden_dim=64)
    model.eval()

    # Inputs: [wind_speed_kt, delta_p_hpa, translation_speed_kt, distance_to_coast_km, water_depth_m, rmw_km, rel_quadrant_azimuth_rad]
    dummy_input = torch.tensor([
        [65.0, 45.0, 12.0, 100.0, 80.0, 35.0, 0.785],
        [110.0, 80.0, 15.0, 30.0, 20.0, 25.0, 0.785],
    ], dtype=torch.float32)

    with torch.no_grad():
        out = model(dummy_input)

    assert "significant_wave_height_m" in out
    assert "peak_wave_period_s" in out
    assert "mean_wave_period_s" in out
    assert "wave_setup_meters" in out
    assert "wave_energy_flux_kw_m" in out
    assert "breaking_danger_index" in out

    assert out["significant_wave_height_m"].shape == (2,)
    assert out["peak_wave_period_s"].shape == (2,)
    assert out["mean_wave_period_s"].shape == (2,)
    assert out["wave_setup_meters"].shape == (2,)
    assert out["wave_energy_flux_kw_m"].shape == (2,)
    assert out["breaking_danger_index"].shape == (2,)

    # Physical lower bound assertions
    assert (out["significant_wave_height_m"] >= 0.5).all()
    assert (out["peak_wave_period_s"] >= 4.0).all()
    assert (out["mean_wave_period_s"] >= 3.0).all()
    assert (out["wave_setup_meters"] >= 0.0).all()
    assert (out["breaking_danger_index"] >= 0.0).all()
    assert (out["breaking_danger_index"] <= 100.0).all()


def test_wavewatch_checkpoint_exists_and_loads():
    """Verify wavewatch_deep_surrogate.pt checkpoint exists and loads cleanly."""
    ckpt_path = CHECKPOINT_DIR / "wavewatch_deep_surrogate.pt"
    meta_path = CHECKPOINT_DIR / "wavewatch_deep_surrogate_metadata.json"

    assert ckpt_path.exists(), "wavewatch_deep_surrogate.pt checkpoint is missing"
    assert meta_path.exists(), "wavewatch_deep_surrogate_metadata.json is missing"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    assert meta.get("status") == "TRAINED"
    assert meta.get("model_name") == "wavewatch_deep_surrogate"

    model = WaveWatchSurrogateNet(input_dim=7, hidden_dim=64)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in ckpt
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_inp = torch.tensor([[85.0, 55.0, 14.0, 50.0, 45.0, 30.0, 0.785]], dtype=torch.float32)
    with torch.no_grad():
        out = model(test_inp)

    hs = float(out["significant_wave_height_m"][0].item())
    tp = float(out["peak_wave_period_s"][0].item())
    assert 2.0 <= hs <= 20.0, f"Expected realistic cyclone wave height, got {hs:.2f} m"
    assert 4.0 <= tp <= 22.0, f"Expected realistic peak period, got {tp:.2f} s"


def test_wavewatch_physical_consistency():
    """Verify that stronger winds generate higher significant wave height."""
    ckpt_path = CHECKPOINT_DIR / "wavewatch_deep_surrogate.pt"
    model = WaveWatchSurrogateNet(input_dim=7, hidden_dim=64)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Low wind vs High wind in deep water
    low_wind = torch.tensor([[35.0, 15.0, 10.0, 200.0, 500.0, 40.0, 0.785]], dtype=torch.float32)
    high_wind = torch.tensor([[120.0, 85.0, 15.0, 200.0, 500.0, 25.0, 0.785]], dtype=torch.float32)

    with torch.no_grad():
        out_low = model(low_wind)
        out_high = model(high_wind)

    hs_low = float(out_low["significant_wave_height_m"][0].item())
    hs_high = float(out_high["significant_wave_height_m"][0].item())

    assert hs_high > hs_low, f"Expected higher wave for stronger cyclone: low={hs_low:.2f}m, high={hs_high:.2f}m"


def test_coastal_risk_analyzer_wave_integration():
    """Verify CoastalRiskAnalyzer incorporates WaveWatch surrogate outputs."""
    result = coastal_risk_analyzer.evaluate(
        central_pressure_hpa=940.0,
        max_wind_knots=95.0,
        distance_to_coast_km=45.0,
    )

    assert result.available is True
    assert "wave_dynamics" in result.metrics
    waves = result.metrics["wave_dynamics"]
    assert waves is not None
    assert "significant_wave_height_m" in waves
    assert "peak_wave_period_s" in waves
    assert "wave_setup_radiation_stress_m" in waves
    assert "wave_energy_flux_kw_m" in waves
    assert "breaking_danger_index" in waves
    assert waves["significant_wave_height_m"] > 1.0


def test_model_registry_wavewatch_registration():
    """Verify wavewatch_deep_surrogate is registered and marked TRAINED in model registry."""
    model_info = model_registry.get_model("wavewatch_deep_surrogate")
    assert model_info is not None
    assert model_info.training_status == "TRAINED"
    assert "WAVEWATCH III" in model_info.description or "Spectral Wave" in model_info.framework


def test_impact_api_catalog_includes_wavewatch(client):
    """Verify /api/impact/models includes wavewatch_deep_surrogate."""
    resp = client.get("/api/impact/models")
    assert resp.status_code == 200
    data = resp.json()
    model_names = [m["name"] for m in data["models"]]
    assert "wavewatch_deep_surrogate" in model_names
