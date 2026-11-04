"""Unit & Integration Tests for Module 23 (Simulation) and Module 24/25 (Analyst) APIs."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_simulation_baseline(client):
    """Test perturbation simulator with zero anomalies matches baseline."""
    payload = {
        "base_wind_kt": 80.0,
        "base_pressure_hpa": 965.0,
        "delta_sst_c": 0.0,
        "delta_shear_kt": 0.0,
        "translation_speed_kt": 12.0,
        "approach_angle_deg": 60.0,
    }
    res = client.post("/api/simulation/perturb", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["baseline"]["vmax_knots"] == 80.0
    assert data["perturbed"]["vmax_knots"] == 80.0
    assert data["impact_deltas"]["delta_vmax_knots"] == 0.0
    assert data["rainfall_amplification"]["accumulation_multiplier"] == 1.0
    assert "COUNTERFACTUAL SIMULATION" in data["provenance"]["disclaimer"]


def test_simulation_marine_heatwave_amplification(client):
    """Test that positive SST anomaly amplifies Kerry Emanuel MPI intensity."""
    payload = {
        "base_wind_kt": 75.0,
        "base_pressure_hpa": 970.0,
        "delta_sst_c": 2.0,
        "delta_shear_kt": 0.0,
        "translation_speed_kt": 10.0,
        "approach_angle_deg": 75.0,
    }
    res = client.post("/api/simulation/perturb", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["perturbed"]["vmax_knots"] > data["baseline"]["vmax_knots"]
    assert data["perturbed"]["central_pressure_hpa"] < data["baseline"]["central_pressure_hpa"]
    assert data["impact_deltas"]["delta_vmax_knots"] > 10.0


def test_simulation_translation_stall_hydrology(client):
    """Test that severe forward translation deceleration scales rain accumulation factor."""
    payload = {
        "base_wind_kt": 70.0,
        "base_pressure_hpa": 975.0,
        "delta_sst_c": 0.0,
        "delta_shear_kt": 0.0,
        "translation_speed_kt": 3.0,
        "approach_angle_deg": 45.0,
    }
    res = client.post("/api/simulation/perturb", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["rainfall_amplification"]["accumulation_multiplier"] == 4.0
    assert data["rainfall_amplification"]["risk_classification"] == "CATASTROPHIC_STALL_INUNDATION"


def test_tactical_briefing_endpoint(client):
    """Test AI analyst tactical decision-support endpoint."""
    res = client.get("/api/analysis/tactical-briefing/test_storm_nio")
    assert res.status_code == 200
    data = res.json()

    assert data["storm_id"] == "test_storm_nio"
    assert "current_status" in data
    assert data["current_status"]["vmax_knots"] > 0
    assert len(data["port_warning_signals"]) >= 2
    for ps in data["port_warning_signals"]:
        assert "port_name" in ps
        assert "signal_number" in ps
        assert ps["signal_number"] in [1, 2, 3, 4, 8, 9, 10, 11]

    assert "evacuation_timeline" in data
    assert data["evacuation_timeline"]["critical_evacuation_window_hours"] > 0
    assert "34 kt" in data["evacuation_timeline"]["zero_hour_threshold"]

    assert len(data["critical_infrastructure"]) > 0
    assert "executive_summary" in data
    assert len(data["executive_summary"]["tactical_actions"]) >= 2

    assert "EXPERIMENTAL DECISION-SUPPORT" in data["provenance"]["disclaimer"]
    assert "NOT AN OFFICIAL MANDATE" in data["provenance"]["disclaimer"]
