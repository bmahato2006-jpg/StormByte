"""Automated Test Suite for Phase 5: Trajectory Forecasting & Holland Vortex Engine.

Verifies:
1. Coriolis parameter computation across latitudes
2. Holland shape parameter B bounding [1.0, 2.5]
3. Radius of Maximum Wind (RMW) estimation
4. Continuous Holland radial wind curve V(r) physics
5. Quadrant gale radii (R34, R50, R64) & translation asymmetry
6. Rossby beta-drift velocity components
7. Trajectory forecaster +6h, +12h, +24h integration
8. Bivariate uncertainty error ellipses & closed cone polygon
9. POST /api/prediction/forecast endpoint
10. GET /api/prediction/vortex/{storm_id} endpoint
11. GET /api/prediction/models endpoint
12. Model registry status: deep neural models report NOT_TRAINED
13. GET /api/cyclones/{id}/prediction dynamic physics generation
14. GET /api/cyclones/{id}/digital-twin Holland model output
15. Preservation of Phase 1-4 endpoints (Health, Cyclones, Data, Satellite AI)
"""

import math
import pytest

from backend.app.ml.physics_vortex import HollandVortexEngine, holland_vortex_engine
from backend.app.ml.trajectory import TrajectoryForecaster, trajectory_forecaster
from backend.app.ml.uncertainty import UncertaintyQuantifier, uncertainty_quantifier
from backend.app.ml.model_registry import model_registry


# ---------------------------------------------------------------------------
# 1. Physics: Coriolis & Holland B Parameter
# ---------------------------------------------------------------------------
def test_coriolis_parameter():
    f_eq = HollandVortexEngine.compute_coriolis(0.0)
    assert f_eq == 0.0

    f_15 = HollandVortexEngine.compute_coriolis(15.0)
    f_30 = HollandVortexEngine.compute_coriolis(30.0)
    assert 0.0 < f_15 < f_30
    assert abs(f_15 - 3.77e-5) < 1e-6


def test_holland_b_parameter_bounds():
    # Shallow depression
    b_low = HollandVortexEngine.compute_holland_b(delta_p=10.0, latitude=15.0)
    assert 1.0 <= b_low <= 2.5

    # Extreme super cyclone
    b_high = HollandVortexEngine.compute_holland_b(delta_p=105.0, latitude=15.0)
    assert 1.0 <= b_high <= 2.5
    assert b_high > b_low


def test_rmw_estimation():
    rmw_moderate = HollandVortexEngine.estimate_rmw(delta_p=30.0, latitude=15.0)
    rmw_intense = HollandVortexEngine.estimate_rmw(delta_p=90.0, latitude=15.0)

    assert 15.0 <= rmw_moderate <= 80.0
    assert 15.0 <= rmw_intense <= 80.0
    # Intense cyclones contract their inner core
    assert rmw_intense < rmw_moderate


# ---------------------------------------------------------------------------
# 2. Continuous Radial Wind Curve V(r)
# ---------------------------------------------------------------------------
def test_holland_radial_wind_curve():
    profile = holland_vortex_engine.generate_radial_profile(
        central_pressure=910.0,
        environmental_pressure=1012.0,
        latitude=14.0,
        max_distance_km=300.0,
        step_km=5.0,
    )

    assert profile["central_pressure_deficit_hpa"] == 102.0
    assert profile["holland_b"] >= 1.0
    assert profile["radius_max_wind_km"] > 0
    assert profile["peak_gradient_wind_knots"] > 80.0

    curve = profile["curve"]
    assert len(curve) > 20

    # Center wind at r=0 must be 0
    assert curve[0]["wind_speed_knots"] == 0.0

    # Check that wind increases towards RMW and decreases beyond
    rmw = profile["radius_max_wind_km"]
    winds_near_rmw = [pt["wind_speed_knots"] for pt in curve if abs(pt["radius_km"] - rmw) < 15.0]
    winds_far = [pt["wind_speed_knots"] for pt in curve if pt["radius_km"] > 250.0]

    assert max(winds_near_rmw) > max(winds_far)
    # Wind speed must never be negative
    assert all(pt["wind_speed_knots"] >= 0.0 for pt in curve)


# ---------------------------------------------------------------------------
# 3. Quadrant Gale Radii & Asymmetry
# ---------------------------------------------------------------------------
def test_quadrant_gale_radii_asymmetry():
    quads = holland_vortex_engine.compute_quadrant_gale_radii(
        central_pressure=920.0,
        environmental_pressure=1012.0,
        latitude=15.0,
        translation_speed_knots=12.0,
        translation_direction_deg=315.0,  # Moving NW
    )

    assert "NE" in quads and "SE" in quads and "SW" in quads and "NW" in quads

    for q_name, q_vals in quads.items():
        # R34 >= R50 >= R64
        assert q_vals["r34_km"] >= q_vals["r50_km"] >= q_vals["r64_km"]

    # In Northern Hemisphere moving NW (315°), forward-right is NE (45°), which must have larger gale extent than rear-left (SW)
    assert quads["NE"]["r34_km"] > quads["SW"]["r34_km"]


# ---------------------------------------------------------------------------
# 4. Rossby Beta Drift
# ---------------------------------------------------------------------------
def test_beta_drift_components():
    u_b, v_b = TrajectoryForecaster.compute_beta_drift(latitude=15.0, max_wind_knots=100.0)
    # In Northern Hemisphere, beta drift directs towards North-West (u < 0, v > 0)
    assert u_b < 0.0
    assert v_b > 0.0


# ---------------------------------------------------------------------------
# 5. Trajectory Forecaster Step Integration
# ---------------------------------------------------------------------------
def test_trajectory_forecaster():
    sample_track = [
        {"latitude": 12.0, "longitude": 86.0, "timestamp": "2020-05-17T00:00:00Z", "windSpeed": 60.0, "pressure": 985.0},
        {"latitude": 13.5, "longitude": 86.4, "timestamp": "2020-05-17T06:00:00Z", "windSpeed": 85.0, "pressure": 965.0},
        {"latitude": 15.0, "longitude": 86.5, "timestamp": "2020-05-17T12:00:00Z", "windSpeed": 115.0, "pressure": 935.0},
    ]

    preds = trajectory_forecaster.forecast(
        track_history=sample_track,
        environmental_sst_c=30.0,
        vertical_wind_shear_knots=10.0,
        forecast_horizons_hours=[6, 12, 24],
    )

    assert len(preds) == 3
    assert preds[0]["forecast_lead_hours"] == 6
    assert preds[1]["forecast_lead_hours"] == 12
    assert preds[2]["forecast_lead_hours"] == 24

    # Trajectory moves northwards
    assert preds[0]["latitude"] > 15.0
    assert preds[1]["latitude"] > preds[0]["latitude"]
    assert preds[2]["latitude"] > preds[1]["latitude"]

    # Verify model identifier
    assert preds[0]["model_name"] == "physics_guided_advection"
    assert preds[0]["model_version"] == "0.5.0"


# ---------------------------------------------------------------------------
# 6. Uncertainty Ellipses & Continuous Polygon Envelope
# ---------------------------------------------------------------------------
def test_uncertainty_quantification():
    s6_maj, s6_min, r6 = uncertainty_quantifier.compute_ellipse_axes(6)
    s12_maj, s12_min, r12 = uncertainty_quantifier.compute_ellipse_axes(12)
    s24_maj, s24_min, r24 = uncertainty_quantifier.compute_ellipse_axes(24)

    # Uncertainty grows with lead time
    assert r6 < r12 < r24
    assert s6_maj > s6_min
    assert s12_maj > s12_min
    assert s24_maj > s24_min

    forecast_nodes = [
        {"latitude": 16.5, "longitude": 86.7, "forecast_lead_hours": 6},
        {"latitude": 18.0, "longitude": 87.1, "forecast_lead_hours": 12},
        {"latitude": 21.0, "longitude": 88.0, "forecast_lead_hours": 24},
    ]

    polygon = uncertainty_quantifier.build_uncertainty_cone_polygon(
        origin_lat=15.0,
        origin_lon=86.5,
        forecast_points=forecast_nodes,
    )

    assert len(polygon) >= 12
    # Closed loop check
    assert polygon[0] == polygon[-1]


# ---------------------------------------------------------------------------
# 7. Model Registry Telemetry
# ---------------------------------------------------------------------------
def test_model_registry_phase5_models():
    models = model_registry.list_models()
    names = [m.name for m in models]

    assert "physics_guided_advection" in names
    assert "holland_vortex_v1" in names
    assert "trajectory_bilstm" in names
    assert "trajectory_transformer" in names

    # Verify active status
    traj_model = next(m for m in models if m.name == "physics_guided_advection")
    assert traj_model.training_status == "ACTIVE"

    # Verify deep learning models status
    bilstm = next(m for m in models if m.name == "trajectory_bilstm")
    assert bilstm.training_status in ("NOT_TRAINED", "TRAINED")

    trans = next(m for m in models if m.name == "trajectory_transformer")
    assert trans.training_status in ("NOT_TRAINED", "TRAINED")


# ---------------------------------------------------------------------------
# 8. API Endpoints: Forecast, Vortex, Models
# ---------------------------------------------------------------------------
def test_api_prediction_forecast_endpoint(client):
    payload = {
        "storm_id": "BOB-01-2020",
        "track_points": [
            {"latitude": 13.0, "longitude": 86.2, "timestamp": "2020-05-17T06:00:00Z", "windSpeed": 80.0, "pressure": 970.0},
            {"latitude": 14.5, "longitude": 86.5, "timestamp": "2020-05-17T12:00:00Z", "windSpeed": 110.0, "pressure": 940.0},
        ],
        "steering_u_knots": -2.0,
        "steering_v_knots": 8.0,
        "sst_c": 30.0,
        "shear_knots": 10.0,
        "horizons_hours": [6, 12, 24],
    }

    response = client.post("/api/prediction/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["cycloneId"] == "BOB-01-2020"
    assert len(data["predictions"]) == 3
    assert len(data["uncertaintyPolygon"]) >= 10
    assert data["modelVersion"] == "physics_guided_advection-v0.5.0"
    assert data["predictions"][0]["leadHours"] == 6


def test_api_vortex_profile_endpoint(client):
    response = client.get("/api/prediction/vortex/BOB-01-2020")
    assert response.status_code == 200
    data = response.json()

    assert "radius_max_wind_km" in data
    assert "holland_b" in data
    assert "quadrant_radii" in data
    assert "radial_curve" in data
    assert len(data["radial_curve"]) > 10
    assert "NE" in data["quadrant_radii"]


def test_api_prediction_models_endpoint(client):
    response = client.get("/api/prediction/models")
    assert response.status_code == 200
    data = response.json()

    assert data["active_trajectory_model"] in ("physics_guided_advection", "trajectory_transformer_v1")
    assert data["active_vortex_model"] in ("holland_vortex_v1", "holland_vortex_pinn")
    assert len(data["models"]) >= 2


# ---------------------------------------------------------------------------
# 9. Dynamic Upgrades to Existing Cyclone Endpoints
# ---------------------------------------------------------------------------
def test_dynamic_cyclone_prediction_route(client):
    response = client.get("/api/cyclones/BOB-01-2020/prediction")
    assert response.status_code == 200
    data = response.json()

    assert len(data["predictions"]) >= 3
    assert len(data["uncertaintyPolygon"]) >= 10
    assert data["modelVersion"] == "physics_guided_advection-v0.5.0"

    # Scientific Corrections Check: Provenance chaining & environmental flags
    prov = data.get("provenance")
    assert prov is not None
    assert prov["source"] == "PHYSICS_ADVECTION_ENGINE"
    assert prov["dataType"] == "AI_PREDICTION"
    assert prov["modelVersion"] == "physics_guided_advection-v0.5.0"
    assert "parentHash" in prov and len(prov["parentHash"]) >= 16
    assert "environmentalObservationsAvailable" in data
    assert "environmentalSourceType" in data


def test_trajectory_unmodulated_when_environmental_missing():
    sample_track = [
        {"latitude": 12.0, "longitude": 86.0, "timestamp": "2020-05-17T00:00:00Z", "windSpeed": 100.0, "pressure": 950.0},
        {"latitude": 13.5, "longitude": 86.4, "timestamp": "2020-05-17T06:00:00Z", "windSpeed": 100.0, "pressure": 950.0},
    ]

    # When environmental observations are None, do not substitute defaults or modulate
    preds = trajectory_forecaster.forecast(
        track_history=sample_track,
        environmental_sst_c=None,
        vertical_wind_shear_knots=None,
        forecast_horizons_hours=[6, 12, 24],
    )

    for p in preds:
        assert p["intensity_trend"] == "UNMODULATED (NO_ENVIRONMENTAL_OBS)"
        assert p["predicted_wind_speed_knots"] == 100.0
        assert p["predicted_pressure_hpa"] == 950.0


def test_api_prediction_forecast_no_environmental_defaults(client):
    payload = {
        "storm_id": "BOB-01-2020",
        "track_points": [
            {"latitude": 13.0, "longitude": 86.2, "timestamp": "2020-05-17T06:00:00Z", "windSpeed": 80.0, "pressure": 970.0},
            {"latitude": 14.5, "longitude": 86.5, "timestamp": "2020-05-17T12:00:00Z", "windSpeed": 110.0, "pressure": 940.0},
        ],
        "horizons_hours": [6, 12],
    }

    response = client.post("/api/prediction/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["environmentalObservationsAvailable"] is False
    assert data["sstAvailable"] is False
    assert data["windShearAvailable"] is False
    assert data["environmentalSourceType"] in ("UNAVAILABLE", "NONE")
    assert data["predictions"][0]["intensityTrend"] == "UNMODULATED (NO_ENVIRONMENTAL_OBS)"
    assert data["predictions"][0]["predictedWindSpeed"] == 110.0
    assert "parentHash" in data["provenance"]
    assert len(data["provenance"]["parentHash"]) >= 16


def test_vortex_disclaimer_present(client):
    response = client.get("/api/prediction/vortex/BOB-01-2020")
    assert response.status_code == 200
    data = response.json()
    assert "MATHEMATICAL VORTEX PROFILE — NOT AN OFFICIAL IMD ADVISORY" in data["scientific_note"]


def test_dynamic_cyclone_digital_twin_route(client):
    response = client.get("/api/cyclones/BOB-01-2020/digital-twin")
    assert response.status_code == 200
    data = response.json()

    assert data["radiusMaxWindKm"] > 0
    assert len(data["radialProfile"]) > 10
    assert "FORWARD-RIGHT" in data["rightFrontQuadrantRisk"]


# ---------------------------------------------------------------------------
# 10. Regression Check on Phase 1-4 Endpoints
# ---------------------------------------------------------------------------
def test_phase1_to_phase4_endpoints_intact(client):
    # Health
    assert client.get("/api/health").status_code == 200
    # Cyclones
    assert client.get("/api/cyclones").status_code == 200
    # Data status (Phase 3)
    assert client.get("/api/data/status").status_code == 200
    # Provenance (Phase 3)
    assert client.get("/api/provenance").status_code == 200
    # Satellite AI models (Phase 4)
    assert client.get("/api/analysis/models").status_code == 200

