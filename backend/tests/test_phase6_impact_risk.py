"""Automated Test Suite for Phase 6: AI Impact & Disaster Risk Intelligence.

Verifies:
1. Wind risk with valid input & threshold classification
2. Wind risk with missing wind (available=False, UNAVAILABLE)
3. Rainfall unavailable behavior (available=False, NO_RAINFALL_DATA)
4. Flood risk unavailable behavior (available=False, INSUFFICIENT_RAINFALL_DATA)
5. Coastal risk unavailable behavior (available=False, NO_VALIDATED_SURGE_MODEL)
6. Coastal risk analytical scenario proxy (EXPERIMENTAL)
7. Population exposure unavailable behavior (available=False, dataset not connected)
8. Infrastructure exposure unavailable behavior (available=False, INFRASTRUCTURE DATA UNAVAILABLE)
9. Composite risk with partial data (deterministic index, zero fake probabilities)
10. Composite risk with insufficient data (UNKNOWN / INSUFFICIENT DATA, risk_index=None)
11. Demo data labeling (DEMO / SCENARIO)
12. User perturbation labeling (USER PERTURBATION, is_scenario_output=True)
13. Provenance creation & parent hash lineage chaining
14. Risk map GeoJSON geometry validity & FeatureCollection structure
15. All 7 /api/impact/* API endpoints
16. Phase 1-5 regression preservation
"""

import pytest
from backend.app.impact.coastal_risk import coastal_risk_analyzer
from backend.app.impact.exposure import exposure_analyzer
from backend.app.impact.flood_risk import flood_risk_analyzer
from backend.app.impact.impact_engine import impact_engine
from backend.app.impact.impact_models import ScenarioRequest
from backend.app.impact.rainfall_risk import rainfall_risk_analyzer
from backend.app.impact.risk_scoring import risk_scoring_engine
from backend.app.impact.wind_risk import wind_risk_analyzer
from backend.app.ml.model_registry import model_registry


# ---------------------------------------------------------------------------
# 1. Wind Risk Module Tests
# ---------------------------------------------------------------------------
def test_wind_risk_valid_inputs():
    res_extreme = wind_risk_analyzer.evaluate(wind_speed_knots=115.0)
    assert res_extreme.available is True
    assert res_extreme.risk_level == "EXTREME"
    assert res_extreme.source_type == "MODEL-DERIVED"
    assert res_extreme.metrics["max_wind_speed_knots"] == 115.0
    assert "Holland" in res_extreme.method

    res_mod = wind_risk_analyzer.evaluate(wind_speed_knots=42.0)
    assert res_mod.available is True
    assert res_mod.risk_level == "MODERATE"

    res_low = wind_risk_analyzer.evaluate(wind_speed_knots=25.0)
    assert res_low.available is True
    assert res_low.risk_level == "LOW"


def test_wind_risk_missing_input():
    # When wind speed is None or <= 0, do not fabricate
    res_none = wind_risk_analyzer.evaluate(wind_speed_knots=None)
    assert res_none.available is False
    assert res_none.risk_level == "UNAVAILABLE"
    assert res_none.status == "MISSING_WIND_DATA"
    assert "unavailable" in res_none.explanation.lower()

    res_zero = wind_risk_analyzer.evaluate(wind_speed_knots=0.0)
    assert res_zero.available is False
    assert res_zero.risk_level == "UNAVAILABLE"


# ---------------------------------------------------------------------------
# 2. Rainfall & Flood Risk Module Tests
# ---------------------------------------------------------------------------
def test_rainfall_risk_unavailable_behavior():
    res = rainfall_risk_analyzer.evaluate(
        measured_rainfall_mm=None,
        scenario_rainfall_mm=None,
        is_demo_mode=False,
    )
    assert res.available is False
    assert res.risk_level == "UNAVAILABLE"
    assert res.status == "NO_RAINFALL_DATA"
    assert "not connected" in res.explanation.lower()


def test_flood_risk_unavailable_when_rainfall_missing():
    unavail_rain = rainfall_risk_analyzer.evaluate(measured_rainfall_mm=None)
    res_flood = flood_risk_analyzer.evaluate(rainfall_risk=unavail_rain)
    assert res_flood.available is False
    assert res_flood.risk_level == "UNAVAILABLE"
    assert res_flood.status == "INSUFFICIENT_RAINFALL_DATA"


def test_rainfall_and_flood_scenario_mode():
    res_rain = rainfall_risk_analyzer.evaluate(scenario_rainfall_mm=210.0)
    assert res_rain.available is True
    assert res_rain.risk_level == "EXTREME"
    assert res_rain.source_type == "USER PERTURBATION"

    res_flood = flood_risk_analyzer.evaluate(res_rain)
    assert res_flood.available is True
    assert res_flood.risk_level == "EXTREME"
    assert res_flood.source_type == "USER PERTURBATION"


# ---------------------------------------------------------------------------
# 3. Coastal Surge Risk Module Tests
# ---------------------------------------------------------------------------
def test_coastal_risk_unavailable_behavior():
    res = coastal_risk_analyzer.evaluate(
        central_pressure_hpa=None,
        max_wind_knots=None,
    )
    assert res.available is False
    assert res.risk_level == "UNAVAILABLE"
    assert res.status == "NO_VALIDATED_SURGE_MODEL"
    assert "not connected" in res.explanation.lower()


def test_coastal_risk_analytical_scenario():
    # Amphan-like central pressure and wind
    res = coastal_risk_analyzer.evaluate(
        central_pressure_hpa=920.0,
        max_wind_knots=120.0,
        distance_to_coast_km=60.0,
    )
    assert res.available is True
    assert res.risk_level in ("HIGH", "EXTREME")
    assert res.source_type in ("EXPERIMENTAL", "DEEP_LEARNING")
    assert res.metrics["estimated_surge_peak_meters"] > 2.0
    assert ("Inverted Barometer" in res.method or "Deep Neural Surrogate" in res.method)
    assert ("NOT AN OFFICIAL STORM-SURGE FORECAST" in res.disclaimer or "TRAINED DEEP HYDRODYNAMIC SURROGATE" in res.disclaimer)


# ---------------------------------------------------------------------------
# 4. Population & Infrastructure Exposure Tests
# ---------------------------------------------------------------------------
def test_population_exposure_unavailable_when_disconnected():
    res = exposure_analyzer.evaluate_population(
        cyclone_lat=20.0,
        cyclone_lon=87.0,
        use_demo_data=False,
    )
    assert res.available is False
    assert res.total_exposed_population is None
    assert "population dataset not connected" in res.explanation.lower()


def test_infrastructure_exposure_unavailable_when_disconnected():
    res = exposure_analyzer.evaluate_infrastructure(
        cyclone_lat=20.0,
        cyclone_lon=87.0,
        use_demo_data=False,
    )
    assert res.available is False
    assert res.critical_count == 0
    assert "INFRASTRUCTURE DATA UNAVAILABLE" in res.explanation


def test_exposure_demo_curated_datasets():
    pop_res = exposure_analyzer.evaluate_population(
        cyclone_lat=21.0,
        cyclone_lon=87.5,
        use_demo_data=True,
        basin="BOB",
    )
    assert pop_res.available is True
    assert pop_res.source_type == "DEMO / SCENARIO"
    assert pop_res.total_exposed_population is not None
    assert pop_res.total_exposed_population > 0
    assert len(pop_res.settlements) > 0

    infra_res = exposure_analyzer.evaluate_infrastructure(
        cyclone_lat=21.0,
        cyclone_lon=87.5,
        use_demo_data=True,
        basin="BOB",
    )
    assert infra_res.available is True
    assert infra_res.source_type == "DEMO / SCENARIO"
    assert infra_res.critical_count > 0


# ---------------------------------------------------------------------------
# 5. Composite Risk Scoring & Zero-Fabrication Contract
# ---------------------------------------------------------------------------
def test_composite_risk_partial_data():
    wind_comp = wind_risk_analyzer.evaluate(wind_speed_knots=90.0)
    coastal_comp = coastal_risk_analyzer.evaluate(central_pressure_hpa=940.0, max_wind_knots=90.0, distance_to_coast_km=80.0)
    unavail_rain = rainfall_risk_analyzer.evaluate(measured_rainfall_mm=None)
    unavail_flood = flood_risk_analyzer.evaluate(unavail_rain)
    unavail_pop = exposure_analyzer.evaluate_population(20.0, 87.0, use_demo_data=False)
    unavail_infra = exposure_analyzer.evaluate_infrastructure(20.0, 87.0, use_demo_data=False)

    level, score, avail, unavail, method = risk_scoring_engine.compute_composite_score(
        wind_risk=wind_comp,
        rainfall_risk=unavail_rain,
        flood_risk=unavail_flood,
        coastal_risk=coastal_comp,
        population_exposure=unavail_pop,
        infrastructure_exposure=unavail_infra,
    )

    assert level in ("HIGH", "EXTREME")
    assert score is not None
    assert 0.0 <= score <= 100.0
    assert len(avail) >= 2
    assert len(unavail) >= 2
    assert "Deterministic Weighted Multi-Hazard Index" in method


def test_composite_risk_insufficient_data():
    # Only 1 component available
    wind_comp = wind_risk_analyzer.evaluate(wind_speed_knots=60.0)
    unavail_coast = coastal_risk_analyzer.evaluate(central_pressure_hpa=None)
    unavail_rain = rainfall_risk_analyzer.evaluate(measured_rainfall_mm=None)
    unavail_flood = flood_risk_analyzer.evaluate(unavail_rain)
    unavail_pop = exposure_analyzer.evaluate_population(20.0, 87.0, use_demo_data=False)
    unavail_infra = exposure_analyzer.evaluate_infrastructure(20.0, 87.0, use_demo_data=False)

    level, score, avail, unavail, method = risk_scoring_engine.compute_composite_score(
        wind_risk=wind_comp,
        rainfall_risk=unavail_rain,
        flood_risk=unavail_flood,
        coastal_risk=unavail_coast,
        population_exposure=unavail_pop,
        infrastructure_exposure=unavail_infra,
    )

    assert level == "UNKNOWN / INSUFFICIENT DATA"
    assert score is None
    assert "Insufficient" in method


# ---------------------------------------------------------------------------
# 6. Impact Orchestrator & Provenance Chaining
# ---------------------------------------------------------------------------
def test_impact_engine_provenance_chaining():
    mock_cyclone = {
        "id": "BOB-01-2020",
        "name": "AMPHAN",
        "currentObservation": {"latitude": 19.85, "longitude": 87.20, "windSpeed": 100.0, "pressure": 950.0},
        "provenance": {"dataHash": "parent_observation_hash_12345"},
    }

    assessment = impact_engine.evaluate_impact(
        cyclone_data=mock_cyclone,
        parent_hash="parent_observation_hash_12345",
    )

    assert assessment.storm_id == "BOB-01-2020"
    assert assessment.provenance.source == "IMPACT_INTELLIGENCE_ENGINE"
    assert assessment.provenance.dataType == "RISK_ASSESSMENT"
    assert assessment.provenance.modelVersion == "impact_risk_engine-v0.6.0"
    assert assessment.provenance.parentHash == "parent_observation_hash_12345"
    assert "EXPERIMENTAL AI DECISION-SUPPORT" in assessment.provenance.disclaimer


def test_impact_engine_risk_map_geojson():
    mock_cyclone = {
        "id": "BOB-01-2020",
        "name": "AMPHAN",
        "currentObservation": {"latitude": 19.85, "longitude": 87.20, "windSpeed": 100.0, "pressure": 950.0},
    }

    assessment = impact_engine.evaluate_impact(cyclone_data=mock_cyclone)
    geojson = impact_engine.generate_risk_map_geojson(assessment, cyclone_data=mock_cyclone)

    assert geojson.type == "FeatureCollection"
    assert len(geojson.features) >= 2

    # Check for wind buffer polygon feature
    wind_feats = [f for f in geojson.features if f["properties"].get("layer") == "WIND_RISK"]
    assert len(wind_feats) >= 1
    assert wind_feats[0]["geometry"]["type"] == "Polygon"


# ---------------------------------------------------------------------------
# 7. Model Registry Phase 6 Models
# ---------------------------------------------------------------------------
def test_model_registry_phase6_models():
    models = model_registry.list_models()
    names = [m.name for m in models]

    assert "impact_rule_engine" in names
    assert "wind_risk_analytic" in names
    assert "flood_risk_baseline" in names
    assert "coastal_risk_baseline" in names
    assert "exposure_intersection_engine" in names
    assert "surge_deep_surrogate" in names

    # Status check: Rule engine and wind analytic are ACTIVE
    assert next(m for m in models if m.name == "impact_rule_engine").training_status == "ACTIVE"
    assert next(m for m in models if m.name == "wind_risk_analytic").training_status == "ACTIVE"

    # Status check: Hydro surrogate neural net is TRAINED or NOT_TRAINED
    assert next(m for m in models if m.name == "surge_deep_surrogate").training_status in ("NOT_TRAINED", "TRAINED")


# ---------------------------------------------------------------------------
# 8. API Endpoints: All 7 /api/impact/* routes
# ---------------------------------------------------------------------------
def test_api_impact_get_storm(client):
    response = client.get("/api/impact/BOB-01-2020")
    assert response.status_code == 200
    data = response.json()

    assert data["stormId"] == "BOB-01-2020"
    assert data["riskLevel"] in ("HIGH", "EXTREME")
    assert data["riskIndex"] is not None
    assert "windRisk" in data
    assert "coastalRisk" in data
    assert "whyExplanation" in data
    assert data["provenance"]["source"] == "IMPACT_INTELLIGENCE_ENGINE"


def test_api_impact_risk_map(client):
    response = client.get("/api/impact/BOB-01-2020/risk-map")
    assert response.status_code == 200
    data = response.json()

    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 0


def test_api_impact_exposure(client):
    response = client.get("/api/impact/BOB-01-2020/exposure")
    assert response.status_code == 200
    data = response.json()

    assert "population_exposure" in data or "populationExposure" in data
    assert "infrastructure_exposure" in data or "infrastructureExposure" in data


def test_api_impact_provenance(client):
    response = client.get("/api/impact/BOB-01-2020/provenance")
    assert response.status_code == 200
    data = response.json()

    assert data["source"] == "IMPACT_INTELLIGENCE_ENGINE"
    assert data["dataType"] == "RISK_ASSESSMENT"


def test_api_impact_models(client):
    response = client.get("/api/impact/models")
    assert response.status_code == 200
    data = response.json()

    assert data["active_composite_engine"] in ("impact_rule_engine", "impact_neural_engine")
    assert len(data["models"]) >= 4


def test_api_impact_analyze_custom(client):
    payload = {
        "id": "TEST-01",
        "name": "CUSTOM_STORM",
        "latitude": 15.0,
        "longitude": 88.0,
        "windSpeed": 75.0,
        "pressure": 975.0,
    }
    response = client.post("/api/impact/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["stormId"] == "TEST-01"
    assert data["windRisk"]["riskLevel"] == "EXTREME"


def test_api_impact_scenario_mode(client):
    payload = {
        "storm_id": "BOB-01-2020",
        "wind_speed_override_knots": 130.0,
        "rainfall_scenario_mm": 300.0,
        "coastal_tide_surge_scenario_m": 5.0,
    }
    response = client.post("/api/impact/scenario", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["isScenarioOutput"] is True
    assert data["windRisk"]["sourceType"] == "USER PERTURBATION"
    assert data["rainfallRisk"]["sourceType"] == "USER PERTURBATION"
    assert data["coastalRisk"]["sourceType"] == "USER PERTURBATION"


# ---------------------------------------------------------------------------
# 9. Phase 1-5 Regression Check
# ---------------------------------------------------------------------------
def test_phase1_to_phase5_regression(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/cyclones").status_code == 200
    assert client.get("/api/cyclones/BOB-01-2020").status_code == 200
    assert client.get("/api/cyclones/BOB-01-2020/prediction").status_code == 200
    assert client.get("/api/cyclones/BOB-01-2020/digital-twin").status_code == 200
    assert client.get("/api/data/status").status_code == 200
    assert client.get("/api/analysis/models").status_code == 200
    assert client.get("/api/prediction/models").status_code == 200
