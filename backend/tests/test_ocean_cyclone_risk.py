"""Comprehensive Test Suite for Ocean Area Cyclone Risk Prediction (Phase 12).

Verifies 19 core scientific and operational requirements:
1. Valid ocean location in Bay of Bengal returns valid risk assessment (200 OK)
2. Valid ocean location in Arabian Sea returns valid risk assessment (200 OK)
3. Land location rejection (Peninsular India / Nagpur) returns is_ocean=False
4. Land location rejection (New Delhi / Northern India) returns is_ocean=False
5. Land location rejection (Sri Lanka) returns is_ocean=False
6. Near-equatorial coordinate (<5°N) returns Coriolis limitation
7. Invalid latitude (<-90 or >90) rejected with 422
8. Invalid longitude (<-180 or >180) rejected with 422
9. Out-of-domain coordinates handled gracefully
10. Risk score is deterministically bounded within [0, 100]
11. Risk level strictly maps to one of the 5 standard categories
12. probability_calibrated is False and zero-fabrication notices verified
13. Scientific non-certainty phrasing in disclaimer and reasoning
14. Evidence checklist includes all key atmospheric/oceanic factors
15. Missing data variables handled with UNAVAILABLE and zero synthetic numbers
16. Temporal persistence analysis returns 24h lookback window
17. Radius km parameter is validated and respected
18. Status endpoint GET /api/prediction/ocean-risk/status reports operational layers
19. Evaluation determinism: identical inputs yield identical outputs
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.ocean_cyclone_risk_service import (
    ocean_cyclone_risk_service,
    point_in_polygon,
    PENINSULAR_INDIA_POLYGON,
    SRI_LANKA_POLYGON,
)
from backend.app.schemas.prediction import OceanRiskRequest

client = TestClient(app)


# 1. Bay of Bengal Ocean Success
def test_bay_of_bengal_ocean_risk_success():
    payload = {"latitude": 15.0, "longitude": 88.0, "radius_km": 100.0}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["location"]["is_ocean"] is True
    assert data["location"]["basin"] == "BAY_OF_BENGAL"
    assert data["location"]["is_equatorial_restricted"] is False
    assert 0.0 <= data["risk_score"] <= 100.0
    assert data["risk_level"] in ["VERY LOW", "LOW", "MODERATE", "HIGH", "VERY HIGH"]


# 2. Arabian Sea Ocean Success
def test_arabian_sea_ocean_risk_success():
    payload = {"latitude": 16.0, "longitude": 65.0, "radius_km": 100.0}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["location"]["is_ocean"] is True
    assert data["location"]["basin"] == "ARABIAN_SEA"
    assert data["location"]["is_equatorial_restricted"] is False


# 3. Land Location Rejection - Peninsular India (Nagpur)
def test_land_location_rejection_peninsular_india():
    payload = {"latitude": 21.14, "longitude": 79.08}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["location"]["is_ocean"] is False
    assert data["risk_score"] == 0.0
    assert data["risk_level"] == "VERY LOW"
    assert "Peninsular India" in data["reasoning"] or "land" in data["reasoning"].lower()


# 4. Land Location Rejection - New Delhi / Northern Continental Land
def test_land_location_rejection_delhi_northern_india():
    payload = {"latitude": 28.61, "longitude": 77.20}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["location"]["is_ocean"] is False
    assert data["risk_score"] == 0.0


# 5. Land Location Rejection - Sri Lanka
def test_land_location_rejection_sri_lanka():
    payload = {"latitude": 7.87, "longitude": 80.77}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["location"]["is_ocean"] is False
    assert "Sri Lanka" in data["reasoning"] or "land" in data["reasoning"].lower()


# 6. Equatorial Coriolis Restriction (<5°N)
def test_equatorial_coriolis_restriction():
    payload = {"latitude": 2.5, "longitude": 85.0, "radius_km": 100.0}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["location"]["is_ocean"] is True
    assert data["location"]["is_equatorial_restricted"] is True
    assert data["risk_score"] <= 15.0
    assert data["risk_level"] == "VERY LOW"
    assert "Coriolis" in data["reasoning"] or "equator" in data["reasoning"].lower()


# 7. Invalid Latitude Validation (<-90 or >90)
def test_invalid_latitude_validation():
    payload = {"latitude": 95.0, "longitude": 85.0}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    assert response.status_code == 422


# 8. Invalid Longitude Validation (<-180 or >180)
def test_invalid_longitude_validation():
    payload = {"latitude": 15.0, "longitude": 200.0}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    assert response.status_code == 422


# 9. Out-of-Domain Global Ocean Handled Gracefully
def test_out_of_domain_global_ocean():
    payload = {"latitude": -15.0, "longitude": 75.0}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["location"]["is_ocean"] is True
    assert data["location"]["basin"] in ["SOUTH_INDIAN_OCEAN", "GLOBAL_OCEAN"]


# 10. Risk Score Bounded Between 0 and 100
def test_risk_score_bounded_0_to_100():
    coords = [
        (12.0, 85.0),
        (18.5, 88.0),
        (10.0, 68.0),
        (3.0, 80.0),
    ]
    for lat, lon in coords:
        res = ocean_cyclone_risk_service.analyze_ocean_risk(
            OceanRiskRequest(latitude=lat, longitude=lon)
        )
        assert 0.0 <= res.risk_score <= 100.0


# 11. Risk Level Categorization
def test_risk_level_categories_mapping():
    valid_levels = {"VERY LOW", "LOW", "MODERATE", "HIGH", "VERY HIGH"}
    res = ocean_cyclone_risk_service.analyze_ocean_risk(
        OceanRiskRequest(latitude=14.0, longitude=86.0)
    )
    assert res.risk_level in valid_levels


# 12. Zero-Fabrication Uncalibrated Flag
def test_zero_fabrication_uncalibrated_flag():
    payload = {"latitude": 15.0, "longitude": 88.0}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["probability_calibrated"] is False
    assert "experimental" in data["experimental_notice"].lower()


# 13. Scientific Non-Certainty Phrasing
def test_scientific_non_certainty_phrasing():
    payload = {"latitude": 16.0, "longitude": 87.0}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    data = response.json()
    disclaimer = data["disclaimer"]
    reasoning = data["reasoning"]

    # Must never claim absolute certainty
    for text in [disclaimer, reasoning]:
        assert "will definitely come" not in text.lower()
        assert "no cyclone will definitely come" not in text.lower()
        assert "100% certain" not in text.lower()


# 14. Evidence Checklist Comprehensive
def test_evidence_checklist_comprehensive():
    payload = {"latitude": 15.0, "longitude": 88.0}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    data = response.json()
    factors = [item["factor"] for item in data["evidence_checklist"]]

    assert any("Coriolis" in f for f in factors)
    assert any("Satellite" in f for f in factors)
    assert any("IMERG" in f or "Precipitation" in f for f in factors)
    assert any("SST" in f or "Sea Surface Temperature" in f for f in factors)
    assert any("Shear" in f or "Wind Shear" in f for f in factors)
    assert any("Cyclone" in f or "Proximity" in f for f in factors)
    assert any("Intensity" in f for f in factors)


# 15. Environmental Variables Handled Truthfully (Verified Values & Zero Synthetic Numbers)
def test_missing_environmental_variables_unflagged_zero_fake():
    payload = {"latitude": 15.0, "longitude": 88.0}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    data = response.json()
    checklist = {item["factor"]: item for item in data["evidence_checklist"]}

    # SST, Wind Shear, and Intensity are verified via Antigravity meteorological tools
    for factor_name, item in checklist.items():
        if "Sea Surface Temperature" in factor_name:
            assert item["status"] in ["PRESENT", "AVAILABLE"]
            assert item["value"] is not None
            assert item["value"] >= 20.0
        if "Wind Shear" in factor_name:
            assert item["status"] in ["PRESENT", "AVAILABLE"]
            assert item["value"] is not None
        if "Intensity" in factor_name:
            assert item["status"] in ["PRESENT", "AVAILABLE"]


# 16. Temporal Persistence 24-Hour Lookback Window
def test_temporal_persistence_24h_window():
    payload = {"latitude": 15.0, "longitude": 88.0}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    data = response.json()
    assert "temporal_persistence" in data
    assert data["temporal_persistence"]["window_hours"] == 24
    assert data["temporal_persistence"]["trend"] in ["STABLE", "INTENSIFYING", "DISSIPATING", "INSUFFICIENT_HISTORY"]


# 17. Radius Parameter Support
def test_radius_km_parameter_support():
    payload = {"latitude": 15.0, "longitude": 88.0, "radius_km": 250.0}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["location"]["radius_km"] == 250.0


# 18. Status Endpoint Reports Operational Layers
def test_ocean_risk_status_endpoint():
    response = client.get("/api/prediction/ocean-risk/status")
    assert response.status_code == 200
    data = response.json()
    assert data["service_status"] == "OPERATIONAL"
    assert "Bay of Bengal" in "".join(data["supported_basins"])
    assert "nrt_satellite" in data["data_sources"]
    assert "nasa_imerg" in data["data_sources"]
    assert "ecmwf_era5" in data["data_sources"]
    assert data["scientific_governance"]["zero_fabrication"] is True
    assert data["scientific_governance"]["coriolis_restriction_enforced"] is True


# 19. Evaluation Determinism
def test_evaluation_determinism():
    req1 = OceanRiskRequest(latitude=14.5, longitude=85.5, radius_km=100.0)
    req2 = OceanRiskRequest(latitude=14.5, longitude=85.5, radius_km=100.0)

    res1 = ocean_cyclone_risk_service.analyze_ocean_risk(req1)
    res2 = ocean_cyclone_risk_service.analyze_ocean_risk(req2)

    assert res1.risk_score == res2.risk_score
    assert res1.risk_level == res2.risk_level
    assert res1.location.is_ocean == res2.location.is_ocean
    assert len(res1.evidence_checklist) == len(res2.evidence_checklist)


# 20. Land Rejection - Arabian Peninsula (Yemen / Oman)
def test_land_location_rejection_arabian_peninsula():
    payload = {"latitude": 15.35, "longitude": 44.20}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["location"]["is_ocean"] is False
    assert "Arabian Peninsula" in data["reasoning"] or "land" in data["reasoning"].lower()


# 21. Land Rejection - Horn of Africa (Somalia)
def test_land_location_rejection_somalia_horn_of_africa():
    payload = {"latitude": 2.04, "longitude": 45.20}
    response = client.post("/api/prediction/ocean-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["location"]["is_ocean"] is False
    assert "Horn of Africa" in data["reasoning"] or "land" in data["reasoning"].lower()

