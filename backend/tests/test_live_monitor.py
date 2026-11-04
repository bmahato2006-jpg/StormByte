"""
Unit and Integration Tests for Live Storm Monitor (North Indian Ocean)
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.live_storm_monitor import (
    _in_nio,
    fetch_active_nio_storms,
    fetch_ibtracs_active_track,
    run_live_pipeline,
    get_basin_environment,
)

client = TestClient(app)


def test_in_nio_bounds():
    """Points in Bay of Bengal and Arabian Sea must be inside NIO."""
    # Bay of Bengal
    assert _in_nio(15.0, 87.0) is True
    # Arabian Sea
    assert _in_nio(18.0, 68.0) is True
    # East Pacific (outside)
    assert _in_nio(18.0, -131.7) is False
    # North Atlantic (outside)
    assert _in_nio(25.0, -75.0) is False
    # South Indian Ocean (outside lat < 0)
    assert _in_nio(-15.0, 80.0) is False


def test_fetch_active_nio_storms_filters_correctly():
    """Mock GDACS containing 1 Pacific storm and 1 Bay of Bengal storm."""
    mock_gdacs_response = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-131.7, 18.0]},
                "properties": {
                    "eventtype": "TC",
                    "eventname": "NORBERT-26",
                    "alertlevel": "Green",
                    "episodealertscore": 1.0,
                    "severitydata": {"severity": 102.0, "severitytext": "Tropical Storm (102 km/h)"},
                    "fromdate": "2026-09-09T21:00:00",
                    "todate": "2026-09-13T09:00:00",
                    "eventid": 1001320,
                    "episodeid": 15,
                    "url": {"report": "https://www.gdacs.org/report.aspx?eventid=1001320"},
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [88.5, 17.5]},
                "properties": {
                    "eventtype": "TC",
                    "eventname": "TEST-CYCLONE",
                    "alertlevel": "Orange",
                    "episodealertscore": 2.0,
                    "severitydata": {"severity": 140.0, "severitytext": "Very Severe Cyclonic Storm (140 km/h)"},
                    "fromdate": "2026-09-12T00:00:00",
                    "todate": "2026-09-13T12:00:00",
                    "eventid": 999999,
                    "episodeid": 1,
                    "url": {"report": "https://www.gdacs.org/report.aspx?eventid=999999"},
                },
            },
        ],
    }

    import json
    with patch("backend.app.services.live_storm_monitor._get", return_value=json.dumps(mock_gdacs_response).encode()):
        storms, prov = fetch_active_nio_storms()
        assert len(storms) == 1
        assert storms[0]["name"] == "TEST-CYCLONE"
        assert storms[0]["basin"] == "BOB"
        assert storms[0]["alert_level"] == "Orange"
        assert "GDACS TC API" in prov


def test_fetch_ibtracs_active_track_parsing():
    """Mock IBTrACS ACTIVE CSV with sample rows."""
    mock_csv = (
        "SID,SEASON,NUMBER,BASIN,SUBBASIN,NAME,ISO_TIME,NATURE,LAT,LON,WMO_WIND,WMO_PRES\n"
        " ,Year, , , , , , ,degrees_north,degrees_east,kts,mb\n"
        "2026250N15088,2026,1,NI,BB,TESTSTORM,2026-09-12 00:00:00,TS,14.0,88.0,45,998\n"
        "2026250N15088,2026,1,NI,BB,TESTSTORM,2026-09-12 06:00:00,TS,14.8,88.2,55,990\n"
        "2026250N15088,2026,1,NI,BB,TESTSTORM,2026-09-12 12:00:00,TS,15.6,88.4,65,980\n"
        "2026250N15088,2026,1,NI,BB,TESTSTORM,2026-09-12 18:00:00,TS,16.5,88.5,75,970\n"
    )

    with patch("backend.app.services.live_storm_monitor._get", return_value=mock_csv.encode("utf-8")):
        fixes, prov = fetch_ibtracs_active_track("TESTSTORM")
        assert len(fixes) == 4
        assert fixes[-1]["latitude"] == 16.5
        assert fixes[-1]["longitude"] == 88.5
        assert fixes[-1]["wind_kt"] == 75.0
        assert "IBTrACS v04r01 ACTIVE" in prov


def test_run_live_pipeline_schema():
    """Test run_live_pipeline on a mock storm generates valid pipeline schema."""
    storm = {
        "name": "TESTSTORM",
        "lat": 17.5,
        "lon": 88.5,
        "basin": "BOB",
        "alert_level": "Orange",
        "severity_kmh": 130.0,
        "severity_text": "Very Severe Cyclonic Storm",
        "from_date": "2026-09-12T00:00:00",
        "to_date": "2026-09-13T12:00:00",
        "gdacs_event_id": 999999,
        "report_url": "https://gdacs.org/test",
    }

    mock_track = [
        {"latitude": 14.0, "longitude": 88.0, "timestamp": "2026-09-12T00:00:00Z"},
        {"latitude": 14.8, "longitude": 88.2, "timestamp": "2026-09-12T06:00:00Z"},
        {"latitude": 15.6, "longitude": 88.4, "timestamp": "2026-09-12T12:00:00Z"},
        {"latitude": 16.5, "longitude": 88.5, "timestamp": "2026-09-12T18:00:00Z"},
    ]

    with patch("backend.app.services.live_storm_monitor.fetch_ibtracs_active_track", return_value=(mock_track, "Mock IBTrACS")):
        with patch("backend.app.services.live_storm_monitor.fetch_env_at", return_value={"sst_celsius": 30.2, "wind_speed_knots": 45.0, "sst_source": "Mock SST", "wind_source": "Mock Wind"}):
            res = run_live_pipeline(storm)
            assert res["detected"] is True
            assert res["label"] == "CYCLONE"
            assert res["confidence"] == 1.0
            assert res["center_fix"]["coordinates_available"] is True
            assert res["center_fix"]["latitude"] == 17.5
            assert res["center_fix"]["longitude"] == 88.5
            assert res["center_fix"]["pixel_x"] is None
            assert res["risk_assessment"]["risk_score"] > 0
            assert res["provenance"]["data_mode"] == "LIVE_NIO_MONITOR"
            assert "detection_model" in res["provenance"]


def test_live_status_endpoint():
    """GET /api/live/status must return valid 200 JSON with basin environment."""
    resp = client.get("/api/live/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "basin_quiet" in data
    assert "active_storms" in data
    assert "basin_environment" in data
    assert "bay_of_bengal" in data["basin_environment"]
    assert "arabian_sea" in data["basin_environment"]
    assert data["basin_environment"]["bay_of_bengal"]["centre_lat"] == 15.0


def test_live_analyze_not_found():
    """POST /api/live/analyze/nonexistent_storm must return 404 with helpful error."""
    resp = client.post("/api/live/analyze/nonexistent_cyclone_xyz")
    assert resp.status_code == 404
    assert "not found in active North Indian Ocean storms" in resp.json()["detail"]
