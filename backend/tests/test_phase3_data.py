"""Automated Test Suite for Phase 3: Scientific Data Intelligence & Provenance.

Verifies:
1. CycloneObservation validation
2. SatelliteObservation validation
3. EnvironmentalObservation validation
4. Historical CSV import
5. Historical JSON import
6. Invalid latitude rejection
7. Invalid longitude rejection
8. Invalid timestamp rejection
9. Invalid file type rejection
10. Satellite upload & dimension extraction
11. Provenance endpoint
12. Data sources endpoint
13. Demo fallback
14. Database unavailable fallback without crash
15. Existing health endpoint preservation
"""

import io
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from backend.app.data.schemas.observations import (
    DataMode,
    CycloneObservation,
    SatelliteObservation,
    EnvironmentalObservation,
)
from backend.app.data.adapters.historical import HistoricalTrackAdapter


# ---------------------------------------------------------------------------
# 1. CycloneObservation Schema Validation
# ---------------------------------------------------------------------------
def test_cyclone_observation_validation():
    obs = CycloneObservation(
        id="obs-test-01",
        storm_id="BOB-01-2020",
        storm_name="AMPHAN",
        timestamp=datetime(2020, 5, 18, 12, 0, tzinfo=timezone.utc),
        latitude=13.9,
        longitude=86.5,
        wind_speed=140.0,
        central_pressure=907.0,
        classification="Super Cyclonic Storm",
        source="IMD_BEST_TRACK",
        data_mode=DataMode.DEMO,
    )
    assert obs.storm_name == "AMPHAN"
    assert obs.latitude == 13.9
    assert obs.data_mode == DataMode.DEMO


# ---------------------------------------------------------------------------
# 2. SatelliteObservation Schema Validation
# ---------------------------------------------------------------------------
def test_satellite_observation_validation():
    sat = SatelliteObservation(
        id="sat-test-01",
        satellite="INSAT-3D",
        sensor="IMAGER",
        product="TIR1",
        timestamp=datetime(2020, 5, 18, 6, 0, tzinfo=timezone.utc),
        file_path="granules/test.png",
        latitude_min=5.0,
        latitude_max=25.0,
        longitude_min=75.0,
        longitude_max=95.0,
        data_mode=DataMode.DEMO,
    )
    assert sat.satellite == "INSAT-3D"
    assert sat.latitude_max >= sat.latitude_min


# ---------------------------------------------------------------------------
# 3. EnvironmentalObservation Schema Validation
# ---------------------------------------------------------------------------
def test_environmental_observation_validation():
    env = EnvironmentalObservation(
        id="env-test-01",
        timestamp=datetime(2020, 5, 18, 0, 0, tzinfo=timezone.utc),
        latitude=15.0,
        longitude=86.5,
        sea_surface_temperature=31.2,
        rainfall=45.0,
        source="NASA_GPM_IMERG",
        data_mode=DataMode.DEMO,
    )
    assert env.sea_surface_temperature == 31.2
    assert env.rainfall == 45.0


# ---------------------------------------------------------------------------
# 4. Historical CSV Import
# ---------------------------------------------------------------------------
def test_historical_csv_import():
    adapter = HistoricalTrackAdapter()
    csv_data = (
        "storm_id,storm_name,timestamp,latitude,longitude,wind_speed,central_pressure,classification\n"
        "BOB-01-2020,AMPHAN,2020-05-18T00:00:00Z,13.2,86.3,120.0,925.0,Extremely Severe Cyclonic Storm\n"
        "BOB-01-2020,AMPHAN,2020-05-18T06:00:00Z,13.7,86.4,130.0,915.0,Super Cyclonic Storm\n"
    )
    observations, report = adapter.import_content(csv_data, file_format="csv")
    assert report.records_received == 2
    assert report.records_imported == 2
    assert report.records_rejected == 0
    assert len(observations) == 2
    assert observations[0].storm_id == "BOB-01-2020"


# ---------------------------------------------------------------------------
# 5. Historical JSON Import
# ---------------------------------------------------------------------------
def test_historical_json_import():
    adapter = HistoricalTrackAdapter()
    json_data = (
        '[{"storm_id": "ARB-01-2023", "storm_name": "BIPARJOY", '
        '"timestamp": "2023-06-08T06:00:00Z", "latitude": 13.9, "longitude": 66.0, '
        '"wind_speed": 75.0, "central_pressure": 980.0, "classification": "Very Severe Cyclonic Storm"}]'
    )
    observations, report = adapter.import_content(json_data, file_format="json")
    assert report.records_received == 1
    assert report.records_imported == 1
    assert report.records_rejected == 0
    assert observations[0].storm_id == "ARB-01-2023"


# ---------------------------------------------------------------------------
# 6. Invalid Latitude Rejection
# ---------------------------------------------------------------------------
def test_invalid_latitude_rejected():
    adapter = HistoricalTrackAdapter()
    csv_data = (
        "storm_id,storm_name,timestamp,latitude,longitude,wind_speed,central_pressure\n"
        "BAD-LAT,INVALID,2020-05-18T00:00:00Z,105.0,86.0,50.0,990.0\n"
    )
    observations, report = adapter.import_content(csv_data, file_format="csv")
    assert report.records_imported == 0
    assert report.records_rejected == 1
    assert any("Latitude 105.0 outside valid range" in err for err in report.validation_errors)


# ---------------------------------------------------------------------------
# 7. Invalid Longitude Rejection
# ---------------------------------------------------------------------------
def test_invalid_longitude_rejected():
    adapter = HistoricalTrackAdapter()
    csv_data = (
        "storm_id,storm_name,timestamp,latitude,longitude,wind_speed,central_pressure\n"
        "BAD-LON,INVALID,2020-05-18T00:00:00Z,15.0,-210.0,50.0,990.0\n"
    )
    observations, report = adapter.import_content(csv_data, file_format="csv")
    assert report.records_imported == 0
    assert report.records_rejected == 1
    assert any("Longitude -210.0 outside valid range" in err for err in report.validation_errors)


# ---------------------------------------------------------------------------
# 8. Invalid Timestamp Rejection
# ---------------------------------------------------------------------------
def test_invalid_timestamp_rejected():
    adapter = HistoricalTrackAdapter()
    csv_data = (
        "storm_id,storm_name,timestamp,latitude,longitude,wind_speed,central_pressure\n"
        "BAD-TS,INVALID,not-a-valid-date-time,15.0,85.0,50.0,990.0\n"
    )
    observations, report = adapter.import_content(csv_data, file_format="csv")
    assert report.records_imported == 0
    assert report.records_rejected == 1
    assert any("Invalid timestamp" in err for err in report.validation_errors)


# ---------------------------------------------------------------------------
# 9. Invalid File Type Rejection
# ---------------------------------------------------------------------------
def test_invalid_file_type_rejected(client):
    fake_sh = io.BytesIO(b"#!/bin/bash\necho bad\n")
    response = client.post(
        "/api/data/upload",
        files={"file": ("malicious.sh", fake_sh, "application/x-sh")},
    )
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 10. Satellite Upload & Dimension Extraction
# ---------------------------------------------------------------------------
def test_satellite_upload(client):
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color="blue")
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/data/upload",
        files={"file": ("test_granule.png", buf, "image/png")},
        data={"satellite_name": "INSAT-3D", "product_type": "TIR1"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["data_mode"] == "USER_UPLOAD"
    assert data["dimensions"]["width"] == 64
    assert data["dimensions"]["height"] == 64
    assert "USER UPLOADED DATA" in data["message"]


# ---------------------------------------------------------------------------
# 11. Provenance Endpoint
# ---------------------------------------------------------------------------
def test_provenance_endpoint(client):
    response = client.get("/api/provenance")
    assert response.status_code == 200
    records = response.json()
    assert isinstance(records, list)
    assert len(records) > 0
    assert "source" in records[0]
    assert "data_mode" in records[0]


# ---------------------------------------------------------------------------
# 12. Data Sources Endpoint
# ---------------------------------------------------------------------------
def test_data_sources_endpoint(client):
    response = client.get("/api/data/sources")
    assert response.status_code == 200
    sources = response.json()
    names = [s["name"] for s in sources]
    assert any("MOSDAC" in n for n in names)
    assert any("IMD" in n for n in names)
    assert any("NASA GPM" in n for n in names)


# ---------------------------------------------------------------------------
# 13. Demo Fallback Operational
# ---------------------------------------------------------------------------
def test_demo_fallback(client):
    response = client.get("/api/data/status")
    assert response.status_code == 200
    status_data = response.json()
    assert status_data["active_data_mode"] in ["LIVE_NRT", "DEMO", "HISTORICAL", "USER_UPLOAD"]
    assert status_data["total_cyclone_observations"] > 0
    assert "EXPERIMENTAL RESEARCH SYSTEM" in status_data["scientific_disclaimer"]


# ---------------------------------------------------------------------------
# 14. Database Unavailable Fallback Without Crashing
# ---------------------------------------------------------------------------
def test_database_unavailable_fallback(client):
    response = client.get("/api/data/status")
    assert response.status_code == 200
    data = response.json()
    assert "STANDALONE_IN_MEMORY_FALLBACK" in data["database_status"]
    assert data["is_database_connected"] is False


# ---------------------------------------------------------------------------
# 15. Existing Health & Cyclones Endpoints Preservation
# ---------------------------------------------------------------------------
def test_existing_endpoints_preserved(client):
    # Health endpoint
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # Cyclones endpoint
    res_cyclones = client.get("/api/cyclones")
    assert res_cyclones.status_code == 200
    cyclones = res_cyclones.json()
    assert len(cyclones) >= 2
    assert any(c["name"] == "AMPHAN" for c in cyclones)

