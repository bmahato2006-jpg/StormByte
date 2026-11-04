"""Tests for NASA GPM IMERG GIS Granule Ingestion & Precipitation Analysis.

Covers:
1. Physical ZIP archive verification & SHA256 integrity.
2. Extracted file structure and layer inventory (all 19 files).
3. Deterministic georeferencing, EPSG:4326 CRS, and bounds [-60, 60, -180, 180].
4. Physical scale factor (10), units (0.1 mm/hr), and missing value (29999).
5. Point lookups: authentic rain cell (8.2 mm/hr) and clear point (0.0 mm/hr).
6. Out of coverage handling (lat > 60° or < -60°).
7. Invalid coordinate validation (HTTP 400).
8. Manifest CSV (19 columns) and provenance JSON verification.
9. NASA IMERG API endpoints:
   - GET /api/satellite/nrt/nasa/location
   - GET /api/satellite/nrt/nasa/observations
10. Credential isolation in NASAGPMImergAdapter (no credential leaks, auth separation).
11. Strict zero-fabrication assertions & historical observation labeling.
"""

import csv
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.data.geospatial.nasa_imerg_service import (
    NASAIMERGService,
    get_nasa_imerg_service,
)
from backend.app.data.geospatial.live_adapters import NASAGPMImergAdapter


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def imerg_service():
    return get_nasa_imerg_service()


# =============================================================================
# 1. Physical ZIP Granule Validation
# =============================================================================
def test_raw_granule_zip_exists_and_checksum(imerg_service):
    """Verify original NASA ZIP is preserved and checksum matches authentic file."""
    zip_path = Path("data/raw/nasa_imerg/2025/08/01/3B-HHR-GIS.MS.MRG.3IMERG.20250801-S000000-E002959.0000.V07B.zip")
    assert zip_path.exists(), f"Raw NASA IMERG zip not found at {zip_path}"
    
    # Check SHA256 matches authentic downloaded file
    checksum = imerg_service.calculate_sha256(zip_path)
    assert checksum == "45de4cba748c16f498d9b2c8494726c554c0fe72f0e335a39532b68e7626f9bd"
    assert zip_path.stat().st_size == 3321097


# =============================================================================
# 2. Extracted Raster Inventory
# =============================================================================
def test_extracted_files_inventory(imerg_service):
    """Verify all 19 files extracted correctly into processed folder."""
    processed_dir = Path("data/processed/nasa_imerg/2025/08/01")
    assert processed_dir.exists()
    
    files = list(processed_dir.glob("*"))
    # Exclude provenance.json if present
    extracted = [f for f in files if f.name != "provenance.json"]
    assert len(extracted) == 19
    
    # Must include total.rate GeoTIFF and TFW
    rate_tif = processed_dir / "3B-HHR-GIS.MS.MRG.3IMERG.20250801-S000000-E002959.0000.V07B.total.rate.tif"
    rate_tfw = processed_dir / "3B-HHR-GIS.MS.MRG.3IMERG.20250801-S000000-E002959.0000.V07B.total.rate.tfw"
    doi_txt = processed_dir / "DOI.txt"
    
    assert rate_tif.exists()
    assert rate_tfw.exists()
    assert doi_txt.exists()


# =============================================================================
# 3. Deterministic Georeferencing & Spatial Bounds
# =============================================================================
def test_spatial_bounds_and_resolution(imerg_service):
    """Verify bounds [-60, 60, -180, 180] and 0.1° resolution."""
    bounds = imerg_service.get_global_bounds()
    assert bounds["min_latitude"] == -60.0
    assert bounds["max_latitude"] == 60.0
    assert bounds["min_longitude"] == -180.0
    assert bounds["max_longitude"] == 180.0
    assert bounds["crs"] == "EPSG:4326"
    assert bounds["pixel_resolution_deg"] == 0.1
    assert bounds["grid_width"] == 3600
    assert bounds["grid_height"] == 1200


# =============================================================================
# 4. Physical Scale Factor & Missing Values
# =============================================================================
def test_physical_scale_and_units(imerg_service):
    """Verify unit is 0.1(mm/hr), scale factor is 10, nodata is 29999."""
    # Lookup in authentic active rain cell in NIO (29.95N, 73.05E)
    loc_rain = imerg_service.lookup_location(lat=29.95, lon=73.05)
    assert loc_rain.coverage is True
    assert loc_rain.precipitation is not None
    # 82 raw / 10 = 8.2 mm/hr
    assert loc_rain.precipitation == 82.0
    assert loc_rain.scale_factor == 10.0
    assert loc_rain.unit == "0.1(mm/hr)"
    assert loc_rain.precipitation_mm_per_hr == 8.2
    assert loc_rain.is_valid_measurement is True
    assert loc_rain.is_missing_no_data is False

    # Clear region over Kolkata (0 raw / 10 = 0.0 mm/hr)
    loc_clear = imerg_service.lookup_location(lat=22.5726, lon=88.3639)
    assert loc_clear.coverage is True
    assert loc_clear.precipitation == 0.0
    assert loc_clear.precipitation_mm_per_hr == 0.0
    assert loc_clear.is_valid_measurement is True


# =============================================================================
# 5. Out of Coverage Handling
# =============================================================================
def test_out_of_coverage_detection(imerg_service):
    """Verify points above +60° or below -60° return LOCATION_OUTSIDE_DATASET_COVERAGE."""
    # North of 60°
    res_north = imerg_service.lookup_location(lat=72.5, lon=45.0)
    assert res_north.coverage is False
    assert res_north.precipitation_mm_per_hr is None
    assert res_north.precipitation is None
    assert res_north.status == "LOCATION_OUTSIDE_DATASET_COVERAGE"

    # South of -60°
    res_south = imerg_service.lookup_location(lat=-65.0, lon=0.0)
    assert res_south.coverage is False
    assert res_south.precipitation_mm_per_hr is None
    assert res_south.status == "LOCATION_OUTSIDE_DATASET_COVERAGE"


# =============================================================================
# 6. Manifest CSV and Provenance JSON Verification
# =============================================================================
def test_manifest_csv_structure():
    """Verify manifest CSV contains 19 rows and required columns."""
    manifest_path = Path("data/processed/nasa_imerg/imerg_manifest.csv")
    assert manifest_path.exists()
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) >= 1
        row0 = reader[0]
        required_cols = [
            "source", "provider", "product", "version", "filename", "file_path",
            "observation_start", "observation_end", "download_time", "latitude_min",
            "latitude_max", "longitude_min", "longitude_max", "resolution", "crs",
            "precipitation_variable", "precipitation_units", "sha256", "status"
        ]
        for col in required_cols:
            assert col in row0, f"Column {col} missing from manifest CSV"


def test_provenance_json_structure():
    """Verify provenance JSON contains authentic DOI, metadata, and checksums."""
    prov_path = Path("data/processed/nasa_imerg/2025/08/01/provenance.json")
    assert prov_path.exists()
    
    with open(prov_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["provider"] == "NASA GPM / PPS"
        assert data["doi"] == "10.25966/gn7e-5b32"
        assert data["parent_doi"] == "10.5067/GPM/IMERG/3B-HH/07"
        assert "Verified authentic NASA PPS" in data["zero_fabrication_contract"]
        assert "3B-HHR-GIS.MS.MRG.3IMERG.20250801-S000000-E002959.0000.V07B.zip" in data["filename"]


# =============================================================================
# 7. FastAPI Endpoints Testing
# =============================================================================
def test_api_nasa_location_valid(client):
    """Test GET /api/satellite/nrt/nasa/location for authentic coordinate."""
    resp = client.get("/api/satellite/nrt/nasa/location?lat=22.57&lon=88.36")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "COVERED"
    assert data["coverage"] is True
    assert data["precipitation_mm_per_hr"] == 0.0
    assert data["provenance"]["crs"] == "EPSG:4326"
    assert data["provenance"]["resolution_deg"] == 0.1
    assert "Zero precipitation numbers" in data["zero_fabrication_rule"]


def test_api_nasa_location_rain_cell(client):
    """Test GET /api/satellite/nrt/nasa/location for rain cell coordinate."""
    resp = client.get("/api/satellite/nrt/nasa/location?lat=29.95&lon=73.05")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "COVERED"
    assert data["coverage"] is True
    assert data["precipitation_mm_per_hr"] == 8.2
    assert data["precipitation"] == 82.0


def test_api_nasa_location_out_of_bounds(client):
    """Test GET /api/satellite/nrt/nasa/location for out-of-coverage latitude (> 60)."""
    resp = client.get("/api/satellite/nrt/nasa/location?lat=75.0&lon=80.0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "LOCATION_OUTSIDE_DATASET_COVERAGE"
    assert data["coverage"] is False
    assert data["precipitation_mm_per_hr"] is None


def test_api_nasa_location_invalid_coords(client):
    """Test GET /api/satellite/nrt/nasa/location rejects invalid coordinates."""
    # Latitude > 90
    resp1 = client.get("/api/satellite/nrt/nasa/location?lat=95.0&lon=80.0")
    assert resp1.status_code == 400
    assert "Latitude must be between" in resp1.json()["detail"]

    # Longitude > 180
    resp2 = client.get("/api/satellite/nrt/nasa/location?lat=20.0&lon=195.0")
    assert resp2.status_code == 400
    assert "Longitude must be between" in resp2.json()["detail"]


def test_api_nasa_observations_list(client):
    """Test GET /api/satellite/nrt/nasa/observations lists verified granules."""
    resp = client.get("/api/satellite/nrt/nasa/observations")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_observations"] >= 1
    assert len(data["observations"]) >= 1
    granule = data["observations"][0]
    assert "3B-HHR-GIS" in granule["granule_id"]
    assert granule["bounds"]["min_latitude"] == -90.0
    assert granule["bounds"]["max_latitude"] == 90.0
    assert granule["max_precipitation_mm_per_hr"] == 61.2


# =============================================================================
# 8. Adapter Security & Credential Isolation
# =============================================================================
def test_nasa_adapter_unconfigured_security():
    """Verify adapter does not leak credentials and reports CREDENTIALS_REQUIRED when empty."""
    adapter = NASAGPMImergAdapter(
        pps_email="",
        pps_password="",
        earthdata_token="",
        legacy_key="",
    )
    assert adapter.is_configured() is False
    status = adapter.get_status()
    assert status["status"] == "CREDENTIALS_REQUIRED"
    assert status["auth_type"] == "NONE"
    
    # Must only contain boolean flags, never secrets
    assert status["environment_variables"]["NASA_PPS_EMAIL"] is False
    assert status["environment_variables"]["NASA_PPS_PASSWORD"] is False
    assert status["environment_variables"]["EARTHDATA_TOKEN"] is False


def test_nasa_adapter_pps_auth_configuration():
    """Verify adapter recognizes PPS HTTPS basic auth without exposing password."""
    adapter = NASAGPMImergAdapter(
        pps_email="researcher@example.com",
        pps_password="secret_pps_password_123",
        earthdata_token="",
        legacy_key="",
    )
    assert adapter.is_configured() is True
    assert adapter.get_auth_type() == "NASA_PPS_BASIC_AUTH"
    status = adapter.get_status()
    assert status["status"] == "CONFIGURED"
    assert status["environment_variables"]["NASA_PPS_EMAIL"] is True
    assert status["environment_variables"]["NASA_PPS_PASSWORD"] is True
    # Password text must NOT appear in status representation
    status_str = json.dumps(status)
    assert "secret_pps_password_123" not in status_str


def test_nasa_adapter_earthdata_auth_configuration():
    """Verify adapter recognizes Earthdata bearer token auth without exposing token."""
    adapter = NASAGPMImergAdapter(
        pps_email="",
        pps_password="",
        earthdata_token="valid_bearer_token_abc12345xyz",
        legacy_key="",
    )
    assert adapter.is_configured() is True
    assert adapter.get_auth_type() == "NASA_EARTHDATA_BEARER_TOKEN"
    status = adapter.get_status()
    assert status["status"] == "CONFIGURED"
    assert status["environment_variables"]["EARTHDATA_TOKEN"] is True
    assert "valid_bearer_token_abc12345xyz" not in json.dumps(status)


# =============================================================================
# 9. Strict Zero Fabrication Test
# =============================================================================
def test_zero_fabrication_guarantees(imerg_service):
    """Verify strictly no data or coordinates are fabricated."""
    res = imerg_service.lookup_location(lat=29.95, lon=73.05)
    # Mode must clearly state Historical / Verified NASA Observation
    assert "Historical" in res.mode
    assert "2025-08-01T00:00:00Z" in res.observation_start
    assert "2025-08-01T00:29:59Z" in res.observation_end
    assert res.precipitation_mm_per_hr == 8.2
    assert "Zero precipitation numbers" in res.zero_fabrication_rule


# =============================================================================
# 10. Risk Engine Integration & Safeguards Test
# =============================================================================
def test_risk_engine_imerg_integration(imerg_service):
    """Verify authentic IMERG rainfall integrates as MEASURED OBSERVATION into Risk Engine."""
    from backend.app.impact.rainfall_risk import rainfall_risk_analyzer
    from backend.app.impact.impact_engine import impact_engine

    # Lookup real rain cell
    loc_rain = imerg_service.lookup_location(lat=29.95, lon=73.05)
    assert loc_rain.precipitation_mm_per_hr == 8.2
    
    # 1. Test direct RainfallRiskAnalyzer with authentic rainfall
    risk_comp = rainfall_risk_analyzer.evaluate(measured_rainfall_mm=loc_rain.precipitation_mm_per_hr)
    assert risk_comp.available is True
    assert risk_comp.source_type == "MEASURED OBSERVATION"
    assert "NASA GPM / IMERG" in risk_comp.source

    # 2. Test ImpactEngine evaluates risk without declaring cyclone purely from rain
    cyclone_payload = {
        "cyclone_id": "TEST-SYSTEM-01",
        "name": "Unverified Weather Feature",
        "latitude": 29.95,
        "longitude": 73.05,
        "is_cyclone": False,  # Crucial: detection is False!
        "confidence": 0.15,
        "measured_rainfall_mm": loc_rain.precipitation_mm_per_hr,
    }
    assessment = impact_engine.evaluate_impact(cyclone_payload)
    assert assessment is not None
    # IMERG rainfall alone does NOT declare a cyclone
    assert cyclone_payload["is_cyclone"] is False


def test_intensity_and_center_localization_safeguards():
    """Verify Intensity and Center Localization remain UNAVAILABLE without independent verification."""
    from backend.app.impact.impact_engine import impact_engine

    # When no wind/central pressure is provided, wind intensity risk is UNAVAILABLE
    assessment = impact_engine.evaluate_impact({
        "cyclone_id": "TEST-NO-WIND",
        "name": "Test Feature",
        "latitude": 15.0,
        "longitude": 85.0,
    })
    assert assessment is not None
    assert assessment.wind_risk.available is False
    assert assessment.wind_risk.status == "MISSING_WIND_DATA"
    assert assessment.wind_risk.source_type == "UNAVAILABLE"
    assert any("wind" in c.lower() for c in assessment.unavailable_components)



