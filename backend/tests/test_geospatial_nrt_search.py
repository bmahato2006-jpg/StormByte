"""Tests for Location-Aware NRT Satellite Geospatial Ingestion & Location Search (Phase 11).

Verifies the 12 mandatory specifications:
1. valid latitude/longitude metadata
2. latitude/longitude grid (1D and 2D)
3. CRS transformation (projected CRS -> WGS84 EPSG:4326)
4. bounding-box calculation (deterministic)
5. requested location inside coverage
6. requested location outside coverage
7. missing geospatial metadata (GEOSPATIAL_METADATA_REQUIRED)
8. invalid latitude rejection ([-90, +90])
9. invalid longitude rejection ([-180, +180])
10. longitude wrapping / dateline handling
11. provenance preservation (MOSDAC/ISRO, dataset_id, granule_id, bounds)
12. no coordinate fabrication (strictly zero pixel inference or fake coordinates)
"""

import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.data.geospatial.nrt_service import (
    NRTObservationService,
    calculate_bounds_from_grid,
    calculate_bounds_from_transform,
    haversine_distance_km,
    is_point_in_bounds,
)
from backend.app.data.geospatial.schemas import (
    LocationAwareNRTObservation,
    NRTLatLonSearchResponse,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def clean_nrt_service():
    """Returns an isolated fresh instance of NRTObservationService."""
    return NRTObservationService()


# =============================================================================
# 1. Valid Latitude/Longitude Metadata Test
# =============================================================================
def test_valid_lat_lon_metadata(clean_nrt_service):
    """Test 1: Preserves authentic latitude and longitude without pixel guessing."""
    ok, obs, status_msg = clean_nrt_service.register_observation(
        granule_id="TEST_GRANULE_001",
        satellite="INSAT-3D",
        sensor="IMAGER",
        dataset_id="3DIMG_L1B_STD",
        acquisition_time="2020-05-18T06:00:00Z",
        provider="MOSDAC/ISRO",
        latitude=15.25,
        longitude=86.50,
        min_latitude=10.0,
        max_latitude=20.0,
        min_longitude=80.0,
        max_longitude=92.0,
        crs="EPSG:4326",
    )
    assert ok is True
    assert obs is not None
    assert obs.latitude == 15.25
    assert obs.longitude == 86.50
    assert obs.min_latitude == 10.0
    assert obs.max_latitude == 20.0
    assert obs.min_longitude == 80.0
    assert obs.max_longitude == 92.0
    assert obs.crs == "EPSG:4326"
    assert obs.is_location_ready is True
    assert status_msg == "CALIBRATED_GEOREFERENCE"


# =============================================================================
# 2. Latitude/Longitude Grid Preservation & Dimension Test
# =============================================================================
def test_lat_lon_grid(clean_nrt_service):
    """Test 2: Preserves 1D/2D coordinate arrays, records dimensions and min/max extents."""
    # 2D curvilinear grid: 10 rows x 20 cols
    lats_2d = np.linspace(12.0, 18.0, 10)[:, None] + np.zeros((10, 20), dtype=np.float32)
    lons_2d = np.zeros((10, 20), dtype=np.float32) + np.linspace(82.0, 88.0, 20)[None, :]

    min_lat, max_lat, min_lon, max_lon, c_lat, c_lon, dims = calculate_bounds_from_grid(lats_2d, lons_2d)
    assert dims == [10, 20]
    assert min_lat == 12.0
    assert max_lat == 18.0
    assert min_lon == 82.0
    assert max_lon == 88.0
    assert c_lat == 15.0
    assert c_lon == 85.0

    # Register via grid
    ok, obs, status_msg = clean_nrt_service.register_observation(
        granule_id="TEST_GRID_002",
        satellite="INSAT-3DR",
        sensor="IMAGER",
        dataset_id="3RIMG_L1B_STD",
        acquisition_time="2023-06-11T03:00:00Z",
        provider="MOSDAC/ISRO",
        lat_grid=lats_2d,
        lon_grid=lons_2d,
    )
    assert ok is True
    assert obs.grid_dimensions == [10, 20]
    assert obs.min_latitude == 12.0
    assert obs.max_latitude == 18.0
    assert obs.latitude == 15.0
    assert obs.longitude == 85.0


# =============================================================================
# 3. CRS Transformation Test (Projected CRS to WGS84)
# =============================================================================
def test_crs_transformation():
    """Test 3: Deterministically transforms coordinates from a projected CRS to EPSG:4326."""
    # UTM Zone 45N (EPSG:32645 covers Bay of Bengal / Kolkata)
    # Affine transform: pixel size 1000m, origin (500000, 2400000)
    transform = [1000.0, 0.0, 500000.0, 0.0, -1000.0, 2400000.0]
    width, height = 100, 100
    crs = "EPSG:32645"

    min_lat, max_lat, min_lon, max_lon, c_lat, c_lon = calculate_bounds_from_transform(
        transform, width, height, crs
    )

    # Verify coordinates fall within realistic Bay of Bengal / Bengal Basin lat/lon
    assert 20.0 <= min_lat < max_lat <= 25.0
    assert 87.0 <= min_lon < max_lon <= 92.0
    assert min_lat <= c_lat <= max_lat
    assert min_lon <= c_lon <= max_lon


# =============================================================================
# 4. Bounding-Box Calculation Test
# =============================================================================
def test_bounding_box_calculation():
    """Test 4: Deterministic calculation of bounding box without pixel estimation."""
    # Direct EPSG:4326 raster transform: dx=0.04 deg, dy=-0.04 deg, top-left=(80.0, 22.0)
    transform = [0.04, 0.0, 80.0, 0.0, -0.04, 22.0]
    width, height = 200, 150  # 8.0 deg wide, 6.0 deg high

    min_lat, max_lat, min_lon, max_lon, c_lat, c_lon = calculate_bounds_from_transform(
        transform, width, height, "EPSG:4326"
    )

    assert min_lat == round(22.0 - 150 * 0.04, 4)  # 16.0
    assert max_lat == 22.0
    assert min_lon == 80.0
    assert max_lon == round(80.0 + 200 * 0.04, 4)  # 88.0
    assert c_lat == 19.0
    assert c_lon == 84.0


# =============================================================================
# 5. Requested Location Inside Coverage Test (API)
# =============================================================================
def test_requested_location_inside_coverage(client):
    """Test 5: Coordinate inside authentic granule returns coverage=true, nearest distance, and detection."""
    # INSAT-3D Amphan sample covers [5.0, 25.0]N and [75.0, 95.0]E
    # Query Bay of Bengal coordinate (15.0°N, 85.0°E)
    response = client.get("/api/satellite/nrt/search?lat=15.0&lon=85.0")
    assert response.status_code == 200
    data = response.json()

    assert data["coverage"] is True
    assert data["status"] == "COVERED"
    assert data["requested_location"]["latitude"] == 15.0
    assert data["requested_location"]["longitude"] == 85.0
    assert data["observation"] is not None
    assert "granule_id" in data["observation"]
    assert "bounds" in data["observation"]
    assert data["observation"]["bounds"]["min_latitude"] <= 15.0 <= data["observation"]["bounds"]["max_latitude"]
    assert data["observation"]["bounds"]["min_longitude"] <= 85.0 <= data["observation"]["bounds"]["max_longitude"]
    assert data["distance_km"] is not None
    assert data["distance_km"] >= 0.0
    assert data["source"] in ("MOSDAC/ISRO", "NASA_EARTHDATA")
    assert "detection" in data


# =============================================================================
# 6. Requested Location Outside Coverage Test (API)
# =============================================================================
def test_requested_location_outside_coverage(client):
    """Test 6: Coordinate outside coverage returns coverage=false and LOCATION_OUTSIDE_GRANULE_COVERAGE."""
    # North Atlantic coordinate (55.0°N, -30.0°E) - completely outside NIO
    response = client.get("/api/satellite/nrt/search?lat=55.0&lon=-30.0")
    assert response.status_code == 200
    data = response.json()

    assert data["coverage"] is False
    assert data["status"] == "LOCATION_OUTSIDE_GRANULE_COVERAGE"
    assert data["observation"] is None
    assert data["distance_km"] is None
    assert data["requested_location"]["latitude"] == 55.0
    assert data["requested_location"]["longitude"] == -30.0
    assert "strictly prohibits fabricating" in data["message"]


# =============================================================================
# 7. Missing Geospatial Metadata Test (Rejection)
# =============================================================================
def test_missing_geospatial_metadata(clean_nrt_service):
    """Test 7: Granule missing geospatial bounds is rejected with status GEOSPATIAL_METADATA_REQUIRED."""
    ok, obs, status_msg = clean_nrt_service.register_observation(
        granule_id="MISSING_GEO_GRANULE",
        satellite="UNKNOWN_SAT",
        sensor="IMAGER",
        dataset_id="UNREFERENCED_DATASET",
        acquisition_time="2024-01-01T00:00:00Z",
        provider="UNKNOWN",
        latitude=None,
        longitude=None,
        min_latitude=None,
        max_latitude=None,
        min_longitude=None,
        max_longitude=None,
    )
    assert ok is False
    assert obs is None
    assert status_msg == "GEOSPATIAL_METADATA_REQUIRED"


# =============================================================================
# 8. Invalid Latitude Test (API)
# =============================================================================
def test_invalid_latitude(client):
    """Test 8: Latitude outside [-90.0, +90.0] is rejected with HTTP 400."""
    response = client.get("/api/satellite/nrt/search?lat=95.5&lon=85.0")
    assert response.status_code == 400
    assert "Invalid latitude" in response.json()["detail"]

    response_neg = client.get("/api/satellite/nrt/search?lat=-91.0&lon=85.0")
    assert response_neg.status_code == 400
    assert "Invalid latitude" in response_neg.json()["detail"]


# =============================================================================
# 9. Invalid Longitude Test (API)
# =============================================================================
def test_invalid_longitude(client):
    """Test 9: Longitude outside [-180.0, +180.0] is rejected with HTTP 400."""
    response = client.get("/api/satellite/nrt/search?lat=15.0&lon=185.0")
    assert response.status_code == 400
    assert "Invalid longitude" in response.json()["detail"]

    response_neg = client.get("/api/satellite/nrt/search?lat=15.0&lon=-185.0")
    assert response_neg.status_code == 400
    assert "Invalid longitude" in response_neg.json()["detail"]


# =============================================================================
# 10. Longitude Wrapping & Dateline Handling Test
# =============================================================================
def test_longitude_wrapping_dateline():
    """Test 10: Correct distance calculation and point-in-bounds check across antimeridian."""
    # Bounding box spanning dateline: [10.0, 20.0]N, [170.0, -170.0]E (170E to 190E / 170W)
    min_lat, max_lat = 10.0, 20.0
    min_lon, max_lon = 170.0, -170.0

    # Point at 175.0E is inside
    assert is_point_in_bounds(15.0, 175.0, min_lat, max_lat, min_lon, max_lon) is True
    # Point at -175.0E (175W) is inside
    assert is_point_in_bounds(15.0, -175.0, min_lat, max_lat, min_lon, max_lon) is True
    # Point at 160.0E is outside
    assert is_point_in_bounds(15.0, 160.0, min_lat, max_lat, min_lon, max_lon) is False

    # Distance calculation wrapping test: 179°E to -179°E is only 2 degrees apart (~222 km)
    dist = haversine_distance_km(0.0, 179.0, 0.0, -179.0)
    assert 220.0 <= dist <= 225.0


# =============================================================================
# 11. Provenance Preservation Test
# =============================================================================
def test_provenance_preservation(client):
    """Test 11: Valid search preserves provider, dataset_id, granule_id, and verified bounds."""
    response = client.get("/api/satellite/nrt/search?lat=15.0&lon=85.0")
    assert response.status_code == 200
    data = response.json()

    assert "provenance" in data
    prov = data["provenance"]
    assert prov is not None
    assert "granule_id" in prov
    assert "provider" in prov
    assert "dataset_id" in prov
    assert "satellite" in prov
    assert "crs" in prov
    assert "verified_bounds" in prov
    assert "analysis_timestamp_utc" in prov


# =============================================================================
# 12. No Coordinate Fabrication Rule Test
# =============================================================================
def test_no_coordinate_fabrication(clean_nrt_service):
    """Test 12: Zero fabrication rule: never assigns default or estimated coordinates."""
    # When a search is conducted for a location with no covering granule
    res = clean_nrt_service.search_by_location(lat=-60.0, lon=-100.0)
    assert res.coverage is False
    assert res.observation is None
    assert res.distance_km is None
    # No fallback coordinates are placed into observation
    assert "Zero coordinates, boundaries, or detections are fabricated" in res.zero_fabrication_rule

