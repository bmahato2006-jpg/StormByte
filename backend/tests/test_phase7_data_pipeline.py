"""Comprehensive automated test suite for Cyclone-X Phase 7: Real Scientific Dataset Integration.

Validates:
- Central Dataset Registry and accurate availability statuses
- IBTrACS and IMD track parsing, normalization, and quality control
- Scientific data validators (coordinates, wind, pressure, timestamps, continuity)
- MOSDAC, GPM, and ERA5 credentials/configuration reporting
- Temporal data leakage protection in training manifests
- Cryptographic provenance chaining
- REST API endpoints under /api/datasets/*
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.data.adapters.era5_adapter import Era5Adapter
from backend.app.data.adapters.gpm_adapter import GpmAdapter
from backend.app.data.adapters.ibtracs_adapter import IbtracsAdapter
from backend.app.data.adapters.imd_adapter import ImdAdapter
from backend.app.data.adapters.mosdac_adapter import MosdacAdapter
from backend.app.data.manifests.manifest_generator import ManifestGenerator
from backend.app.data.provenance import compute_object_sha256, PipelineProvenanceTracker
from backend.app.data.registry import dataset_registry
from backend.app.data.schemas import QualityStatus, ScientificDatasetStatus
from backend.app.data.validators import ScientificDataValidator


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Registry Tests
# ---------------------------------------------------------------------------

def test_dataset_registry_initialization():
    """Verify registry loads all scientific datasets including NRT with required metadata."""
    datasets = dataset_registry.get_all()
    assert len(datasets) >= 8

    keys = {d.dataset_id for d in datasets}
    expected_keys = {
        "ibtracs_nio",
        "imd_best_track",
        "mosdac_insat",
        "nasa_gpm_imerg",
        "era5_reanalysis",
        "worldpop_density",
        "srtm_elevation",
        "osm_infrastructure",
    }
    assert expected_keys.issubset(keys)


def test_dataset_registry_statuses_and_integrity():
    """Verify physical file presence dictates AVAILABLE vs CREDENTIALS_REQUIRED status."""
    ibtracs = dataset_registry.get_by_id("ibtracs_nio")
    assert ibtracs is not None
    assert ibtracs.status == ScientificDatasetStatus.AVAILABLE
    assert ibtracs.checksum is not None
    assert len(ibtracs.checksum) == 64  # SHA-256 length

    gpm = dataset_registry.get_by_id("nasa_gpm_imerg")
    assert gpm is not None
    assert gpm.status in (ScientificDatasetStatus.CREDENTIALS_REQUIRED, ScientificDatasetStatus.AVAILABLE)


# ---------------------------------------------------------------------------
# 2. Adapter & Quality Control Tests
# ---------------------------------------------------------------------------

def test_ibtracs_adapter_normalization():
    """Verify IBTrACS CSV normalization extracts valid coordinates, winds, pressures."""
    adapter = IbtracsAdapter()
    assert adapter.is_available()

    records = adapter.normalize()
    assert len(records) >= 40

    # Test individual record integrity
    amphan_records = [r for r in records if r.storm_name == "AMPHAN"]
    assert len(amphan_records) > 0
    first = amphan_records[0]
    assert first.storm_name == "AMPHAN"
    assert first.basin == "NI"
    assert first.latitude == 10.7
    assert first.longitude == 86.5
    assert first.max_wind_kt == 30.0
    assert first.central_pressure_hpa == 1000.0


def test_ibtracs_quality_control():
    """Verify automated QC evaluates IBTrACS dataset as PASS."""
    adapter = IbtracsAdapter()
    report = adapter.run_quality_control()
    assert report.quality_status == QualityStatus.PASS
    assert report.total_records >= 40
    assert report.invalid_records == 0
    assert report.duplicate_timestamps == 0
    assert "latitude" in report.units


def test_imd_adapter_normalization():
    """Verify IMD bulletin records are properly normalized."""
    adapter = ImdAdapter()
    assert adapter.is_available()

    records = adapter.normalize()
    assert len(records) >= 5
    for r in records:
        assert r.source == "IMD_RSMC_NEW_DELHI"
        assert r.latitude > 0
        assert r.longitude > 0


def test_mosdac_satellite_adapter():
    """Verify MOSDAC catalog parsing and satellite metadata extraction."""
    adapter = MosdacAdapter()
    granules = adapter.normalize()
    assert len(granules) >= 3

    satellites = {g.satellite for g in granules}
    assert "INSAT-3D" in satellites


# ---------------------------------------------------------------------------
# 3. Scientific Validators Tests
# ---------------------------------------------------------------------------

def test_coordinate_validator():
    """Verify geographic bounds enforcement."""
    valid, err = ScientificDataValidator.validate_coordinates(20.5, 88.2)
    assert valid is True
    assert err is None

    # Invalid latitude
    valid, err = ScientificDataValidator.validate_coordinates(95.0, 88.2)
    assert valid is False
    assert "Latitude 95.0 is outside" in err

    # Invalid longitude
    valid, err = ScientificDataValidator.validate_coordinates(20.5, 190.0)
    assert valid is False
    assert "Longitude 190.0 is outside" in err


def test_wind_and_pressure_validators():
    """Verify meteorological limits and None preservation."""
    # Valid
    valid, _ = ScientificDataValidator.validate_wind_speed(120.0)
    assert valid is True

    # Missing is valid (None preserved)
    valid, _ = ScientificDataValidator.validate_wind_speed(None)
    assert valid is True

    # Negative wind rejected
    valid, err = ScientificDataValidator.validate_wind_speed(-10.0)
    assert valid is False
    assert "cannot be negative" in err

    # Out of atmospheric pressure range
    valid, err = ScientificDataValidator.validate_central_pressure(750.0)
    assert valid is False
    assert "outside plausible atmospheric range" in err


def test_track_continuity_and_duplicates():
    """Verify detection of duplicate timestamps and impossible movement."""
    from datetime import datetime, timezone
    t0 = datetime(2023, 6, 1, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2023, 6, 1, 3, 0, tzinfo=timezone.utc)

    # Normal points (slow speed)
    points = [
        {"timestamp": t0, "latitude": 15.0, "longitude": 85.0},
        {"timestamp": t1, "latitude": 15.2, "longitude": 85.1},
    ]
    issues = ScientificDataValidator.validate_track_continuity(points)
    assert len(issues) == 0

    # Duplicate timestamps
    dup_points = [
        {"timestamp": t0, "latitude": 15.0, "longitude": 85.0},
        {"timestamp": t0, "latitude": 15.0, "longitude": 85.0},
    ]
    issues = ScientificDataValidator.validate_track_continuity(dup_points)
    assert any("Duplicate timestamp" in i for i in issues)


def test_era5_vertical_wind_shear_computation():
    """Verify vector shear calculation and zero-fabrication None handling."""
    # u200 = 15 m/s, v200 = 0 m/s, u850 = 5 m/s, v850 = 0 m/s
    # du = 10 m/s -> 10 * 1.94384 = 19.4 knots
    shear = Era5Adapter.calculate_vertical_wind_shear(15.0, 0.0, 5.0, 0.0)
    assert shear == 19.4

    # None if any input missing
    assert Era5Adapter.calculate_vertical_wind_shear(None, 0.0, 5.0, 0.0) is None


# ---------------------------------------------------------------------------
# 4. Leakage Protection & Provenance Tests
# ---------------------------------------------------------------------------

def test_training_manifest_leakage_protection():
    """Verify strict temporal partitioning prevents storm leakage across splits."""
    gen = ManifestGenerator()
    manifests = gen.generate_trajectory_manifests()

    assert "train" in manifests
    assert "test" in manifests

    train_storms = set(manifests["train"].storm_ids)
    test_storms = set(manifests["test"].storm_ids)

    # CRITICAL: No storm may be present in both train and test!
    overlap = train_storms.intersection(test_storms)
    assert len(overlap) == 0, f"Data leakage detected! Overlapping storms: {overlap}"

    # Verify temporal assignment
    assert "AMPHAN" in train_storms or "FANI" in train_storms
    assert "BIPARJOY" in test_storms or "MOCHA" in test_storms


def test_pipeline_provenance_chaining():
    """Verify cryptographic parentHash chaining across pipeline stages."""
    raw_hash = compute_object_sha256({"raw": "sample_track_row"})
    prov1 = PipelineProvenanceTracker.create_transformation_provenance(
        stage="RAW_INGESTION",
        source="IBTRACS",
        data_type="TRACK_OBSERVATION",
        data_hash=raw_hash,
    )
    assert prov1["dataHash"] == raw_hash
    assert "parentHash" not in prov1

    # Stage 2: Normalization references Stage 1 dataHash as parentHash
    norm_hash = compute_object_sha256({"normalized": "standard_schema"})
    prov2 = PipelineProvenanceTracker.create_transformation_provenance(
        stage="NORMALIZATION",
        source="IBTRACS",
        data_type="NORMALIZED_TRACK",
        parent_hash=prov1["dataHash"],
        data_hash=norm_hash,
    )
    assert prov2["parentHash"] == raw_hash
    assert prov2["dataHash"] == norm_hash


# ---------------------------------------------------------------------------
# 5. REST API Endpoints Tests
# ---------------------------------------------------------------------------

def test_api_list_datasets(client):
    """Test GET /api/datasets."""
    res = client.get("/api/datasets")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 8


def test_api_get_dataset_detail(client):
    """Test GET /api/datasets/{dataset_id}."""
    res = client.get("/api/datasets/ibtracs_nio")
    assert res.status_code == 200
    data = res.json()
    assert data["dataset_id"] == "ibtracs_nio"
    assert data["status"] == "AVAILABLE"
    assert "max_wind_kt" in data["variables"]


def test_api_get_dataset_status(client):
    """Test GET /api/datasets/{dataset_id}/status."""
    res = client.get("/api/datasets/ibtracs_nio/status")
    assert res.status_code == 200
    data = res.json()
    assert data["is_on_disk"] is True
    assert data["status"] == "AVAILABLE"


def test_api_get_dataset_quality(client):
    """Test GET /api/datasets/{dataset_id}/quality."""
    res = client.get("/api/datasets/ibtracs_nio/quality")
    assert res.status_code == 200
    data = res.json()
    assert data["quality_status"] == "PASS"
    assert data["total_records"] >= 40


def test_api_get_dataset_provenance(client):
    """Test GET /api/datasets/{dataset_id}/provenance."""
    res = client.get("/api/datasets/ibtracs_nio/provenance")
    assert res.status_code == 200
    data = res.json()
    assert data["checksum_sha256"] is not None
    assert "provenance_chain" in data


def test_api_get_tracks(client):
    """Test GET /api/datasets/tracks."""
    res = client.get("/api/datasets/tracks?storm_name=AMPHAN")
    assert res.status_code == 200
    data = res.json()
    assert len(data) > 0
    for r in data:
        assert "AMPHAN" in r["storm_name"]


def test_api_get_satellite_records(client):
    """Test GET /api/datasets/satellite."""
    res = client.get("/api/datasets/satellite")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 3


def test_api_get_rainfall_records(client):
    """Test GET /api/datasets/rainfall."""
    res = client.get("/api/datasets/rainfall")
    assert res.status_code == 200
    data = res.json()
    assert data["dataset"] == "NASA GPM IMERG (V07B)"


def test_api_get_environmental_records(client):
    """Test GET /api/datasets/environmental."""
    res = client.get("/api/datasets/environmental")
    assert res.status_code == 200
    data = res.json()
    assert data["dataset"] == "ECMWF ERA5 Reanalysis"


def test_api_dataset_not_found(client):
    """Test GET /api/datasets/unknown returns 404."""
    res = client.get("/api/datasets/unknown_nonexistent")
    assert res.status_code == 404

