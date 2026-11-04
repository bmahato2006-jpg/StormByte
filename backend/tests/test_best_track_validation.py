"""Tests for Best Track Data Validation and Quarantine Routing."""

from datetime import datetime, timezone
import pytest
from backend.app.data.best_track.normalizer import BestTrackNormalizer
from backend.app.data.best_track.validator import BestTrackValidator


def test_valid_fix_passes_audit():
    raw = {
        "storm_id": "IMD-2020-AMPHAN",
        "storm_name": "AMPHAN",
        "timestamp": "2020-05-16T06:00:00Z",
        "latitude": 10.9,
        "longitude": 86.3,
        "max_wind_kt": 45.0,
        "central_pressure_hpa": 990.0,
    }
    fix = BestTrackNormalizer.normalize_record(raw, source="IMD_RSMC_NEW_DELHI", source_file="amphan.xlsx")
    assert fix is not None
    is_valid, reason = BestTrackValidator.validate_fix(fix)
    assert is_valid is True
    assert reason is None


def test_out_of_bounds_coords_quarantined():
    raw = {
        "storm_id": "IMD-2020-TEST",
        "storm_name": "TEST",
        "timestamp": "2020-05-16T06:00:00Z",
        "latitude": 75.0,  # Outside North Indian Ocean
        "longitude": 86.3,
    }
    fix = BestTrackNormalizer.normalize_record(raw, source="IMD_RSMC_NEW_DELHI", source_file="test.xlsx")
    assert fix is not None
    is_valid, reason = BestTrackValidator.validate_fix(fix)
    assert is_valid is False
    assert "outside North Indian Ocean" in reason


def test_unphysical_wind_quarantined():
    raw = {
        "storm_id": "IMD-2020-TEST",
        "storm_name": "TEST",
        "timestamp": "2020-05-16T06:00:00Z",
        "latitude": 12.0,
        "longitude": 85.0,
        "max_wind_kt": 450.0,  # Unphysical
    }
    fix = BestTrackNormalizer.normalize_record(raw, source="IMD_RSMC_NEW_DELHI", source_file="test.xlsx")
    assert fix is not None
    is_valid, reason = BestTrackValidator.validate_fix(fix)
    assert is_valid is False
    assert "Unphysical wind speed" in reason
