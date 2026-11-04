"""Tests for Best Track Normalization Rules (Zero Fabrication & UTC)."""

from datetime import datetime, timezone
import pytest
from backend.app.data.best_track.normalizer import BestTrackNormalizer


def test_longitude_normalization_range():
    assert BestTrackNormalizer.normalize_longitude(88.5) == 88.5
    assert BestTrackNormalizer.normalize_longitude(-88.5) == -88.5
    assert BestTrackNormalizer.normalize_longitude(190.0) == -170.0
    assert BestTrackNormalizer.normalize_longitude(-190.0) == 170.0


def test_latitude_normalization_bounds():
    assert BestTrackNormalizer.normalize_latitude(15.2) == 15.2
    assert BestTrackNormalizer.normalize_latitude(95.0) == 90.0
    assert BestTrackNormalizer.normalize_latitude(-95.0) == -90.0


def test_timestamp_utc_conversion():
    dt_naive = datetime(2020, 5, 16, 12, 0)
    utc_dt = BestTrackNormalizer.normalize_timestamp(dt_naive)
    assert utc_dt.tzinfo == timezone.utc
    assert utc_dt.hour == 12

    iso_str = "2020-05-16T12:00:00Z"
    parsed_utc = BestTrackNormalizer.normalize_timestamp(iso_str)
    assert parsed_utc.tzinfo == timezone.utc
    assert parsed_utc.year == 2020


def test_zero_fabrication_preserves_none():
    raw_record = {
        "storm_id": "IMD-2020-TEST",
        "storm_name": "TEST",
        "timestamp": "2020-05-16T00:00:00Z",
        "latitude": 12.5,
        "longitude": 85.0,
        "max_wind_kt": None,
        "central_pressure_hpa": "NaN",
    }
    fix = BestTrackNormalizer.normalize_record(
        raw_record,
        source="IMD_RSMC_NEW_DELHI",
        source_file="test.xlsx",
    )
    assert fix is not None
    assert fix.max_wind_kt is None
    assert fix.central_pressure_hpa is None
    assert fix.latitude == 12.5
    assert fix.longitude == 85.0
