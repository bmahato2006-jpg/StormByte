"""Tests for Best Track Deduplication and Cross-Source Discrepancies."""

from datetime import datetime, timezone
import pytest
from backend.app.data.best_track.deduplicator import BestTrackDeduplicator, haversine
from backend.app.data.best_track.normalizer import BestTrackNormalizer


def test_exact_duplicate_elimination():
    raw = {
        "storm_id": "IMD-2020-TEST",
        "storm_name": "TEST",
        "timestamp": "2020-05-16T00:00:00Z",
        "latitude": 12.0,
        "longitude": 85.0,
    }
    f1 = BestTrackNormalizer.normalize_record(raw, source="IMD_RSMC_NEW_DELHI", source_file="test.xlsx")
    f2 = BestTrackNormalizer.normalize_record(raw, source="IMD_RSMC_NEW_DELHI", source_file="test.xlsx")
    assert f1 is not None and f2 is not None

    deduped, count = BestTrackDeduplicator.deduplicate([f1, f2])
    assert len(deduped) == 1
    assert count == 1


def test_haversine_accuracy():
    # Distance between Chennai (13.0827, 80.2707) and Kolkata (22.5726, 88.3639) is ~1360 km
    d = haversine(13.0827, 80.2707, 22.5726, 88.3639)
    assert 1340.0 < d < 1380.0
