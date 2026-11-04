"""Tests for NOAA IBTrACS Adapter."""

import pytest
from backend.app.data.best_track.ibtracs_adapter import IbtracsBestTrackAdapter


def test_ibtracs_adapter_local_archive_presence():
    adapter = IbtracsBestTrackAdapter()
    assert adapter.is_available() is True
    assert adapter.csv_path.stat().st_size > 10_000_000


def test_ibtracs_parse_single_year():
    adapter = IbtracsBestTrackAdapter()
    fixes = adapter.parse_all(target_years=[2020])
    assert len(fixes) > 100
    amphan_fixes = [f for f in fixes if f.storm_name == "AMPHAN"]
    assert len(amphan_fixes) >= 40
