"""Tests for IMD Best Track Adapter and Dynamic Header Detection."""

import pytest
from backend.app.data.best_track.imd_adapter import ImdBestTrackAdapter


def test_imd_adapter_local_archive_presence():
    adapter = ImdBestTrackAdapter()
    assert adapter.is_available() is True
    assert adapter.excel_path.stat().st_size > 5_000_000


def test_imd_time_parsing():
    assert ImdBestTrackAdapter._parse_time_string("0300") == (3, 0)
    assert ImdBestTrackAdapter._parse_time_string(300) == (3, 0)
    assert ImdBestTrackAdapter._parse_time_string("1200") == (12, 0)
    assert ImdBestTrackAdapter._parse_time_string("00:30") == (0, 30)
