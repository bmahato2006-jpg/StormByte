"""Tests for Spatiotemporal Satellite-to-Track Association Logic."""

import pytest
from backend.app.data.best_track.service import BestTrackService

service = BestTrackService()


def test_unverified_when_coords_missing():
    res = service.match_satellite_image_to_track("IMD-2020-AMPHAN", image_timestamp_utc="2020-05-16T12:00:00Z")
    assert res.status == "UNVERIFIED"
    assert "UNVERIFIED" in res.details


def test_verified_match_when_spatiotemporally_close():
    # AMPHAN had fix at (10.9, 86.3) at 2020-05-16T09:00:00+00:00
    res = service.match_satellite_image_to_track(
        "IMD-2020-AMPHAN",
        image_timestamp_utc="2020-05-16T09:15:00Z",
        image_latitude=10.95,
        image_longitude=86.35,
    )
    assert res.status == "VERIFIED_MATCH"
    assert res.time_difference_hours is not None
    assert res.time_difference_hours <= 1.5
    assert res.spatial_distance_km is not None
    assert res.spatial_distance_km <= 50.0


def test_no_match_when_spatially_far():
    res = service.match_satellite_image_to_track(
        "IMD-2020-AMPHAN",
        image_timestamp_utc="2020-05-16T09:00:00Z",
        image_latitude=25.0,  # Inland Northern India far from system
        image_longitude=75.0,
    )
    assert res.status == "NO_MATCH"
