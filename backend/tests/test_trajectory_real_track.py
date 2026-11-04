"""Tests for Trajectory LSTM Integration with Real Historical Tracks."""

import pytest
from backend.app.data.best_track.service import BestTrackService

service = BestTrackService()


def test_amphan_real_backtest():
    res = service.evaluate_forecast("IMD-2020-AMPHAN", lead_steps=4)
    assert res.storm_name == "AMPHAN"
    assert res.history_fixes_count == 4
    assert res.future_fixes_count >= 40
    assert res.overall_mae_km is not None
    assert res.overall_mae_km < 100.0  # Realistic forecast error bound
    assert "6h" in res.metrics
    assert "12h" in res.metrics
    assert "24h" in res.metrics


def test_insufficient_history_enforced():
    with pytest.raises(ValueError) as exc:
        service.evaluate_forecast("IMD-2020-AMPHAN", lead_steps=2)  # Under 4 fixes
    assert "INSUFFICIENT_TRACK_HISTORY" in str(exc.value)
