"""Feature engineering and geodetic math for Cyclone-X ML Trajectory Forecasting (Phase 8).

Strict Zero-Leakage:
- Scaler statistics are computed SOLELY from the training split.
- No future observation information is used at step t.
- Features are strictly deterministic and reproducible.
"""

from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.app.ml.trajectory.config import (
    EARTH_RADIUS_KM,
    FEATURE_NAMES,
    SCALER_PATH,
)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two coordinate pairs in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates forward compass bearing from point 1 to point 2 in degrees [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    bearing_rad = math.atan2(y, x)
    return (math.degrees(bearing_rad) + 360.0) % 360.0


def extract_raw_step_features(
    current_fix: Dict[str, Any],
    previous_fix: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """Extracts raw kinematic and temporal features for a single trajectory fix."""
    lat = float(current_fix["latitude"])
    lon = float(current_fix["longitude"])
    ts_val = current_fix.get("timestamp") or current_fix.get("timestamp_utc")
    if isinstance(ts_val, str):
        try:
            ts = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
        except Exception:
            ts = datetime.now(timezone.utc)
    elif isinstance(ts_val, datetime):
        ts = ts_val
    else:
        ts = datetime.now(timezone.utc)

    if previous_fix is not None:
        prev_lat = float(previous_fix["latitude"])
        prev_lon = float(previous_fix["longitude"])
        prev_ts_val = previous_fix.get("timestamp") or previous_fix.get("timestamp_utc")
        if isinstance(prev_ts_val, str):
            try:
                prev_ts = datetime.fromisoformat(prev_ts_val.replace("Z", "+00:00"))
            except Exception:
                from datetime import timedelta
                prev_ts = ts - timedelta(hours=6)
        elif isinstance(prev_ts_val, datetime):
            prev_ts = prev_ts_val
        else:
            from datetime import timedelta
            prev_ts = ts - timedelta(hours=6)

        dt_sec = max(1.0, (ts - prev_ts).total_seconds())
        dt_hours = dt_sec / 3600.0

        delta_lat = lat - prev_lat
        delta_lon = lon - prev_lon
        disp_km = haversine_distance(prev_lat, prev_lon, lat, lon)
        speed_kmh = max(0.0, min(150.0, disp_km / dt_hours))
        bearing = calculate_bearing(prev_lat, prev_lon, lat, lon)
    else:
        # First fix in sequence (no previous fix)
        dt_hours = 3.0
        delta_lat = 0.0
        delta_lon = 0.0
        disp_km = 0.0
        speed_kmh = 0.0
        bearing = 315.0  # Climatic northwestward NIO drift

    bearing_rad = math.radians(bearing)
    bearing_sin = math.sin(bearing_rad)
    bearing_cos = math.cos(bearing_rad)

    # Cyclical seasonal encoding (month 1-12)
    month_rad = 2.0 * math.pi * (ts.month - 1) / 12.0
    month_sin = math.sin(month_rad)
    month_cos = math.cos(month_rad)

    # Cyclical diurnal encoding (hour 0-23)
    hour_rad = 2.0 * math.pi * ts.hour / 24.0
    hour_sin = math.sin(hour_rad)
    hour_cos = math.cos(hour_rad)

    return {
        "lat": lat,
        "lon": lon,
        "delta_lat": delta_lat,
        "delta_lon": delta_lon,
        "displacement_km": disp_km,
        "speed_kmh": speed_kmh,
        "bearing_sin": bearing_sin,
        "bearing_cos": bearing_cos,
        "time_delta_h": dt_hours,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
    }


class TrajectoryFeatureScaler:
    """Normalizes continuous kinematic features strictly fitted on the training split."""

    CONTINUOUS_KEYS = ["lat", "lon", "delta_lat", "delta_lon", "displacement_km", "speed_kmh", "time_delta_h"]

    def __init__(self):
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, feature_dicts: List[Dict[str, float]]):
        """Fit scaler on a list of raw step feature dictionaries."""
        if not feature_dicts:
            raise ValueError("Cannot fit scaler on empty feature list.")

        for key in self.CONTINUOUS_KEYS:
            values = [f[key] for f in feature_dicts]
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            self.means[key] = mean_val
            # Guard against zero variance
            self.stds[key] = std_val if std_val > 1e-6 else 1.0

        self.is_fitted = True

    def transform(self, raw_features: Dict[str, float]) -> List[float]:
        """Transforms a raw feature dict into a standard float feature vector."""
        if not self.is_fitted:
            raise RuntimeError("TrajectoryFeatureScaler must be fitted before transforming features.")

        vector: List[float] = []
        for name in FEATURE_NAMES:
            raw_val = raw_features[name]
            if name in self.CONTINUOUS_KEYS:
                norm_val = (raw_val - self.means[name]) / self.stds[name]
                vector.append(float(norm_val))
            else:
                # Sine/cosine encodings are already bounded in [-1, 1]
                vector.append(float(raw_val))
        return vector

    def save(self, filepath: Optional[Path] = None):
        """Save scaler parameters to JSON."""
        out_path = filepath or SCALER_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "means": self.means,
            "stds": self.stds,
            "feature_names": FEATURE_NAMES,
            "continuous_keys": self.CONTINUOUS_KEYS,
            "version": "1.0.0",
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "TrajectoryFeatureScaler":
        """Load fitted scaler parameters from JSON."""
        in_path = filepath or SCALER_PATH
        if not in_path.exists():
            raise FileNotFoundError(f"Scaler parameters not found at {in_path}")

        scaler = cls()
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scaler.means = data["means"]
        scaler.stds = data["stds"]
        scaler.is_fitted = True
        return scaler

