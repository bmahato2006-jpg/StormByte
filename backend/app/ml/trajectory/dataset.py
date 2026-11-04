"""PyTorch Dataset and sequence builder for Cyclone-X ML Trajectory Forecasting (Phase 8).

Ensures:
- Strict storm-level sequence construction without cross-storm contamination.
- Sliding history window of length SEQ_LENGTH (default 4 fixes).
- Multi-horizon targets (+6h, +12h, +24h) relative displacement matching.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset

from backend.app.ml.trajectory.config import (
    HORIZONS,
    INPUT_DIM,
    OUTPUT_DIM,
    SEQ_LENGTH,
)
from backend.app.ml.trajectory.features import (
    extract_raw_step_features,
    TrajectoryFeatureScaler,
)


class CycloneSequenceDataset(Dataset):
    """PyTorch Dataset for multi-step cyclone trajectory forecasting."""

    def __init__(
        self,
        csv_path: Path,
        scaler: Optional[TrajectoryFeatureScaler] = None,
        fit_scaler: bool = False,
        require_all_horizons: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.scaler = scaler or TrajectoryFeatureScaler()
        self.require_all_horizons = require_all_horizons

        self.samples: List[Dict[str, Any]] = []
        self._load_and_build_sequences(fit_scaler=fit_scaler)

    def _load_and_build_sequences(self, fit_scaler: bool):
        """Parse manifest rows, sort chronologically per storm, and extract sequence windows."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.csv_path}")

        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Group fixes strictly by unique storm_id
        storms: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            sid = r["storm_id"]
            if sid not in storms:
                storms[sid] = []
            storms[sid].append(r)

        # If fit_scaler requested, collect all step features across storms first
        if fit_scaler:
            raw_features_for_fit: List[Dict[str, float]] = []
            for sid, storm_fixes in storms.items():
                storm_fixes.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]))
                for i in range(len(storm_fixes)):
                    prev = storm_fixes[i - 1] if i > 0 else None
                    feat = extract_raw_step_features(storm_fixes[i], prev)
                    raw_features_for_fit.append(feat)
            self.scaler.fit(raw_features_for_fit)

        # Build sequence samples (sliding window of length SEQ_LENGTH)
        for sid, storm_fixes in storms.items():
            storm_fixes.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]))

            # Need at least SEQ_LENGTH + 1 fixes to have input and future targets
            if len(storm_fixes) < SEQ_LENGTH + 1:
                continue

            # Compute raw features for all fixes in this storm once
            storm_raw_feats = []
            for i in range(len(storm_fixes)):
                prev = storm_fixes[i - 1] if i > 0 else None
                storm_raw_feats.append(extract_raw_step_features(storm_fixes[i], prev))

            # Slide window: index i is the forecast initialization point
            for i in range(SEQ_LENGTH - 1, len(storm_fixes)):
                init_fix = storm_fixes[i]
                init_time = datetime.fromisoformat(init_fix["timestamp"])
                init_lat = float(init_fix["latitude"])
                init_lon = float(init_fix["longitude"])

                # Input history window: fixes [i - SEQ_LENGTH + 1, ..., i]
                window_raw_feats = storm_raw_feats[i - SEQ_LENGTH + 1 : i + 1]
                input_tensor = torch.tensor(
                    [self.scaler.transform(f) for f in window_raw_feats],
                    dtype=torch.float32,
                )

                # Search ahead for targets at +6h, +12h, +24h
                targets_map: Dict[int, Tuple[float, float]] = {}  # horizon -> (target_lat, target_lon)

                for j in range(i + 1, len(storm_fixes)):
                    future_fix = storm_fixes[j]
                    future_time = datetime.fromisoformat(future_fix["timestamp"])
                    dt_h = (future_time - init_time).total_seconds() / 3600.0
                    f_lat = float(future_fix["latitude"])
                    f_lon = float(future_fix["longitude"])

                    # Match +6h (5.0h - 7.0h)
                    if 6 not in targets_map and 5.0 <= dt_h <= 7.0:
                        targets_map[6] = (f_lat, f_lon)
                    # Match +12h (11.0h - 13.0h)
                    if 12 not in targets_map and 11.0 <= dt_h <= 13.0:
                        targets_map[12] = (f_lat, f_lon)
                    # Match +24h (22.0h - 26.0h)
                    if 24 not in targets_map and 22.0 <= dt_h <= 26.0:
                        targets_map[24] = (f_lat, f_lon)

                # Check if required horizons exist
                if self.require_all_horizons:
                    if len(targets_map) < len(HORIZONS):
                        continue
                elif len(targets_map) == 0:
                    continue

                # Build target relative displacement vector: [dlat_6, dlon_6, dlat_12, dlon_12, dlat_24, dlon_24]
                target_vector: List[float] = []
                actual_coords: Dict[int, Dict[str, float]] = {}
                for h in HORIZONS:
                    if h in targets_map:
                        t_lat, t_lon = targets_map[h]
                        target_vector.extend([t_lat - init_lat, t_lon - init_lon])
                        actual_coords[h] = {"latitude": t_lat, "longitude": t_lon}
                    else:
                        target_vector.extend([0.0, 0.0])

                self.samples.append({
                    "input": input_tensor,
                    "target": torch.tensor(target_vector, dtype=torch.float32),
                    "storm_id": sid,
                    "storm_name": init_fix.get("storm_name", "UNNAMED"),
                    "init_time": init_time.isoformat(),
                    "origin_lat": init_lat,
                    "origin_lon": init_lon,
                    "actual_coords": actual_coords,
                    "available_horizons": list(targets_map.keys()),
                    "track_history": [
                        {
                            "latitude": float(storm_fixes[k]["latitude"]),
                            "longitude": float(storm_fixes[k]["longitude"]),
                            "timestamp": storm_fixes[k]["timestamp"],
                            "windSpeed": float(storm_fixes[k]["max_wind_kt"]) if storm_fixes[k].get("max_wind_kt") else None,
                            "pressure": float(storm_fixes[k]["central_pressure_hpa"]) if storm_fixes[k].get("central_pressure_hpa") else None,
                        }
                        for k in range(i - SEQ_LENGTH + 1, i + 1)
                    ],
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

