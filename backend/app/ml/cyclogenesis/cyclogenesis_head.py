"""Early Tropical Cyclogenesis Classification Head.

Introduces a deep neural classification head that evaluates early pre-cyclonic disturbances
(wind < 34 kt, Tropical Cloud Clusters, and Low-Pressure Areas) before intensity regression.
Classifies whether a tropical disturbance will intensify into a cyclonic storm (wind >= 34 kt)
within the next 24 to 48 hours.
"""

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from backend.app.ml.cyclogenesis.tcc_tracker import TCCSequenceProfile, EnvironmentalParameters

logger = logging.getLogger("cyclone_x.cyclogenesis_head")

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
HEAD_CHECKPOINT_PATH = CHECKPOINT_DIR / "early_cyclogenesis_head.pt"
HEAD_METADATA_PATH = CHECKPOINT_DIR / "early_cyclogenesis_head_metadata.json"


class EarlyCyclogenesisNeuralHead(nn.Module):
    """Deep binary cyclogenesis classification network."""

    def __init__(self, in_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle single unbatched sample
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return self.net(x).squeeze(0)
        return self.net(x)


@dataclass
class EarlyCyclogenesisOutput:
    """Output contract for early tropical cyclogenesis prediction."""
    probability: float  # [0.0, 1.0]
    probability_percent: float  # [0.0, 100.0]
    risk_tier: str  # "LOW", "MODERATE", "HIGH", "CRITICAL"
    intensification_24to48h: bool  # True if P >= 0.50
    pre_cyclone_stage: str  # e.g., TROPICAL_CLOUD_CLUSTER, DEPRESSION, CYCLONIC_STORM
    disturbance_radius_km: float  # Envelope radius (100 - 250 km)
    circulation_center: Dict[str, float]  # {"latitude": ..., "longitude": ...}
    confidence_tier: str
    headline: str
    environmental_summary: Dict[str, Any]
    feature_vector: Dict[str, float]


class EarlyCyclogenesisClassifier:
    """Production service for early tropical cyclogenesis detection."""

    FEATURE_NAMES = [
        "coriolis_f",
        "sst_celsius",
        "vertical_wind_shear_kt",
        "relative_vorticity_850",
        "relative_humidity_700",
        "tcc_area_growth",
        "tcc_persistence_hours",
        "convective_radius_km",
        "curvature_score",
        "current_wind_speed",
    ]

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = Path(checkpoint_path or HEAD_CHECKPOINT_PATH)
        self.model = EarlyCyclogenesisNeuralHead(in_features=10)
        self._init_or_load_weights()

    def _init_or_load_weights(self):
        """Initializes weights or loads saved PyTorch state dictionary."""
        self.model.eval()
        if self.checkpoint_path.exists():
            try:
                state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
                self.model.load_state_dict(state_dict)
                logger.info("Loaded EarlyCyclogenesisNeuralHead from %s", self.checkpoint_path)
                return
            except Exception as e:
                logger.warning("Could not load EarlyCyclogenesisNeuralHead: %s. Re-training synthetic weights.", e)

        # Build and train calibrated baseline weights if checkpoint is absent
        self._train_and_save_baseline_head()

    def _train_and_save_baseline_head(self):
        """Trains an initial calibrated cyclogenesis head on physics-informed synthetic profiles."""
        rng = np.random.RandomState(42)
        X_list = []
        y_list = []

        # 1. Positive Genesis Cases (favorable thermodynamic & dynamic conditions)
        for _ in range(1200):
            f = rng.uniform(1.8e-5, 5.5e-5)
            sst = rng.uniform(27.8, 31.5)
            shear = rng.uniform(5.0, 18.0)  # low shear favorable
            vort = rng.uniform(2.5, 7.5)   # high low-level vorticity
            rh = rng.uniform(70.0, 92.0)   # moist mid-troposphere
            growth = rng.uniform(500.0, 4500.0)  # expanding convective core
            persist = rng.uniform(14.0, 24.0)    # persistent convective cluster > 12h
            radius = rng.uniform(120.0, 260.0)
            curv = rng.uniform(0.55, 0.95)       # visible curved convective banding
            wind = rng.uniform(15.0, 32.0)       # pre-cyclone stage
            X_list.append([f, sst, shear, vort, rh, growth, persist, radius, curv, wind])
            y_list.append(1.0)

        # 2. Non-Genesis Disorganizing Cases (unfavorable conditions: high shear, cold SST, weak vorticity)
        for _ in range(1800):
            failure_type = rng.choice(["shear", "dry", "cold", "quiescent"])
            if failure_type == "shear":
                shear = rng.uniform(24.0, 50.0)  # shears convective chimney apart
                sst = rng.uniform(27.5, 30.0)
                vort = rng.uniform(0.8, 2.5)
                rh = rng.uniform(55.0, 75.0)
                growth = rng.uniform(-2000.0, 500.0)
                curv = rng.uniform(0.15, 0.45)
            elif failure_type == "dry":
                shear = rng.uniform(12.0, 28.0)
                sst = rng.uniform(27.0, 29.5)
                vort = rng.uniform(1.0, 3.0)
                rh = rng.uniform(35.0, 58.0)  # dry air entrainment suppresses convection
                growth = rng.uniform(-1500.0, 200.0)
                curv = rng.uniform(0.10, 0.40)
            elif failure_type == "cold":
                shear = rng.uniform(10.0, 30.0)
                sst = rng.uniform(22.0, 25.8)  # below 26.5°C threshold
                vort = rng.uniform(0.5, 2.0)
                rh = rng.uniform(50.0, 75.0)
                growth = rng.uniform(-3000.0, 0.0)
                curv = rng.uniform(0.05, 0.35)
            else:
                shear = rng.uniform(15.0, 32.0)
                sst = rng.uniform(26.5, 28.5)
                vort = rng.uniform(0.2, 1.5)
                rh = rng.uniform(50.0, 70.0)
                growth = rng.uniform(-1000.0, 1000.0)
                curv = rng.uniform(0.10, 0.35)

            f = rng.uniform(0.5e-5, 4.5e-5)
            persist = rng.uniform(2.0, 12.0)
            radius = rng.uniform(60.0, 180.0)
            wind = rng.uniform(10.0, 25.0)
            X_list.append([f, sst, shear, vort, rh, growth, persist, radius, curv, wind])
            y_list.append(0.0)

        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)[:, None]

        # Normalization parameters
        mean = np.mean(X_arr, axis=0)
        std = np.std(X_arr, axis=0)
        std[std == 0] = 1.0
        self.mean = mean
        self.std = std

        X_norm = (X_arr - mean) / std

        # Train model
        net = EarlyCyclogenesisNeuralHead(in_features=10)
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.005, weight_decay=1e-4)
        criterion = nn.BCELoss()

        x_t = torch.tensor(X_norm, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32)

        for _ in range(35):
            optimizer.zero_grad()
            preds = net(x_t)
            loss = criterion(preds, y_t)
            loss.backward()
            optimizer.step()

        net.eval()
        self.model = net

        # Save checkpoint & metadata
        try:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), self.checkpoint_path)
            meta = {
                "model_name": "EarlyCyclogenesisNeuralHead",
                "version": "1.0.0",
                "in_features": 10,
                "feature_names": self.FEATURE_NAMES,
                "feature_mean": mean.tolist(),
                "feature_std": std.tolist(),
                "threshold_wind_kt": 34.0,
                "target": "24_48h_cyclonic_storm_intensification",
            }
            with open(HEAD_METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            logger.info("Successfully saved EarlyCyclogenesisNeuralHead checkpoint: %s", self.checkpoint_path)
        except Exception as e:
            logger.warning("Could not persist head checkpoint: %s", e)

    def predict(
        self,
        tcc_profile: TCCSequenceProfile,
        current_wind_speed_kt: float = 25.0,
        manual_curvature_score: Optional[float] = None,
    ) -> EarlyCyclogenesisOutput:
        """Classifies cyclogenesis potential and 24-48 hour intensification likelihood."""
        env = tcc_profile.environmental
        curv_score = manual_curvature_score if manual_curvature_score is not None else (
            tcc_profile.clusters_history[-1].curvature_vorticity_score if tcc_profile.clusters_history else 0.45
        )

        raw_vec = np.array([
            env.coriolis_f,
            env.sea_surface_temperature_c,
            env.vertical_wind_shear_kt,
            env.relative_vorticity_850,
            env.relative_humidity_700,
            tcc_profile.area_growth_rate_km2_hr,
            tcc_profile.persistence_hours,
            tcc_profile.mean_radius_km,
            curv_score,
            current_wind_speed_kt,
        ], dtype=np.float32)

        # Baseline normalization parameters
        mean = getattr(self, "mean", np.array([2.5e-5, 29.0, 14.0, 3.5, 75.0, 500.0, 14.0, 160.0, 0.50, 25.0], dtype=np.float32))
        std = getattr(self, "std", np.array([1.5e-5, 2.0, 6.0, 2.0, 12.0, 1500.0, 6.0, 60.0, 0.25, 8.0], dtype=np.float32))
        norm_vec = (raw_vec - mean) / std

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(norm_vec, dtype=torch.float32)
            prob_raw = float(self.model(x_tensor).item())

        # Physical constraint adjustments:
        # Near-equator (< 5°N) suppresses cyclogenesis due to negligible Coriolis force
        if env.latitude < 5.0:
            prob_raw *= max(0.1, (env.latitude / 5.0))

        # SST < 26.0°C severely restricts thermodynamic energy
        if env.sea_surface_temperature_c < 26.0:
            prob_raw *= 0.25

        prob = float(np.clip(prob_raw, 0.02, 0.98))
        prob_pct = round(prob * 100.0, 1)

        # Risk tiering
        if prob >= 0.80:
            tier = "CRITICAL"
            conf = "HIGH"
        elif prob >= 0.60:
            tier = "HIGH"
            conf = "HIGH"
        elif prob >= 0.35:
            tier = "MODERATE"
            conf = "MEDIUM"
        else:
            tier = "LOW"
            conf = "MEDIUM"

        # Pre-cyclone stage naming
        if current_wind_speed_kt >= 34.0:
            stage = "CYCLONIC_STORM"
            intensify = True
        elif current_wind_speed_kt >= 28.0:
            stage = "DEPRESSION"
            intensify = (prob >= 0.45)
        elif current_wind_speed_kt >= 17.0:
            stage = "LOW_PRESSURE_AREA"
            intensify = (prob >= 0.50)
        else:
            stage = "TROPICAL_CLOUD_CLUSTER"
            intensify = (prob >= 0.50)

        # Disturbance envelope radius (enclosing the pre-cyclone TCC circulation)
        disturbance_radius = round(max(100.0, min(260.0, tcc_profile.mean_radius_km * 1.2)), 1)

        headline = f"Genesis Risk: {tier} - {prob_pct:.0f}% ({'Intensification to Cyclonic Storm likely in 24–48h' if intensify else 'Development unlikely / suppressed'})"

        return EarlyCyclogenesisOutput(
            probability=round(prob, 4),
            probability_percent=prob_pct,
            risk_tier=tier,
            intensification_24to48h=intensify,
            pre_cyclone_stage=stage,
            disturbance_radius_km=disturbance_radius,
            circulation_center={"latitude": env.latitude, "longitude": env.longitude},
            confidence_tier=conf,
            headline=headline,
            environmental_summary={
                "seaSurfaceTemperature": env.sea_surface_temperature_c,
                "verticalWindShearKt": env.vertical_wind_shear_kt,
                "shearFavorable": env.shear_favorable,
                "relativeVorticity850": env.relative_vorticity_850,
                "relativeHumidity700": env.relative_humidity_700,
                "oceanHeatContent": env.ocean_heat_content_kj,
                "tccGrowthRateKm2Hr": tcc_profile.area_growth_rate_km2_hr,
                "tccPersistenceHours": tcc_profile.persistence_hours,
            },
            feature_vector={name: float(val) for name, val in zip(self.FEATURE_NAMES, raw_vec)},
        )


early_cyclogenesis_classifier = EarlyCyclogenesisClassifier()
