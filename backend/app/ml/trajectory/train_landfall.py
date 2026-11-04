"""Landfall Prediction Neural Network Training Pipeline.

Trains LandfallPredictorNet on authentic NOAA IBTrACS North Indian Ocean cyclone tracks (1,300+ historical landfall storms).
Predicts:
  1. Coastal Landfall Coordinates (Latitude, Longitude)
  2. Arrival Lead Time (Hours to Coastal Impact)
  3. Landfall Probability (0.0 to 1.0)
Evaluates Haversine coastal distance error (km) and lead-time MAE (hours) on held-out test storms.
"""

import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ml.training_framework.extended_models import LandfallPredictorNet

logger = logging.getLogger("cyclone_x.landfall_trainer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

CHECKPOINT_DIR = PROJECT_ROOT / "backend" / "app" / "ml" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = CHECKPOINT_DIR / "landfall_net.pt"
METADATA_PATH = CHECKPOINT_DIR / "landfall_net_metadata.json"
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "best_track" / "ibtracs" / "ibtracs.NI.list.v04r01.csv"


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Computes great-circle distance between coordinate arrays in kilometers."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return R * c


class LandfallDataset(Dataset):
    def __init__(self, features: np.ndarray, coords: np.ndarray, lead_times: np.ndarray, probs: np.ndarray):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.coords = torch.from_numpy(coords.astype(np.float32))
        self.lead_times = torch.from_numpy(lead_times.astype(np.float32))
        self.probs = torch.from_numpy(probs.astype(np.float32))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.coords[idx], self.lead_times[idx], self.probs[idx]


class LandfallTrainer:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LandfallPredictorNet(input_dim=16, hidden_dim=64).to(self.device)
        logger.info(f"Initialized LandfallPredictorNet on {self.device}")

    def load_data(self) -> Tuple[LandfallDataset, LandfallDataset, LandfallDataset, Dict[str, np.ndarray]]:
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"IBTrACS dataset missing at {DATASET_PATH}")

        logger.info(f"Loading IBTrACS North Indian Ocean tracks from {DATASET_PATH.name}...")
        df = pd.read_csv(DATASET_PATH, low_memory=False, skiprows=[1])
        df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
        df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
        df["DIST2LAND"] = pd.to_numeric(df["DIST2LAND"], errors="coerce")
        df["USA_WIND"] = pd.to_numeric(df["USA_WIND"], errors="coerce").fillna(35.0)
        df["USA_PRES"] = pd.to_numeric(df["USA_PRES"], errors="coerce").fillna(1000.0)
        df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")

        df = df.dropna(subset=["SID", "ISO_TIME", "LAT", "LON", "DIST2LAND"]).sort_values(["SID", "ISO_TIME"])

        # Partition storms by SID
        unique_sids = df["SID"].unique()
        np.random.seed(42)
        shuffled_sids = np.random.permutation(unique_sids)

        n_storms = len(shuffled_sids)
        n_train = int(n_storms * 0.8)
        n_val = int(n_storms * 0.1)

        train_sids = set(shuffled_sids[:n_train])
        val_sids = set(shuffled_sids[n_train:n_train + n_val])
        test_sids = set(shuffled_sids[n_train + n_val:])

        groups = {}
        for row in df[["SID", "LAT", "LON", "DIST2LAND", "USA_WIND", "USA_PRES", "ISO_TIME"]].itertuples(index=False):
            sid = row[0]
            if sid not in groups:
                groups[sid] = []
            groups[sid].append(row[1:])

        OMEGA = 7.2921e-5

        splits = {"train": ([], [], [], []), "val": ([], [], [], []), "test": ([], [], [], [])}

        for sid, fixes in groups.items():
            n = len(fixes)
            if n < 4:
                continue

            if sid in train_sids:
                target_split = "train"
            elif sid in val_sids:
                target_split = "val"
            else:
                target_split = "test"

            # Find landfall index
            lf_idx = -1
            for idx, f in enumerate(fixes):
                if f[2] <= 0:  # dist2land == 0
                    lf_idx = idx
                    break

            has_lf = (lf_idx >= 0)
            target_idx = lf_idx if has_lf else n - 1
            lf_lat, lf_lon = fixes[target_idx][0], fixes[target_idx][1]
            lf_time = fixes[target_idx][5]

            end_idx = lf_idx if has_lf else n
            for i in range(end_idx):
                lat, lon, dist2land, vmax, pres, curr_time = fixes[i]
                if i > 0:
                    prev_lat, prev_lon, _, prev_vmax, prev_pres, prev_time = fixes[i - 1]
                    dt_h = max(0.5, (curr_time - prev_time).total_seconds() / 3600.0)
                    dlat = lat - prev_lat
                    dlon = lon - prev_lon
                    u_trans = (dlon * 111.0 * np.cos(np.radians(lat))) / dt_h
                    v_trans = (dlat * 111.0) / dt_h
                    speed = np.sqrt(u_trans ** 2 + v_trans ** 2)
                    heading = np.arctan2(u_trans, v_trans)
                    dv_dt = (vmax - prev_vmax) / dt_h
                    dp_dt = (pres - prev_pres) / dt_h
                else:
                    u_trans, v_trans, speed, heading, dv_dt, dp_dt = 5.0, 5.0, 7.0, 0.78, 0.0, 0.0

                lag6_i = max(0, i - 2)
                lag12_i = max(0, i - 4)
                lat_lag6, lon_lag6 = fixes[lag6_i][0], fixes[lag6_i][1]
                lat_lag12, lon_lag12 = fixes[lag12_i][0], fixes[lag12_i][1]

                f_cor = 2.0 * OMEGA * np.sin(np.radians(lat))
                lead_h = max(0.0, (lf_time - curr_time).total_seconds() / 3600.0) if has_lf else 96.0
                prob_lf = 1.0 if (has_lf and lead_h <= 96.0) else 0.0

                feat = [
                    lat / 30.0,
                    lon / 100.0,
                    vmax / 150.0,
                    (pres - 900.0) / 120.0,
                    dist2land / 1500.0,
                    np.clip(u_trans / 50.0, -1.0, 1.0),
                    np.clip(v_trans / 50.0, -1.0, 1.0),
                    heading / (2.0 * np.pi),
                    np.clip(speed / 60.0, 0.0, 1.0),
                    np.clip(dv_dt / 20.0, -1.0, 1.0),
                    np.clip(dp_dt / 20.0, -1.0, 1.0),
                    f_cor * 1e4,
                    lat_lag6 / 30.0,
                    lon_lag6 / 100.0,
                    lat_lag12 / 30.0,
                    lon_lag12 / 100.0,
                ]

                s_feats, s_coords, s_times, s_probs = splits[target_split]
                s_feats.append(feat)
                s_coords.append((lf_lat, lf_lon))
                s_times.append(lead_h)
                s_probs.append(prob_lf)

        train_ds = LandfallDataset(
            np.array(splits["train"][0]), np.array(splits["train"][1]),
            np.array(splits["train"][2]), np.array(splits["train"][3])
        )
        val_ds = LandfallDataset(
            np.array(splits["val"][0]), np.array(splits["val"][1]),
            np.array(splits["val"][2]), np.array(splits["val"][3])
        )
        test_ds = LandfallDataset(
            np.array(splits["test"][0]), np.array(splits["test"][1]),
            np.array(splits["test"][2]), np.array(splits["test"][3])
        )

        test_arrays = {
            "coords": np.array(splits["test"][1]),
            "times": np.array(splits["test"][2]),
            "probs": np.array(splits["test"][3]),
        }

        logger.info(f"Loaded samples | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
        return train_ds, val_ds, test_ds, test_arrays

    def train(self, num_epochs: int = 12, batch_size: int = 64, lr: float = 1e-3) -> Dict[str, Any]:
        train_ds, val_ds, test_ds, test_arrays = self.load_data()

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
        huber = nn.SmoothL1Loss()
        bce = nn.BCELoss()

        best_val_loss = float("inf")
        best_state = None
        t0 = time.time()

        for ep in range(1, num_epochs + 1):
            self.model.train()
            train_loss = 0.0
            total = 0

            for x, coords, lead_time, prob in train_loader:
                x = x.to(self.device)
                coords = coords.to(self.device)
                lead_time = lead_time.to(self.device)
                prob = prob.to(self.device)

                optimizer.zero_grad()
                out = self.model(x)

                pred_coords = torch.stack([out["landfall_latitude"], out["landfall_longitude"]], dim=1)
                l_coords = huber(pred_coords, coords)
                l_time = huber(out["arrival_lead_time_hours"], lead_time) * 0.05
                l_prob = bce(out["landfall_probability"], prob)

                loss = l_coords + l_time + l_prob
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(x)
                total += len(x)

            scheduler.step()
            avg_train_loss = train_loss / max(1, total)

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_total = 0
            with torch.no_grad():
                for x, coords, lead_time, prob in val_loader:
                    x = x.to(self.device)
                    coords = coords.to(self.device)
                    lead_time = lead_time.to(self.device)
                    prob = prob.to(self.device)

                    out = self.model(x)
                    pred_coords = torch.stack([out["landfall_latitude"], out["landfall_longitude"]], dim=1)
                    l_coords = huber(pred_coords, coords)
                    l_time = huber(out["arrival_lead_time_hours"], lead_time) * 0.05
                    l_prob = bce(out["landfall_probability"], prob)
                    loss = l_coords + l_time + l_prob

                    val_loss += loss.item() * len(x)
                    val_total += len(x)

            avg_val_loss = val_loss / max(1, val_total)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            logger.info(f"Epoch [{ep:02d}/{num_epochs:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        elapsed = time.time() - t0

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Save checkpoint
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "architecture": "LandfallPredictorNet",
                "epochs": num_epochs,
                "best_val_loss": round(best_val_loss, 5),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            CHECKPOINT_PATH,
        )
        logger.info(f"Saved trained LandfallPredictorNet checkpoint to {CHECKPOINT_PATH}")

        # Evaluate on test set
        test_metrics = self.evaluate(test_ds, test_arrays)

        metadata = {
            "model_name": "landfall_net",
            "architecture": "LandfallPredictorNet (Multi-Task Coastal Point & Lead Time Regressor)",
            "status": "TRAINED",
            "epochs": num_epochs,
            "training_time_seconds": round(elapsed, 2),
            "best_val_loss": round(best_val_loss, 5),
            "test_metrics": test_metrics,
            "dataset": "NOAA IBTrACS North Indian Ocean Historical Tracks (1982-2024)",
            "checkpoint_path": str(CHECKPOINT_PATH),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def evaluate(self, test_ds: LandfallDataset, test_arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        self.model.eval()

        pred_lats, pred_lons = [], []
        pred_times, pred_probs = [], []

        with torch.no_grad():
            for x, coords, lead_time, prob in test_loader:
                x = x.to(self.device)
                out = self.model(x)

                pred_lats.extend(out["landfall_latitude"].cpu().numpy())
                pred_lons.extend(out["landfall_longitude"].cpu().numpy())
                pred_times.extend(out["arrival_lead_time_hours"].cpu().numpy())
                pred_probs.extend(out["landfall_probability"].cpu().numpy())

        pred_lats = np.array(pred_lats)
        pred_lons = np.array(pred_lons)
        pred_times = np.array(pred_times)
        pred_probs = np.array(pred_probs)

        true_coords = test_arrays["coords"]
        true_times = test_arrays["times"]
        true_probs = test_arrays["probs"]

        # Great circle error in km
        distances_km = haversine_km(pred_lats, pred_lons, true_coords[:, 0], true_coords[:, 1])
        mean_coord_error_km = float(np.mean(distances_km))
        median_coord_error_km = float(np.median(distances_km))
        time_mae_hours = float(np.mean(np.abs(pred_times - true_times)))

        bin_pred = (pred_probs >= 0.5).astype(int)
        bin_true = (true_probs >= 0.5).astype(int)
        acc = float(np.mean(bin_pred == bin_true))

        metrics = {
            "mean_coastal_error_km": round(mean_coord_error_km, 2),
            "median_coastal_error_km": round(median_coord_error_km, 2),
            "lead_time_mae_hours": round(time_mae_hours, 2),
            "landfall_probability_accuracy": round(acc, 4),
            "total_test_samples": len(test_ds),
        }
        logger.info(f"LandfallPredictorNet Held-out Test Metrics: {metrics}")
        return metrics


def main():
    trainer = LandfallTrainer()
    metadata = trainer.train(num_epochs=12, batch_size=64)
    print("Training finished. Metadata:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
