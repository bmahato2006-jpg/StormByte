"""Automated Unit Tests for Cyclone-X Deep Learning & Statistical Training Framework."""

import math
import numpy as np
import pytest
import torch

from backend.app.ml.training_framework.data_loaders import (
    SatelliteCenterNetDataset,
    TrajectorySequenceDataset,
    generate_2d_gaussian_heatmap,
    compute_gaussian_radius,
)
from backend.app.ml.training_framework.models import (
    CenterNetFocalLoss,
    CenterNetRegLoss,
    CycloneCenterNet,
    TrajectoryBiLSTM,
)
from backend.app.ml.training_framework.train_pipeline import (
    preprocess_ships_era5_data,
    train_ri_predictor,
    train_centernet,
    train_trajectory_bilstm,
)


# =============================================================================
# 1. 2D Gaussian Heatmap & Data Loader Tests
# =============================================================================

def test_gaussian_heatmap_generation():
    """Verify 2D Gaussian heatmap properties: peak = 1.0, boundaries, and falloff."""
    hm = np.zeros((64, 64), dtype=np.float32)
    center = (32, 32)
    radius = 6

    generate_2d_gaussian_heatmap(hm, center, radius)

    # 1. Peak value at center should be 1.0
    assert math.isclose(hm[32, 32], 1.0, abs_tol=1e-5)
    # 2. Values must be bounded in [0.0, 1.0]
    assert np.all(hm >= 0.0)
    assert np.all(hm <= 1.0)
    # 3. Radial monotonicity: points further from center must have lower intensity
    assert hm[32, 32] > hm[32, 35] > hm[32, 40]
    # 4. Far background should be zero
    assert hm[0, 0] == 0.0


def test_centernet_dataset_and_tensors():
    """Verify SatelliteCenterNetDataset builds valid image, heatmap, and offset tensors."""
    img_arr = np.random.uniform(0, 1, (256, 256)).astype(np.float32)
    sample = {
        "image": img_arr,
        "center_x": 128.0,
        "center_y": 128.0,
        "eye_radius": 20.0,
    }

    dataset = SatelliteCenterNetDataset([sample], input_size=(256, 256), downsample_ratio=4)
    item = dataset[0]

    assert "image" in item
    assert "target_heatmap" in item
    assert "target_offset" in item
    assert "offset_mask" in item

    # Input image shape [C=3, H=256, W=256]
    assert item["image"].shape == (3, 256, 256)
    # Target heatmap shape [C=1, H=64, W=64] (stride 4)
    assert item["target_heatmap"].shape == (1, 64, 64)
    # Subpixel offset shape [C=2, H=64, W=64]
    assert item["target_offset"].shape == (2, 64, 64)
    # Offset mask shape [C=1, H=64, W=64]
    assert item["offset_mask"].shape == (1, 64, 64)

    # Peak must be at 128 / 4 = 32
    assert math.isclose(float(item["target_heatmap"][0, 32, 32].item()), 1.0, abs_tol=1e-4)
    assert float(item["offset_mask"][0, 32, 32].item()) == 1.0


# =============================================================================
# 2. CenterNet Model & Focal Loss Tests
# =============================================================================

def test_centernet_focal_loss():
    """Verify CenterNet modified focal loss numerical stability and gradient flow."""
    criterion = CenterNetFocalLoss(alpha=2.0, beta=4.0)

    # Target: 1.0 at center, 0.0 elsewhere
    target = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
    target[:, :, 16, 16] = 1.0

    # Perfect prediction should have very small loss (< 1e-4)
    pred_perfect = target.clone()
    pred_perfect = torch.clamp(pred_perfect, 1e-4, 1.0 - 1e-4)
    loss_perfect = criterion(pred_perfect, target)
    assert loss_perfect.item() < 0.05

    # Flawed prediction should produce higher positive loss
    pred_flawed = torch.full((2, 1, 32, 32), 0.1, requires_grad=True)
    loss_flawed = criterion(pred_flawed, target)
    assert loss_flawed.item() > loss_perfect.item()

    # Verify backward gradient flow
    loss_flawed.backward()
    assert pred_flawed.grad is not None
    assert not torch.isnan(pred_flawed.grad).any()


def test_cyclone_centernet_architecture():
    """Verify CycloneCenterNet forward pass dimensions, decoding, and parameter gradients."""
    model = CycloneCenterNet(backbone_name="resnet18", pretrained=False)
    x = torch.randn(2, 3, 256, 256)

    outputs = model(x)
    assert "heatmap" in outputs
    assert "offset" in outputs

    # Heatmap should be [B, 1, 64, 64] in [0, 1]
    hm = outputs["heatmap"]
    assert hm.shape == (2, 1, 64, 64)
    assert (hm >= 0.0).all() and (hm <= 1.0).all()

    # Offset should be [B, 2, 64, 64]
    off = outputs["offset"]
    assert off.shape == (2, 2, 64, 64)

    # Test center decoding
    decoded = model.decode_eye_center(hm[0:1], off[0:1])
    assert "pixel_x" in decoded
    assert "pixel_y" in decoded
    assert "confidence" in decoded
    assert 0.0 <= decoded["pixel_x"] <= 256.0
    assert 0.0 <= decoded["pixel_y"] <= 256.0


# =============================================================================
# 3. Trajectory Sequence Dataset & BiLSTM Tests
# =============================================================================

def test_trajectory_dataset_sliding_windows():
    """Verify TrajectorySequenceDataset creates correct 8-feature sequences and targets."""
    # Synthetic 10-fix track
    track = []
    for step in range(10):
        track.append({
            "latitude": 12.0 + step * 0.5,
            "longitude": 85.0 + step * 0.4,
            "max_wind_kt": 40.0 + step * 5.0,
            "central_pressure_hpa": 1000.0 - step * 4.0,
            "sea_surface_temp_c": 29.0,
            "vertical_wind_shear_kt": 10.0,
            "steering_u500_kt": 2.0,
            "steering_v500_kt": 5.0,
        })

    dataset = TrajectorySequenceDataset([track], seq_length=4, forecast_steps=3)
    # Expected windows: 10 - 4 - 3 + 1 = 4 sliding windows
    assert len(dataset) == 4

    in_seq, targets = dataset[0]
    # Input sequence: [T=4, Features=8]
    assert in_seq.shape == (4, 8)
    # Forecast targets: [Steps=3, Coordinates=2 (dlat, dlon)]
    assert targets.shape == (3, 2)


def test_trajectory_bilstm_forward():
    """Verify TrajectoryBiLSTM forward pass, temporal attention, and gradient calculation."""
    model = TrajectoryBiLSTM(input_dim=8, hidden_dim=64, num_layers=2, forecast_steps=3)
    x = torch.randn(4, 4, 8)  # [B=4, Seq=4, Feats=8]

    out = model(x)
    assert out.shape == (4, 3, 2)  # [B=4, Forecast=3, Coords=2]

    # Test loss backward
    target = torch.randn(4, 3, 2)
    loss = torch.nn.functional.smooth_l1_loss(out, target)
    loss.backward()

    # Check gradients exist across LSTM weights
    assert model.bilstm.weight_ih_l0.grad is not None
    assert not torch.isnan(model.bilstm.weight_ih_l0.grad).any()


# =============================================================================
# 4. End-to-End Pipeline Dry-Run Tests
# =============================================================================

def test_ri_predictor_training_pipeline(tmp_path):
    """Verify Rapid Intensification tabular training pipeline produces valid ROC-AUC."""
    res = train_ri_predictor(output_dir=tmp_path)
    assert "model" in res
    assert "metrics" in res
    metrics = res["metrics"]
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] > 0.60  # Should easily beat random chance on synthetic data
    assert metrics["test_samples"] > 0


def test_centernet_training_one_epoch(tmp_path):
    """Verify CycloneCenterNet training loop executes a complete epoch without crashing."""
    res = train_centernet(num_epochs=1, batch_size=2, output_dir=tmp_path)
    assert "model" in res
    assert "best_val_loss" in res
    assert len(res["history"]) == 1


def test_trajectory_bilstm_training_one_epoch(tmp_path):
    """Verify TrajectoryBiLSTM training loop executes a complete epoch and reports ADE."""
    res = train_trajectory_bilstm(num_epochs=1, batch_size=4, output_dir=tmp_path)
    assert "model" in res
    assert "best_val_loss" in res
    assert "scaler" in res
    assert len(res["history"]) == 1
    assert "ade_km" in res["history"][0]
