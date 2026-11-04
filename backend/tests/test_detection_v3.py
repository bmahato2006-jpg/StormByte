"""Comprehensive Unit and Integration Tests for Cyclone Detector V3 Pipeline."""

import math
import numpy as np
from PIL import Image
import pytest
import torch

from backend.app.data.satellite.qc_dedup_v3 import compute_dhash, hamming_distance
from backend.app.ml.detection.evaluate_v3 import GradCAM, compute_roc_auc
from backend.app.ml.detection.model_v3 import CycloneDetectorCNN_V3, CLASS_NAMES


class TestModelV3Architecture:
    """Validates structural and computational properties of CycloneDetectorCNN_V3."""

    def test_forward_and_shapes(self):
        model = CycloneDetectorCNN_V3()
        model.eval()

        dummy_batch = torch.randn(4, 3, 256, 256)
        with torch.no_grad():
            logits = model(dummy_batch)
            probs = model.predict_proba(dummy_batch)
            features = model.extract_features(dummy_batch)

        assert logits.shape == (4, 2), f"Expected logits shape (4, 2), got {logits.shape}"
        assert probs.shape == (4, 2), f"Expected probs shape (4, 2), got {probs.shape}"
        assert features.shape == (4, 256, 16, 16), f"Expected feature map (4, 256, 16, 16), got {features.shape}"

        # Probs must sum to 1 across classes
        sums = probs.sum(dim=-1).numpy()
        assert np.allclose(sums, np.ones(4), atol=1e-5), "Softmax probabilities must sum to 1.0"

    def test_parameter_count(self):
        model = CycloneDetectorCNN_V3()
        params = model.count_parameters()
        assert 2_000_000 <= params <= 3_000_000, f"Expected parameter count ~2.5M, got {params:,}"

    def test_classify_single(self):
        model = CycloneDetectorCNN_V3()
        dummy_img = torch.randn(3, 256, 256)

        res_low_thresh = model.classify_single(dummy_img, threshold=0.01)
        assert "is_cyclone" in res_low_thresh
        assert "predicted_label" in res_low_thresh
        assert "confidence" in res_low_thresh
        assert "class_probabilities" in res_low_thresh
        assert "structural_type" in res_low_thresh
        assert res_low_thresh["is_cyclone"] is True

        res_high_thresh = model.classify_single(dummy_img, threshold=0.99)
        assert res_high_thresh["is_cyclone"] is False


class TestQualityControlAndDeduplication:
    """Validates exact and perceptual hashing and Hamming distance calculations."""

    def test_dhash_and_hamming(self):
        # Create test images: base, identical copy, and inverted
        img1 = Image.new("RGB", (256, 256), color=(100, 150, 200))
        img2 = Image.new("RGB", (256, 256), color=(100, 150, 200))
        img3 = Image.new("RGB", (256, 256), color=(255, 0, 0))

        h1 = compute_dhash(img1)
        h2 = compute_dhash(img2)
        h3 = compute_dhash(img3)

        assert isinstance(h1, int)
        # Identical images must have zero Hamming distance
        assert hamming_distance(h1, h2) == 0

    def test_roc_auc_calculation(self):
        y_true = np.array([1, 1, 0, 0])
        # Perfect predictions
        y_score_perfect = np.array([0.9, 0.8, 0.2, 0.1])
        auc = compute_roc_auc(y_true, y_score_perfect)
        assert auc == 1.0

        # Inverted predictions
        y_score_inverted = np.array([0.1, 0.2, 0.8, 0.9])
        auc_inv = compute_roc_auc(y_true, y_score_inverted)
        assert auc_inv == 0.0


class TestGradCAMExplainability:
    """Validates native Grad-CAM attention heatmap generation."""

    def test_gradcam_generation(self):
        model = CycloneDetectorCNN_V3()
        gradcam = GradCAM(model)

        dummy_tensor = torch.randn(1, 3, 256, 256)
        cam = gradcam.generate(dummy_tensor, target_class=1)

        assert cam.shape == (256, 256), f"Expected CAM shape (256, 256), got {cam.shape}"
        assert cam.min() >= 0.0, "CAM minimum must be >= 0.0"
        assert cam.max() <= 1.0, "CAM maximum must be <= 1.0"

