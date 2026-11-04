"""Cyclone-X Detection & Classification Package (Phase 9).

Exports:
- CycloneDetectorCNN: PyTorch Convolutional Neural Network
- DetectionTrainer: Deterministic training loop
- DetectionEvaluator, detection_evaluator: Test benchmark engine
- DetectionInferenceService, detection_inference_service: Production inference pipeline
- CycloneImageDataset, DetectionDatasetBuilder: Dataset and authentic tile builder
"""

from backend.app.ml.detection.dataset import (
    CycloneImageDataset,
    DetectionDatasetBuilder,
)
from backend.app.ml.detection.evaluator import DetectionEvaluator, detection_evaluator
from backend.app.ml.detection.inference import (
    DetectionInferenceService,
    detection_inference_service,
    detection_inference_service_v1,
    detection_inference_service_v2,
)
from backend.app.ml.detection.model import CycloneDetectorCNN
from backend.app.ml.detection.preprocessing import (
    compute_cloud_fraction,
    preprocess_for_inference,
    standardize_channels_to_rgb,
    validate_image_payload,
)
from backend.app.ml.detection.trainer import DetectionTrainer

__all__ = [
    "CycloneDetectorCNN",
    "DetectionTrainer",
    "DetectionEvaluator",
    "detection_evaluator",
    "DetectionInferenceService",
    "detection_inference_service",
    "detection_inference_service_v1",
    "detection_inference_service_v2",
    "CycloneImageDataset",
    "DetectionDatasetBuilder",
    "validate_image_payload",
    "standardize_channels_to_rgb",
    "preprocess_for_inference",
    "compute_cloud_fraction",
]

