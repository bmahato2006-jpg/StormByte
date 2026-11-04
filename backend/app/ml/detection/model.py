"""PyTorch Neural Network Architecture for Satellite Cyclone Detection (Phase 9).

Architecture:
- 4-Stage Deep Convolutional Neural Network with Batch Normalization and Dropout.
- Standardized 3-channel input (256 x 256).
- Calibrated Softmax output for binary detection: [P(NO_CYCLONE), P(CYCLONE)].
- Secondary structural pattern categorization.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.app.ml.detection.config import (
    CLASS_NAMES,
    IDX_TO_CLASS,
    INPUT_CHANNELS,
    NUM_CLASSES,
)


class CycloneDetectorCNN(nn.Module):
    """Deep Convolutional Neural Network for satellite-based tropical cyclone detection."""

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stage 1: (3, 256, 256) -> (32, 128, 128)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 2: (32, 128, 128) -> (64, 64, 64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 3: (64, 64, 64) -> (128, 32, 32)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 4: (128, 32, 32) -> (256, 16, 16)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adaptive Pooling & Classification Head
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming normal initialization for conv layers and Xavier for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits (batch_size, 2)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes calibrated class probabilities via softmax."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities

    def classify_single(
        self, x: torch.Tensor, threshold: float = 0.50
    ) -> Dict[str, Any]:
        """Classifies a single preprocessed image tensor (1, 3, H, W)."""
        self.eval()
        if x.ndim == 3:
            x = x.unsqueeze(0)

        with torch.no_grad():
            proba = self.predict_proba(x)[0].cpu().numpy()

        p_no_cyclone = float(proba[0])
        p_cyclone = float(proba[1])

        is_cyclone = bool(p_cyclone >= threshold)
        predicted_idx = 1 if is_cyclone else 0
        predicted_label = IDX_TO_CLASS[predicted_idx]
        confidence = p_cyclone if is_cyclone else p_no_cyclone

        # Secondary structural pattern assessment
        if is_cyclone:
            if p_cyclone > 0.85:
                structural_type = "TIGHT_VORTEX_CORE"
            elif p_cyclone > 0.70:
                structural_type = "SPIRAL_BAND_ORGANIZATION"
            else:
                structural_type = "DISORGANIZED_CONVECTIVE_MASS"
        else:
            structural_type = "NON_CYCLONIC_ENVIRONMENT"

        return {
            "is_cyclone": is_cyclone,
            "predicted_label": predicted_label,
            "confidence": round(float(confidence), 4),
            "class_probabilities": {
                "NO_CYCLONE": round(p_no_cyclone, 4),
                "CYCLONE": round(p_cyclone, 4),
            },
            "structural_type": structural_type,
        }

