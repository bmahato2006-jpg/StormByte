"""Cyclone Detector CNN V3: Deep Residual Architecture for North Indian Ocean Satellite Detection.

Architecture:
- 5-Stage Deep Residual Convolutional Neural Network (Stem + 4 Residual Stages).
- Residual skip connections with 1x1 projection shortcuts for stable gradient flow.
- Pure PyTorch implementation (zero torchvision dependency).
- Global Adaptive Average Pooling with 2-layer MLP classification head and dropout regularization.
- Native hook for Grad-CAM feature map extraction and attention visualization.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ["NO_CYCLONE", "CYCLONE"]
NUM_CLASSES = 2
INPUT_CHANNELS = 3
DEFAULT_THRESHOLD = 0.50


class ResidualBlock(nn.Module):
    """Residual Convolutional Block with 1x1 projection shortcut for downsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut projection if spatial dimension or channel count changes
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class CycloneDetectorCNN_V3(nn.Module):
    """5-Stage Deep Residual Convolutional Neural Network for Cyclone Detection V3."""

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stem: (3, 256, 256) -> (32, 128, 128)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 1: (32, 128, 128) -> (64, 64, 64)
        self.stage1 = ResidualBlock(32, 64, stride=2)

        # Stage 2: (64, 64, 64) -> (128, 32, 32)
        self.stage2 = ResidualBlock(64, 128, stride=2)

        # Stage 3: (128, 32, 32) -> (256, 16, 16)
        self.stage3 = ResidualBlock(128, 256, stride=2)

        # Stage 4: (256, 16, 16) -> (256, 16, 16) - Rich 16x16 resolution for Grad-CAM
        self.stage4 = ResidualBlock(256, 256, stride=1)

        # Head: Global Average Pooling + MLP
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming normal initialization for conv layers and Xavier for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts final convolutional feature maps (for Grad-CAM and diagnostic visualization)."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        features = self.stage4(x)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits (batch_size, 2)."""
        features = self.extract_features(x)
        pooled = self.global_pool(features)
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes calibrated class probabilities via Softmax (batch_size, 2)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def classify_single(
        self, x: torch.Tensor, threshold: float = 0.50
    ) -> Dict[str, Any]:
        """Classifies a single preprocessed image tensor (1, 3, H, W) or (3, H, W)."""
        self.eval()
        if x.ndim == 3:
            x = x.unsqueeze(0)

        with torch.no_grad():
            proba = self.predict_proba(x)[0].cpu().numpy()

        p_no_cyclone = float(proba[0])
        p_cyclone = float(proba[1])

        is_cyclone = bool(p_cyclone >= threshold)
        predicted_idx = 1 if is_cyclone else 0
        predicted_label = CLASS_NAMES[predicted_idx]
        confidence = p_cyclone if is_cyclone else p_no_cyclone

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


if __name__ == "__main__":
    # Smoke test architecture
    model = CycloneDetectorCNN_V3()
    dummy_input = torch.randn(2, 3, 256, 256)
    logits = model(dummy_input)
    probs = model.predict_proba(dummy_input)
    feats = model.extract_features(dummy_input)
    print(f"Model V3 initialized successfully.")
    print(f"Trainable parameters: {model.count_parameters():,}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Feature map shape: {feats.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape} | Row sum: {probs.sum(dim=-1)}")
