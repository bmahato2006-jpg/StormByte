"""Pure PyTorch ResNet-18 Architecture for Tropical Cyclone Intensity Regression.

Zero-torchvision dependency: Built entirely with standard PyTorch (torch.nn) primitives
to ensure 100% portability, stable CPU execution, and immunity to dynamic link errors.
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.app.ml.intensity.config import (
    INPUT_CHANNELS,
    TARGET_MEAN,
    TARGET_STD,
    knots_to_central_pressure,
    knots_to_imd_category,
)


class BasicBlock(nn.Module):
    """Standard ResNet-18 BasicBlock with 3x3 convolutions and skip connection."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
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
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CycloneIntensityResNet18(nn.Module):
    """ResNet-18 Deep Residual Network for Tropical Cyclone Wind Speed Regression."""

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        target_mean: float = TARGET_MEAN,
        target_std: float = TARGET_STD,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.target_mean = target_mean
        self.target_std = target_std

        # Stem: 256x256 -> 64x64
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 Residual Stages (2 blocks each = 8 blocks = 16 conv layers + stem + head = ResNet-18)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Global Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

        self._initialize_weights()

    def _make_layer(
        self, in_channels: int, out_channels: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1, downsample=None))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting normalized target value."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        normalized_pred = self.regressor(x)  # shape: (B, 1)
        return normalized_pred

    def predict_knots(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting denormalized wind speed in knots."""
        norm_pred = self.forward(x)
        knots = norm_pred * self.target_std + self.target_mean
        return knots

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_single(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Runs end-to-end inference on a single standardized tensor (1, 3, 256, 256)."""
        self.eval()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            knots_val = float(self.predict_knots(tensor)[0, 0].item())

        # Clamp to physical range of historical observations
        knots_clamped = float(max(20.0, min(140.0, knots_val)))
        knots_rounded = round(knots_clamped, 1)
        pressure_val = knots_to_central_pressure(knots_rounded)
        category_val = knots_to_imd_category(knots_rounded)

        return {
            "wind_speed_knots": knots_rounded,
            "central_pressure_hpa": pressure_val,
            "category": category_val,
        }
