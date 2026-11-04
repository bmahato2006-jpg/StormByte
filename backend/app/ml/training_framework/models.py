"""Production PyTorch Neural Network Architectures for Cyclone-X.

Implements:
1. CycloneCenterNet: Pure PyTorch ResNet-backbone deep keypoint detector for automated cyclone eye localization,
   with transposed convolution upsampling, heatmap classification head, and sub-pixel offset regression head.
2. CenterNetFocalLoss: Exact modified Gaussian focal loss for keypoint heatmap training.
3. TrajectoryBiLSTM: Bidirectional Recurrent Neural Network with temporal attention / residual projection
   for 8-feature atmospheric environmental steering trajectory forecasting.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. CenterNet Modified Focal Loss
# =============================================================================

class CenterNetFocalLoss(nn.Module):
    """Modified Focal Loss for keypoint heatmap regression.

    Reference:
        CornerNet: Detecting Objects as Paired Keypoints (Law & Deng, ECCV 2018)
        Objects as Points (Zhou et al., arXiv 2019 / CenterNet)

    Formula:
        L_k = - (1 / N) * sum_{xy}:
            (1 - y_hat)^alpha * log(y_hat)                    if y == 1
            (1 - y)^beta * (y_hat)^alpha * log(1 - y_hat)     otherwise
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted heatmaps in (0, 1) of shape [B, C, H, W] (already activated by sigmoid).
            target: Ground truth Gaussian heatmaps in [0, 1] of shape [B, C, H, W].

        Returns:
            Scalar tensor loss.
        """
        # Clamp to avoid numerical underflow/log(0)
        pred = torch.clamp(pred, min=self.eps, max=1.0 - self.eps)

        pos_mask = target.eq(1.0).float()
        neg_mask = target.lt(1.0).float()

        # Positive term: (1 - pred)^alpha * log(pred)
        pos_loss = torch.pow(1.0 - pred, self.alpha) * torch.log(pred) * pos_mask

        # Negative term: (1 - target)^beta * pred^alpha * log(1 - pred)
        neg_weight = torch.pow(1.0 - target, self.beta)
        neg_loss = neg_weight * torch.pow(pred, self.alpha) * torch.log(1.0 - pred) * neg_mask

        num_pos = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss


class CenterNetRegLoss(nn.Module):
    """L1 Loss with spatial masking for sub-pixel offset regression."""

    def __init__(self):
        super().__init__()

    def forward(
        self, pred_offset: torch.Tensor, target_offset: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_offset: [B, 2, H, W]
            target_offset: [B, 2, H, W]
            mask: [B, 1, H, W] (1.0 at peak cells, 0.0 elsewhere)
        """
        mask_expanded = mask.expand_as(pred_offset)
        diff = torch.abs(pred_offset - target_offset) * mask_expanded
        num_pos = mask.sum()
        if num_pos > 0:
            return diff.sum() / (num_pos * 2.0)
        return torch.tensor(0.0, device=pred_offset.device)


# =============================================================================
# 2. CycloneCenterNet Architecture (Pure PyTorch ResNet Backbone)
# =============================================================================

class ResNetBasicBlock(nn.Module):
    """Standard ResNet Residual BasicBlock with skip connection."""

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


class ConvBlock(nn.Module):
    """Standard Convolution + BatchNorm + ReLU building block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class UpSampleBlock(nn.Module):
    """Transposed Convolution decoder block for upsampling feature maps."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.deconv(x)))


class CycloneCenterNet(nn.Module):
    """Deep Sub-pixel Cyclone Eye CenterNet with pure PyTorch ResNet Backbone.

    Replaces the algorithmic FRST (Fast Radial Symmetry Transform) heuristic with
    an end-to-end differentiable keypoint detection network.

    Features:
        - Backbone: ResNet-18 feature extractor (Stem + 4 Residual Stages)
        - Upsampling Decoder: 3-stage transposed convolution expanding feature stride from 32x down to 4x
        - Heatmap Head: Sub-pixel Gaussian eye centroid probability [B, 1, H/4, W/4]
        - Offset Head: Discretization coordinate offset regression [B, 2, H/4, W/4]
    """

    def __init__(
        self,
        in_channels: int = 3,
        backbone_name: str = "resnet18",
        pretrained: bool = False,
        downsample_ratio: int = 4,
        head_conv_channels: int = 64,
    ):
        super().__init__()
        self.downsample_ratio = downsample_ratio

        # 1. Stem: H x W -> H/4 x W/4
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2. Four Residual Stages (ResNet-18: [2, 2, 2, 2])
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)    # Stride 4
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)   # Stride 8
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)  # Stride 16
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)  # Stride 32

        # 3. Deconvolution Upsampling Decoder (Stride 32 -> 16 -> 8 -> 4)
        self.up1 = UpSampleBlock(512, 256)
        self.up2 = UpSampleBlock(256, 128)
        self.up3 = UpSampleBlock(128, 64)

        # 4. Task Prediction Heads (Stride 4)
        # Eye Keypoint Heatmap Head (1 class = Cyclone Eye)
        self.heatmap_head = nn.Sequential(
            ConvBlock(64, head_conv_channels, kernel_size=3),
            nn.Conv2d(head_conv_channels, 1, kernel_size=1),
        )
        # Continuous Sub-pixel Offset Head (dx, dy)
        self.offset_head = nn.Sequential(
            ConvBlock(64, head_conv_channels, kernel_size=3),
            nn.Conv2d(head_conv_channels, 2, kernel_size=1),
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

        layers = [ResNetBasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResNetBasicBlock(out_channels, out_channels, stride=1, downsample=None))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # CenterNet special initialization: bias = -2.19 so sigmoid(bias) ~ 0.1 at start
        self.heatmap_head[-1].bias.data.fill_(-2.19)
        nn.init.normal_(self.offset_head[-1].weight, std=0.001)
        nn.init.constant_(self.offset_head[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input satellite tensor [B, 3, H, W] in [0, 1].

        Returns:
            Dict containing:
                - 'heatmap': [B, 1, H/4, W/4] activated via torch.sigmoid
                - 'offset': [B, 2, H/4, W/4] continuous sub-pixel offset
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Backbone
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # Decoder
        u1 = self.up1(c4)
        u2 = self.up2(u1)
        u3 = self.up3(u2)  # Stride 4

        # Heads
        heatmap_raw = self.heatmap_head(u3)
        heatmap = torch.sigmoid(heatmap_raw)
        offset = self.offset_head(u3)

        return {"heatmap": heatmap, "offset": offset}

    @torch.no_grad()
    def decode_eye_center(
        self,
        heatmap: torch.Tensor,
        offset: torch.Tensor,
        top_k: int = 1,
        kernel_size: int = 3,
    ) -> Dict[str, Any]:
        """Decode peak eye coordinates in original image space using 2D Max-Pooling NMS.

        Args:
            heatmap: [1, 1, out_h, out_w]
            offset: [1, 2, out_h, out_w]
            top_k: Number of candidate centers to return (default 1).
            kernel_size: NMS neighborhood size.

        Returns:
            Dict with 'pixel_x', 'pixel_y', 'confidence', 'score'.
        """
        pad = (kernel_size - 1) // 2
        hmax = F.max_pool2d(heatmap, kernel_size=kernel_size, stride=1, padding=pad)
        keep = (hmax == heatmap).float()
        filtered_heatmap = heatmap * keep

        b, c, h, w = filtered_heatmap.shape
        flat = filtered_heatmap.view(b, -1)
        top_scores, top_indices = torch.topk(flat, top_k)

        y_indices = (top_indices // w).squeeze(0)
        x_indices = (top_indices % w).squeeze(0)
        scores = top_scores.squeeze(0)

        results = []
        for i in range(top_k):
            xi = int(x_indices[i].item())
            yi = int(y_indices[i].item())
            score = float(scores[i].item())

            dx = float(offset[0, 0, yi, xi].item())
            dy = float(offset[0, 1, yi, xi].item())

            orig_x = (xi + dx) * float(self.downsample_ratio)
            orig_y = (yi + dy) * float(self.downsample_ratio)

            results.append({"pixel_x": orig_x, "pixel_y": orig_y, "confidence": score})

        return results[0] if results else {"pixel_x": 0.0, "pixel_y": 0.0, "confidence": 0.0}

    @torch.no_grad()
    def decode_keypoints(
        self,
        heatmap: torch.Tensor,
        offset: torch.Tensor,
        k: int = 1,
        kernel_size: int = 3,
    ) -> List[Dict[str, float]]:
        """Decodes keypoints returning list of dicts with 'x', 'y', 'score'."""
        pad = (kernel_size - 1) // 2
        hmax = F.max_pool2d(heatmap, kernel_size=kernel_size, stride=1, padding=pad)
        keep = (hmax == heatmap).float()
        filtered_heatmap = heatmap * keep

        b, c, h, w = filtered_heatmap.shape
        flat = filtered_heatmap.view(b, -1)
        top_scores, top_indices = torch.topk(flat, k)

        y_indices = (top_indices // w).squeeze(0)
        x_indices = (top_indices % w).squeeze(0)
        scores = top_scores.squeeze(0)

        results = []
        for i in range(k):
            xi = int(x_indices[i].item())
            yi = int(y_indices[i].item())
            score = float(scores[i].item())

            dx = float(offset[0, 0, yi, xi].item())
            dy = float(offset[0, 1, yi, xi].item())

            orig_x = (xi + dx) * float(self.downsample_ratio)
            orig_y = (yi + dy) * float(self.downsample_ratio)

            results.append({"x": orig_x, "y": orig_y, "score": score, "pixel_x": orig_x, "pixel_y": orig_y, "confidence": score})

        return results


# =============================================================================
# 3. TrajectoryBiLSTM Architecture (Environmental Steering)
# =============================================================================

class TrajectoryBiLSTM(nn.Module):
    """Bidirectional Recurrent Neural Network for Tropical Cyclone Trajectory Forecasting.

    Accepts 8-dimensional multi-source atmospheric and kinematic sequences:
    [latitude, longitude, max_wind_kt, central_pressure_hpa, sea_surface_temp_c,
     vertical_wind_shear_kt, steering_u500_kt, steering_v500_kt].

    Projects future relative displacements (dlat, dlon) across multi-horizon steps:
    +6h (Step 1), +12h (Step 2), +24h (Step 3).
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        forecast_steps: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_steps = forecast_steps

        # Input feature projection
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Bidirectional Recurrent Core
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention pooling over temporal sequence
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Multi-horizon projection head (predicting [dlat, dlon] for each horizon)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_steps * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence tensor [Batch, Seq_Length, 8].

        Returns:
            Displacement tensor [Batch, Forecast_Steps, 2] in degrees (dlat, dlon).
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project input features
        h = self.input_encoder(x)  # [B, T, hidden_dim]

        # 2. BiLSTM recurrent forward pass
        lstm_out, _ = self.bilstm(h)  # [B, T, hidden_dim * 2]

        # 3. Temporal Attention Pooling
        attn_weights = self.temporal_attention(lstm_out)  # [B, T, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(lstm_out * attn_weights, dim=1)  # [B, hidden_dim * 2]

        # 4. Multi-horizon displacement projection
        out = self.head(pooled)  # [B, forecast_steps * 2]
        return out.view(batch_size, self.forecast_steps, 2)
