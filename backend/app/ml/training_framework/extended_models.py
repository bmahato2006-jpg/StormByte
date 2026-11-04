"""Extended Deep Learning Model Architectures for Cyclone-X.

Implements production-grade, device-agnostic PyTorch models for:
1. CycloneViT: Vision Transformer with 16x16 patch embeddings & self-attention for synoptic satellite analysis.
2. ObjectiveDvorakNet: Multi-scale curved-band & central-dense-overcast (CDO) network for objective Dvorak T-number estimation.
3. PrecipitationUNet: 4-stage encoder-decoder U-Net mapping satellite IR brightness to surface rain flux (mm/hr).
4. SatelliteRadarFusionNet: Dual-branch cross-attention network fusing satellite TIR1 and Doppler Weather Radar.
5. HollandPINN: Physics-Informed Neural Network solving the gradient wind balance equation.
6. PhysicsAdvectionNet: Neural advection network parameterizing inertial momentum and planetary Rossby beta-drift deflection.
7. SurgeDeepSurrogate: Deep surrogate neural network emulating 2D hydrodynamic storm surge (ADCIRC/SLOSH).
8. FloodRiskNet: Hydrological runoff & drainage basin inundation deep network.
9. LandfallPredictorNet: Multi-task coastal crossing point and arrival lead-time regression network.
10. ImpactNeuralEngine: Multi-hazard composite risk and critical infrastructure vulnerability deep network.
11. WaveWatchSurrogateNet: Coupled 3rd-generation spectral ocean wave surrogate (WAVEWATCH III / SWAN).
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. CycloneViT: Vision Transformer for Synoptic Satellite Analysis
# =============================================================================

class PatchEmbedding(nn.Module):
    """Splits image into non-overlapping patches and projects to embedding dimension."""

    def __init__(self, img_size: int = 256, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> proj -> [B, embed_dim, H/patch, W/patch] -> flatten -> [B, embed_dim, num_patches] -> transpose -> [B, num_patches, embed_dim]
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerEncoderBlock(nn.Module):
    """Pre-LN Multi-Head Self-Attention + MLP Transformer Encoder Block."""

    def __init__(self, embed_dim: int = 192, num_heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class CycloneViT(nn.Module):
    """Vision Transformer (ViT) for automated tropical cyclone presence and circulation analysis."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 2,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_dim]
        x = self.pos_drop(x + self.pos_embed)

        attn_maps = []
        for block in self.blocks:
            x, weights = block(x)
            attn_maps.append(weights)

        x = self.norm(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probabilities": probs,
            "attention_maps": attn_maps,
            "cyclone_probability": probs[:, 1],
        }


# =============================================================================
# 2. ObjectiveDvorakNet: Deep Objective Dvorak T-Number CNN
# =============================================================================

class ObjectiveDvorakNet(nn.Module):
    """Multi-Scale Deep Convolutional Network for Objective Dvorak T-Number (T1.0–T8.0) Estimation."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Branch 1: Core Eye & Central Dense Overcast (CDO) features (Fine-scale 3x3)
        self.cdo_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Branch 2: Curved Band Spiral Pattern (Coarse-scale 7x7)
        self.band_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Merged deep feature representation
        self.merged_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Dual Regression Heads:
        # 1. Dvorak T-number: [1.0, 8.0]
        self.t_number_head = nn.Sequential(
            nn.Linear(128 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # scaled to 1.0 - 8.0
        )

        # 2. Maximum Sustained Wind (kt) and Central Pressure (hPa)
        self.wind_head = nn.Sequential(
            nn.Linear(128 * 16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cdo_feat = self.cdo_branch(x)
        band_feat = self.band_branch(x)
        merged = torch.cat([cdo_feat, band_feat], dim=1)
        feat = self.merged_conv(merged)
        flat = feat.view(feat.size(0), -1)

        # Scale Sigmoid output to Dvorak range [1.0, 8.0]
        t_num = 1.0 + 7.0 * self.t_number_head(flat).squeeze(-1)

        # Physical Atkinson-Holliday relation from predicted T-number:
        # T1.0 ~ 25kt, T4.0 ~ 65kt, T6.0 ~ 115kt, T8.0 ~ 170kt
        wind_pres = self.wind_head(flat)
        vmax_kt = 25.0 + (t_num - 1.0) * 20.7 + wind_pres[:, 0]
        pc_hpa = 1010.0 - (vmax_kt * 0.72) + wind_pres[:, 1]

        return {
            "dvorak_t_number": t_num,
            "max_wind_kt": vmax_kt,
            "central_pressure_hpa": pc_hpa,
        }


# =============================================================================
# 3. PrecipitationUNet: Satellite-to-Rainfall Flux AI
# =============================================================================

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2 block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrecipitationUNet(nn.Module):
    """4-stage U-Net architecture predicting 2D surface precipitation flux (mm/hr) from thermal IR."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Decoder with Skip Connections
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)

        # Output Rain Rate Head (mm/hr >= 0)
        self.outc = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),  # Rainfall rate is strictly non-negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        u1 = self.up1(x4)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv_up3(u3)

        rain_rate_mm_h = self.outc(u3)
        return rain_rate_mm_h


# =============================================================================
# 4. SatelliteRadarFusionNet: Multi-Modal Cross-Attention Fusion
# =============================================================================

class SatelliteRadarFusionNet(nn.Module):
    """Dual-branch cross-attention fusion network combining Satellite IR with Coastal Doppler Radar."""

    def __init__(self, sat_channels: int = 3, radar_channels: int = 2, feature_dim: int = 128):
        super().__init__()
        # Satellite Branch (TIR1/VIS)
        self.sat_encoder = nn.Sequential(
            DoubleConv(sat_channels, 32),
            nn.MaxPool2d(2),
            DoubleConv(32, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, feature_dim),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        # Radar Branch (Reflectivity Z and Radial Velocity Vr)
        self.radar_encoder = nn.Sequential(
            DoubleConv(radar_channels, 32),
            nn.MaxPool2d(2),
            DoubleConv(32, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, feature_dim),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        # Cross-Attention: Radar queries Satellite features
        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

        # Eyewall Diagnostics & Intensity Head
        self.diagnostic_head = nn.Sequential(
            nn.Linear(feature_dim * 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 4),  # [Vmax_kt, Eyewall_Diameter_km, Max_Reflectivity_dBZ, Asymmetry_Index]
        )

    def forward(self, sat_img: torch.Tensor, radar_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = sat_img.size(0)
        # 1. Encode spatial features
        sat_feat = self.sat_encoder(sat_img).flatten(2).transpose(1, 2)    # [B, 64, feature_dim]
        radar_feat = self.radar_encoder(radar_img).flatten(2).transpose(1, 2)  # [B, 64, feature_dim]

        # 2. Cross-modal attention
        fused_attn, weights = self.cross_attn(query=radar_feat, key=sat_feat, value=sat_feat)
        fused = self.norm(radar_feat + fused_attn).flatten(1)  # [B, 64 * feature_dim]

        # 3. Predict diagnostics
        out = self.diagnostic_head(fused)
        return {
            "vmax_kt": F.relu(out[:, 0] + 45.0),
            "eyewall_diameter_km": F.relu(out[:, 1] + 35.0),
            "max_reflectivity_dbz": F.relu(out[:, 2] + 48.0),
            "asymmetry_index": torch.sigmoid(out[:, 3]),
            "attention_weights": weights,
        }


# =============================================================================
# 5. HollandPINN: Physics-Informed Neural Network (Vortex Momentum Equation)
# =============================================================================

class HollandPINN(nn.Module):
    """Physics-Informed Neural Network (PINN) for Tropical Cyclone Radial Wind & Pressure Dynamics.

    Enforces the Navier-Stokes Gradient Wind Momentum Equilibrium in the loss function:
        F(r) = V(r)^2 / r + f * V(r) - (1 / rho) * (dP / dr) = 0
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        # Input features: [normalized_r, latitude, central_pressure_drop, translation_speed]
        layers = [nn.Linear(4, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 2))  # Outputs: [V(r) in m/s, P(r) in hPa]
        self.net = nn.Sequential(*layers)

    def forward(self, r_norm: torch.Tensor, lat: torch.Tensor, delta_p: torch.Tensor, v_trans: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            r_norm: [B, 1] normalized radius r / 500 km
            lat: [B, 1] latitude in degrees
            delta_p: [B, 1] pressure drop (Pn - Pc) in hPa
            v_trans: [B, 1] translation speed in m/s

        Returns:
            v_radial: [B, 1] gradient wind speed V(r) in m/s
            p_radial: [B, 1] surface pressure P(r) in hPa
        """
        inp = torch.cat([r_norm, lat, delta_p, v_trans], dim=-1)
        out = self.net(inp)
        v_radial = F.relu(out[:, 0:1])  # Wind speed >= 0
        p_radial = 1010.0 - out[:, 1:2]
        return v_radial, p_radial

    def compute_pinn_loss(
        self, r: torch.Tensor, lat: torch.Tensor, delta_p: torch.Tensor, v_trans: torch.Tensor, rho: float = 1.15
    ) -> torch.Tensor:
        """Compute physics residual loss: F = V^2 / r + f * V - (1 / rho) * dP/dr."""
        r.requires_grad_(True)
        v, p = self.forward(r, lat, delta_p, v_trans)

        # dP / dr (Pa / m)
        dp_dr = torch.autograd.grad(p, r, grad_outputs=torch.ones_like(p), create_graph=True)[0] * 100.0 / 500000.0

        # Coriolis parameter f = 2 * Omega * sin(phi)
        omega = 7.2921e-5
        f_coriolis = 2.0 * omega * torch.sin(torch.deg2rad(lat))

        # Real radius in meters (r_norm * 500 km)
        r_meters = torch.clamp(r * 500000.0, min=5000.0)

        # Physical equilibrium residual
        centrifugal = (v ** 2) / r_meters
        coriolis = f_coriolis * v
        pgf = (1.0 / rho) * dp_dr
        residual = centrifugal + coriolis - pgf

        physics_loss = torch.mean(residual ** 2)
        return physics_loss


# =============================================================================
# 6. PhysicsAdvectionNet: Neural Advection & Rossby Beta-Drift
# =============================================================================

class PhysicsAdvectionNet(nn.Module):
    """Neural Advection Network fusing Inertial Translation with Rossby Beta-Drift dynamics."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # Inputs: [u_trans, v_trans, u_steering500, v_steering500, latitude, vmax_kt, pc_hpa] (7 features)
        self.mlp = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6),  # [+6h, +12h, +24h displacement vectors (dlat, dlon)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Features: [B, 7] -> outputs [B, 3, 2] displacements."""
        out = self.mlp(features)
        return out.view(features.size(0), 3, 2)


# =============================================================================
# 7. SurgeDeepSurrogate: Hydrodynamic Coastal Storm Surge Surrogate
# =============================================================================

class SurgeDeepSurrogate(nn.Module):
    """Deep Surrogate Neural Network emulating 2D ADCIRC/SLOSH Hydrodynamic Storm Surge Inundation."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # Inputs: [central_pressure_drop_hpa, max_wind_kt, coastal_distance_km, bathymetry_slope, onshore_wind_u, onshore_wind_v, forward_speed_kt] (7 inputs)
        self.net = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),  # [Peak Surge Height (m), Inundation Extent (km), Surge Energy Index]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.net(x)
        surge_height_m = F.relu(out[:, 0])  # Non-negative surge
        inundation_km = F.relu(out[:, 1])
        energy_idx = torch.sigmoid(out[:, 2]) * 100.0

        return {
            "surge_height_meters": surge_height_m,
            "inundation_extent_km": inundation_km,
            "surge_hazard_index": energy_idx,
        }


# =============================================================================
# 8. FloodRiskNet: Hydrological Runoff & Drainage Basin Inundation
# =============================================================================

class FloodRiskNet(nn.Module):
    """Deep hydrological runoff model linking precipitation flux to river basin flood stress."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        # Inputs: [24h_accumulated_rain_mm, max_hourly_rain_mm, soil_moisture_index, elevation_m, distance_to_river_km] (5 inputs)
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),  # [Runoff Potential Index (0-1), Inundation Depth (m)]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.net(x)
        runoff_score = torch.sigmoid(out[:, 0])
        inundation_m = F.relu(out[:, 1])
        return {"runoff_potential": runoff_score, "inundation_depth_m": inundation_m}


# =============================================================================
# 9. LandfallPredictorNet: Direct Coastal Crossing Point & Lead Time
# =============================================================================

class LandfallPredictorNet(nn.Module):
    """Multi-task neural network predicting coastal landfall coordinates and lead time arrival."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Landfall coordinates (lat, lon)
        self.coord_head = nn.Linear(hidden_dim, 2)
        # Landfall lead-time hours (>= 0)
        self.time_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.ReLU())
        # Landfall probability (0 - 1)
        self.prob_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x)
        coords = self.coord_head(feat)
        time_h = self.time_head(feat)
        prob = self.prob_head(feat)

        return {
            "landfall_latitude": coords[:, 0],
            "landfall_longitude": coords[:, 1],
            "arrival_lead_time_hours": time_h.squeeze(-1),
            "landfall_probability": prob.squeeze(-1),
        }


# =============================================================================
# 10. ImpactNeuralEngine: Multi-Hazard Vulnerability Deep Network
# =============================================================================

class ImpactNeuralEngine(nn.Module):
    """Deep tabular vulnerability model replacing static rule weights with trained multi-hazard loss."""

    def __init__(self, num_hazards: int = 6, hidden_dim: int = 64):
        super().__init__()
        # Inputs: [wind_hazard_score, surge_hazard_score, flood_hazard_score, population_density_log, critical_infra_count, coastal_distance_km] (6 inputs)
        self.net = nn.Sequential(
            nn.Linear(num_hazards, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # [Composite Risk Index (0-100), Population at Extreme Risk, Damage Ratio (0-1)]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.net(x)
        composite_risk = torch.clamp(out[:, 0], 0.0, 100.0)
        pop_at_risk = F.relu(out[:, 1])
        damage_ratio = torch.sigmoid(out[:, 2])

        return {
            "composite_risk_score": composite_risk,
            "population_at_risk": pop_at_risk,
            "estimated_damage_ratio": damage_ratio,
        }


# =============================================================================
# 11. WaveWatchSurrogateNet: Coupled 3rd-Gen Spectral Ocean Wave Surrogate (WW3/SWAN)
# =============================================================================

class WaveWatchSurrogateNet(nn.Module):
    """Deep Surrogate Neural Network emulating 3rd-generation spectral ocean wave models (WAVEWATCH III / SWAN).

    Predicts integrated spectral sea-state variables from tropical cyclone kinematic forcing:
    - Significant Wave Height Hs (m)
    - Peak Wave Period Tp (s)
    - Mean Wave Period Tm (s)
    - Coastal Radiation Stress Wave Setup (m)
    - Wave Energy Flux Pw (kW/m)
    - Breaking Wave Hazard Index (0-100)
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 6),  # [hs, tp, tm, wave_setup, wave_power, danger_idx]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.net(x)
        hs = F.softplus(out[:, 0]) + 0.5  # smooth positive wave height >= 0.5m
        tp = F.softplus(out[:, 1]) + 4.0  # smooth peak period >= 4.0s
        tm = F.softplus(out[:, 2]) + 3.0  # smooth mean period >= 3.0s
        setup = F.softplus(out[:, 3])     # wave setup >= 0
        power = F.softplus(out[:, 4])     # energy flux >= 0
        danger = torch.sigmoid(out[:, 5]) * 100.0  # 0 to 100%

        return {
            "significant_wave_height_m": hs,
            "peak_wave_period_s": tp,
            "mean_wave_period_s": tm,
            "wave_setup_meters": setup,
            "wave_energy_flux_kw_m": power,
            "breaking_danger_index": danger,
        }

