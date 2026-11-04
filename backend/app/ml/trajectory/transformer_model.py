"""PyTorch Spatio-Temporal Trajectory Transformer for Cyclone-X (Phase 13).

A clean, reproducible, self-attention Transformer encoder architecture designed for
multi-horizon (+6h, +12h, +24h) tropical cyclone trajectory displacement prediction.
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from backend.app.ml.trajectory.config import (
    HIDDEN_DIM,
    HORIZONS,
    INPUT_DIM,
    NUM_LAYERS,
    OUTPUT_DIM,
    SEQ_LENGTH,
)


class TrajectoryTransformer(nn.Module):
    """Spatio-Temporal Transformer Encoder for tropical cyclone trajectory forecasting.
    
    Processes sliding windows of sequential track telemetry (seq_len=4, input_dim=13)
    and predicts relative latitude/longitude displacements for multiple forecast horizons.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = HIDDEN_DIM,
        nhead: int = 4,
        num_layers: int = NUM_LAYERS,
        dim_feedforward: int = 128,
        output_dim: int = OUTPUT_DIM,
        dropout: float = 0.1,
        seq_len: int = SEQ_LENGTH,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.seq_len = seq_len

        # 1. Feature Projection & LayerNorm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # 2. Learnable Temporal Positional Embedding
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.pos_dropout = nn.Dropout(dropout)

        # 3. Transformer Encoder (Pre-LN for training stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(d_model)

        # 4. Multi-Horizon Non-Linear Regression Head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using Xavier uniform and normal distributions."""
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Tensor of shape (batch_size, 6): [dlat_6, dlon_6, dlat_12, dlon_12, dlat_24, dlon_24]
        """
        B, T, D = x.shape

        # Project features
        h = self.input_proj(x)  # (B, T, d_model)

        # Add temporal positional embedding
        if T <= self.seq_len:
            h = h + self.pos_embedding[:, :T, :]
        else:
            # Fallback if variable length passed
            h = h + self.pos_embedding[:, -1:, :]

        h = self.pos_dropout(h)

        # Transformer encoder pass
        encoded = self.transformer_encoder(h)  # (B, T, d_model)
        encoded = self.final_norm(encoded)

        # Extract last observation timestep's contextualized representation
        last_step_repr = encoded[:, -1, :]  # (B, d_model)

        # Regression head to produce [dlat_6, dlon_6, dlat_12, dlon_12, dlat_24, dlon_24]
        out = self.head(last_step_repr)
        return out

    def predict_displacements(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method returning reshaped displacement tensor (batch, 3, 2)."""
        out = self.forward(x)
        return out.view(-1, len(HORIZONS), 2)

    def predict_coordinates(
        self,
        x: torch.Tensor,
        origin_lat: float,
        origin_lon: float,
    ) -> Dict[int, Tuple[float, float]]:
        """Predicts actual future coordinates (+6h, +12h, +24h) given an input sequence."""
        self.eval()
        with torch.no_grad():
            if x.ndim == 2:
                x = x.unsqueeze(0)
            displacements = self.predict_displacements(x).squeeze(0).cpu().numpy()

        coords: Dict[int, Tuple[float, float]] = {}
        for idx, h in enumerate(HORIZONS):
            dlat = float(displacements[idx, 0])
            dlon = float(displacements[idx, 1])
            coords[h] = (round(origin_lat + dlat, 4), round(origin_lon + dlon, 4))
        return coords
