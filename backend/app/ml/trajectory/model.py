"""PyTorch TrajectoryLSTM Neural Network Architecture for Cyclone-X (Phase 8).

A compact, reproducible, 2-layer Recurrent Neural Network designed for
multi-horizon (+6h, +12h, +24h) tropical cyclone trajectory displacement prediction.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from backend.app.ml.trajectory.config import (
    HIDDEN_DIM,
    HORIZONS,
    INPUT_DIM,
    NUM_LAYERS,
    OUTPUT_DIM,
)


class TrajectoryLSTM(nn.Module):
    """Deep Recurrent Neural Network for tropical cyclone trajectory forecasting."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        output_dim: int = OUTPUT_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # 2-layer LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Non-linear regression projection head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier uniform initialization for linear layers."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tensor of shape (batch_size, 6): [dlat_6, dlon_6, dlat_12, dlon_12, dlat_24, dlon_24]
        """
        lstm_out, (hn, cn) = self.lstm(x)
        # Take the hidden representation of the final time-step
        last_step_hidden = lstm_out[:, -1, :]
        out = self.head(last_step_hidden)
        return out

    def predict_displacements(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method returning reshaped displacement tensor (batch, 3, 2)."""
        out = self.forward(x)
        return out.view(-1, len(HORIZONS), 2)

    def predict_coordinates(
        self,
        x: torch.Tensor,
        origin_lat: float,
        origin_lon: float
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

