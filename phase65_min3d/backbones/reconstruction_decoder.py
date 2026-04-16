from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionDecoder(nn.Module):
    """Simple RGB reconstruction baseline from decoder adapter features."""

    def __init__(self, in_dim: int = 128, hidden_dim: int = 128, out_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, decoder_cond: torch.Tensor) -> torch.Tensor:
        return self.net(decoder_cond)
