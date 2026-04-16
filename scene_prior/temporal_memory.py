"""scene_prior/temporal_memory.py
================================
Temporal slot memory for Phase 64 entity fields.

Each entity gets its own TemporalSlotMemory instance.  Given the previous
hidden GRU state and the current context feature map, it produces an updated
state that captures temporal dynamics across frames.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalSlotMemory(nn.Module):
    """GRU-based temporal memory for a single entity slot.

    Compresses the current context feature map to an observation vector via
    AdaptiveAvgPool → Linear, then updates the hidden state with a GRUCell.

    Parameters
    ----------
    slot_dim : int
        GRU hidden state dimension.
    obs_dim : int
        Observation (input) dimension for the GRUCell.  This must equal the
        channel count of ``field_feat``.
    """

    def __init__(self, slot_dim: int = 128, obs_dim: int = 128) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.obs_dim  = obs_dim

        # Observation projector: (B, C, H, W) → (B, obs_dim)
        # We learn a 1×1 linear after spatial pooling.
        # The input channel count equals obs_dim (= ctx_dim from the encoder).
        self.obs_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, 1, 1)
            nn.Flatten(),             # (B, C)
            nn.Linear(obs_dim, obs_dim),
            nn.GELU(),
        )

        # GRU cell: maps obs → new hidden state
        self.gru = nn.GRUCell(obs_dim, slot_dim)

    # ------------------------------------------------------------------

    def forward(
        self,
        prev_state: torch.Tensor,    # (B, slot_dim)
        field_feat: torch.Tensor,    # (B, C, H, W)
    ) -> torch.Tensor:               # (B, slot_dim)
        """Update the slot state given the current entity context feature.

        Parameters
        ----------
        prev_state : (B, slot_dim)
            Previous GRU hidden state.  Use ``init_state`` for the first frame.
        field_feat : (B, C, H, W)
            Context feature map from ImageContextEncoder (or any feature map
            with C == obs_dim).

        Returns
        -------
        new_state : (B, slot_dim)
        """
        obs = self.obs_proj(field_feat)        # (B, obs_dim)
        new_state = self.gru(obs, prev_state)  # (B, slot_dim)
        return new_state

    # ------------------------------------------------------------------

    def init_state(self, B: int, device: torch.device) -> torch.Tensor:
        """Return a zeroed initial hidden state.

        Parameters
        ----------
        B      : batch size
        device : target device

        Returns
        -------
        (B, slot_dim) zeros
        """
        return torch.zeros(B, self.slot_dim, device=device)
