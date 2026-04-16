"""scene_prior/motion_model.py
================================
Lightweight motion model that maps entity identity + time → pose code.

Two classes are provided:

  MotionModel      — single-frame: (entity_id, t_scalar) → pose_code
  TrajectoryPrior  — multi-frame:  entity_id             → (B, T, pose_dim)

The design is backbone-agnostic: no AnimateDiff or diffusion dependencies.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MotionModel
# ---------------------------------------------------------------------------

class MotionModel(nn.Module):
    """Maps entity identity and a time scalar to a pose code.

    Parameters
    ----------
    id_dim : int
        Dimension of the incoming entity identity embedding.
    hidden_dim : int
        Width of intermediate MLP layers.
    pose_dim : int
        Output pose code dimension.
    n_frames : int
        Number of video frames (used by TrajectoryPrior).

    Input
    -----
    entity_id : (B, id_dim)
    t_scalar  : (B, 1)        normalised time in [0, 1]

    Output
    ------
    pose_code : (B, pose_dim)
    """

    def __init__(
        self,
        id_dim: int = 128,
        hidden_dim: int = 256,
        pose_dim: int = 32,
        n_frames: int = 8,
    ) -> None:
        super().__init__()
        self.id_dim = id_dim
        self.hidden_dim = hidden_dim
        self.pose_dim = pose_dim
        self.n_frames = n_frames

        # Time embedding: scalar → small Fourier-style feature
        self.time_dim = 32
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        # Main MLP: [id_dim + time_dim] → pose_dim
        in_dim = id_dim + self.time_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pose_dim),
        )

    def forward(
        self,
        entity_id: torch.Tensor,   # (B, id_dim)
        t_scalar:  torch.Tensor,   # (B, 1)   normalised ∈ [0, 1]
    ) -> torch.Tensor:             # (B, pose_dim)
        t_feat = self.time_embed(t_scalar.float())     # (B, time_dim)
        x = torch.cat([entity_id.float(), t_feat], dim=-1)  # (B, id_dim+time_dim)
        return self.mlp(x)                             # (B, pose_dim)

    # ------------------------------------------------------------------
    def forward_sequence(
        self,
        entity_id: torch.Tensor,   # (B, id_dim)
        T: Optional[int] = None,
    ) -> torch.Tensor:             # (B, T, pose_dim)
        """Generate pose codes for a sequence of T frames.

        Equivalent to calling ``forward`` T times with t ∈ {0/T-1, …, 1}.

        Parameters
        ----------
        entity_id : (B, id_dim)
        T         : int — number of frames; defaults to self.n_frames.
        """
        if T is None:
            T = self.n_frames
        B = entity_id.shape[0]
        device = entity_id.device

        # Normalised time steps: (T,)
        t_vals = torch.linspace(0.0, 1.0, T, device=device)  # (T,)

        # Expand entity_id: (B, T, id_dim)
        id_exp = entity_id.unsqueeze(1).expand(B, T, -1)      # (B, T, id_dim)
        id_flat = id_exp.reshape(B * T, self.id_dim)           # (BT, id_dim)

        # Expand t: (B*T, 1)
        t_exp = t_vals.unsqueeze(0).expand(B, T)               # (B, T)
        t_flat = t_exp.reshape(B * T, 1)                       # (BT, 1)

        pose_flat = self.forward(id_flat, t_flat)              # (BT, pose_dim)
        return pose_flat.reshape(B, T, self.pose_dim)          # (B, T, pose_dim)


# ---------------------------------------------------------------------------
# TrajectoryPrior (convenience wrapper)
# ---------------------------------------------------------------------------

class TrajectoryPrior(nn.Module):
    """Wrapper around MotionModel that directly exposes ``forward_sequence``.

    Can be used independently when only trajectory generation is needed,
    without the single-frame API.
    """

    def __init__(
        self,
        id_dim: int = 128,
        hidden_dim: int = 256,
        pose_dim: int = 32,
        n_frames: int = 8,
    ) -> None:
        super().__init__()
        self.motion = MotionModel(
            id_dim=id_dim,
            hidden_dim=hidden_dim,
            pose_dim=pose_dim,
            n_frames=n_frames,
        )
        self.pose_dim = pose_dim

    def forward(
        self,
        entity_id: torch.Tensor,   # (B, id_dim)
        T: Optional[int] = None,
    ) -> torch.Tensor:             # (B, T, pose_dim)
        return self.motion.forward_sequence(entity_id, T)
