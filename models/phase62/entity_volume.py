"""
Phase 62 — Entity Volume Predictor
====================================

Predicts a 3D entity volume V_logits (B, N+1, K, H, W) from UNet
cross-attention features, where N+1 = 3 classes (bg=0, entity0=1, entity1=2).

This v2 predictor strengthens Rule-1-style topology bias by:
  1. Building a deeper shared 3D trunk over (K, H, W)
  2. Splitting bg / entity0 / entity1 into separate 3D branches
  3. Letting each entity branch see both global context and its own slot feature

The goal is to reduce one-entity winner collapse and encourage more coherent
3D blobs before first-hit projection.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Residual3DBlock(nn.Module):
    """Simple residual 3D block for stronger spatial-depth continuity bias."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=True),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


def _make_branch(in_ch: int, hidden: int, n_blocks: int) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv3d(in_ch, hidden, kernel_size=3, padding=1, bias=True),
        nn.GELU(),
    ]
    for _ in range(n_blocks):
        layers.append(Residual3DBlock(hidden))
    return nn.Sequential(*layers)


class EntityVolumePredictor(nn.Module):
    """
    Predict 3D entity volume from UNet cross-attention features.

    Input:
        F_g  (B, S, D) — global attention features
        F_0  (B, S, D) — entity-0 slot features
        F_1  (B, S, D) — entity-1 slot features

    Output:
        V_logits (B, 3, K, H, W) — per-voxel class logits
    """

    def __init__(
        self,
        feat_dim: int = 640,
        n_classes: int = 3,
        depth_bins: int = 8,
        spatial_h: int = 16,
        spatial_w: int = 16,
        hidden: int = 64,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_classes = n_classes
        self.depth_bins = depth_bins
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.hidden = hidden

        # Feature projectors.
        self.proj_g = nn.Linear(feat_dim, hidden)
        self.proj_e0 = nn.Linear(feat_dim, hidden)
        self.proj_e1 = nn.Linear(feat_dim, hidden)

        # Shared 2D fusion before depth expansion.
        self.shared_2d = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden * 2, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden * 2, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )

        # Separate seeds for shared / bg / entity branches.
        self.expand_shared = nn.Conv2d(hidden, hidden * depth_bins, kernel_size=1, bias=True)
        self.expand_bg = nn.Conv2d(hidden, hidden * depth_bins, kernel_size=1, bias=True)
        self.expand_e0 = nn.Conv2d(hidden * 2, hidden * depth_bins, kernel_size=1, bias=True)
        self.expand_e1 = nn.Conv2d(hidden * 2, hidden * depth_bins, kernel_size=1, bias=True)

        # Deeper shared 3D trunk to strengthen topology continuity bias.
        self.shared_3d_in = nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, bias=True)
        self.shared_3d = nn.Sequential(
            Residual3DBlock(hidden),
            Residual3DBlock(hidden),
            Residual3DBlock(hidden),
            Residual3DBlock(hidden),
        )

        # Per-class branches. Each branch sees the shared trunk plus its own seed.
        self.bg_branch = _make_branch(hidden * 2, hidden, n_blocks=2)
        self.e0_branch = _make_branch(hidden * 2, hidden, n_blocks=2)
        self.e1_branch = _make_branch(hidden * 2, hidden, n_blocks=2)

        self.bg_head = nn.Conv3d(hidden, 1, kernel_size=1, bias=True)
        self.e0_head = nn.Conv3d(hidden, 1, kernel_size=1, bias=True)
        self.e1_head = nn.Conv3d(hidden, 1, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        # All heads zero-init: no entity starts favored or disfavored
        for head in (self.bg_head, self.e0_head, self.e1_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _to_2d(self, feat: torch.Tensor) -> torch.Tensor:
        B = feat.shape[0]
        H, W = self.spatial_h, self.spatial_w
        return feat.permute(0, 2, 1).reshape(B, self.hidden, H, W)

    def _expand_3d(self, feat_2d: torch.Tensor, proj: nn.Conv2d) -> torch.Tensor:
        B = feat_2d.shape[0]
        K = self.depth_bins
        h_3d = proj(feat_2d)
        return h_3d.reshape(B, self.hidden, K, self.spatial_h, self.spatial_w)

    def forward(
        self,
        F_g: torch.Tensor,
        F_0: torch.Tensor,
        F_1: torch.Tensor,
    ) -> torch.Tensor:
        B = F_g.shape[0]

        h_g = self._to_2d(self.proj_g(F_g.float()))
        h_e0 = self._to_2d(self.proj_e0(F_0.float()))
        h_e1 = self._to_2d(self.proj_e1(F_1.float()))

        h_cat = torch.cat([h_g, h_e0, h_e1], dim=1)
        h_shared_2d = self.shared_2d(h_cat)

        shared_seed = self._expand_3d(h_shared_2d, self.expand_shared)
        shared_3d = self.shared_3d(self.shared_3d_in(shared_seed))

        bg_seed = self._expand_3d(h_g, self.expand_bg)
        e0_seed = self._expand_3d(torch.cat([h_shared_2d, h_e0], dim=1), self.expand_e0)
        e1_seed = self._expand_3d(torch.cat([h_shared_2d, h_e1], dim=1), self.expand_e1)

        bg_feat = self.bg_branch(torch.cat([shared_3d, bg_seed], dim=1))
        e0_feat = self.e0_branch(torch.cat([shared_3d, e0_seed], dim=1))
        e1_feat = self.e1_branch(torch.cat([shared_3d, e1_seed], dim=1))

        bg_logit = self.bg_head(bg_feat)
        e0_logit = self.e0_head(e0_feat)
        e1_logit = self.e1_head(e1_feat)

        return torch.cat([bg_logit, e0_logit, e1_logit], dim=1)
