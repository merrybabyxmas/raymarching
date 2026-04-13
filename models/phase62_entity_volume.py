"""
Phase 62 — Entity Volume Predictor
====================================

Predicts a 3D entity volume V_logits (B, N+1, K, H, W) from UNet
cross-attention features, where N+1 = 3 classes (bg=0, entity0=1, entity1=2).

Architecture:
  1. Project each of F_g, F_0, F_1: (B, S, D) -> (B, hidden, H, W)
  2. Concat -> (B, hidden*3, H, W)
  3. 2D->3D expand via Conv2d -> reshape to (B, hidden, K, H, W)
  4. 3D conv refinement (depth-axis continuity)
  5. Classifier Conv3d -> (B, 3, K, H, W) logits

No transparency, no alpha. Pure class logits with voxel-wise softmax.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EntityVolumePredictor(nn.Module):
    """
    Predicts 3D entity volume from UNet cross-attention features.

    Input:
        F_g  (B, S, D) — global attention features
        F_0  (B, S, D) — entity-0 slot features
        F_1  (B, S, D) — entity-1 slot features
        where S = spatial_h * spatial_w

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

        # Feature projectors: one per stream (F_g, F_0, F_1)
        self.proj_g = nn.Linear(feat_dim, hidden)  # (B, S, D) -> (B, S, hidden)
        self.proj_e0 = nn.Linear(feat_dim, hidden)
        self.proj_e1 = nn.Linear(feat_dim, hidden)

        # 2D -> 3D expansion: (B, hidden*3, H, W) -> (B, hidden*K, H, W)
        # then reshape to (B, hidden, K, H, W)
        self.expand = nn.Conv2d(
            in_channels=hidden * 3,
            out_channels=hidden * depth_bins,
            kernel_size=1,
            bias=True,
        )

        # 3D conv refinement for depth-axis continuity
        self.refine_3d = nn.Sequential(
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )

        # Classifier: (B, hidden, K, H, W) -> (B, n_classes, K, H, W)
        self.classifier = nn.Conv3d(hidden, n_classes, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize classifier bias to favor background at start."""
        # Small positive bias for bg class (index 0) so initial predictions
        # default to background, which is the dominant class
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.classifier.bias.data[0] = 1.0  # bg logit starts higher

    def forward(
        self,
        F_g: torch.Tensor,   # (B, S, D)
        F_0: torch.Tensor,   # (B, S, D)
        F_1: torch.Tensor,   # (B, S, D)
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            V_logits: (B, 3, K, H, W) — per-voxel class logits
        """
        B = F_g.shape[0]
        H, W = self.spatial_h, self.spatial_w
        K = self.depth_bins

        # 1. Project each feature stream to hidden dim
        h_g = self.proj_g(F_g.float())    # (B, S, hidden)
        h_e0 = self.proj_e0(F_0.float())  # (B, S, hidden)
        h_e1 = self.proj_e1(F_1.float())  # (B, S, hidden)

        # 2. Reshape to spatial maps: (B, S, hidden) -> (B, hidden, H, W)
        h_g = h_g.permute(0, 2, 1).reshape(B, self.hidden, H, W)    # (B, hidden, H, W)
        h_e0 = h_e0.permute(0, 2, 1).reshape(B, self.hidden, H, W)  # (B, hidden, H, W)
        h_e1 = h_e1.permute(0, 2, 1).reshape(B, self.hidden, H, W)  # (B, hidden, H, W)

        # 3. Concat: (B, hidden*3, H, W)
        h_cat = torch.cat([h_g, h_e0, h_e1], dim=1)  # (B, hidden*3, H, W)

        # 4. Expand to 3D: Conv2d -> (B, hidden*K, H, W) -> reshape to (B, hidden, K, H, W)
        h_3d = self.expand(h_cat)  # (B, hidden*K, H, W)
        h_3d = h_3d.reshape(B, self.hidden, K, H, W)  # (B, hidden, K, H, W)

        # 5. 3D conv refinement
        h_3d = self.refine_3d(h_3d)  # (B, hidden, K, H, W)

        # 6. Classify: (B, n_classes, K, H, W)
        V_logits = self.classifier(h_3d)  # (B, 3, K, H, W)

        return V_logits
