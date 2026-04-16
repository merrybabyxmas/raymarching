"""scene_prior/renderer.py
================================
Transmittance-based differentiable renderer for Phase 64.

This is a standalone renderer that does NOT depend on any diffusion backbone.
It consumes two independent density fields and produces the canonical
SceneOutputs.

Compositing algorithm (blueprint §8.2)
---------------------------------------
Let k index depth bins (k=0 = nearest).

  occ_k   = 1 - (1 - d0_k)(1 - d1_k)         combined occupancy
  empty_k = 1 - occ_k                          = (1-d0_k)(1-d1_k)

  T_0   = 1
  T_k   = prod_{j<k} empty_j                   exclusive cumprod

  visible_ei = sum_k T_k * d_ei_k              in [0, 1]
  amodal_ei  = 1 - prod_k (1 - d_ei_k)         in [0, 1]
  depth_map  = sum_k T_k*(d0+d1)*depth_k       (normalised)
  sep_map    = visible_e0 - visible_e1
  hidden_ei  = relu(amodal_ei - visible_ei)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from scene_prior.scene_outputs import SceneOutputs, RendererOutputs


class EntityRenderer(nn.Module):
    """Differentiable transmittance renderer for two independent entity fields.

    Parameters
    ----------
    depth_bins : int
        Number of depth bins (K).  Must match the depth axis of the input
        density tensors.
    eps : float
        Small constant added to denominators to avoid division by zero.
    """

    def __init__(self, depth_bins: int = 8, eps: float = 1e-6) -> None:
        super().__init__()
        self.depth_bins = depth_bins
        self.eps = eps

    # ------------------------------------------------------------------

    def forward(
        self,
        density_e0: torch.Tensor,   # (B, K, H, W) values in [0, 1]
        density_e1: torch.Tensor,   # (B, K, H, W)
    ) -> SceneOutputs:
        """Render two entity density fields to SceneOutputs.

        Parameters
        ----------
        density_e0 : (B, K, H, W)
        density_e1 : (B, K, H, W)

        Returns
        -------
        SceneOutputs  (all fields are (B, H, W))
        """
        assert density_e0.shape == density_e1.shape, (
            f"density shape mismatch: {density_e0.shape} vs {density_e1.shape}"
        )
        assert density_e0.dim() == 4, (
            f"expected (B, K, H, W), got {density_e0.shape}"
        )

        # Work in float32 for numerically stable cumprod
        d0 = density_e0.float().clamp(0.0, 1.0)
        d1 = density_e1.float().clamp(0.0, 1.0)
        B, K, H, W = d0.shape

        # ------------------------------------------------------------------
        # 1. Combined occupancy and transmittance
        # ------------------------------------------------------------------
        empty = (1.0 - d0) * (1.0 - d1)     # (B, K, H, W)
        # occ = 1.0 - empty                  # not stored, used implicitly

        # Exclusive cumprod: T_k = prod_{j<k} empty_j
        #   inclusive[:, k] = prod_{j<=k} empty_j
        inclusive = torch.cumprod(empty, dim=1)              # (B, K, H, W)
        ones = torch.ones(B, 1, H, W, device=d0.device, dtype=d0.dtype)
        T = torch.cat([ones, inclusive[:, :-1]], dim=1)      # (B, K, H, W)

        # ------------------------------------------------------------------
        # 2. Visible (first-hit)
        # ------------------------------------------------------------------
        visible_e0 = (T * d0).sum(dim=1)    # (B, H, W)
        visible_e1 = (T * d1).sum(dim=1)    # (B, H, W)

        # ------------------------------------------------------------------
        # 3. Amodal (total presence)
        # ------------------------------------------------------------------
        amodal_e0 = 1.0 - (1.0 - d0).prod(dim=1)   # (B, H, W)
        amodal_e1 = 1.0 - (1.0 - d1).prod(dim=1)   # (B, H, W)

        # ------------------------------------------------------------------
        # 4. Depth map
        # ------------------------------------------------------------------
        if K > 1:
            k_norm = torch.linspace(
                0.0, 1.0, K, device=d0.device, dtype=d0.dtype,
            ).view(1, K, 1, 1)
        else:
            k_norm = torch.zeros(1, 1, 1, 1, device=d0.device, dtype=d0.dtype)

        # weighted depth by combined foreground contribution
        fg_weight = T * (d0 + d1)           # (B, K, H, W)
        depth_num = (fg_weight * k_norm).sum(dim=1)          # (B, H, W)
        depth_den = fg_weight.sum(dim=1) + self.eps
        depth_map = depth_num / depth_den   # (B, H, W)

        # ------------------------------------------------------------------
        # 5. Derived channels
        # ------------------------------------------------------------------
        sep_map  = visible_e0 - visible_e1                          # (B, H, W)
        hidden_e0 = torch.relu(amodal_e0 - visible_e0)             # (B, H, W)
        hidden_e1 = torch.relu(amodal_e1 - visible_e1)             # (B, H, W)

        return SceneOutputs(
            visible_e0=visible_e0,
            visible_e1=visible_e1,
            amodal_e0=amodal_e0,
            amodal_e1=amodal_e1,
            depth_map=depth_map,
            sep_map=sep_map,
            hidden_e0=hidden_e0,
            hidden_e1=hidden_e1,
        )
