"""
Phase 63 — Differentiable Transmittance Renderer
================================================

Replaces Phase 62's ``FirstHitProjector``.  Given two *independent* entity
density fields ``sigma_0, sigma_1 : (B, K, H, W)`` — each produced by
``EntityField`` — this module renders 2D visible / amodal masks and an
expected depth map in a fully differentiable manner.

Notation
--------
Let k index depth bins (k=0 is nearest the camera).  Combined occupancy of a
voxel is::

    occ_k = 1 - (1 - sigma_0_k) * (1 - sigma_1_k)

Transmittance (probability that a ray has NOT yet hit anything before bin k)::

    T_k = prod_{j < k} (1 - occ_j)             exclusive cumprod
    T_0 = 1

Visible (first-hit) contribution of entity i at a pixel::

    V_i = sum_k T_k * sigma_i_k                in [0, 1]

Amodal (total presence) of entity i at a pixel, i.e. probability that the
pixel is covered by entity i at ANY depth::

    A_i = 1 - prod_k (1 - sigma_i_k)           in [0, 1]

Expected depth (only among foreground voxels)::

    D   = sum_k T_k * occ_k * (k / (K - 1))    in [0, 1]

All operations are implemented with ``torch.cumprod`` and element-wise ops,
so they are fully differentiable.

Backwards-compatibility
-----------------------
``RendererOutputs`` also exposes the Phase 62-shaped ``front_probs``,
``back_probs``, ``visible``, ``amodal`` and ``visible_class`` fields so that
existing losses / evaluators can consume the new renderer without change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class RendererOutputs:
    visible_e0: torch.Tensor     # (B, H, W) soft first-hit contribution
    visible_e1: torch.Tensor     # (B, H, W)
    amodal_e0: torch.Tensor      # (B, H, W)
    amodal_e1: torch.Tensor      # (B, H, W)
    depth: torch.Tensor          # (B, H, W) expected depth in [0, 1]
    transmittance: torch.Tensor  # (B, K, H, W) T_k

    # Phase 62 compatibility --------------------------------------------------
    front_probs: torch.Tensor    # (B, 3, H, W)  [bg, e0, e1]
    back_probs: torch.Tensor     # (B, 3, H, W)
    visible: Dict[str, torch.Tensor] = field(default_factory=dict)    # {"e0", "e1"}
    amodal: Dict[str, torch.Tensor] = field(default_factory=dict)
    visible_class: torch.Tensor = field(default=None)                 # (B, H, W) int64


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
class TransmittanceRenderer(nn.Module):
    """Differentiable volume renderer for two independent entity fields."""

    def __init__(self, bg_class: int = 0):
        super().__init__()
        self.bg_class = bg_class

    def forward(
        self,
        density_e0: torch.Tensor,   # (B, K, H, W) in [0, 1]
        density_e1: torch.Tensor,   # (B, K, H, W) in [0, 1]
    ) -> RendererOutputs:
        assert density_e0.shape == density_e1.shape, \
            f"shape mismatch: {density_e0.shape} vs {density_e1.shape}"
        assert density_e0.dim() == 4, f"expected (B,K,H,W), got {density_e0.shape}"

        # Cast to float32 for numerically stable cumprod; the caller handles
        # amp-aware downcasting at the loss boundary.
        s0 = density_e0.float().clamp(0.0, 1.0)
        s1 = density_e1.float().clamp(0.0, 1.0)
        B, K, H, W = s0.shape

        # --- Combined occupancy and transmittance ---------------------------
        # occ_k = 1 - (1 - s0_k)(1 - s1_k)
        empty = (1.0 - s0) * (1.0 - s1)                  # (B, K, H, W)  P(voxel empty)
        occ = 1.0 - empty                                # (B, K, H, W)

        # Exclusive cumprod of (1 - occ) along K.  We want
        #   T_k = prod_{j<k} empty_j
        # so we pad a "1" at the front and drop the last slice.
        ones = torch.ones(B, 1, H, W, device=s0.device, dtype=s0.dtype)
        # inclusive_cumprod[:, k] = prod_{j <= k} empty_j
        inclusive = torch.cumprod(empty, dim=1)                   # (B, K, H, W)
        # exclusive cumprod: shift right, pad with 1 at k=0
        T = torch.cat([ones, inclusive[:, :-1]], dim=1)           # (B, K, H, W)

        # --- Visible (first-hit) --------------------------------------------
        visible_e0 = (T * s0).sum(dim=1)                          # (B, H, W)
        visible_e1 = (T * s1).sum(dim=1)                          # (B, H, W)

        # --- Amodal ----------------------------------------------------------
        # A_i = 1 - prod_k (1 - s_i_k).  Using cumprod's final slice would also
        # work but .prod is fine and clearer.
        amodal_e0 = 1.0 - (1.0 - s0).prod(dim=1)                  # (B, H, W)
        amodal_e1 = 1.0 - (1.0 - s1).prod(dim=1)                  # (B, H, W)

        # --- Depth -----------------------------------------------------------
        # Normalised bin index in [0, 1].
        if K > 1:
            k_norm = torch.linspace(
                0.0, 1.0, K, device=s0.device, dtype=s0.dtype,
            ).view(1, K, 1, 1)
        else:
            k_norm = torch.zeros(1, 1, 1, 1, device=s0.device, dtype=s0.dtype)
        depth = (T * occ * k_norm).sum(dim=1)                     # (B, H, W)

        # --- Phase 62 compatible front_probs --------------------------------
        # Total background prob = transmittance through the whole volume.
        bg_prob = inclusive[:, -1]                                # (B, H, W)  = prod_k empty_k
        front_probs = torch.stack(
            [bg_prob, visible_e0, visible_e1], dim=1,
        )                                                          # (B, 3, H, W)
        # Normalise for safety.
        front_probs = front_probs / front_probs.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # --- Phase 62 compatible back_probs ---------------------------------
        # "Has something occluded this pixel before bin k?"  Equivalent to
        # (1 - T_k).  Accumulated entity contribution behind the first hit.
        has_front_before = 1.0 - T                                # (B, K, H, W)
        back_e0 = (has_front_before * s0).sum(dim=1)              # (B, H, W)
        back_e1 = (has_front_before * s1).sum(dim=1)
        back_bg = torch.zeros_like(bg_prob)                       # no bg "behind" itself
        back_probs = torch.stack([back_bg, back_e0, back_e1], dim=1)
        back_sum = back_probs.sum(dim=1, keepdim=True)
        back_probs = torch.where(
            back_sum > 1e-6, back_probs / back_sum.clamp(min=1e-6), back_probs,
        )

        # --- Phase 62 compatible visible_class ------------------------------
        # Hard classification (no gradients — kept for metrics only).
        with torch.no_grad():
            # Class prob stack: [bg, e0, e1], pick argmax.
            visible_class = front_probs.argmax(dim=1).long()      # (B, H, W)

        return RendererOutputs(
            visible_e0=visible_e0,
            visible_e1=visible_e1,
            amodal_e0=amodal_e0,
            amodal_e1=amodal_e1,
            depth=depth,
            transmittance=T,
            front_probs=front_probs,
            back_probs=back_probs,
            visible={"e0": visible_e0, "e1": visible_e1},
            amodal={"e0": amodal_e0, "e1": amodal_e1},
            visible_class=visible_class,
        )
