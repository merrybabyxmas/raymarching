"""
Phase 63 — Independent Entity Field
====================================

Replaces Phase 62's ``EntityVolumePredictor``.

Key design difference versus Phase 62:

Phase 62 used a *class-competition* formulation for the entity volume.  The
foreground / identity logits were normalised with a softmax over the class
dimension, which meant that the probability mass of entity-0 and entity-1 had
to *compete* for the same pixels.  In practice this produced "single-entity
collapse": one entity grabbed most of the mass and the other was pushed
towards the background class.

Phase 63 instead uses a *fully independent* density per entity:

- two independent decoders (no shared final MLP / softmax)
- sigmoid on the final density logit (no softmax coupling entity-0 and
  entity-1)
- no background class — occupancy of empty space is simply
  ``(1-sigma_0) * (1-sigma_1)`` and is computed by the renderer, never by
  this module.

Each decoder consumes a concatenation of the global backbone feature F_g and
its own entity feature F_ei.  The feature map is expanded to the depth-bin
axis with a 1x1 conv (2D→3D in einops terms) and then refined with a stack of
``Residual3DBlock`` modules before the density / appearance heads.

Outputs mirror the dataclass declared in this file so that downstream modules
(renderer, guide encoder, losses) get named tensors, but we also populate a
backwards-compatible ``entity_probs`` tensor shaped ``(B, 2, K, H, W)`` for
any legacy Phase 62 loss that still consumes that shape.

Tensor shapes
-------------
Inputs:
    F_g  : (B, S, feat_dim)       S = spatial_h * spatial_w
    F_e0 : (B, S, feat_dim)
    F_e1 : (B, S, feat_dim)
    depth_hint : (B, H, W) in [0, 1]  (optional)

Outputs (``EntityFieldOutputs``):
    density_e0    : (B, K, H, W)
    density_e1    : (B, K, H, W)
    appearance_e0 : (B, app_dim, H, W)
    appearance_e1 : (B, app_dim, H, W)
    entity_probs  : (B, 2, K, H, W)   # stacked densities, bw-compat
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class EntityFieldOutputs:
    """Structured output of ``EntityField``.

    density_{e0,e1} are independent sigmoids — they do NOT sum to 1 across
    entity dim.  The renderer decides what to do with them.
    """
    density_e0: torch.Tensor      # (B, K, H, W), float32, values in [0, 1]
    density_e1: torch.Tensor      # (B, K, H, W)
    appearance_e0: torch.Tensor   # (B, app_dim, K, H, W) after collapse → (B, app_dim, H, W)
    appearance_e1: torch.Tensor   # (B, app_dim, K, H, W) collapsed to 2D
    entity_probs: torch.Tensor    # (B, 2, K, H, W) — stack(density_e0, density_e1)


# ---------------------------------------------------------------------------
# Basic blocks
# ---------------------------------------------------------------------------
class Residual3DBlock(nn.Module):
    """Two 3x3x3 convs + GELU + residual.  Same as Phase 62's block."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=True),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, K, H, W) -> (B, C, K, H, W)
        return self.act(x + self.net(x))


class EntityFieldDecoder(nn.Module):
    """
    Per-entity decoder:  (B, hidden*2, H, W)  ->  density (B, K, H, W)
                                              +  appearance (B, app_dim, H, W)

    There is one instance per entity — the two decoders do NOT share weights,
    which removes any architectural bias towards a single dominant entity.
    """

    def __init__(
        self,
        hidden: int,
        depth_bins: int,
        app_dim: int,
        n_refine_blocks: int = 3,
    ):
        super().__init__()
        self.hidden = hidden
        self.depth_bins = depth_bins
        self.app_dim = app_dim

        # 2D -> 3D expansion.  Conv2d(hidden*2 -> hidden*K) then reshape.
        self.expand_3d = nn.Conv2d(
            in_channels=hidden * 2,
            out_channels=hidden * depth_bins,
            kernel_size=1,
            bias=True,
        )

        # Sinusoidal depth positional embedding so that each K-bin starts with
        # a distinct feature vector — same trick as Phase 62, just scoped per
        # decoder.  Shape: (1, hidden, K, 1, 1).
        self.register_buffer(
            "depth_pos_emb",
            self._make_sinusoidal_depth_emb(depth_bins, hidden),
            persistent=False,
        )

        # 3D refinement.
        self.refine_3d = nn.Sequential(
            *[Residual3DBlock(hidden) for _ in range(n_refine_blocks)]
        )

        # Density head.  Sigmoid applied in forward.
        self.density_head = nn.Conv3d(hidden, 1, kernel_size=1, bias=True)
        # Appearance head.
        self.appearance_head = nn.Conv3d(hidden, app_dim, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        # Zero-init density head so that at step 0 density is sigmoid(0)=0.5
        # everywhere — neutral starting point, no dead entity.
        nn.init.zeros_(self.density_head.weight)
        nn.init.zeros_(self.density_head.bias)
        # Appearance head small random — needs SOMETHING to break symmetry.
        nn.init.normal_(self.appearance_head.weight, std=0.01)
        nn.init.zeros_(self.appearance_head.bias)

    @staticmethod
    def _make_sinusoidal_depth_emb(depth_bins: int, hidden: int) -> torch.Tensor:
        """Transformer-style sinusoidal PE along the depth axis.

        Returns shape (1, hidden, K, 1, 1), scaled to ±0.3.
        """
        K = depth_bins
        H = hidden
        pe = torch.zeros(H, K)
        position = torch.arange(K, dtype=torch.float).unsqueeze(1)                  # (K, 1)
        div_term = torch.exp(torch.arange(0, H, 2, dtype=torch.float) *
                             -(math.log(10.0) / H))                                  # (H/2,)
        pe[0::2, :] = torch.sin(position * div_term).T                               # (H/2, K)
        pe[1::2, :] = torch.cos(position * div_term).T
        pe = pe * 0.3
        return pe.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)                           # (1,H,K,1,1)

    def forward(
        self,
        feat_2d: torch.Tensor,        # (B, hidden*2, H, W)
        depth_feat: Optional[torch.Tensor] = None,  # (B, hidden, K, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            density    (B, K, H, W)      — sigmoid, independent
            appearance (B, app_dim, H, W) — collapsed across K via weighted sum
        """
        B, _, H, W = feat_2d.shape
        K = self.depth_bins

        # 2D -> 3D: expand then reshape with einops for clarity.
        h3d = self.expand_3d(feat_2d)                                  # (B, hidden*K, H, W)
        h3d = rearrange(
            h3d, "b (c k) h w -> b c k h w", c=self.hidden, k=K,
        )                                                              # (B, hidden, K, H, W)

        # Add depth positional embedding.
        h3d = h3d + self.depth_pos_emb.to(h3d.dtype)

        # Optional depth hint feature (from the outer module).
        if depth_feat is not None:
            h3d = h3d + depth_feat.to(h3d.dtype)

        # Refinement.
        h3d = self.refine_3d(h3d)                                      # (B, hidden, K, H, W)

        # Density head — NO softmax across entity dim, NO softmax across K.
        # Sigmoid gives each voxel an independent "is this entity here?" prob.
        density_logit = self.density_head(h3d.float())                 # (B, 1, K, H, W)
        density = torch.sigmoid(density_logit).squeeze(1)              # (B, K, H, W)

        # Appearance in voxel form.  Collapse to 2D by weighting by density
        # (so that appearance is concentrated wherever the entity actually
        # exists), then L1-normalise across K to get a spatial feature.
        app_3d = self.appearance_head(h3d.float())                     # (B, app, K, H, W)

        # Weighted pool over K: sum_k (density_k * appearance_k) / sum_k density_k
        density_kexpand = rearrange(density, "b k h w -> b 1 k h w")   # (B, 1, K, H, W)
        weight_sum = reduce(density_kexpand, "b 1 k h w -> b 1 h w", "sum").clamp(min=1e-6)
        appearance = reduce(app_3d * density_kexpand, "b c k h w -> b c h w", "sum") / weight_sum

        return density, appearance


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class EntityField(nn.Module):
    """
    Predict two *independent* entity density fields from backbone features.

    Parameters
    ----------
    feat_dim : int
        Dimensionality of the per-token backbone features (F_g, F_e0, F_e1).
    hidden : int
        Internal channel width shared by all projections / decoders.
    depth_bins : int
        Number of depth bins K in the volume.
    spatial_h, spatial_w : int
        Spatial resolution of the token grid, i.e. S = spatial_h * spatial_w.
    app_dim : int
        Number of appearance-feature channels per entity.
    n_refine_blocks : int
        Number of ``Residual3DBlock`` modules in each decoder.
    """

    def __init__(
        self,
        feat_dim: int = 640,
        hidden: int = 64,
        depth_bins: int = 8,
        spatial_h: int = 16,
        spatial_w: int = 16,
        app_dim: int = 32,
        n_refine_blocks: int = 3,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.depth_bins = depth_bins
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.app_dim = app_dim

        # Token-level projections to hidden.
        self.proj_g = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU())
        self.proj_e0 = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU())
        self.proj_e1 = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU())

        # Per-entity identity bias — breaks symmetry between e0 and e1.
        self.entity_id_e0 = nn.Parameter(torch.randn(1, hidden, 1, 1) * 0.1)
        self.entity_id_e1 = nn.Parameter(torch.randn(1, hidden, 1, 1) * 0.1)

        # Two INDEPENDENT decoders.  Important: no weight sharing.
        self.decoder_e0 = EntityFieldDecoder(
            hidden=hidden,
            depth_bins=depth_bins,
            app_dim=app_dim,
            n_refine_blocks=n_refine_blocks,
        )
        self.decoder_e1 = EntityFieldDecoder(
            hidden=hidden,
            depth_bins=depth_bins,
            app_dim=app_dim,
            n_refine_blocks=n_refine_blocks,
        )

        # Depth-hint encoder — reused by both entities (depth is a property of
        # the scene, not of a specific entity).  Block-diagonal init so each
        # output group k responds to input bin k.
        self.depth_encoder = nn.Conv2d(
            depth_bins, hidden * depth_bins, kernel_size=1, bias=True,
        )
        self._init_depth_encoder()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------
    def _init_depth_encoder(self) -> None:
        """Block-diagonal init — output channels (k*hidden, (k+1)*hidden)
        react to input channel k with weight 0.3, all other entries zero.
        Bias is zero so the initial depth_feat is purely proportional to the
        depth_hint, with no global offset.
        """
        K = self.depth_bins
        h = self.hidden
        nn.init.zeros_(self.depth_encoder.weight)
        nn.init.zeros_(self.depth_encoder.bias)
        with torch.no_grad():
            for k in range(K):
                self.depth_encoder.weight[k * h:(k + 1) * h, k, 0, 0] = 0.3

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _tokens_to_2d(self, x: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        """(B, S, feat_dim) -> (B, hidden, H, W) using einops."""
        B = x.shape[0]
        h = proj(x.float())                                            # (B, S, hidden)
        return rearrange(
            h, "b (h w) c -> b c h w", h=self.spatial_h, w=self.spatial_w,
        )

    def _depth_hint_feat(self, depth_hint: torch.Tensor) -> torch.Tensor:
        """
        Convert scene depth ∈ [0,1] at (B, H, W) into a (B, hidden, K, H, W)
        feature via a soft Gaussian assignment to depth bins followed by the
        block-diagonal depth_encoder.
        """
        B, H, W = depth_hint.shape
        K = self.depth_bins

        depth_bin_cont = depth_hint.unsqueeze(1) * (K - 1)             # (B, 1, H, W)
        bin_idx = torch.arange(
            K, device=depth_hint.device, dtype=depth_hint.dtype,
        ).view(1, K, 1, 1)
        sigma = 1.0
        raw = torch.exp(-(depth_bin_cont - bin_idx) ** 2 / (2.0 * sigma ** 2))
        dw = raw / raw.sum(dim=1, keepdim=True).clamp(min=1e-6)        # (B, K, H, W)

        feat = self.depth_encoder(dw)                                  # (B, hidden*K, H, W)
        return rearrange(feat, "b (c k) h w -> b c k h w",
                         c=self.hidden, k=K)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        F_g: torch.Tensor,                                      # (B, S, feat_dim)
        F_e0: torch.Tensor,                                     # (B, S, feat_dim)
        F_e1: torch.Tensor,                                     # (B, S, feat_dim)
        depth_hint: Optional[torch.Tensor] = None,              # (B, H, W) in [0,1]
    ) -> EntityFieldOutputs:
        assert F_g.dim() == 3, f"F_g must be (B, S, D), got {F_g.shape}"
        assert F_g.shape[:2] == F_e0.shape[:2] == F_e1.shape[:2], \
            "F_g / F_e0 / F_e1 must share (B, S)"

        # Project tokens to 2D feature maps.
        h_g = self._tokens_to_2d(F_g, self.proj_g)                     # (B, hidden, H, W)
        h_e0 = self._tokens_to_2d(F_e0, self.proj_e0) + self.entity_id_e0
        h_e1 = self._tokens_to_2d(F_e1, self.proj_e1) + self.entity_id_e1

        # Concat per entity with global — each decoder sees the full scene
        # context and its own entity signal.
        feat_e0 = torch.cat([h_g, h_e0], dim=1)                        # (B, hidden*2, H, W)
        feat_e1 = torch.cat([h_g, h_e1], dim=1)

        # Optional depth prior (shared across entities).
        depth_feat = None
        if depth_hint is not None:
            # Interpolate depth_hint to token grid if needed.
            if depth_hint.shape[-2:] != (self.spatial_h, self.spatial_w):
                depth_hint = F.interpolate(
                    depth_hint.unsqueeze(1).float(),
                    size=(self.spatial_h, self.spatial_w),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)
            depth_feat = self._depth_hint_feat(depth_hint)              # (B, hidden, K, H, W)

        # Independent decoding — no competition between entities.
        density_e0, appearance_e0 = self.decoder_e0(feat_e0, depth_feat)
        density_e1, appearance_e1 = self.decoder_e1(feat_e1, depth_feat)

        # Backwards-compat stacked tensor for Phase 62 losses that still expect
        # a (B, 2, K, H, W) entity_probs.
        entity_probs = torch.stack([density_e0, density_e1], dim=1)    # (B, 2, K, H, W)

        return EntityFieldOutputs(
            density_e0=density_e0.float(),
            density_e1=density_e1.float(),
            appearance_e0=appearance_e0.float(),
            appearance_e1=appearance_e1.float(),
            entity_probs=entity_probs.float(),
        )
