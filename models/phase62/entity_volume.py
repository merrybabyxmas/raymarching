"""
Phase 62 — Entity Volume Predictor
====================================

Supports multiple volume representations:

1. 'independent': bg/e0/e1 as 3 independent heads (original)
2. 'factorized_fg_id': fg_head + id_head (factorized foreground + identity)
3. 'center_offset': fg_head + id_head + offset heads for center regression
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from training.phase62.objectives.base import VolumeOutputs


class Residual3DBlock(nn.Module):

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

    Representation modes:
      - 'independent': 3 class heads (bg, e0, e1) with independent sigmoid
      - 'factorized_fg_id': fg_head (1ch) + id_head (2ch) — factorized
      - 'center_offset': fg + id + per-entity 3D offset heads
    """

    def __init__(
        self,
        feat_dim: int = 640,
        n_classes: int = 3,
        depth_bins: int = 8,
        spatial_h: int = 16,
        spatial_w: int = 16,
        hidden: int = 64,
        representation: str = "independent",
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_classes = n_classes
        self.depth_bins = depth_bins
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.hidden = hidden
        self.representation = representation

        self.proj_g = nn.Linear(feat_dim, hidden)
        self.proj_e0 = nn.Linear(feat_dim, hidden)
        self.proj_e1 = nn.Linear(feat_dim, hidden)

        # Symmetric entity init (v40r patch):
        # Independent proj_e0/e1 randn init causes seed-dependent magnitude asymmetry:
        #   seed=42: |proj_e0(F)| ≈ |proj_e1(F)| → entity competition balanced
        #   seed=7:  |proj_e0(F)| >> |proj_e1(F)| → entity 0 dominates from ep0
        # Fix: tie proj_e1 weights to proj_e0 at init → same extraction strength.
        # Anti-symmetric entity_id: entity_id_e1 = -entity_id_e0 → both start with
        #   equal magnitude, differentiated only by sign → no init-level dominance.
        # Nets can diverge freely after init via backprop.
        with torch.no_grad():
            self.proj_e1.weight.copy_(self.proj_e0.weight)
            self.proj_e1.bias.copy_(self.proj_e0.bias)
        self.entity_id_e0 = nn.Parameter(torch.randn(1, hidden, 1, 1) * 0.1)
        self.entity_id_e1 = nn.Parameter(-self.entity_id_e0.data.clone())

        self.shared_2d = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden * 2, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden * 2, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )

        self.expand_shared = nn.Conv2d(hidden, hidden * depth_bins, kernel_size=1, bias=True)
        self.expand_bg = nn.Conv2d(hidden, hidden * depth_bins, kernel_size=1, bias=True)
        self.expand_e0 = nn.Conv2d(hidden * 2, hidden * depth_bins, kernel_size=1, bias=True)
        self.expand_e1 = nn.Conv2d(hidden * 2, hidden * depth_bins, kernel_size=1, bias=True)

        self.shared_3d_in = nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, bias=True)
        self.shared_3d = nn.Sequential(
            Residual3DBlock(hidden),
            Residual3DBlock(hidden),
            Residual3DBlock(hidden),
            Residual3DBlock(hidden),
        )

        # Depth hint injection: optional per-pixel scene depth → depth-bin features.
        # Input: (B, K, H, W) soft depth bin weights derived from scene depth map.
        # Output: (B, hidden, K, H, W) added to shared_seed to concentrate entity_probs.
        #
        # Block-diagonal init: output group k is active when input channel k is high.
        # This gives an IMMEDIATE useful depth signal from epoch 0 (unlike zero-init).
        self.depth_encoder = nn.Conv2d(depth_bins, hidden * depth_bins, kernel_size=1, bias=True)
        self._init_depth_encoder()

        # Learnable depth positional embedding (K, hidden) — scene-invariant depth bias.
        # Allows each depth bin to develop a unique feature representation, helping
        # the shared_3d blocks converge quickly to depth-specific predictions.
        # Sinusoidal init: each K bin gets a DISTINCT feature fingerprint immediately,
        # unlike zero-init which provides no depth signal until after many gradient steps.
        # sin/cos waves of different frequencies ensure orthogonal representations.
        self.depth_pos_emb = nn.Parameter(
            self._make_sinusoidal_depth_emb(depth_bins, hidden))

        if representation == "independent":
            self.bg_branch = _make_branch(hidden * 2, hidden, n_blocks=2)
            self.e0_branch = _make_branch(hidden * 2, hidden, n_blocks=2)
            self.e1_branch = _make_branch(hidden * 2, hidden, n_blocks=2)
            self.bg_head = nn.Conv3d(hidden, 1, kernel_size=1, bias=True)
            self.e0_head = nn.Conv3d(hidden, 1, kernel_size=1, bias=True)
            self.e1_head = nn.Conv3d(hidden, 1, kernel_size=1, bias=True)
        elif representation in ("factorized_fg_id", "center_offset"):
            self.fg_branch = _make_branch(hidden * 2, hidden, n_blocks=2)
            self.id_branch = _make_branch(hidden * 2, hidden, n_blocks=2)
            self.fg_head = nn.Conv3d(hidden, 1, kernel_size=1, bias=True)
            self.id_head = nn.Conv3d(hidden, 2, kernel_size=1, bias=True)
            # Separate 2D spatial fg head: "Is fg present at this (h,w) pixel?"
            # Decoupled from fg_head (which now controls depth_attn only).
            # fg_magnitude = sigmoid(fg_spatial_head(h_shared_2d)) — no K coupling.
            # L_fg_spatial trains this to be 1 at fg pixels, 0 at bg.
            self.fg_spatial_head = nn.Conv2d(hidden, 1, kernel_size=1, bias=True)
            # id_branch needs BOTH entity features to classify e0 vs e1
            self.expand_id = nn.Conv2d(hidden * 3, hidden * depth_bins, kernel_size=1, bias=True)

            if representation == "center_offset":
                self.offset_e0_branch = _make_branch(hidden * 2, hidden, n_blocks=2)
                self.offset_e1_branch = _make_branch(hidden * 2, hidden, n_blocks=2)
                self.offset_e0_head = nn.Conv3d(hidden, 3, kernel_size=1, bias=True)
                self.offset_e1_head = nn.Conv3d(hidden, 3, kernel_size=1, bias=True)
        else:
            raise ValueError(f"Unknown representation: {representation}")

        self._init_weights()

    def _init_weights(self) -> None:
        if self.representation == "independent":
            for head in (self.bg_head, self.e0_head, self.e1_head):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        elif self.representation in ("factorized_fg_id", "center_offset"):
            nn.init.zeros_(self.fg_head.weight)
            nn.init.zeros_(self.fg_head.bias)
            nn.init.zeros_(self.id_head.weight)
            nn.init.zeros_(self.id_head.bias)
            nn.init.zeros_(self.fg_spatial_head.weight)
            nn.init.zeros_(self.fg_spatial_head.bias)
            if self.representation == "center_offset":
                for head in (self.offset_e0_head, self.offset_e1_head):
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)

    @staticmethod
    def _make_sinusoidal_depth_emb(depth_bins: int, hidden: int) -> torch.Tensor:
        """
        Sinusoidal positional encoding for depth bins → (1, hidden, K, 1, 1).

        Each K bin gets a UNIQUE feature vector of sin/cos waves at different
        frequencies, identical to transformer positional encodings. This provides
        orthogonal depth representations from epoch 0, unlike zero-init which
        provides no differentiation until gradients push the embedding.

        Scaled to ±0.3 so it's useful but doesn't overpower learned features.
        """
        K = depth_bins
        H = hidden
        pe = torch.zeros(H, K)
        position = torch.arange(K, dtype=torch.float).unsqueeze(1)  # (K, 1)
        div_term = torch.exp(torch.arange(0, H, 2, dtype=torch.float) *
                             -(math.log(10.0) / H))  # (H/2,)
        pe[0::2, :] = torch.sin(position * div_term).T   # (H/2, K)
        pe[1::2, :] = torch.cos(position * div_term).T   # (H/2, K)
        # Scale to ±0.3
        pe = pe * 0.3
        return pe.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, H, K, 1, 1)

    def _init_depth_encoder(self) -> None:
        """
        Block-diagonal initialisation for depth_encoder.
        Output group k (channels k*hidden : (k+1)*hidden) gets weight 0.3
        on input channel k, and 0 elsewhere.  This immediately produces a
        depth-bin-specific feature bias from the soft depth weights, giving
        useful signal from epoch 0 rather than waiting for the encoder to learn.
        Bias is zero-initialised (no global bias — pure depth-conditional signal).
        """
        K = self.depth_bins
        h = self.hidden
        nn.init.zeros_(self.depth_encoder.weight)
        nn.init.zeros_(self.depth_encoder.bias)
        with torch.no_grad():
            for k in range(K):
                # output channels [k*h : (k+1)*h] should activate for input ch k
                self.depth_encoder.weight[k * h:(k + 1) * h, k, 0, 0] = 0.3

    def _to_2d(self, feat: torch.Tensor) -> torch.Tensor:
        B = feat.shape[0]
        return feat.permute(0, 2, 1).reshape(B, self.hidden, self.spatial_h, self.spatial_w)

    def _expand_3d(self, feat_2d: torch.Tensor, proj: nn.Conv2d) -> torch.Tensor:
        B = feat_2d.shape[0]
        h_3d = proj(feat_2d)
        return h_3d.reshape(B, self.hidden, self.depth_bins, self.spatial_h, self.spatial_w)

    def _depth_hint_feat(
        self,
        depth_hint: torch.Tensor,  # (B, H, W) normalized depth in [0,1]
    ) -> torch.Tensor:
        """
        Convert per-pixel scene depth to a (B, hidden, K, H, W) feature bias.

        Uses a soft Gaussian assignment of depth values to depth bins:
          depth_weights[b, k, h, w] = softmax_k(exp(-(d - bin_center_k)^2 / sigma^2))

        The resulting per-spatial-pixel depth distribution is encoded by
        depth_encoder into a (B, hidden*K, H, W) feature, reshaped to 3D.
        With zero-init, initial output is 0 → safe initialisation.
        """
        B, H, W = depth_hint.shape
        K = self.depth_bins

        # Continuous depth in bin units: 0 → bin 0 centre, K-1 → bin K-1 centre
        depth_bin_cont = depth_hint.unsqueeze(1) * (K - 1)   # (B, 1, H, W)
        bin_idx = torch.arange(K, device=depth_hint.device,
                               dtype=depth_hint.dtype).view(1, K, 1, 1)  # (1,K,1,1)
        sigma = 1.0   # ≈ 1 bin width smoothness
        raw = torch.exp(-(depth_bin_cont - bin_idx) ** 2 / (2.0 * sigma ** 2))  # (B,K,H,W)
        dw = raw / raw.sum(dim=1, keepdim=True).clamp(min=1e-6)                 # normalised

        feat = self.depth_encoder(dw)               # (B, hidden*K, H, W)
        return feat.reshape(B, self.hidden, K, H, W)

    def forward(
        self,
        F_g: torch.Tensor,
        F_0: torch.Tensor,
        F_1: torch.Tensor,
        depth_hint: Optional[torch.Tensor] = None,  # (B, H_vol, W_vol) scene depth [0,1]
    ) -> VolumeOutputs:
        B = F_g.shape[0]

        h_g = self._to_2d(self.proj_g(F_g.float()))
        h_e0 = self._to_2d(self.proj_e0(F_0.float())) + self.entity_id_e0
        h_e1 = self._to_2d(self.proj_e1(F_1.float())) + self.entity_id_e1

        h_cat = torch.cat([h_g, h_e0, h_e1], dim=1)
        h_shared_2d = self.shared_2d(h_cat)

        shared_seed = self._expand_3d(h_shared_2d, self.expand_shared)

        # Depth positional embedding: scene-invariant per-bin bias.
        # Allows shared_3d to quickly develop depth-specific representations.
        shared_seed = shared_seed + self.depth_pos_emb

        # Inject scene depth hint (if available) as a depth-bin feature bias.
        # depth_hint provides per-pixel depth → helps concentrate entity_probs
        # at the correct depth bins instead of spreading uniformly.
        if depth_hint is not None:
            depth_feat = self._depth_hint_feat(depth_hint)  # (B, hidden, K, H_vol, W_vol)
            shared_seed = shared_seed + depth_feat

        shared_3d = self.shared_3d(self.shared_3d_in(shared_seed))

        bg_seed = self._expand_3d(h_g, self.expand_bg)
        e0_seed = self._expand_3d(torch.cat([h_shared_2d, h_e0], dim=1), self.expand_e0)
        e1_seed = self._expand_3d(torch.cat([h_shared_2d, h_e1], dim=1), self.expand_e1)

        shared_3d_d = shared_3d.detach()

        if self.representation == "independent":
            bg_feat = self.bg_branch(torch.cat([shared_3d, bg_seed], dim=1))
            e0_feat = self.e0_branch(torch.cat([shared_3d_d, e0_seed], dim=1))
            e1_feat = self.e1_branch(torch.cat([shared_3d_d, e1_seed], dim=1))

            bg_logit = self.bg_head(bg_feat)
            e0_logit = self.e0_head(e0_feat)
            e1_logit = self.e1_head(e1_feat)

            V_logits = torch.cat([bg_logit, e0_logit, e1_logit], dim=1)  # (B, 3, K, H, W)
            entity_probs = torch.sigmoid(V_logits[:, 1:3].float())

            return VolumeOutputs(
                entity_logits=V_logits[:, 1:3],
                entity_probs=entity_probs,
            )

        elif self.representation in ("factorized_fg_id", "center_offset"):
            fg_feat = self.fg_branch(torch.cat([shared_3d, bg_seed], dim=1))
            # id_branch sees BOTH e0 and e1 features + shared context
            id_seed = self._expand_3d(
                torch.cat([h_shared_2d, h_e0, h_e1], dim=1), self.expand_id)
            id_feat = self.id_branch(torch.cat([shared_3d_d, id_seed], dim=1))

            fg_logit = self.fg_head(fg_feat)   # (B, 1, K, H, W)
            id_logits = self.id_head(id_feat)  # (B, 2, K, H, W)

            # Factorized entity_probs with DECOUPLED fg/depth heads:
            #
            # Problem with depth-softmax (fg_max + softmax coupling):
            #   fg_magnitude = sigmoid(max_k(fg_logit)) couples fg and depth gradients.
            #   BCE at non-front K bins flows back through sigmoid(max) and fights depth
            #   concentration → compact oscillates and stalls at ~0.15.
            #
            # Fix: Separate fg spatial detection from depth localization:
            #   fg_spatial_logit = fg_spatial_head(h_shared_2d)   "Is fg at (h,w)?"
            #   fg_magnitude = sigmoid(fg_spatial_logit)          NO K coupling
            #   depth_attn = softmax_K(fg_logit[:, 0])            pure depth localization
            #   p_fg = fg_magnitude × depth_attn
            #
            # L_fg_spatial: BCE(fg_spatial_logit, Y_fg_any) → trains fg_magnitude
            # L_depth_ce: CE(fg_logit_vol at fg pixels, k_front) → trains depth_attn
            # These are fully decoupled — no gradient interference.
            fg_spatial_logit = self.fg_spatial_head(h_shared_2d)   # (B, 1, H, W)
            fg_magnitude = torch.sigmoid(fg_spatial_logit)          # (B, 1, H, W)

            fg_logit_vol = fg_logit[:, 0].float()          # (B, K, H, W)

            # Depth prior: concentrate depth_attn at the scene depth from epoch 0.
            #
            # Without this, gradient learning alone needs 50+ epochs to concentrate
            # depth_attn (gradient clipping at 0.3 over 2M+ params means fg_logit_vol
            # moves ~0.007 per 300 steps — far too slow for compact ≥ 0.20).
            #
            # Fix: add depth_hint Gaussian prior directly to fg_logit_vol in logit space.
            # This gives depth_attn[k_front] ≈ 0.60 and compact ≈ 0.40 from epoch 0.
            # The learned fg_logit_vol (from fg_head) refines the prior as training proceeds.
            #
            # Math: depth_prior = log(Gaussian(depth_hint, sigma=1.5 bins)) added to logit.
            # After softmax: depth_attn ≈ Gaussian^scale / Z → concentrates at depth_hint.
            # scale=5.0: depth_attn[k_front] ≈ 0.60 (center bin) to 0.74 (edge bins).
            if depth_hint is not None:
                K_d = self.depth_bins
                depth_bin_cont = depth_hint.unsqueeze(1).float() * (K_d - 1)   # (B, 1, H, W)
                bin_idx = torch.arange(K_d, device=depth_hint.device,
                                       dtype=torch.float32).view(1, K_d, 1, 1)
                sigma = 1.5  # ≈ 1.5 bin widths → smooth concentration
                raw = torch.exp(-(depth_bin_cont - bin_idx) ** 2 / (2.0 * sigma ** 2))  # (B, K, H, W)
                log_depth_prior = torch.log(
                    raw / raw.sum(dim=1, keepdim=True).clamp(min=1e-8))          # (B, K, H, W)
                depth_prior_scale = 10.0  # v16: 5.0→10.0. Gives compact≈0.50 from ep0
                # (was 5.0: compact≈0.34, fg_magnitude needed >0.53).
                # With 10.0: depth_attn[front]≈0.73 → need fg_magnitude>0.41 (achievable).
                fg_logit_vol = fg_logit_vol + depth_prior_scale * log_depth_prior

            depth_attn = torch.softmax(fg_logit_vol, dim=1)  # (B, K, H, W)
            p_fg = fg_magnitude * depth_attn               # (B, K, H, W)

            q = torch.softmax(id_logits.float(), dim=1)    # (B, 2, K, H, W)
            entity_probs = p_fg.unsqueeze(1) * q           # (B, 2, K, H, W)

            outputs = VolumeOutputs(
                fg_logit=fg_logit,
                fg_spatial_logit=fg_spatial_logit,
                id_logits=id_logits,
                entity_probs=entity_probs,
            )

            if self.representation == "center_offset":
                off_e0_feat = self.offset_e0_branch(torch.cat([shared_3d_d, e0_seed], dim=1))
                off_e1_feat = self.offset_e1_branch(torch.cat([shared_3d_d, e1_seed], dim=1))
                offset_e0 = self.offset_e0_head(off_e0_feat)  # (B, 3, K, H, W)
                offset_e1 = self.offset_e1_head(off_e1_feat)  # (B, 3, K, H, W)
                outputs.amodal["offset_e0"] = offset_e0
                outputs.amodal["offset_e1"] = offset_e1

            return outputs

        raise RuntimeError(f"Unknown representation: {self.representation}")
