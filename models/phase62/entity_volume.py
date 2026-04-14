"""
Phase 62 — Entity Volume Predictor
====================================

Supports multiple volume representations:

1. 'independent': bg/e0/e1 as 3 independent heads (original)
2. 'factorized_fg_id': fg_head + id_head (factorized foreground + identity)
3. 'center_offset': fg_head + id_head + offset heads for center regression
"""
from __future__ import annotations

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

        self.entity_id_e0 = nn.Parameter(torch.randn(1, hidden, 1, 1) * 0.1)
        self.entity_id_e1 = nn.Parameter(torch.randn(1, hidden, 1, 1) * 0.1)

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
            if self.representation == "center_offset":
                for head in (self.offset_e0_head, self.offset_e1_head):
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)

    def _to_2d(self, feat: torch.Tensor) -> torch.Tensor:
        B = feat.shape[0]
        return feat.permute(0, 2, 1).reshape(B, self.hidden, self.spatial_h, self.spatial_w)

    def _expand_3d(self, feat_2d: torch.Tensor, proj: nn.Conv2d) -> torch.Tensor:
        B = feat_2d.shape[0]
        h_3d = proj(feat_2d)
        return h_3d.reshape(B, self.hidden, self.depth_bins, self.spatial_h, self.spatial_w)

    def forward(
        self,
        F_g: torch.Tensor,
        F_0: torch.Tensor,
        F_1: torch.Tensor,
    ) -> VolumeOutputs:
        B = F_g.shape[0]

        h_g = self._to_2d(self.proj_g(F_g.float()))
        h_e0 = self._to_2d(self.proj_e0(F_0.float())) + self.entity_id_e0
        h_e1 = self._to_2d(self.proj_e1(F_1.float())) + self.entity_id_e1

        h_cat = torch.cat([h_g, h_e0, h_e1], dim=1)
        h_shared_2d = self.shared_2d(h_cat)

        shared_seed = self._expand_3d(h_shared_2d, self.expand_shared)
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

            # p_fg = sigmoid(z_fg)
            p_fg = torch.sigmoid(fg_logit[:, 0].float())  # (B, K, H, W)
            # q_n = softmax(z_id)_n
            q = torch.softmax(id_logits.float(), dim=1)   # (B, 2, K, H, W)
            # p_n = p_fg * q_n
            entity_probs = p_fg.unsqueeze(1) * q           # (B, 2, K, H, W)

            outputs = VolumeOutputs(
                fg_logit=fg_logit,
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
