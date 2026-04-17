from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayeredEntityDecoder(nn.Module):
    """Decode a per-entity layered 2.5D representation.

    Emits raw visible, hidden, and depth logits plus a latent feature map.
    The decoder is slot- and layout-conditioned and intentionally avoids
    early fusion between entities.
    """

    def __init__(
        self,
        slot_dim: int = 256,
        feat_dim: int = 64,
        hidden_dim: int = 128,
        Hs: int = 64,
        Ws: int = 64,
        Hf: int = 32,
        Wf: int = 32,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.Hs = Hs
        self.Ws = Ws
        self.Hf = Hf
        self.Wf = Wf

        base_ch = slot_dim * 2 + 5  # slot, mem, x/y grid (2), center heatmap (1), scale (1), frontness (1)
        self.trunk = nn.Sequential(
            nn.Conv2d(base_ch, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head_visible = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.head_hidden = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.head_depth = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.head_feat = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, feat_dim, kernel_size=1),
        )

    def _build_coord_grid(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, self.Hs, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, self.Ws, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return grid

    def _layout_maps(self, layout_i: Dict[str, torch.Tensor], B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        center = layout_i["center"]  # (B, 2)
        scale = layout_i["scale"]    # (B, 1)
        frontness = layout_i["frontness"]  # (B, 1)

        grid = self._build_coord_grid(B, device, dtype)
        cx = center[:, 0].view(B, 1, 1, 1)
        cy = center[:, 1].view(B, 1, 1, 1)
        sx = scale.view(B, 1, 1, 1) * 1.25 + 1e-3
        sy = scale.view(B, 1, 1, 1) * 1.25 + 1e-3
        dist = ((grid[:, 0:1] - cx) ** 2) / (sx ** 2) + ((grid[:, 1:2] - cy) ** 2) / (sy ** 2)
        center_heatmap = torch.exp(-dist)
        scale_map = scale.view(B, 1, 1, 1).expand(B, 1, self.Hs, self.Ws)
        front_map = frontness.view(B, 1, 1, 1).expand(B, 1, self.Hs, self.Ws)
        return torch.cat([grid, center_heatmap, scale_map, front_map], dim=1)

    def forward(
        self,
        slot_i: torch.Tensor,
        mem_i: torch.Tensor,
        layout_i: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = slot_i.shape[0]
        device = slot_i.device
        dtype = slot_i.dtype

        slot_map = slot_i.view(B, self.slot_dim, 1, 1).expand(B, self.slot_dim, self.Hs, self.Ws)
        mem_map = mem_i.view(B, self.slot_dim, 1, 1).expand(B, self.slot_dim, self.Hs, self.Ws)
        layout_maps = self._layout_maps(layout_i, B, device, dtype)
        x = torch.cat([slot_map, mem_map, layout_maps], dim=1)

        h = self.trunk(x)
        raw_visible = self.head_visible(h)
        raw_hidden = self.head_hidden(h)
        raw_depth = self.head_depth(h)
        feat = self.head_feat(h)
        if feat.shape[-2:] != (self.Hf, self.Wf):
            feat = F.interpolate(feat, size=(self.Hf, self.Wf), mode="bilinear", align_corners=False)
        return raw_visible, raw_hidden, raw_depth, feat
