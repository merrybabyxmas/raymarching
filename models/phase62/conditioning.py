"""
Phase 62 — Entity-Conditioned Guide Assembly + UNet Injection
==============================================================

Guide families:
  - 'none':        G = 0, no guide injection
  - 'front_only':  G = [V_0 * F_0, V_1 * F_1]
  - 'dual':        G = [F_front, F_back] (mixed front/back)
  - 'four_stream': G = [V_0*F_0, V_1*F_1, (A_0-V_0)*F_0, (A_1-V_1)*F_1]
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.phase62.objectives.base import VolumeOutputs


BLOCK_DIMS: Dict[str, int] = {
    "mid":  1280,
    "up1":  1280,
    "up2":  640,
    "up3":  320,
}

BLOCK_SPATIAL: Dict[str, tuple] = {
    "mid":  (4, 4),
    "up1":  (8, 8),
    "up2":  (16, 16),
    "up3":  (32, 32),
}

INJECT_CONFIGS: Dict[str, List[str]] = {
    "mid_only":   ["mid"],
    "mid_up2":    ["mid", "up2"],
    "multiscale": ["mid", "up1", "up2", "up3"],
}

GUIDE_FAMILIES = ("none", "front_only", "dual", "four_stream")


class GuideFeatureAssembler(nn.Module):
    """
    Assemble guide features from entity projections and backbone features.

    Supports multiple guide families:
      - 'dual' (default): front+back mixed streams (2 * hidden channels)
      - 'front_only': entity-specific visible weighted (2 * hidden channels)
      - 'four_stream': front_e0, front_e1, back_e0, back_e1 (4 * hidden channels)
      - 'none': no guide (returns empty dict)
    """

    def __init__(
        self,
        feat_dim: int = 640,
        hidden: int = 64,
        spatial_h: int = 16,
        spatial_w: int = 16,
        n_classes: int = 3,
        inject_config: str = "mid_up2",
        guide_family: str = "dual",
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.n_classes = n_classes
        self.guide_family = guide_family or "none"

        if self.guide_family == "none":
            self.block_names = []
            return

        self.proj_g = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU())
        self.proj_e0 = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU())
        self.proj_e1 = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU())

        block_names = INJECT_CONFIGS.get(inject_config, ["mid", "up2"])
        self.block_names = block_names

        if guide_family == "four_stream":
            in_ch = hidden * 4
        else:
            in_ch = hidden * 2

        self.block_projectors = nn.ModuleDict()
        for block_name in block_names:
            block_dim = BLOCK_DIMS[block_name]
            self.block_projectors[block_name] = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
                nn.Conv2d(in_ch, block_dim, kernel_size=1, bias=True),
            )

        for block_name in block_names:
            proj = self.block_projectors[block_name]
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)

    def forward(
        self,
        vol_outputs: VolumeOutputs,
        F_g: torch.Tensor,
        F_0: torch.Tensor,
        F_1: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.guide_family == "none":
            return {}

        B = F_g.shape[0]
        H, W = self.spatial_h, self.spatial_w

        h_g = self.proj_g(F_g.float()).permute(0, 2, 1).reshape(B, self.hidden, H, W)
        h_e0 = self.proj_e0(F_0.float()).permute(0, 2, 1).reshape(B, self.hidden, H, W)
        h_e1 = self.proj_e1(F_1.float()).permute(0, 2, 1).reshape(B, self.hidden, H, W)

        if self.guide_family == "dual":
            feat_stack = torch.stack([h_g, h_e0, h_e1], dim=1)  # (B, 3, hidden, H, W)
            front_probs = vol_outputs.front_probs   # (B, C, H, W)
            back_probs = vol_outputs.back_probs     # (B, C, H, W)
            front_layer = (feat_stack * front_probs.unsqueeze(2)).sum(dim=1)
            back_layer = (feat_stack * back_probs.unsqueeze(2)).sum(dim=1)
            guide_base = torch.cat([front_layer, back_layer], dim=1)  # (B, hidden*2, H, W)

        elif self.guide_family == "front_only":
            # G = [V_0 * F_0, V_1 * F_1]
            vis_e0 = vol_outputs.visible["e0"].unsqueeze(1)  # (B, 1, H, W)
            vis_e1 = vol_outputs.visible["e1"].unsqueeze(1)  # (B, 1, H, W)
            front_e0 = vis_e0 * h_e0   # (B, hidden, H, W)
            front_e1 = vis_e1 * h_e1   # (B, hidden, H, W)
            guide_base = torch.cat([front_e0, front_e1], dim=1)  # (B, hidden*2, H, W)

        elif self.guide_family == "four_stream":
            # G = [V_0*F_0, V_1*F_1, (A_0-V_0)*F_0, (A_1-V_1)*F_1]
            vis_e0 = vol_outputs.visible["e0"].unsqueeze(1)
            vis_e1 = vol_outputs.visible["e1"].unsqueeze(1)
            amo_e0 = vol_outputs.amodal["e0"].unsqueeze(1)
            amo_e1 = vol_outputs.amodal["e1"].unsqueeze(1)
            front_e0 = vis_e0 * h_e0
            front_e1 = vis_e1 * h_e1
            back_e0 = (amo_e0 - vis_e0).clamp(min=0) * h_e0
            back_e1 = (amo_e1 - vis_e1).clamp(min=0) * h_e1
            guide_base = torch.cat([front_e0, front_e1, back_e0, back_e1], dim=1)

        else:
            raise ValueError(f"Unknown guide_family: {self.guide_family}")

        guides: Dict[str, torch.Tensor] = {}
        for block_name in self.block_names:
            proj = self.block_projectors[block_name]
            h_block, w_block = BLOCK_SPATIAL[block_name]
            if h_block != H or w_block != W:
                guide_resized = F.interpolate(guide_base, size=(h_block, w_block),
                                              mode="bilinear", align_corners=False)
            else:
                guide_resized = guide_base
            guides[block_name] = proj(guide_resized)

        return guides


def inject_guide_into_unet_features(
    hidden_states: torch.Tensor,
    guide: torch.Tensor,
    max_ratio: float = 0.1,
) -> torch.Tensor:
    hs_std = hidden_states.float().std().clamp(min=1e-6)
    guide_std = guide.float().std().clamp(min=1e-8)
    if guide_std > max_ratio * hs_std:
        guide = guide * (max_ratio * hs_std / guide_std)

    if hidden_states.dim() == 5:
        B_hs = hidden_states.shape[0]
        T = hidden_states.shape[2]
        H_block, W_block = hidden_states.shape[3], hidden_states.shape[4]
        if guide.shape[2] != H_block or guide.shape[3] != W_block:
            guide = F.interpolate(guide, size=(H_block, W_block), mode='nearest')
        if guide.shape[0] != B_hs:
            guide = guide.repeat(B_hs // max(guide.shape[0], 1), 1, 1, 1)
        guide_5d = guide.unsqueeze(2).expand(-1, -1, T, -1, -1)
        return hidden_states + guide_5d
    else:
        B_hs = hidden_states.shape[0]
        H_block, W_block = hidden_states.shape[2], hidden_states.shape[3]
        if guide.shape[2] != H_block or guide.shape[3] != W_block:
            guide = F.interpolate(guide, size=(H_block, W_block), mode='nearest')
        if guide.shape[0] != B_hs:
            guide = guide.repeat(B_hs // max(guide.shape[0], 1), 1, 1, 1)
        return hidden_states + guide


class GuideInjectionManager:

    def __init__(self, inject_config: str = "mid_up2"):
        self.inject_config = inject_config
        self.block_names = INJECT_CONFIGS.get(inject_config, ["mid", "up2"])
        self._hooks: list = []
        self._guides: Dict[str, torch.Tensor] = {}

    def set_guides(self, guides: Dict[str, torch.Tensor]) -> None:
        self._guides = guides

    def clear_guides(self) -> None:
        self._guides = {}

    def _make_hook(self, block_name: str):
        def hook_fn(module, input, output):
            if block_name not in self._guides:
                return output
            guide = self._guides[block_name]
            if isinstance(output, tuple):
                h = output[0]
                h = inject_guide_into_unet_features(h, guide)
                return (h,) + output[1:]
            else:
                return inject_guide_into_unet_features(output, guide)
        return hook_fn

    def register_hooks(self, unet) -> None:
        block_map = {"mid": unet.mid_block}
        for i, up_block in enumerate(unet.up_blocks):
            block_map[f"up{i}"] = up_block
        for block_name in self.block_names:
            if block_name in block_map and block_map[block_name] is not None:
                h = block_map[block_name].register_forward_hook(self._make_hook(block_name))
                self._hooks.append(h)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
