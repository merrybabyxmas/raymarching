"""
Phase 62 — Entity-Conditioned Guide Assembly + UNet Injection
==============================================================

v22 gate gradient fix (2026-04-14):
  PROBLEM: inject_guide_into_unet_features normalised by guide_std = gate * proj_std,
           which cancelled gate from the gradient path entirely:
             guide_eff = proj(x)*gate * (max_ratio*hs_std) / (gate*proj_std)
                       = proj(x) * max_ratio*hs_std / proj_std   ← gate cancels!
           → dL/d(gate_param) ≈ 0 → gate stuck at equilibrium ~0.042.
  FIX: Normalise proj(x) BEFORE gate multiplication. Gate applied after normalisation:
           guide_eff = normalize(proj(x)) * gate
           dL/d(gate_param) = dL/d(guide_eff) * normalize(proj(x)) * sech²(gate_param) ≠ 0

  Implementation:
    - GuideFeatureAssembler.forward() outputs proj(x) WITHOUT gate multiplication.
    - Gate values stored in self._current_gates dict.
    - inject_guide_into_unet_features() accepts optional gate arg; applies AFTER norm.
    - GuideInjectionManager.set_guides() accepts gate_fn for hook retrieval.

Guide families:
  - 'none':        G = 0, no guide injection
  - 'front_only':  G = [V_0 * F_0, V_1 * F_1]
  - 'dual':        G = [F_front, F_back] (mixed front/back)
  - 'four_stream': G = [V_0*F_0, V_1*F_1, (A_0-V_0)*F_0, (A_1-V_1)*F_1]
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

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

    v22 change: forward() outputs proj(x) WITHOUT gate multiplication.
    Gate is stored in self._current_gates and applied at injection time
    AFTER amplitude normalisation — preserving gate gradient path.
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
        gate_warm_start: float = 0.0,
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

        # Learnable scale gate.
        # gate_warm_start=0.0: tanh(0)=0 → guide=0 at init (conservative, default).
        # gate_warm_start>0: atanh(warm_start) → gate starts at warm_start value.
        # v22: gate applied AFTER amplitude normalisation → clean gradient path.
        # v39b: gate_warm_start support — prevents gate from being stuck near 0 in stage2.
        _ws = float(gate_warm_start)
        _ws = max(0.0, min(_ws, 0.95))   # clamp to valid tanh range
        _gate_init = math.atanh(_ws) if _ws > 0.0 else 0.0
        self.guide_gates = nn.ParameterDict({
            bn: nn.Parameter(torch.full((1,), _gate_init)) for bn in block_names
        })

        # Current-batch gate values (set during forward, read by injection hooks).
        # These are tensors on the computation graph — hooks retrieve them for injection.
        self._current_gates: Dict[str, torch.Tensor] = {}

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
            vis_e0 = vol_outputs.visible["e0"].unsqueeze(1)  # (B, 1, H, W)
            vis_e1 = vol_outputs.visible["e1"].unsqueeze(1)  # (B, 1, H, W)
            front_e0 = vis_e0 * h_e0   # (B, hidden, H, W)
            front_e1 = vis_e1 * h_e1   # (B, hidden, H, W)
            guide_base = torch.cat([front_e0, front_e1], dim=1)  # (B, hidden*2, H, W)

        elif self.guide_family == "four_stream":
            vis_e0 = vol_outputs.visible["e0"].unsqueeze(1)
            vis_e1 = vol_outputs.visible["e1"].unsqueeze(1)
            amo_e0 = vol_outputs.amodal["e0"].unsqueeze(1)
            amo_e1 = vol_outputs.amodal["e1"].unsqueeze(1)

            # Per-stream L2 normalisation — prevents entity-0 scale dominance.
            # Without this, if vis_e0 >> vis_e1, front_e0 >> front_e1 in absolute
            # magnitude, causing the guide to be e0-dominated. UNet then amplifies
            # e0 further, creating a positive-feedback collapse spiral (winner→0.86+).
            # Normalise each stream independently so both entities contribute equally
            # regardless of their current visible/amodal field magnitudes.
            def _norm_stream(x: torch.Tensor) -> torch.Tensor:
                return x / x.norm(dim=1, keepdim=True).clamp(min=1e-6)

            front_e0 = _norm_stream(vis_e0 * h_e0)
            front_e1 = _norm_stream(vis_e1 * h_e1)
            back_e0  = _norm_stream((amo_e0 - vis_e0).clamp(min=0) * h_e0)
            back_e1  = _norm_stream((amo_e1 - vis_e1).clamp(min=0) * h_e1)
            guide_base = torch.cat([front_e0, front_e1, back_e0, back_e1], dim=1)

            # Store stream norms for diagnostics (detached, no-grad)
            self._diag_stream_norms = {
                "front_e0": (vis_e0 * h_e0).norm(dim=1).mean().item(),
                "front_e1": (vis_e1 * h_e1).norm(dim=1).mean().item(),
                "back_e0":  ((amo_e0 - vis_e0).clamp(0) * h_e0).norm(dim=1).mean().item(),
                "back_e1":  ((amo_e1 - vis_e1).clamp(0) * h_e1).norm(dim=1).mean().item(),
            }

        else:
            raise ValueError(f"Unknown guide_family: {self.guide_family}")

        guides: Dict[str, torch.Tensor] = {}
        self._current_gates = {}
        for block_name in self.block_names:
            proj = self.block_projectors[block_name]
            h_block, w_block = BLOCK_SPATIAL[block_name]
            if h_block != H or w_block != W:
                guide_resized = F.interpolate(guide_base, size=(h_block, w_block),
                                              mode="bilinear", align_corners=False)
            else:
                guide_resized = guide_base

            # v22: output proj(x) WITHOUT gate multiplication.
            # Gate is stored in _current_gates and applied by injection manager
            # AFTER amplitude normalisation, preserving gate gradient.
            proj_out = proj(guide_resized)           # (B, block_dim, H_b, W_b)
            gate = torch.tanh(self.guide_gates[block_name])
            guides[block_name] = proj_out            # gate-free
            self._current_gates[block_name] = gate   # scalar tensor, grad-connected

        return guides

    def get_gate(self, block_name: str) -> Optional[torch.Tensor]:
        """Return the gate tensor for a given block (set during last forward call)."""
        return self._current_gates.get(block_name, None)


def inject_guide_into_unet_features(
    hidden_states: torch.Tensor,
    guide: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    max_ratio: float = 0.1,
) -> torch.Tensor:
    """
    Inject guide features into UNet hidden states.

    v22 change: gate applied AFTER amplitude normalisation.

    Old (broken):
        guide_eff = proj(x) * gate                     # gate in numerator
        → normalize by guide_std = gate * proj_std     # gate cancels!
        → dL/d(gate) ≈ 0                               # gate stuck

    New (correct):
        proj_normalised = proj(x) * (max_ratio * hs_std / proj_std)  # gate-independent
        guide_eff = proj_normalised * gate                            # gate preserved
        → dL/d(gate_param) = dL/d(guide_eff) * proj_normalised * sech²(gate_param) ≠ 0

    Args:
        hidden_states: UNet intermediate features (B, C, H, W) or (B, C, T, H, W)
        guide: gate-free proj output (B, C, H_b, W_b)
        gate: optional scalar gate tensor; if None falls back to old behavior
        max_ratio: max guide amplitude as fraction of hs_std
    """
    hs_std = hidden_states.float().std().clamp(min=1e-6)
    guide_std = guide.float().std().clamp(min=1e-8)

    if gate is not None:
        # v22 path: normalise proj THEN apply gate.
        # proj_std is gate-independent → gate gradient flows cleanly.
        guide_normalized = guide * (max_ratio * hs_std / guide_std)
        guide_eff = guide_normalized * gate
    else:
        # Legacy path (gate already baked into guide): clip if too large.
        if guide_std > max_ratio * hs_std:
            guide = guide * (max_ratio * hs_std / guide_std)
        guide_eff = guide

    guide_eff = guide_eff.to(dtype=hidden_states.dtype)

    if hidden_states.dim() == 5:
        B_hs = hidden_states.shape[0]
        T = hidden_states.shape[2]
        H_block, W_block = hidden_states.shape[3], hidden_states.shape[4]
        if guide_eff.shape[2] != H_block or guide_eff.shape[3] != W_block:
            guide_eff = F.interpolate(guide_eff, size=(H_block, W_block), mode='nearest')
        if guide_eff.shape[0] != B_hs:
            guide_eff = guide_eff.repeat(B_hs // max(guide_eff.shape[0], 1), 1, 1, 1)
        guide_5d = guide_eff.unsqueeze(2).expand(-1, -1, T, -1, -1)
        return hidden_states + guide_5d
    else:
        B_hs = hidden_states.shape[0]
        H_block, W_block = hidden_states.shape[2], hidden_states.shape[3]
        if guide_eff.shape[2] != H_block or guide_eff.shape[3] != W_block:
            guide_eff = F.interpolate(guide_eff, size=(H_block, W_block), mode='nearest')
        if guide_eff.shape[0] != B_hs:
            guide_eff = guide_eff.repeat(B_hs // max(guide_eff.shape[0], 1), 1, 1, 1)
        return hidden_states + guide_eff


class GuideInjectionManager:

    def __init__(self, inject_config: str = "mid_up2", guide_max_ratio: float = 0.1):
        self.inject_config = inject_config
        self.block_names = INJECT_CONFIGS.get(inject_config, ["mid", "up2"])
        self._hooks: list = []
        self._guides: Dict[str, torch.Tensor] = {}
        self._gate_fn: Optional[Callable[[str], torch.Tensor]] = None
        # guide_max_ratio: max guide amplitude as fraction of UNet hidden-state std.
        # Default 0.1 is conservative (1-8% effective at gate=0.1-0.4).
        # Set 0.2-0.5 for stronger spatial control at the cost of potential artifacts.
        self.guide_max_ratio: float = float(guide_max_ratio)

    def set_guides(
        self,
        guides: Dict[str, torch.Tensor],
        gate_fn: Optional[Callable[[str], torch.Tensor]] = None,
    ) -> None:
        """
        Store guides and optional gate retrieval function.

        gate_fn: callable(block_name) → scalar gate tensor (on computation graph).
                 If provided, gate is applied AFTER amplitude normalisation in hooks.
                 If None, legacy behavior (gate assumed baked into guide).
        """
        self._guides = guides
        self._gate_fn = gate_fn

    def clear_guides(self) -> None:
        self._guides = {}
        self._gate_fn = None

    def _make_hook(self, block_name: str):
        def hook_fn(module, input, output):
            if block_name not in self._guides:
                return output
            guide = self._guides[block_name]
            # Retrieve gate if available (v22 path)
            gate = self._gate_fn(block_name) if self._gate_fn is not None else None
            if isinstance(output, tuple):
                h = output[0]
                h = inject_guide_into_unet_features(
                    h, guide, gate=gate, max_ratio=self.guide_max_ratio)
                return (h,) + output[1:]
            else:
                return inject_guide_into_unet_features(
                    output, guide, gate=gate, max_ratio=self.guide_max_ratio)
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
