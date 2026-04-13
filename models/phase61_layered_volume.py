"""
Phase 61 — Depth-Layered Volume Diffusion
==========================================

Core insight: Instead of predicting one alpha + one depth per entity per pixel,
predict K depth bins per entity. Each bin has alpha (occupancy) and feature delta.
Final composition uses NeRF-style front-to-back alpha rendering.

This lets two entities coexist at the same pixel but at different depths —
solving the collision identity collapse problem that plagued Phase 60's
single-alpha ownership model.

Architecture per injected block:
  - Shared LoRA on K, V, Out (from SlotLoRA)
  - Slot adapters for entity-masked attention
  - PRIMARY block only:
      * DepthVolumeHead for e0, e1: predict per-bin alpha + feature delta
      * VolumeCompositor: NeRF-style front-to-back rendering
  - Non-primary blocks: entity-presence gating blend

Forward pass (Phase61Processor.__call__):
  1. K, V with LoRA augmentation
  2. Q from hidden_states
  3. F_g = global cross-attention
  4. F_0, F_1 = masked attention → slot adapters
  5. (PRIMARY) alpha_bins, feat_bins = DepthVolumeHead(F_0), DepthVolumeHead(F_1)
  6. (PRIMARY) composed, w0, w1, w_bg = VolumeCompositor(F_g, feat0, feat1, alpha0, alpha1)
  7. Output projection with LoRA
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.entity_slot import SlotAdapter
from models.entity_slot_phase40 import (
    SlotLoRA,
    BLOCK_INNER_DIMS,
    CROSS_ATTN_DIM,
    DEFAULT_INJECT_KEYS,
)


# =============================================================================
# DepthVolumeHead: per-entity alpha bins + feature delta bins
# =============================================================================

class DepthVolumeHead(nn.Module):
    """
    Predict per-depth-bin alpha and feature delta from entity features.

    For K depth bins:
      alpha_head: feat_dim → hidden → K (sigmoid)
      feat_head:  feat_dim → hidden → feat_dim * K (raw delta)

    Output feature = base feature + delta (residual connection).
    Zero-init on last layers → alpha starts at sigmoid(0) = 0.5, delta = 0.
    """

    def __init__(self, feat_dim: int, depth_bins: int = 2, hidden: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.depth_bins = depth_bins

        self.alpha_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, depth_bins),
        )
        self.feat_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat_dim * depth_bins),
        )

        # Zero-init last layers
        nn.init.zeros_(self.alpha_head[-1].weight)
        nn.init.zeros_(self.alpha_head[-1].bias)
        nn.init.zeros_(self.feat_head[-1].weight)
        nn.init.zeros_(self.feat_head[-1].bias)

    def forward(
        self, feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: (B, S, D) entity-specific attention features

        Returns:
            alpha_bins: (B, S, K) occupancy probability per bin in [0, 1]
            feat_bins:  (B, S, K, D) feature per bin = feat + delta
        """
        B, S, D = feat.shape
        K = self.depth_bins

        alpha_bins = torch.sigmoid(self.alpha_head(feat))  # (B, S, K)

        delta = self.feat_head(feat)  # (B, S, D*K)
        delta = delta.view(B, S, K, D)  # (B, S, K, D)

        # Residual: each bin feature = base feature + learned delta
        feat_bins = feat.unsqueeze(2) + delta  # (B, S, K, D)

        return alpha_bins, feat_bins


# =============================================================================
# VolumeCompositor: NeRF-style front-to-back alpha rendering
# =============================================================================

class VolumeCompositor(nn.Module):
    """
    Compose two entities + background via NeRF-style front-to-back rendering.

    At each depth bin z (z=0 is front, z=K-1 is back):
      alpha_total(z) = 1 - (1 - alpha0(z)) * (1 - alpha1(z))
      T(z) = prod_{k<z} (1 - alpha_total(k))
      w0(z) = T(z) * alpha0(z)
      w1(z) = T(z) * alpha1(z)
    w_bg = prod_all (1 - alpha_total(z))
    composed = sum_z (w0(z)*feat0(z) + w1(z)*feat1(z)) + w_bg * feat_bg
    """

    def __init__(self, depth_bins: int = 2):
        super().__init__()
        self.depth_bins = depth_bins

    def forward(
        self,
        feat_bg:     torch.Tensor,  # (B, S, D) background/global features
        feat0_bins:  torch.Tensor,  # (B, S, K, D) entity 0 features per bin
        feat1_bins:  torch.Tensor,  # (B, S, K, D) entity 1 features per bin
        alpha0_bins: torch.Tensor,  # (B, S, K) entity 0 alpha per bin
        alpha1_bins: torch.Tensor,  # (B, S, K) entity 1 alpha per bin
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            composed:  (B, S, D) composited features
            w0_bins:   (B, S, K) entity 0 rendering weights per bin
            w1_bins:   (B, S, K) entity 1 rendering weights per bin
            w_bg:      (B, S) background weight
        """
        B, S, K = alpha0_bins.shape
        dtype = feat_bg.dtype

        # Compute in float for numerical stability
        a0 = alpha0_bins.float()  # (B, S, K)
        a1 = alpha1_bins.float()  # (B, S, K)

        # alpha_total(z) = 1 - (1 - alpha0(z)) * (1 - alpha1(z))
        alpha_total = 1.0 - (1.0 - a0) * (1.0 - a1)  # (B, S, K)

        # T(z) = prod_{k<z} (1 - alpha_total(k))
        # T(0) = 1.0
        # T(z) = T(z-1) * (1 - alpha_total(z-1))
        one_minus_alpha = 1.0 - alpha_total  # (B, S, K)
        # Cumulative product shifted by 1: [1, (1-a0), (1-a0)*(1-a1), ...]
        # Use cumprod on (1 - alpha_total) then shift
        cumprod = torch.cumprod(one_minus_alpha, dim=2)  # (B, S, K)
        # T(z=0) = 1, T(z=1) = (1-alpha_total(0)), ...
        T = torch.ones_like(cumprod)  # (B, S, K)
        T[:, :, 1:] = cumprod[:, :, :-1]  # (B, S, K)

        # Per-bin rendering weights
        w0_bins = T * a0  # (B, S, K)
        w1_bins = T * a1  # (B, S, K)

        # Background weight = transmittance after all bins
        w_bg = cumprod[:, :, -1]  # (B, S)

        # Compose features
        # sum_z(w0(z)*feat0(z) + w1(z)*feat1(z)) + w_bg*feat_bg
        feat0_f = feat0_bins.float()  # (B, S, K, D)
        feat1_f = feat1_bins.float()  # (B, S, K, D)
        feat_bg_f = feat_bg.float()   # (B, S, D)

        # Weighted sum across bins: (B, S, K, 1) * (B, S, K, D) → sum over K
        weighted_e0 = (w0_bins.unsqueeze(-1) * feat0_f).sum(dim=2)  # (B, S, D)
        weighted_e1 = (w1_bins.unsqueeze(-1) * feat1_f).sum(dim=2)  # (B, S, D)
        weighted_bg = w_bg.unsqueeze(-1) * feat_bg_f                # (B, S, D)

        composed = (weighted_e0 + weighted_e1 + weighted_bg).to(dtype)  # (B, S, D)

        return composed, w0_bins.to(dtype), w1_bins.to(dtype), w_bg.to(dtype)


# =============================================================================
# Phase61Processor: cross-attention with depth-layered volume composition
# =============================================================================

class Phase61Processor(nn.Module):
    """
    Cross-attention processor with depth-layered volume composition.

    All blocks get:
      - Shared LoRA on K, V, Out
      - Slot adapters for entity-masked attention

    Primary block additionally gets:
      - DepthVolumeHead for e0 and e1
      - VolumeCompositor (NeRF-style front-to-back rendering)

    Non-primary blocks use simple entity-presence gating blend.
    """

    def __init__(
        self,
        inner_dim:           int,
        adapter_rank:        int  = 64,
        lora_rank:           int  = 4,
        depth_bins:          int  = 2,
        cross_attention_dim: int  = CROSS_ATTN_DIM,
        is_primary:          bool = False,
    ):
        super().__init__()
        self._inner_dim = inner_dim
        self.is_primary = is_primary
        self.depth_bins = depth_bins

        # Shared LoRA (K, V, Out)
        self.lora_k   = SlotLoRA(cross_attention_dim, inner_dim, rank=lora_rank)
        self.lora_v   = SlotLoRA(cross_attention_dim, inner_dim, rank=lora_rank)
        self.lora_out = SlotLoRA(inner_dim, inner_dim, rank=lora_rank)

        # Slot adapters
        self.slot0_adapter = SlotAdapter(inner_dim, r=adapter_rank)
        self.slot1_adapter = SlotAdapter(inner_dim, r=adapter_rank)

        # Volume heads (primary only)
        if is_primary:
            self.e0_volume = DepthVolumeHead(
                feat_dim=inner_dim, depth_bins=depth_bins, hidden=128)
            self.e1_volume = DepthVolumeHead(
                feat_dim=inner_dim, depth_bins=depth_bins, hidden=128)
            self.compositor = VolumeCompositor(depth_bins=depth_bins)

        # Entity token indices (set externally)
        self._toks_e0: Optional[torch.Tensor] = None
        self._toks_e1: Optional[torch.Tensor] = None

        # Stored predictions (primary only, filled per forward)
        self._last_alpha0_bins: Optional[torch.Tensor] = None
        self._last_alpha1_bins: Optional[torch.Tensor] = None
        self._last_feat0_bins:  Optional[torch.Tensor] = None
        self._last_feat1_bins:  Optional[torch.Tensor] = None
        self._last_w0_bins:     Optional[torch.Tensor] = None
        self._last_w1_bins:     Optional[torch.Tensor] = None
        self._last_w_bg:        Optional[torch.Tensor] = None

    def set_entity_tokens(
        self, toks_e0: Optional[torch.Tensor], toks_e1: Optional[torch.Tensor],
    ):
        self._toks_e0 = toks_e0
        self._toks_e1 = toks_e1

    def reset(self):
        self._last_alpha0_bins = None
        self._last_alpha1_bins = None
        self._last_feat0_bins  = None
        self._last_feat1_bins  = None
        self._last_w0_bins     = None
        self._last_w1_bins     = None
        self._last_w_bg        = None

    def _masked_attn(
        self,
        q_mh:    torch.Tensor,  # (B, H, S, D_h)
        k_full:  torch.Tensor,  # (B, H, T_seq, D_h)
        v_full:  torch.Tensor,  # (B, H, T_seq, D_h)
        tok_idx: torch.Tensor,  # (N_tok,) token indices
        scale:   float,
        B: int, S: int, inner_dim: int,
    ) -> torch.Tensor:
        """
        Masked attention: attend only to selected entity tokens.

        Returns:
            out: (B, S, inner_dim)
        """
        k_e = k_full[:, :, tok_idx, :]  # (B, H, N_tok, D_h)
        v_e = v_full[:, :, tok_idx, :]  # (B, H, N_tok, D_h)

        scores = torch.matmul(q_mh, k_e.transpose(-1, -2)) * scale  # (B, H, S, N_tok)
        w = scores.softmax(dim=-1)                                    # (B, H, S, N_tok)
        out = torch.matmul(w, v_e)                                    # (B, H, S, D_h)

        n_heads = q_mh.shape[1]
        head_dim = inner_dim // n_heads
        out = out.permute(0, 2, 1, 3).reshape(B, S, inner_dim)  # (B, S, inner_dim)
        return out

    def __call__(
        self,
        attn,
        hidden_states:         torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask=None,
        temb=None,
        **kwargs,
    ) -> torch.Tensor:
        B, S, D = hidden_states.shape
        dtype = hidden_states.dtype
        enc_hs = (encoder_hidden_states
                  if encoder_hidden_states is not None
                  else hidden_states)

        # AnimateDiff broadcast: enc_hs (1, 77, 768) → (B, 77, 768)
        if enc_hs.shape[0] == 1 and B > 1:
            enc_hs = enc_hs.expand(B, -1, -1)

        T_seq = enc_hs.shape[1]

        inner_dim = attn.to_q.weight.shape[0]
        n_heads   = attn.heads
        head_dim  = inner_dim // n_heads
        scale     = head_dim ** -0.5

        def _mh(x: torch.Tensor, seq_len: int) -> torch.Tensor:
            """Reshape to multi-head: (B, seq, D) → (B, H, seq, D_h)"""
            return x.reshape(B, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)

        # K, V with LoRA
        enc_hs_f = enc_hs.float()
        k = attn.to_k(enc_hs) + self.lora_k(enc_hs_f).to(dtype=enc_hs.dtype)
        v = attn.to_v(enc_hs) + self.lora_v(enc_hs_f).to(dtype=enc_hs.dtype)
        k_mh = _mh(k, T_seq)  # (B, H, T_seq, D_h)
        v_mh = _mh(v, T_seq)  # (B, H, T_seq, D_h)

        # Q
        q    = attn.to_q(hidden_states)
        q_mh = _mh(q, S)  # (B, H, S, D_h)

        # Global cross-attention → F_g
        scores_g = torch.matmul(q_mh, k_mh.transpose(-1, -2)) * scale  # (B, H, S, T_seq)
        w_g = scores_g.softmax(dim=-1)
        F_g = (torch.matmul(w_g, v_mh)
               .permute(0, 2, 1, 3)
               .reshape(B, S, inner_dim))  # (B, S, inner_dim)

        # Entity-specific masked attention
        has_entity_tokens = (
            self._toks_e0 is not None
            and self._toks_e1 is not None
            and len(self._toks_e0) > 0
            and len(self._toks_e1) > 0
        )

        if has_entity_tokens:
            F_0 = self._masked_attn(
                q_mh, k_mh, v_mh, self._toks_e0, scale,
                B, S, inner_dim)  # (B, S, inner_dim)
            F_1 = self._masked_attn(
                q_mh, k_mh, v_mh, self._toks_e1, scale,
                B, S, inner_dim)  # (B, S, inner_dim)

            # Slot adapters
            F_0 = self.slot0_adapter(F_0.float()).to(dtype)  # (B, S, inner_dim)
            F_1 = self.slot1_adapter(F_1.float()).to(dtype)  # (B, S, inner_dim)

            if self.is_primary:
                # Predict per-bin alpha + feature from entity features
                alpha0_bins, feat0_bins = self.e0_volume(F_0.float())
                # alpha0_bins: (B, S, K), feat0_bins: (B, S, K, D)
                alpha1_bins, feat1_bins = self.e1_volume(F_1.float())

                # NeRF-style front-to-back composition
                composed, w0_bins, w1_bins, w_bg = self.compositor(
                    F_g, feat0_bins.to(dtype), feat1_bins.to(dtype),
                    alpha0_bins, alpha1_bins)

                # Store for loss computation
                self._last_alpha0_bins = alpha0_bins
                self._last_alpha1_bins = alpha1_bins
                self._last_feat0_bins  = feat0_bins
                self._last_feat1_bins  = feat1_bins
                self._last_w0_bins     = w0_bins
                self._last_w1_bins     = w1_bins
                self._last_w_bg        = w_bg

                out = composed
            else:
                # Non-primary: simple entity-presence gating
                entity_presence = torch.maximum(
                    F_0.float().abs().mean(dim=-1, keepdim=True),
                    F_1.float().abs().mean(dim=-1, keepdim=True),
                ).clamp(0.0, 1.0).to(dtype)  # (B, S, 1)

                composed = 0.5 * F_0 + 0.5 * F_1  # (B, S, inner_dim)
                blend = 0.3
                out = (1.0 - blend) * F_g + blend * composed
        else:
            out = F_g

        # Output projection with LoRA
        result = (attn.to_out[0](out)
                  + self.lora_out(out.float()).to(dtype=out.dtype))
        result = attn.to_out[1](result)  # dropout
        return result.to(dtype)


# =============================================================================
# Phase61Manager
# =============================================================================

class Phase61Manager:
    """
    Manages Phase61Processors across multiple attention blocks.
    """

    def __init__(
        self,
        procs:       List[Phase61Processor],
        keys:        List[str],
        primary_idx: int = 1,
    ):
        self.procs = procs
        self.keys  = keys
        self.primary_idx = primary_idx
        self.primary = procs[primary_idx]
        assert self.primary.is_primary, (
            f"Processor at index {primary_idx} ({keys[primary_idx]}) "
            f"must be the primary block")

    def set_entity_tokens(
        self, toks_e0: torch.Tensor, toks_e1: torch.Tensor,
    ):
        for p in self.procs:
            p.set_entity_tokens(toks_e0, toks_e1)

    def reset(self):
        for p in self.procs:
            p.reset()

    @property
    def volume_predictions(
        self,
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor],  # alpha0_bins, alpha1_bins
        Optional[torch.Tensor], Optional[torch.Tensor],  # feat0_bins, feat1_bins
        Optional[torch.Tensor], Optional[torch.Tensor],  # w0_bins, w1_bins
        Optional[torch.Tensor],                           # w_bg
    ]:
        """Get volume predictions from the primary block."""
        p = self.primary
        return (
            p._last_alpha0_bins, p._last_alpha1_bins,
            p._last_feat0_bins, p._last_feat1_bins,
            p._last_w0_bins, p._last_w1_bins,
            p._last_w_bg,
        )

    def train(self):
        for p in self.procs:
            p.train()

    def eval(self):
        for p in self.procs:
            p.eval()

    # ── Parameter groups ────────────────────────────────────────────────

    def all_params(self) -> List[nn.Parameter]:
        params = []
        for p in self.procs:
            params += list(p.parameters())
        return params

    def shared_lora_params(self) -> List[nn.Parameter]:
        params = []
        for p in self.procs:
            params += list(p.lora_k.parameters())
            params += list(p.lora_v.parameters())
            params += list(p.lora_out.parameters())
        return params

    def adapter_params(self) -> List[nn.Parameter]:
        params = []
        for p in self.procs:
            params += list(p.slot0_adapter.parameters())
            params += list(p.slot1_adapter.parameters())
        return params

    def volume_head_params(self) -> List[nn.Parameter]:
        """Volume head parameters (primary block only)."""
        params = []
        params += list(self.primary.e0_volume.parameters())
        params += list(self.primary.e1_volume.parameters())
        return params


# =============================================================================
# Injection
# =============================================================================

def inject_phase61(
    pipe,
    inject_keys:  Optional[List[str]] = None,
    adapter_rank: int = 64,
    lora_rank:    int = 4,
    depth_bins:   int = 2,
    primary_key:  Optional[str] = None,
) -> Tuple[Phase61Manager, Dict]:
    """
    Inject Phase61Processors into the UNet.

    Returns:
        manager:    Phase61Manager with all processors
        orig_procs: original processor dict for restoration
    """
    if inject_keys is None:
        inject_keys = DEFAULT_INJECT_KEYS

    if primary_key is None:
        primary_key = inject_keys[1]  # up_blocks.2, inner_dim=640

    unet       = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))
    new_procs  = dict(unet.attn_processors)
    procs      = []
    primary_idx = -1

    for i, key in enumerate(inject_keys):
        inner_dim  = BLOCK_INNER_DIMS.get(key, 640)
        is_primary = (key == primary_key)
        if is_primary:
            primary_idx = i

        proc = Phase61Processor(
            inner_dim=inner_dim,
            adapter_rank=adapter_rank,
            lora_rank=lora_rank,
            depth_bins=depth_bins,
            cross_attention_dim=CROSS_ATTN_DIM,
            is_primary=is_primary,
        )
        new_procs[key] = proc
        procs.append(proc)

    assert primary_idx >= 0, (
        f"Primary key {primary_key} not found in inject_keys {inject_keys}")

    unet.set_attn_processor(new_procs)
    manager = Phase61Manager(procs, inject_keys, primary_idx=primary_idx)
    return manager, orig_procs


# =============================================================================
# Checkpoint restore (compatible with Phase52/60 checkpoints)
# =============================================================================

def restore_phase61(
    manager: Phase61Manager,
    ckpt:    dict,
    device:  str,
) -> None:
    """
    Load checkpoint into Phase61Manager.

    Handles:
      - Phase61 native checkpoint (full state including volume heads)
      - Phase60 checkpoint (shared LoRA + adapters + branch heads → volume heads new)
      - Phase52-era checkpoint (shared LoRA + adapters only)

    Volume heads are always freshly initialized (zero-init) when loading
    from older checkpoints.
    """
    procs_state = ckpt.get("procs_state", [])
    if not procs_state:
        print("[restore_p61] No procs_state in checkpoint, skipping.", flush=True)
        return

    for i, p in enumerate(manager.procs):
        if i >= len(procs_state):
            break
        state = procs_state[i]

        # Shared LoRA
        if "lora_k" in state:
            p.lora_k.load_state_dict(state["lora_k"])
        if "lora_v" in state:
            p.lora_v.load_state_dict(state["lora_v"])
        if "lora_out" in state:
            p.lora_out.load_state_dict(state["lora_out"])

        # Slot adapters
        if "slot0_adapter" in state:
            p.slot0_adapter.load_state_dict(state["slot0_adapter"])
        if "slot1_adapter" in state:
            p.slot1_adapter.load_state_dict(state["slot1_adapter"])

        # Volume heads (Phase61 native only)
        if p.is_primary:
            if "e0_volume" in state:
                p.e0_volume.load_state_dict(state["e0_volume"])
                p.e1_volume.load_state_dict(state["e1_volume"])
                print(f"  [restore_p61] block[{i}]: loaded volume heads",
                      flush=True)
            else:
                print(f"  [restore_p61] block[{i}]: volume heads freshly "
                      f"initialized (zero-init)", flush=True)

        p.to(device)

    print(f"[restore_p61] Loaded {min(len(procs_state), len(manager.procs))} "
          f"block(s) from checkpoint.", flush=True)


# =============================================================================
# Depth bin target construction
# =============================================================================

def build_depth_bin_targets(
    entity_masks_BS: torch.Tensor,  # (B, 2, S) GT entity masks
    depth_orders:    list,          # [(front_idx, back_idx), ...] per frame
    depth_bins:      int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-bin alpha targets for depth-layered supervision.

    For K=2 bins: bin 0 = front layer, bin 1 = back layer.

    Rules:
      - Exclusive pixels (only one entity present):
          Entity alpha in bin 0 = 1.0, bin 1 = 0.0
          (entity occupies the front layer when alone)
      - Overlap pixels (both entities present):
          Front entity: alpha in bin 0 = 1.0, bin 1 = 0.0
          Back entity:  alpha in bin 0 = 0.0, bin 1 = 1.0

    Returns:
        alpha0_targets: (B, S, K) per-bin alpha targets for entity 0
        alpha1_targets: (B, S, K) per-bin alpha targets for entity 1
    """
    B, _, S = entity_masks_BS.shape
    K = depth_bins
    device = entity_masks_BS.device

    m0 = entity_masks_BS[:, 0, :].float()  # (B, S)
    m1 = entity_masks_BS[:, 1, :].float()  # (B, S)

    alpha0_tgt = torch.zeros(B, S, K, device=device)  # (B, S, K)
    alpha1_tgt = torch.zeros(B, S, K, device=device)  # (B, S, K)

    # Overlap mask: both entities present
    overlap = ((m0 > 0.5) & (m1 > 0.5))  # (B, S) bool

    # Exclusive masks: only one entity
    only0 = ((m0 > 0.5) & ~(m1 > 0.5))  # (B, S)
    only1 = (~(m0 > 0.5) & (m1 > 0.5))  # (B, S)

    # Exclusive pixels: entity in bin 0 (front)
    alpha0_tgt[:, :, 0][only0] = 1.0
    alpha1_tgt[:, :, 0][only1] = 1.0

    # Overlap pixels: assign by depth order (per-frame)
    for b in range(B):
        fi = min(b, len(depth_orders) - 1) if depth_orders else 0
        if fi >= len(depth_orders):
            front = 0
        else:
            front = int(depth_orders[fi][0])

        ov_b = overlap[b]  # (S,) bool

        if front == 0:
            # Entity 0 in front (bin 0), entity 1 in back (bin 1)
            alpha0_tgt[b, :, 0][ov_b] = 1.0
            alpha1_tgt[b, :, 1][ov_b] = 1.0
        else:
            # Entity 1 in front (bin 0), entity 0 in back (bin 1)
            alpha1_tgt[b, :, 0][ov_b] = 1.0
            alpha0_tgt[b, :, 1][ov_b] = 1.0

    return alpha0_tgt, alpha1_tgt
