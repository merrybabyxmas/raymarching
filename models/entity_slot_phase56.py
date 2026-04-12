"""
Phase 56 — T-COMP Style 2-Pass Entity LoRA
============================================

LISA T-COMP inspired approach: 3-pass noise composition training.

Instead of single-pass decomposition heads (Phase 46-53 all failed),
run UNet 3 times with entity-mode switching:
  1. bg mode  : shared LoRA, full prompt  → noise_pred_bg
  2. e0 mode  : entity0 LoRA, e0 prompt   → noise_pred_e0
  3. e1 mode  : entity1 LoRA, e1 prompt   → noise_pred_e1

Transmittance compositing with GT masks produces composite noise
that is directly supervised against the original noise.

Architecture per block:
  - Shared: lora_k, lora_v, lora_out (from Phase 40)
  - Entity0: lora_k_e0, lora_v_e0, lora_out_e0
  - Entity1: lora_k_e1, lora_v_e1, lora_out_e1
  - Slot adapters: slot0_adapter (e0 mode), slot1_adapter (e1 mode)

entity_mode switching:
  'bg'  → shared LoRA, no slot adapter, no masked attention
  'e0'  → entity0 LoRA + slot0_adapter, full text attention
  'e1'  → entity1 LoRA + slot1_adapter, full text attention
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
# Phase56Processor
# =============================================================================

class Phase56Processor(nn.Module):
    """
    Simplified multi-mode attention processor for T-COMP training.

    entity_mode determines which LoRA set and slot adapter to use:
      'bg' : shared lora_k/v/out, no slot adapter
      'e0' : lora_k_e0/v_e0/out_e0, slot0_adapter
      'e1' : lora_k_e1/v_e1/out_e1, slot1_adapter

    No VCA, no decomposition heads, no blend maps, no weight routing.
    Just LoRA-augmented cross-attention with mode switching.
    """

    def __init__(
        self,
        inner_dim:           int,
        adapter_rank:        int   = 64,
        lora_rank:           int   = 4,
        cross_attention_dim: int   = CROSS_ATTN_DIM,
    ):
        super().__init__()
        self._inner_dim = inner_dim
        self.lora_rank = lora_rank

        # ── Shared LoRA (bg mode) ────────────────────────────────────────
        self.lora_k   = SlotLoRA(cross_attention_dim, inner_dim, rank=lora_rank)
        self.lora_v   = SlotLoRA(cross_attention_dim, inner_dim, rank=lora_rank)
        self.lora_out = SlotLoRA(inner_dim, inner_dim, rank=lora_rank)

        # ── Entity 0 LoRA ────────────────────────────────────────────────
        self.lora_k_e0   = SlotLoRA(cross_attention_dim, inner_dim, rank=lora_rank)
        self.lora_v_e0   = SlotLoRA(cross_attention_dim, inner_dim, rank=lora_rank)
        self.lora_out_e0 = SlotLoRA(inner_dim, inner_dim, rank=lora_rank)

        # ── Entity 1 LoRA ────────────────────────────────────────────────
        self.lora_k_e1   = SlotLoRA(cross_attention_dim, inner_dim, rank=lora_rank)
        self.lora_v_e1   = SlotLoRA(cross_attention_dim, inner_dim, rank=lora_rank)
        self.lora_out_e1 = SlotLoRA(inner_dim, inner_dim, rank=lora_rank)

        # ── Slot adapters ────────────────────────────────────────────────
        self.slot0_adapter = SlotAdapter(inner_dim, r=adapter_rank)
        self.slot1_adapter = SlotAdapter(inner_dim, r=adapter_rank)

        # ── Entity mode state ────────────────────────────────────────────
        self._entity_mode: str = 'bg'

    @property
    def entity_mode(self) -> str:
        return self._entity_mode

    @entity_mode.setter
    def entity_mode(self, mode: str):
        assert mode in ('bg', 'e0', 'e1'), f"Invalid entity_mode: {mode}"
        self._entity_mode = mode

    def _get_active_lora(self) -> Tuple[SlotLoRA, SlotLoRA, SlotLoRA]:
        """Return (lora_k, lora_v, lora_out) for current entity_mode."""
        if self._entity_mode == 'e0':
            return self.lora_k_e0, self.lora_v_e0, self.lora_out_e0
        elif self._entity_mode == 'e1':
            return self.lora_k_e1, self.lora_v_e1, self.lora_out_e1
        else:  # 'bg'
            return self.lora_k, self.lora_v, self.lora_out

    def __call__(
        self,
        attn,
        hidden_states:          torch.Tensor,
        encoder_hidden_states:  Optional[torch.Tensor] = None,
        attention_mask          = None,
        temb                    = None,
        **kwargs,
    ) -> torch.Tensor:
        B, S, D = hidden_states.shape
        dtype   = hidden_states.dtype
        enc_hs  = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        # AnimateDiff broadcast: enc_hs (1, 77, 768) → (B, 77, 768)
        if enc_hs.shape[0] == 1 and B > 1:
            enc_hs = enc_hs.expand(B, -1, -1)

        T_seq = enc_hs.shape[1]

        inner_dim = attn.to_q.weight.shape[0]
        n_heads   = attn.heads
        head_dim  = inner_dim // n_heads
        scale     = head_dim ** -0.5

        def _mh(x: torch.Tensor, seq_len: int) -> torch.Tensor:
            return x.reshape(B, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)

        # ── K, V: bg mode uses NO LoRA (vanilla UNet), entity modes use LoRA
        enc_hs_f = enc_hs.float()
        if self._entity_mode == 'bg':
            k = attn.to_k(enc_hs)
            v = attn.to_v(enc_hs)
        else:
            active_k, active_v, _ = self._get_active_lora()
            k = attn.to_k(enc_hs) + active_k(enc_hs_f).to(dtype=enc_hs.dtype)
            v = attn.to_v(enc_hs) + active_v(enc_hs_f).to(dtype=enc_hs.dtype)
        k_mh = _mh(k, T_seq)
        v_mh = _mh(v, T_seq)

        # ── Q ────────────────────────────────────────────────────────────
        q    = attn.to_q(hidden_states)
        q_mh = _mh(q, S)

        # ── Standard cross-attention ─────────────────────────────────────
        scores = torch.matmul(q_mh, k_mh.transpose(-1, -2)) * scale
        w      = scores.softmax(dim=-1)
        out    = (torch.matmul(w, v_mh)
                  .permute(0, 2, 1, 3)
                  .reshape(B, S, inner_dim))

        # ── Slot adapter (entity modes only) ─────────────────────────────
        if self._entity_mode == 'e0':
            out = self.slot0_adapter(out.float()).to(dtype)
        elif self._entity_mode == 'e1':
            out = self.slot1_adapter(out.float()).to(dtype)
        # bg mode: no adapter, pass through

        # ── Output projection: bg uses vanilla, entity uses LoRA ─────────
        if self._entity_mode == 'bg':
            result = attn.to_out[0](out)
        else:
            _, _, active_out = self._get_active_lora()
            result = (attn.to_out[0](out)
                      + active_out(out.float()).to(dtype=out.dtype))
        result = attn.to_out[1](result)
        return result.to(dtype)


# =============================================================================
# Injection
# =============================================================================

def inject_multi_block_entity_slot_p56(
    pipe,
    inject_keys:  Optional[List[str]] = None,
    adapter_rank: int = 64,
    lora_rank:    int = 4,
) -> Tuple[List[Phase56Processor], Dict]:
    """
    Inject Phase56Processor into specified attention blocks.

    Returns
    -------
    procs      : List[Phase56Processor]
    orig_procs : original processor dict for restoration
    """
    if inject_keys is None:
        inject_keys = DEFAULT_INJECT_KEYS

    unet       = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))
    new_procs  = dict(unet.attn_processors)
    procs      = []

    for key in inject_keys:
        inner_dim = BLOCK_INNER_DIMS.get(key, 640)
        proc = Phase56Processor(
            inner_dim=inner_dim,
            adapter_rank=adapter_rank,
            lora_rank=lora_rank,
            cross_attention_dim=CROSS_ATTN_DIM,
        )
        new_procs[key] = proc
        procs.append(proc)

    unet.set_attn_processor(new_procs)
    return procs, orig_procs


# =============================================================================
# Manager
# =============================================================================

class MultiBlockSlotManagerP56:
    """
    Manages multiple Phase56Processors with unified entity_mode switching.
    """

    def __init__(self, procs: List[Phase56Processor], keys: List[str]):
        self.procs = procs
        self.keys  = keys

    def set_entity_mode(self, mode: str):
        """Set entity mode for all processors: 'bg', 'e0', or 'e1'."""
        for p in self.procs:
            p.entity_mode = mode

    def train(self):
        for p in self.procs:
            p.train()

    def eval(self):
        for p in self.procs:
            p.eval()

    def all_params(self) -> List[nn.Parameter]:
        """All trainable parameters."""
        params = []
        for p in self.procs:
            params += list(p.parameters())
        return params

    def shared_lora_params(self) -> List[nn.Parameter]:
        """Shared (bg) LoRA parameters."""
        params = []
        for p in self.procs:
            params += list(p.lora_k.parameters())
            params += list(p.lora_v.parameters())
            params += list(p.lora_out.parameters())
        return params

    def entity_lora_params(self) -> List[nn.Parameter]:
        """Entity-specific LoRA parameters (e0 + e1)."""
        params = []
        for p in self.procs:
            params += list(p.lora_k_e0.parameters())
            params += list(p.lora_v_e0.parameters())
            params += list(p.lora_out_e0.parameters())
            params += list(p.lora_k_e1.parameters())
            params += list(p.lora_v_e1.parameters())
            params += list(p.lora_out_e1.parameters())
        return params

    def adapter_params(self) -> List[nn.Parameter]:
        """Slot adapter parameters."""
        params = []
        for p in self.procs:
            params += list(p.slot0_adapter.parameters())
            params += list(p.slot1_adapter.parameters())
        return params


# =============================================================================
# Checkpoint restore
# =============================================================================

def restore_multiblock_state_p56(
    manager:  MultiBlockSlotManagerP56,
    ckpt:     dict,
    device:   str,
) -> None:
    """
    Load Phase56 checkpoint into manager.

    If loading from a Phase52-era checkpoint, initializes entity LoRA
    from the shared LoRA weights and skips decomposition heads.
    """
    procs_state = ckpt.get("procs_state", [])
    if not procs_state:
        print("[restore_p56] No procs_state in checkpoint, skipping.", flush=True)
        return

    for i, p in enumerate(manager.procs):
        if i >= len(procs_state):
            break
        state = procs_state[i]

        # ── Shared LoRA ──────────────────────────────────────────────────
        if "lora_k" in state:
            p.lora_k.load_state_dict(state["lora_k"])
        if "lora_v" in state:
            p.lora_v.load_state_dict(state["lora_v"])
        if "lora_out" in state:
            p.lora_out.load_state_dict(state["lora_out"])

        # ── Entity LoRA ──────────────────────────────────────────────────
        if "lora_k_e0" in state:
            # Full Phase56 checkpoint
            p.lora_k_e0.load_state_dict(state["lora_k_e0"])
            p.lora_k_e1.load_state_dict(state["lora_k_e1"])
            p.lora_v_e0.load_state_dict(state["lora_v_e0"])
            p.lora_v_e1.load_state_dict(state["lora_v_e1"])
            p.lora_out_e0.load_state_dict(state["lora_out_e0"])
            p.lora_out_e1.load_state_dict(state["lora_out_e1"])
        else:
            # Phase52-era checkpoint: init entity LoRA from shared
            print(f"  [restore_p56] block[{i}]: init entity LoRA from shared",
                  flush=True)
            p.lora_k_e0.load_state_dict(p.lora_k.state_dict())
            p.lora_k_e1.load_state_dict(p.lora_k.state_dict())
            p.lora_v_e0.load_state_dict(p.lora_v.state_dict())
            p.lora_v_e1.load_state_dict(p.lora_v.state_dict())
            p.lora_out_e0.load_state_dict(p.lora_out.state_dict())
            p.lora_out_e1.load_state_dict(p.lora_out.state_dict())

        # ── Slot adapters ────────────────────────────────────────────────
        if "slot0_adapter" in state:
            p.slot0_adapter.load_state_dict(state["slot0_adapter"])
        if "slot1_adapter" in state:
            p.slot1_adapter.load_state_dict(state["slot1_adapter"])

        p.to(device)

    print(f"[restore_p56] Loaded {min(len(procs_state), len(manager.procs))} "
          f"block(s) from checkpoint.", flush=True)


# =============================================================================
# Transmittance compositing (noise space)
# =============================================================================

def transmittance_composite_absolute(
    noise_e0:     torch.Tensor,   # (1, 4, T, H, W)
    noise_e1:     torch.Tensor,   # (1, 4, T, H, W)
    noise_bg:     torch.Tensor,   # (1, 4, T, H, W)
    m0:           torch.Tensor,   # (1, 1, T, H, W) GT masks at latent resolution
    m1:           torch.Tensor,   # (1, 1, T, H, W)
    depth_orders: list,           # list of (front_entity, back_entity) per frame
    blend_strength: float = 0.7,  # how much entity replaces bg (0=pure bg, 1=full replace)
) -> torch.Tensor:
    """
    Soft transmittance composition with configurable blend strength.

    Entity passes provide identity-specific noise, but are blended with
    bg noise using soft masks to avoid latent-space discontinuities.

    composite = (1 - strength*v_union) * noise_bg
              + strength * (v0 * noise_e0 + v1 * noise_e1)

    At blend_strength < 1.0, the bg scene foundation shows through even
    in entity regions, maintaining spatial coherence while adding identity.
    """
    T = noise_bg.shape[2]
    composite = torch.zeros_like(noise_bg)

    for fi in range(T):
        front = int(depth_orders[fi][0]) if fi < len(depth_orders) else 0
        a0 = m0[:, :, fi:fi+1]   # (1, 1, 1, H, W)
        a1 = m1[:, :, fi:fi+1]

        if front == 0:
            v0 = a0
            v1 = a1 * (1.0 - a0)
        else:
            v1 = a1
            v0 = a0 * (1.0 - a1)

        # Soft blend: entity contributions scaled by blend_strength
        v_entity = (v0 + v1).clamp(0, 1) * blend_strength
        bg_weight = 1.0 - v_entity

        entity_noise = v0 * noise_e0[:, :, fi:fi+1] + v1 * noise_e1[:, :, fi:fi+1]
        entity_sum = (v0 + v1).clamp(min=1e-6)
        entity_noise_normalized = entity_noise / entity_sum

        composite[:, :, fi:fi+1] = (
            bg_weight * noise_bg[:, :, fi:fi+1]
            + v_entity * entity_noise_normalized
        )

    return composite


# Keep residual version as fallback (not used in v4)
def residual_transmittance_composite(
    noise_e0, noise_e1, noise_bg, m0, m1, depth_orders
):
    """Legacy residual version — kept for backward compat."""
    return transmittance_composite_absolute(
        noise_e0, noise_e1, noise_bg, m0, m1, depth_orders)


def outside_mask_suppression(
    noise_entity: torch.Tensor,   # (1, 4, T, H, W)
    noise_bg:     torch.Tensor,   # (1, 4, T, H, W)
    mask:         torch.Tensor,   # (1, 1, T, H, W) entity mask
    eps:          float = 1e-6,
) -> torch.Tensor:
    """
    Penalize entity pass for predicting noise DIFFERENT from bg OUTSIDE its mask.

    delta = noise_entity - noise_bg
    loss = MSE(delta * (1-mask), 0) / n_outside

    Forces entity passes to only contribute within their own mask region,
    preventing entity identity from leaking into other regions.
    """
    delta = (noise_entity - noise_bg).float()
    outside = (1.0 - mask).float()
    diff_sq = delta.pow(2) * outside
    n_out = outside.sum() * noise_entity.shape[1] + eps
    return diff_sq.sum() / n_out


# =============================================================================
# Masked MSE loss
# =============================================================================

def masked_mse(
    pred:   torch.Tensor,   # (1, C, T, H, W)
    target: torch.Tensor,   # (1, C, T, H, W)
    mask:   torch.Tensor,   # (1, 1, T, H, W)
    eps:    float = 1e-6,
) -> torch.Tensor:
    """MSE loss weighted by spatial mask."""
    diff = (pred - target).pow(2)
    return (diff * mask).sum() / (mask.sum() * pred.shape[1] + eps)


def solo_entity_anchor(
    pipe,
    noisy_solo:   torch.Tensor,   # (1, 4, T, H, W) noisy solo latent
    noise_solo:   torch.Tensor,   # (1, 4, T, H, W) GT noise for solo
    t:            torch.Tensor,   # timestep
    enc_entity:   torch.Tensor,   # (1, 77, 768) entity text embedding
) -> torch.Tensor:
    """
    Solo entity denoising anchor: entity pass should correctly denoise
    a solo-rendered frame of its own object.

    This forces entity-specific LoRA to learn actual object identity,
    not just scene-level noise prediction.
    """
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        pred = pipe.unet(noisy_solo, t, encoder_hidden_states=enc_entity).sample
    return F.mse_loss(pred.float(), noise_solo.float())
