"""
Phase 60 — Depth-Ordered Layered Video Diffusion (Single-Pass)
===============================================================

Key insight: Phase 46-53 decomposition heads all failed because explicit
weight prediction is structurally unstable. Phase 56 (3-pass) works but
costs 3x UNet forward per step. Phase 60 achieves single-pass decomposition
via FEATURE-LEVEL ownership composition inside the cross-attention processor.

Architecture per injected block:
  - Shared LoRA on K, V, Out (from Phase 40/52)
  - Slot adapters: slot0_adapter (entity 0), slot1_adapter (entity 1)
  - Entity branch heads (PRIMARY block only):
      * e0_alpha_head: F_0 → alpha_0 occupancy probability
      * e1_alpha_head: F_1 → alpha_1 occupancy probability
      * e0_depth_head: F_0 → depth_logit_0
      * e1_depth_head: F_1 → depth_logit_1
  - OwnershipComputer: alpha + depth → own0, own1, own_bg (no learnable params)

Forward pass (inside Phase60Processor.__call__):
  1. Compute Q from hidden_states
  2. Compute K, V with LoRA augmentation
  3. F_g = softmax(Q @ K.T / sqrt(d)) @ V            — global features
  4. F_0 = masked_attn(Q, K[e0_toks], V[e0_toks])    — entity 0 slot
  5. F_1 = masked_attn(Q, K[e1_toks], V[e1_toks])    — entity 1 slot
  6. F_0 = slot0_adapter(F_0), F_1 = slot1_adapter(F_1)
  7. (PRIMARY only) Predict alpha, depth from F_0, F_1
  8. (PRIMARY only) Compute ownership: own0, own1, own_bg
  9. (PRIMARY only) composed = own_bg * F_g + own0 * F_0 + own1 * F_1
  10. Non-primary blocks: use slot_blend as in Phase 40 (entity_presence gating)
  11. Output projection with LoRA

The ownership predictions from the PRIMARY block are stored for loss computation.
Non-primary blocks use a simple blend since ownership needs only be computed once.
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
# EntityBranchHead: lightweight per-entity alpha + depth predictor
# =============================================================================

class EntityBranchHead(nn.Module):
    """
    Predict occupancy alpha and depth logit from entity-specific features.

    Architecture:
      alpha_head: feat_dim → hidden → 1 (sigmoid)
      depth_head: feat_dim → hidden → 1 (raw logit, no activation)

    Zero-init on last layer → sigmoid(0) = 0.5 at init.
    """

    def __init__(self, feat_dim: int = 640, hidden: int = 128):
        super().__init__()
        self.alpha_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.depth_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init last layers → starts at sigmoid(0) = 0.5
        nn.init.zeros_(self.alpha_head[-1].weight)
        nn.init.zeros_(self.alpha_head[-1].bias)
        nn.init.zeros_(self.depth_head[-1].weight)
        nn.init.zeros_(self.depth_head[-1].bias)

    def forward(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: (B, S, feat_dim) entity-specific attention features

        Returns:
            alpha:       (B, S) occupancy probability in [0, 1]
            depth_logit: (B, S) raw depth logit (lower = closer to camera)
        """
        alpha = torch.sigmoid(self.alpha_head(feat)).squeeze(-1)  # (B, S)
        depth_logit = self.depth_head(feat).squeeze(-1)           # (B, S)
        return alpha, depth_logit


# =============================================================================
# OwnershipComputer: alpha + depth → ownership maps (no learnable params)
# =============================================================================

class OwnershipComputer(nn.Module):
    """
    Compute visible ownership from alpha + depth using Porter-Duff semantics.

    own0 = alpha0 * ((1 - alpha1) + alpha1 * front0)
    own1 = alpha1 * ((1 - alpha0) + alpha0 * (1 - front0))
    own_bg = clamp(1 - max(alpha0, alpha1), 0, 1)

    where front0 = sigmoid(sharpness * (depth1 - depth0))
    (entity0 is in front when depth0 < depth1)

    Soft-normalize so own0 + own1 + own_bg = 1.
    """

    def __init__(self, sharpness: float = 10.0):
        super().__init__()
        self.sharpness = sharpness

    def forward(
        self,
        alpha0: torch.Tensor,  # (B, S)
        alpha1: torch.Tensor,  # (B, S)
        depth0: torch.Tensor,  # (B, S)
        depth1: torch.Tensor,  # (B, S)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            own0, own1, own_bg: (B, S) each, summing to 1.0
        """
        # front0: probability that entity0 is in front of entity1
        front0 = torch.sigmoid(
            self.sharpness * (depth1 - depth0)
        )  # (B, S)

        # Porter-Duff ownership
        own0 = alpha0 * ((1.0 - alpha1) + alpha1 * front0)       # (B, S)
        own1 = alpha1 * ((1.0 - alpha0) + alpha0 * (1.0 - front0))  # (B, S)
        own_bg = (1.0 - torch.maximum(alpha0, alpha1)).clamp(0, 1)   # (B, S)

        # Soft normalize to ensure own0 + own1 + own_bg = 1
        total = (own0 + own1 + own_bg).clamp(min=1e-6)  # (B, S)
        own0 = own0 / total
        own1 = own1 / total
        own_bg = own_bg / total

        return own0, own1, own_bg


# =============================================================================
# Phase60Processor: single-pass cross-attention with ownership composition
# =============================================================================

class Phase60Processor(nn.Module):
    """
    Cross-attention processor with feature-level entity composition.

    All blocks get:
      - Shared LoRA on K, V, Out
      - Slot adapters for entity-masked attention
      - Entity-presence gating (simple blend for non-primary)

    Primary block additionally gets:
      - EntityBranchHead for e0 and e1
      - OwnershipComputer
      - Feature-level composition: composed = own_bg*F_g + own0*F_0 + own1*F_1

    The primary block stores its predictions (alpha, depth, ownership) for
    external loss computation via the manager.
    """

    def __init__(
        self,
        inner_dim:           int,
        adapter_rank:        int   = 64,
        lora_rank:           int   = 4,
        cross_attention_dim: int   = CROSS_ATTN_DIM,
        is_primary:          bool  = False,
        depth_sharpness:     float = 10.0,
    ):
        super().__init__()
        self._inner_dim = inner_dim
        self.is_primary = is_primary

        # ── Shared LoRA (K, V, Out) ──────────────────────────────────────
        self.lora_k   = SlotLoRA(cross_attention_dim, inner_dim, rank=lora_rank)
        self.lora_v   = SlotLoRA(cross_attention_dim, inner_dim, rank=lora_rank)
        self.lora_out = SlotLoRA(inner_dim, inner_dim, rank=lora_rank)

        # ── Slot adapters ────────────────────────────────────────────────
        self.slot0_adapter = SlotAdapter(inner_dim, r=adapter_rank)
        self.slot1_adapter = SlotAdapter(inner_dim, r=adapter_rank)

        # ── Entity branch heads (primary only) ──────────────────────────
        if is_primary:
            self.e0_branch = EntityBranchHead(feat_dim=inner_dim, hidden=128)
            self.e1_branch = EntityBranchHead(feat_dim=inner_dim, hidden=128)
            self.ownership = OwnershipComputer(sharpness=depth_sharpness)

        # ── Entity token indices (set externally) ────────────────────────
        self._toks_e0: Optional[torch.Tensor] = None
        self._toks_e1: Optional[torch.Tensor] = None

        # ── Stored predictions (primary only, filled per forward) ────────
        self._last_alpha0:      Optional[torch.Tensor] = None
        self._last_alpha1:      Optional[torch.Tensor] = None
        self._last_depth0:      Optional[torch.Tensor] = None
        self._last_depth1:      Optional[torch.Tensor] = None
        self._last_own0:        Optional[torch.Tensor] = None
        self._last_own1:        Optional[torch.Tensor] = None
        self._last_own_bg:      Optional[torch.Tensor] = None
        self._last_F0:          Optional[torch.Tensor] = None
        self._last_F1:          Optional[torch.Tensor] = None
        self._last_Fg:          Optional[torch.Tensor] = None

    def set_entity_tokens(
        self, toks_e0: Optional[torch.Tensor], toks_e1: Optional[torch.Tensor]
    ):
        """Set entity token indices for masked attention."""
        self._toks_e0 = toks_e0
        self._toks_e1 = toks_e1

    def reset(self):
        """Clear stored predictions."""
        self._last_alpha0 = None
        self._last_alpha1 = None
        self._last_depth0 = None
        self._last_depth1 = None
        self._last_own0   = None
        self._last_own1   = None
        self._last_own_bg = None
        self._last_F0     = None
        self._last_F1     = None
        self._last_Fg     = None

    def _masked_attn(
        self,
        q_mh: torch.Tensor,    # (B, H, S, D_h)
        k_full: torch.Tensor,  # (B, H, T_seq, D_h)
        v_full: torch.Tensor,  # (B, H, T_seq, D_h)
        tok_idx: torch.Tensor, # (N_tok,) token indices
        scale: float,
        B: int, S: int, inner_dim: int,
        n_heads: int,
    ) -> torch.Tensor:
        """
        Masked attention: attend only to selected entity tokens.

        Args:
            q_mh:     (B, H, S, D_h) query in multi-head form
            k_full:   (B, H, T_seq, D_h) full key
            v_full:   (B, H, T_seq, D_h) full value
            tok_idx:  (N_tok,) indices into T_seq dimension
            scale:    attention scale factor

        Returns:
            out: (B, S, inner_dim) entity-specific attention output
        """
        # Select entity tokens: (B, H, N_tok, D_h)
        k_e = k_full[:, :, tok_idx, :]
        v_e = v_full[:, :, tok_idx, :]

        scores = torch.matmul(q_mh, k_e.transpose(-1, -2)) * scale  # (B, H, S, N_tok)
        w = scores.softmax(dim=-1)                                    # (B, H, S, N_tok)
        out = torch.matmul(w, v_e)                                    # (B, H, S, D_h)
        out = out.permute(0, 2, 1, 3).reshape(B, S, inner_dim)       # (B, S, inner_dim)
        return out

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
        enc_hs  = (encoder_hidden_states
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
            """Reshape to multi-head: (B, seq_len, D) → (B, H, seq_len, D_h)"""
            return x.reshape(B, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)

        # ── K, V with LoRA ──────────────────────────────────────────────
        enc_hs_f = enc_hs.float()
        k = attn.to_k(enc_hs) + self.lora_k(enc_hs_f).to(dtype=enc_hs.dtype)
        v = attn.to_v(enc_hs) + self.lora_v(enc_hs_f).to(dtype=enc_hs.dtype)
        k_mh = _mh(k, T_seq)   # (B, H, T_seq, D_h)
        v_mh = _mh(v, T_seq)   # (B, H, T_seq, D_h)

        # ── Q ────────────────────────────────────────────────────────────
        q    = attn.to_q(hidden_states)
        q_mh = _mh(q, S)       # (B, H, S, D_h)

        # ── Global cross-attention: F_g ──────────────────────────────────
        scores_g = torch.matmul(q_mh, k_mh.transpose(-1, -2)) * scale  # (B, H, S, T_seq)
        w_g      = scores_g.softmax(dim=-1)
        F_g      = (torch.matmul(w_g, v_mh)
                    .permute(0, 2, 1, 3)
                    .reshape(B, S, inner_dim))  # (B, S, inner_dim)

        # ── Entity-specific masked attention ─────────────────────────────
        has_entity_tokens = (self._toks_e0 is not None and
                             self._toks_e1 is not None and
                             len(self._toks_e0) > 0 and
                             len(self._toks_e1) > 0)

        if has_entity_tokens:
            F_0 = self._masked_attn(
                q_mh, k_mh, v_mh, self._toks_e0, scale,
                B, S, inner_dim, n_heads)   # (B, S, inner_dim)
            F_1 = self._masked_attn(
                q_mh, k_mh, v_mh, self._toks_e1, scale,
                B, S, inner_dim, n_heads)   # (B, S, inner_dim)

            # Apply slot adapters
            F_0 = self.slot0_adapter(F_0.float()).to(dtype)  # (B, S, inner_dim)
            F_1 = self.slot1_adapter(F_1.float()).to(dtype)  # (B, S, inner_dim)

            if self.is_primary:
                # ── Predict alpha + depth from entity features ────────
                alpha0, depth0 = self.e0_branch(F_0.float())  # (B, S), (B, S)
                alpha1, depth1 = self.e1_branch(F_1.float())  # (B, S), (B, S)

                # ── Compute ownership ─────────────────────────────────
                own0, own1, own_bg = self.ownership(
                    alpha0, alpha1, depth0, depth1)  # each (B, S)

                # ── Feature-level composition ─────────────────────────
                # composed = own_bg * F_g + own0 * F_0 + own1 * F_1
                own0_e = own0.unsqueeze(-1).to(dtype)   # (B, S, 1)
                own1_e = own1.unsqueeze(-1).to(dtype)   # (B, S, 1)
                own_bg_e = own_bg.unsqueeze(-1).to(dtype)  # (B, S, 1)

                composed = (own_bg_e * F_g
                            + own0_e * F_0
                            + own1_e * F_1)  # (B, S, inner_dim)

                # ── Store predictions for loss computation ────────────
                self._last_alpha0 = alpha0
                self._last_alpha1 = alpha1
                self._last_depth0 = depth0
                self._last_depth1 = depth1
                self._last_own0   = own0
                self._last_own1   = own1
                self._last_own_bg = own_bg
                self._last_F0     = F_0
                self._last_F1     = F_1
                self._last_Fg     = F_g

                out = composed
            else:
                # Non-primary blocks: simple entity_presence gating
                entity_presence = torch.maximum(
                    F_0.float().abs().mean(dim=-1, keepdim=True),
                    F_1.float().abs().mean(dim=-1, keepdim=True),
                ).clamp(0.0, 1.0).to(dtype)  # (B, S, 1)

                # Simple average composition for non-primary
                composed = 0.5 * F_0 + 0.5 * F_1  # (B, S, inner_dim)
                blend = 0.3  # conservative blend for non-primary blocks
                out = (1.0 - blend) * F_g + blend * composed
        else:
            # No entity tokens → pure global attention
            out = F_g

        # ── Output projection with LoRA ──────────────────────────────────
        result = (attn.to_out[0](out)
                  + self.lora_out(out.float()).to(dtype=out.dtype))
        result = attn.to_out[1](result)   # dropout
        return result.to(dtype)


# =============================================================================
# Phase60Manager
# =============================================================================

class Phase60Manager:
    """
    Manages Phase60Processors across multiple attention blocks.

    Provides unified interface for:
      - Setting entity token indices
      - Accessing predictions from the primary block
      - Collecting parameters by group for optimizer
      - Reset/eval/train mode switching
    """

    def __init__(
        self,
        procs: List[Phase60Processor],
        keys:  List[str],
        primary_idx: int = 1,  # up_blocks.2 = index 1 in DEFAULT_INJECT_KEYS
    ):
        self.procs = procs
        self.keys  = keys
        self.primary_idx = primary_idx
        self.primary = procs[primary_idx]
        assert self.primary.is_primary, (
            f"Processor at index {primary_idx} ({keys[primary_idx]}) "
            f"must be the primary block"
        )

    def set_entity_tokens(
        self, toks_e0: torch.Tensor, toks_e1: torch.Tensor
    ):
        """Set entity token indices for all processors."""
        for p in self.procs:
            p.set_entity_tokens(toks_e0, toks_e1)

    def reset(self):
        """Clear stored predictions in all processors."""
        for p in self.procs:
            p.reset()

    @property
    def entity_predictions(
        self,
    ) -> Tuple[
        torch.Tensor, torch.Tensor,  # alpha0, alpha1
        torch.Tensor, torch.Tensor,  # depth0, depth1
        torch.Tensor, torch.Tensor, torch.Tensor,  # own0, own1, own_bg
        torch.Tensor, torch.Tensor, torch.Tensor,  # F_0, F_1, F_g
    ]:
        """
        Get entity predictions from the primary block.

        Returns (alpha0, alpha1, depth0, depth1,
                 own0, own1, own_bg, F_0, F_1, F_g)
        """
        p = self.primary
        return (
            p._last_alpha0, p._last_alpha1,
            p._last_depth0, p._last_depth1,
            p._last_own0, p._last_own1, p._last_own_bg,
            p._last_F0, p._last_F1, p._last_Fg,
        )

    def train(self):
        for p in self.procs:
            p.train()

    def eval(self):
        for p in self.procs:
            p.eval()

    # ── Parameter groups ─────────────────────────────────────────────────

    def all_params(self) -> List[nn.Parameter]:
        params = []
        for p in self.procs:
            params += list(p.parameters())
        return params

    def shared_lora_params(self) -> List[nn.Parameter]:
        """Shared LoRA (K, V, Out) parameters across all blocks."""
        params = []
        for p in self.procs:
            params += list(p.lora_k.parameters())
            params += list(p.lora_v.parameters())
            params += list(p.lora_out.parameters())
        return params

    def adapter_params(self) -> List[nn.Parameter]:
        """Slot adapter parameters."""
        params = []
        for p in self.procs:
            params += list(p.slot0_adapter.parameters())
            params += list(p.slot1_adapter.parameters())
        return params

    def entity_branch_params(self) -> List[nn.Parameter]:
        """Entity branch head parameters (primary block only)."""
        params = []
        params += list(self.primary.e0_branch.parameters())
        params += list(self.primary.e1_branch.parameters())
        return params

    def ownership_head_params(self) -> List[nn.Parameter]:
        """Ownership computer has no learnable params, but keep for API."""
        return []


# =============================================================================
# Injection
# =============================================================================

def inject_phase60(
    pipe,
    inject_keys:      Optional[List[str]] = None,
    adapter_rank:     int   = 64,
    lora_rank:        int   = 4,
    primary_key:      Optional[str] = None,
    depth_sharpness:  float = 10.0,
) -> Tuple[Phase60Manager, Dict]:
    """
    Inject Phase60Processors into the UNet.

    Args:
        pipe:            diffusers pipeline with UNet
        inject_keys:     attention block keys to inject (default: up_blocks.1/2/3)
        adapter_rank:    slot adapter bottleneck rank
        lora_rank:       LoRA rank for K/V/Out
        primary_key:     which block is primary (default: up_blocks.2 = inner_dim 640)
        depth_sharpness: sigmoid sharpness for depth ordering

    Returns:
        manager:    Phase60Manager with all processors
        orig_procs: original processor dict for restoration
    """
    if inject_keys is None:
        inject_keys = DEFAULT_INJECT_KEYS

    if primary_key is None:
        # Default primary: up_blocks.2 (inner_dim=640)
        primary_key = inject_keys[1]  # up_blocks.2

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

        proc = Phase60Processor(
            inner_dim=inner_dim,
            adapter_rank=adapter_rank,
            lora_rank=lora_rank,
            cross_attention_dim=CROSS_ATTN_DIM,
            is_primary=is_primary,
            depth_sharpness=depth_sharpness,
        )
        new_procs[key] = proc
        procs.append(proc)

    assert primary_idx >= 0, (
        f"Primary key {primary_key} not found in inject_keys {inject_keys}")

    unet.set_attn_processor(new_procs)

    manager = Phase60Manager(procs, inject_keys, primary_idx=primary_idx)
    return manager, orig_procs


# =============================================================================
# Checkpoint restore
# =============================================================================

def restore_phase60(
    manager: Phase60Manager,
    ckpt:    dict,
    device:  str,
) -> None:
    """
    Load checkpoint into Phase60Manager.

    Handles:
      - Phase60 native checkpoint (full state)
      - Phase52-era checkpoint (shared LoRA + adapters only)
      - Phase56 checkpoint (shared + entity LoRA, no branch heads)

    Entity branch heads are always freshly initialized (zero-init)
    when loading from older checkpoints.
    """
    procs_state = ckpt.get("procs_state", [])
    if not procs_state:
        print("[restore_p60] No procs_state in checkpoint, skipping.", flush=True)
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

        # ── Slot adapters ────────────────────────────────────────────────
        if "slot0_adapter" in state:
            p.slot0_adapter.load_state_dict(state["slot0_adapter"])
        if "slot1_adapter" in state:
            p.slot1_adapter.load_state_dict(state["slot1_adapter"])

        # ── Entity branch heads (Phase60 native only) ────────────────────
        if p.is_primary:
            if "e0_branch" in state:
                p.e0_branch.load_state_dict(state["e0_branch"])
                p.e1_branch.load_state_dict(state["e1_branch"])
                print(f"  [restore_p60] block[{i}]: loaded entity branch heads",
                      flush=True)
            else:
                print(f"  [restore_p60] block[{i}]: entity branch heads "
                      f"freshly initialized (zero-init)", flush=True)

        p.to(device)

    print(f"[restore_p60] Loaded {min(len(procs_state), len(manager.procs))} "
          f"block(s) from checkpoint.", flush=True)
