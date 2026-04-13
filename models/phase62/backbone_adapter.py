"""
Phase 62 — Backbone Feature Extractor
=======================================

Self-contained cross-attention processor that extracts F_g, F_0, F_1
from UNet cross-attention blocks. Replaces Phase40 slot processor
dependency entirely — no imports from entity_slot_phase40.py.

Contains its own:
  - LoRALayer for K/V/Out projections
  - FeatureAdapter for per-entity slot adaptation
  - Masked attention for entity-specific feature extraction

The processor hooks into UNet's attn2 (text cross-attention) blocks.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as Func


# =============================================================================
# Lightweight LoRA (self-contained, no external import)
# =============================================================================

class LoRALayer(nn.Module):
    """
    Low-rank adaptation layer.

    Δout = lora_B(lora_A(x)) * scale

    Zero-init on lora_B -> identity at start.
    """

    def __init__(self, in_f: int, out_f: int, rank: int = 4):
        super().__init__()
        self.rank = rank
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.scale = 1.0 / rank

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_f) -> (..., out_f)
        return self.lora_B(self.lora_A(x)) * self.scale


# =============================================================================
# Feature Adapter (self-contained, no external import)
# =============================================================================

class FeatureAdapter(nn.Module):
    """
    Residual adapter: Linear(dim, r) -> GELU -> Linear(r, dim).

    Zero-init output -> identity at start (x + 0 = x).
    """

    def __init__(self, dim: int, r: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, r)
        self.up = nn.Linear(r, dim)

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim) -> (..., dim)
        return x + self.up(Func.gelu(self.down(x)))


# =============================================================================
# SD1.5 UNet block dimensions
# =============================================================================

BLOCK_INNER_DIMS: Dict[str, int] = {
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor": 1280,
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor": 640,
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor": 320,
}

CROSS_ATTN_DIM = 768  # SD1.5 text encoder output dim

DEFAULT_INJECT_KEYS = list(BLOCK_INNER_DIMS.keys())


# =============================================================================
# BackboneFeatureExtractor: the cross-attention processor
# =============================================================================

class BackboneFeatureExtractor(nn.Module):
    """
    Wraps UNet cross-attention processors to extract F_g, F_0, F_1.

    Self-contained: has its own LoRA K/V/Out and slot adapters.
    Does NOT import from entity_slot_phase40.

    When called by the UNet, performs:
      1. Compute K, V with LoRA: K = attn.to_k(enc) + lora_k(enc)
      2. Compute Q from hidden states
      3. Global attention -> F_g
      4. Masked attention over entity-0 tokens -> F_0_raw
      5. Masked attention over entity-1 tokens -> F_1_raw
      6. Slot adapters: F_0 = adapter_0(F_0_raw), F_1 = adapter_1(F_1_raw)
      7. Output path follows F_g only (no old slot-blend compositing)
      8. Output projection with LoRA

    Features F_g, F_0, F_1 are stored for the volume predictor to read.
    """

    def __init__(
        self,
        inner_dim: int,
        adapter_rank: int = 64,
        lora_rank: int = 4,
        cross_attn_dim: int = CROSS_ATTN_DIM,
    ):
        super().__init__()
        self.inner_dim = inner_dim
        self.cross_attn_dim = cross_attn_dim

        # LoRA on K, V (cross-attention: in=cross_attn_dim, out=inner_dim)
        self.lora_k = LoRALayer(cross_attn_dim, inner_dim, rank=lora_rank)
        self.lora_v = LoRALayer(cross_attn_dim, inner_dim, rank=lora_rank)
        # LoRA on output projection (in=inner_dim, out=inner_dim)
        self.lora_out = LoRALayer(inner_dim, inner_dim, rank=lora_rank)

        # Per-entity slot adapters
        self.slot0_adapter = FeatureAdapter(inner_dim, r=adapter_rank)
        self.slot1_adapter = FeatureAdapter(inner_dim, r=adapter_rank)

        # Entity token positions (set before forward)
        self.toks_e0: Optional[torch.Tensor] = None  # (n_tok,) int
        self.toks_e1: Optional[torch.Tensor] = None

        # Feature store (read by volume predictor after forward)
        self.last_Fg: Optional[torch.Tensor] = None   # (B, S, D)
        self.last_F0: Optional[torch.Tensor] = None
        self.last_F1: Optional[torch.Tensor] = None

    def set_entity_tokens(
        self,
        toks_e0: torch.Tensor,  # (n_tok,) int
        toks_e1: torch.Tensor,
    ) -> None:
        """Set entity token positions for masked attention."""
        self.toks_e0 = toks_e0
        self.toks_e1 = toks_e1

    def reset_slot_store(self) -> None:
        """Clear stored features from previous forward pass."""
        self.last_Fg = None
        self.last_F0 = None
        self.last_F1 = None

    def _masked_attn(
        self,
        q_mh: torch.Tensor,      # (B, n_heads, S, head_dim)
        k_mh: torch.Tensor,      # (B, n_heads, T_seq, head_dim)
        v_mh: torch.Tensor,      # (B, n_heads, T_seq, head_dim)
        tok_idx: torch.Tensor,   # (n_tok,) int64
        T_seq: int,
        scale: float,
        B: int,
        S: int,
        n_heads: int,
        head_dim: int,
        inner_dim: int,
        fallback: torch.Tensor,  # (B, S, inner_dim)
    ) -> torch.Tensor:
        """Compute attention over a subset of tokens (entity-specific)."""
        if tok_idx is None or len(tok_idx) == 0:
            return fallback

        valid = tok_idx[(tok_idx >= 0) & (tok_idx < T_seq)]
        if len(valid) == 0:
            return fallback

        k_sub = k_mh[:, :, valid, :]   # (B, n_heads, n_tok, head_dim)
        v_sub = v_mh[:, :, valid, :]    # (B, n_heads, n_tok, head_dim)

        scores = torch.matmul(q_mh, k_sub.transpose(-1, -2)) * scale  # (B, H, S, n_tok)
        w = scores.softmax(dim=-1)                                      # (B, H, S, n_tok)
        out = torch.matmul(w, v_sub)                                    # (B, H, S, head_dim)

        return out.permute(0, 2, 1, 3).reshape(B, S, inner_dim)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,                    # (B, S, D)
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask=None,
        temb=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Cross-attention forward with LoRA and entity slot extraction.

        Args:
            attn: the diffusers Attention module (has to_q, to_k, to_v, to_out)
            hidden_states: (B, S, D) — UNet spatial hidden states
            encoder_hidden_states: (B, T_seq, D_text) — text encoder output

        Returns:
            output: (B, S, D) — processed hidden states
        """
        B, S, D = hidden_states.shape
        dtype = hidden_states.dtype
        enc_hs = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        T_seq = enc_hs.shape[1]

        inner_dim = attn.to_q.weight.shape[0]
        n_heads = attn.heads
        head_dim = inner_dim // n_heads
        scale = head_dim ** -0.5

        def _mh(x: torch.Tensor, seq_len: int) -> torch.Tensor:
            """Reshape to multi-head: (B, seq, D) -> (B, n_heads, seq, head_dim)."""
            return x.reshape(B, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)

        # ── K, V with LoRA ──────────────────────────────────────────────
        enc_hs_f = enc_hs.float()
        k = attn.to_k(enc_hs) + self.lora_k(enc_hs_f).to(dtype=enc_hs.dtype)  # (B, T, D)
        v = attn.to_v(enc_hs) + self.lora_v(enc_hs_f).to(dtype=enc_hs.dtype)  # (B, T, D)
        k_mh = _mh(k, T_seq)   # (B, n_heads, T_seq, head_dim)
        v_mh = _mh(v, T_seq)

        # ── Q ────────────────────────────────────────────────────────────
        q = attn.to_q(hidden_states)   # (B, S, D)
        q_mh = _mh(q, S)               # (B, n_heads, S, head_dim)

        # ── Global attention (F_g) ───────────────────────────────────────
        scores_g = torch.matmul(q_mh, k_mh.transpose(-1, -2)) * scale  # (B, H, S, T)
        w_g = scores_g.softmax(dim=-1)
        F_g = (torch.matmul(w_g, v_mh)
               .permute(0, 2, 1, 3)
               .reshape(B, S, inner_dim))  # (B, S, D)

        # ── Entity slot attention ────────────────────────────────────────
        F_0_raw = self._masked_attn(
            q_mh, k_mh, v_mh, self.toks_e0, T_seq,
            scale, B, S, n_heads, head_dim, inner_dim, fallback=F_g)

        F_1_raw = self._masked_attn(
            q_mh, k_mh, v_mh, self.toks_e1, T_seq,
            scale, B, S, n_heads, head_dim, inner_dim, fallback=F_g)

        # ── Slot adapters ────────────────────────────────────────────────
        F_0 = self.slot0_adapter(F_0_raw.float()).to(dtype)  # (B, S, D)
        F_1 = self.slot1_adapter(F_1_raw.float()).to(dtype)  # (B, S, D)

        # ── Store features for volume predictor ──────────────────────────
        self.last_Fg = F_g   # (B, S, D)
        self.last_F0 = F_0   # (B, S, D)
        self.last_F1 = F_1   # (B, S, D)

        # ── Output projection with LoRA ──────────────────────────────────
        # Phase62 mainline should not reintroduce phase40-style slot blending
        # into the backbone output path. We extract F_0/F_1 for topology
        # prediction, but the actual attention output follows the global stream
        # and is later conditioned by the projected 2D guide.
        out = (attn.to_out[0](F_g)
               + self.lora_out(F_g.float()).to(dtype=F_g.dtype))
        out = attn.to_out[1](out)  # dropout

        return out.to(dtype)


# =============================================================================
# Multi-block injection + manager
# =============================================================================

def inject_backbone_extractors(
    pipe,
    adapter_rank: int = 64,
    lora_rank: int = 4,
    inject_keys: Optional[List[str]] = None,
) -> Tuple[List[BackboneFeatureExtractor], Dict]:
    """
    Inject BackboneFeatureExtractor processors into UNet cross-attention blocks.

    Replaces inject_multi_block_entity_slot from phase40 — no external
    model dependencies.

    Args:
        pipe: AnimateDiffPipeline with UNet
        adapter_rank: slot adapter bottleneck dimension
        lora_rank: LoRA rank for K/V/Out
        inject_keys: list of attn processor keys to inject

    Returns:
        extractors: list of BackboneFeatureExtractor (one per inject key)
        orig_procs: dict of original attn processors (for restoration)
    """
    import copy

    if inject_keys is None:
        inject_keys = DEFAULT_INJECT_KEYS

    unet = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))
    new_procs = dict(unet.attn_processors)
    extractors: List[BackboneFeatureExtractor] = []

    for key in inject_keys:
        inner_dim = BLOCK_INNER_DIMS.get(key, 640)
        ext = BackboneFeatureExtractor(
            inner_dim=inner_dim,
            adapter_rank=adapter_rank,
            lora_rank=lora_rank,
            cross_attn_dim=CROSS_ATTN_DIM,
        )
        new_procs[key] = ext
        extractors.append(ext)

    unet.set_attn_processor(new_procs)
    return extractors, orig_procs


class BackboneManager:
    """
    Manages multiple BackboneFeatureExtractor processors across UNet blocks.

    Provides a unified interface for setting entity tokens, resetting stores,
    and collecting trainable parameters.

    primary_idx selects which extractor's features are used by the volume predictor.
    """

    def __init__(
        self,
        extractors: List[BackboneFeatureExtractor],
        keys: List[str],
        primary_idx: int = 1,
    ):
        self.extractors = extractors
        self.keys = keys
        self.primary_idx = min(primary_idx, len(extractors) - 1)

    @property
    def primary(self) -> BackboneFeatureExtractor:
        """The primary extractor whose features feed the volume predictor."""
        return self.extractors[self.primary_idx]

    def set_entity_tokens(
        self,
        toks_e0: torch.Tensor,
        toks_e1: torch.Tensor,
    ) -> None:
        """Set entity token positions on all extractors."""
        for ext in self.extractors:
            ext.set_entity_tokens(toks_e0, toks_e1)

    def reset_slot_store(self) -> None:
        """Clear stored features on all extractors."""
        for ext in self.extractors:
            ext.reset_slot_store()

    def train(self) -> None:
        """Set all extractors to training mode."""
        for ext in self.extractors:
            ext.train()

    def eval(self) -> None:
        """Set all extractors to eval mode."""
        for ext in self.extractors:
            ext.eval()

    def adapter_params(self) -> List[nn.Parameter]:
        """Collect all slot adapter parameters."""
        params = []
        for ext in self.extractors:
            params.extend(ext.slot0_adapter.parameters())
            params.extend(ext.slot1_adapter.parameters())
        return params

    def lora_params(self) -> List[nn.Parameter]:
        """Collect all LoRA parameters."""
        params = []
        for ext in self.extractors:
            params.extend(ext.lora_k.parameters())
            params.extend(ext.lora_v.parameters())
            params.extend(ext.lora_out.parameters())
        return params

    def blend_params(self) -> List[nn.Parameter]:
        """Phase62 no longer trains old slot-blend parameters."""
        return []

    def to(self, device) -> "BackboneManager":
        """Move all extractors to device."""
        for ext in self.extractors:
            ext.to(device)
        return self
