"""
Phase 32 — Chimera Reduction via Depth-aware Attention Occlusion Masking
========================================================================

Method
------
During inference, use VCA sigma values (from a trained Phase-31 VCA checkpoint)
to suppress the *back* entity's cross-attention weights at each spatial token.
This prevents text features of an occluded entity (e.g. the blue ball) from
leaking into spatial locations already claimed by the front entity (red ball),
which is one of the main sources of chimera artefacts in multi-entity video
generation.

Pipeline
--------
1. AnimateDiff (`scripts.run_animatediff.load_pipeline`)
2. Phase-31 VCA injected at `up_blocks.2.attentions.0.transformer_blocks.0.attn2`
3. Replacement processor:
     OcclusionVCAProcessor — does text cross-attention explicitly (Q,K,V)
     and pre-softmax-suppresses the loser entity's text tokens using the VCA
     sigma field cached from the previous denoising step (lagged). After the
     (occluded) text attention output is computed, the Phase-31 additive VCA
     depth-delta is added, exactly as in `AdditiveVCAInferProcessor`.
4. `OcclusionPropagator` (no VCA delta, attention-mask-only) is applied to
   `mid_block` attn2 (8×8) and `up_blocks.3` attn2 (32×32) so that the
   occlusion mask acts on *other* cross-attention layers as well. The lagged
   sigma from step 3 is bilinearly resampled to the target resolution.

Outputs
-------
In `args.debug_dir`:
  * p32_baseline_frames.gif      — raw AnimateDiff (no VCA / no occlusion)
  * p32_occlusion_frames.gif     — with VCA + occlusion masking
  * p32_chimera_mask.gif         — yellow overlay on chimera pixels
  * p32_sigma_overlay.gif        — sigma field overlay (E0 red, E1 blue)

Chimera metric
--------------
`chimera_score(frames)` returns |chimera pixels| / |entity pixels|, where
    chimera_pixel = R>80 AND B>80  (both colors co-located → chimera)
    entity_pixel  = R>80 OR  B>80
The seed with the lowest VCA-ON chimera score is picked as the winner
per prompt.

CLI
---
  --ckpt checkpoints/phase31/best.pt
  --debug-dir debug/chimera
  --n-seeds 5
  --suppression 0.0
  --height 256 --width 256
  --n-frames 8
  --n-inference-steps 20
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import layer_norm
import imageio.v2 as iio2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers.models.attention_processor import AttnProcessor2_0

from models.vca_attention import VCALayer
from scripts.run_animatediff import load_pipeline
from scripts.train_phase31 import (
    VCA_ALPHA,
    INJECT_KEY,
    INJECT_QUERY_DIM,
    ATTN_CAPTURE_KEY,
    AdditiveVCAProcessor,
    AdditiveVCAInferProcessor,
    inject_vca_p21,
    inject_vca_p21_infer,
    restore_procs,
    measure_generation_diff,
)


# =============================================================================
# Collision prompts / seeds
# =============================================================================
COLLISION_PROMPTS = [
    {
        "prompt": "a red ball and a blue ball rolling toward each other on a "
                  "wooden table, they collide in the center, cinematic lighting, "
                  "photorealistic, high quality",
        "negative": "low quality, blurry, deformed, watermark, text, chimera, "
                    "merged, fused, cartoon, painting",
        "entity0": "red ball",
        "entity1": "blue ball",
        "color0_rgb": (200, 50, 50),
        "color1_rgb": (50, 50, 200),
    },
    {
        "prompt": "a red cat and a blue cat running toward each other on a "
                  "grassy field, they meet in the middle, cinematic, "
                  "photorealistic, high detail",
        "negative": "low quality, blurry, deformed, watermark, text, chimera, "
                    "merged, fused, painting",
        "entity0": "red cat",
        "entity1": "blue cat",
        "color0_rgb": (200, 50, 50),
        "color1_rgb": (50, 50, 200),
    },
]
SEEDS = [42, 123, 456, 789, 1337]


# =============================================================================
# Chimera metric
# =============================================================================
def chimera_score(frames: list[np.ndarray]) -> float:
    """
    frames: list of (H, W, 3) uint8 arrays.

    chimera_pixel : R > 80 AND B > 80   (both colours overlap)
    overlap_pixel : R > 80 OR  B > 80   (any entity present)
    score         : chimera / overlap   ∈ [0, 1]
    """
    total_ch = 0
    total_ov = 0
    for f in frames:
        r = f[..., 0].astype(np.int32)
        b = f[..., 2].astype(np.int32)
        ch = (r > 80) & (b > 80)
        ov = (r > 80) | (b > 80)
        total_ch += int(ch.sum())
        total_ov += int(ov.sum())
    if total_ov == 0:
        return 0.0
    return total_ch / total_ov


def chimera_masks(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Return per-frame bool chimera masks (H, W)."""
    out = []
    for f in frames:
        r = f[..., 0].astype(np.int32)
        b = f[..., 2].astype(np.int32)
        out.append((r > 80) & (b > 80))
    return out


# =============================================================================
# GIF helpers
# =============================================================================
def add_label(frame_arr: np.ndarray, text: str, font_size: int = 12) -> np.ndarray:
    img = Image.fromarray(frame_arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((3, 3), text, fill=(255, 255, 255), font=font,
              stroke_width=1, stroke_fill=(0, 0, 0))
    return np.array(img)


def make_chimera_overlay(frame: np.ndarray, chimera_mask: np.ndarray,
                         alpha: float = 0.6) -> np.ndarray:
    overlay = frame.copy().astype(float)
    yellow = np.array([255, 220, 0], dtype=float)
    overlay[chimera_mask] = overlay[chimera_mask] * (1 - alpha) + yellow * alpha
    return overlay.clip(0, 255).astype(np.uint8)


def make_sigma_overlay(frame: np.ndarray,
                       sigma_e0: np.ndarray,
                       sigma_e1: np.ndarray,
                       hw: int = 16, P: int = 256, alpha: float = 0.4
                       ) -> np.ndarray:
    e0 = sigma_e0.reshape(hw, hw)
    e1 = sigma_e1.reshape(hw, hw)
    e0 = (e0 - e0.min()) / (e0.max() - e0.min() + 1e-8)
    e1 = (e1 - e1.min()) / (e1.max() - e1.min() + 1e-8)
    e0_up = np.array(
        Image.fromarray((e0 * 255).astype(np.uint8)).resize((P, P), Image.BILINEAR)
    ) / 255.0
    e1_up = np.array(
        Image.fromarray((e1 * 255).astype(np.uint8)).resize((P, P), Image.BILINEAR)
    ) / 255.0
    f = frame.astype(float) / 255.0
    overlay = f + alpha * np.stack(
        [e0_up, np.zeros_like(e0_up), e1_up], axis=-1
    )
    return (overlay.clip(0, 1) * 255).astype(np.uint8)


# =============================================================================
# Text-token index finder
# =============================================================================
def find_entity_tokens(tokenizer, full_prompt: str, entity_text: str) -> list[int]:
    full_ids = tokenizer(full_prompt, add_special_tokens=True)["input_ids"]
    kw_ids = tokenizer(entity_text, add_special_tokens=False)["input_ids"]
    for i in range(len(full_ids) - len(kw_ids) + 1):
        if full_ids[i:i + len(kw_ids)] == kw_ids:
            return list(range(i, i + len(kw_ids)))
    # Fallback: individual words
    for word in entity_text.split():
        wids = tokenizer(word, add_special_tokens=False)["input_ids"]
        for i in range(len(full_ids) - len(wids) + 1):
            if full_ids[i:i + len(wids)] == wids:
                return list(range(i, i + len(wids)))
    return [1]


def get_entity_ctx_simple(pipe, entity0_text: str, entity1_text: str, device: str
                          ) -> torch.Tensor:
    """
    Color-qualified CLIP mean-pooled embeddings for two entities.
    Returns (1, 2, 768) float32 on `device`.
    """
    embs = []
    for text in [entity0_text, entity1_text]:
        tokens = pipe.tokenizer(
            text, return_tensors="pt",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            out = pipe.text_encoder(**tokens).last_hidden_state  # (1, 77, 768)
        ids = tokens.input_ids[0]
        mask = (ids != pipe.tokenizer.pad_token_id) & (ids != pipe.tokenizer.eos_token_id)
        mask[0] = False  # drop BOS
        emb = out[0][mask].mean(0)
        embs.append(emb)
    return torch.stack(embs, dim=0).unsqueeze(0).float().to(device)  # (1, 2, 768)


# =============================================================================
# VCA checkpoint loader
# =============================================================================
def load_vca_checkpoint(ckpt_path: str, pipe, device: str):
    """
    Loads Phase-31 best.pt → VCALayer (on device, eval mode) + gamma_trained.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    vca_layer = VCALayer(
        query_dim=INJECT_QUERY_DIM,
        context_dim=768,
        n_heads=8,
        n_entities=2,
        z_bins=2,
        lora_rank=8,
        use_softmax=False,
        depth_pe_init_scale=0.3,
    ).to(device)
    vca_layer.load_state_dict(ckpt["vca_state_dict"])
    vca_layer.eval()
    gamma_trained = float(ckpt.get("gamma_trained", VCA_ALPHA))
    print(f"[phase32] loaded VCA ckpt: {ckpt_path} | gamma_trained={gamma_trained:.4f}",
          flush=True)
    return vca_layer, gamma_trained


# =============================================================================
# Occlusion processors
# =============================================================================
class OcclusionVCAProcessor:
    """
    Replaces AdditiveVCAInferProcessor with:
      1. Explicit Q/K/V cross-attention
      2. Pre-softmax suppression of the *back* entity's text-token columns,
         driven by the VCA sigma field cached from the previous step (lagged).
      3. Post-softmax, the Phase-31 additive VCA depth-delta is added exactly
         like AdditiveVCAInferProcessor.

    Sigma cache shape: (BF, S=256, N=2, Z=2); we use sigma[:, :, :, 0] (near bin).
        sigma[fi, s, 0] > sigma[fi, s, 1]  → E0 in front at (fi, s)
                                              → suppress E1 text cols
        else                                → suppress E0 text cols
    """
    def __init__(self, vca_layer: VCALayer, entity_ctx: torch.Tensor,
                 orig_processor: AttnProcessor2_0,
                 tok_e0: list[int], tok_e1: list[int],
                 gamma_trained: float,
                 suppression: float = 0.0,
                 spatial_hw: int = 16):
        self.vca = vca_layer
        self.ctx = entity_ctx           # (1, 2, 768) fp32 on device
        self.orig = orig_processor      # kept only as a fallback
        self.tok_e0 = list(tok_e0)
        self.tok_e1 = list(tok_e1)
        self.gamma_trained = float(gamma_trained)
        self.suppression = float(suppression)
        self.spatial_hw = spatial_hw
        # Lagged sigma cache: (BF, S, N, Z) or None on the first forward
        self.cached_sigma: Optional[torch.Tensor] = None
        self.last_chimera_score: Optional[float] = None
        self.call_count = 0

    # ------------------------------------------------------------------ utils
    def _build_column_scale(self, scores_shape, device, dtype,
                             sigma_for_layer: torch.Tensor,
                             nheads: int):
        """
        Build a multiplicative mask over attention `scores`:
            scale[bh, s, t] ∈ {1, suppression}

        scores_shape : (BF*nheads, S, T)   (head-batched)
        sigma_for_layer : (BF, S, N, Z) — already resampled to this layer's S.
        """
        BFh, S, T = scores_shape
        BF = BFh // nheads
        scale = torch.ones(scores_shape, device=device, dtype=dtype)

        # Near-bin sigma per entity
        sig_e0 = sigma_for_layer[:, :, 0, 0]  # (BF, S)
        sig_e1 = sigma_for_layer[:, :, 1, 0]  # (BF, S)

        # Where E0 wins we suppress E1 tokens; elsewhere suppress E0 tokens.
        e0_wins = sig_e0 > sig_e1          # (BF, S) bool
        e1_wins = ~e0_wins

        tok_e0 = [t for t in self.tok_e0 if 0 <= t < T]
        tok_e1 = [t for t in self.tok_e1 if 0 <= t < T]

        if len(tok_e0) == 0 and len(tok_e1) == 0:
            return scale

        # Expand per-head: repeat the (BF, S) mask `nheads` times along batch.
        # scale layout (BF, nheads, S, T) → viewed as (BF*nheads, S, T)
        scale4 = scale.view(BF, nheads, S, T)
        supp = self.suppression
        if len(tok_e1) > 0:
            # suppress E1 columns at positions where E0 wins
            idx_e1 = torch.tensor(tok_e1, device=device, dtype=torch.long)
            mask = e0_wins.unsqueeze(1).unsqueeze(-1)         # (BF,1,S,1)
            col_mask = torch.zeros(T, device=device, dtype=torch.bool)
            col_mask[idx_e1] = True                            # (T,)
            col_mask = col_mask.view(1, 1, 1, T)
            apply = mask & col_mask                            # (BF,1,S,T)
            scale4 = torch.where(apply, torch.full_like(scale4, supp), scale4)
        if len(tok_e0) > 0:
            idx_e0 = torch.tensor(tok_e0, device=device, dtype=torch.long)
            mask = e1_wins.unsqueeze(1).unsqueeze(-1)
            col_mask = torch.zeros(T, device=device, dtype=torch.bool)
            col_mask[idx_e0] = True
            col_mask = col_mask.view(1, 1, 1, T)
            apply = mask & col_mask
            scale4 = torch.where(apply, torch.full_like(scale4, supp), scale4)

        return scale4.view(BFh, S, T)

    # ------------------------------------------------------------------ call
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        self.call_count += 1
        BF, S, D = hidden_states.shape
        device = hidden_states.device
        in_dtype = hidden_states.dtype

        # --- residual / group norm (as in AttnProcessor2_0) -------------------
        residual = hidden_states
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if getattr(attn, "group_norm", None) is not None:
            hs_gn = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        else:
            hs_gn = hidden_states

        if encoder_hidden_states is None:
            encoder_hidden_states = hs_gn
        if getattr(attn, "norm_cross", None) is not None:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # --- Q, K, V (head-batched) ------------------------------------------
        q = attn.to_q(hs_gn)
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)
        q = attn.head_to_batch_dim(q)   # (BF*nH, S, d)
        k = attn.head_to_batch_dim(k)   # (BF*nH, T, d)
        v = attn.head_to_batch_dim(v)   # (BF*nH, T, d)
        nheads = attn.heads
        T = k.shape[1]

        # --- raw pre-softmax scores in fp32 for stability --------------------
        qf = q.float()
        kf = k.float()
        scores = torch.baddbmm(
            torch.empty(qf.shape[0], qf.shape[1], kf.shape[1],
                        device=device, dtype=torch.float32),
            qf, kf.transpose(-1, -2),
            beta=0.0, alpha=float(attn.scale),
        )  # (BF*nH, S, T)

        # --- occlusion: multiplicative scale on scores -----------------------
        if self.cached_sigma is not None:
            sig_layer = self.cached_sigma.to(device=device, dtype=torch.float32)
            # sig_layer spatial S should already match this layer (it is the
            # sigma produced *by* this same layer on the previous step).
            if sig_layer.shape[1] == S:
                col_scale = self._build_column_scale(
                    scores.shape, device, scores.dtype, sig_layer, nheads,
                )
                # scores <0 normally; suppression==0 collapses softmax mass
                # away from these columns by multiplying pre-softmax values
                # by 0 (preserving sign) then softmax; equivalently we can
                # set them to -inf when suppression==0.
                if self.suppression == 0.0:
                    neg_inf = torch.finfo(scores.dtype).min
                    scores = torch.where(col_scale == 0.0,
                                         torch.full_like(scores, neg_inf),
                                         scores)
                else:
                    scores = scores * col_scale

        # --- softmax & output -------------------------------------------------
        attn_probs = torch.softmax(scores, dim=-1).to(v.dtype)
        text_out_occl = torch.bmm(attn_probs, v)             # (BF*nH, S, d)
        text_out_occl = attn.batch_to_head_dim(text_out_occl)  # (BF, S, D)
        text_out_occl = attn.to_out[0](text_out_occl)
        text_out_occl = attn.to_out[1](text_out_occl)

        if getattr(attn, "residual_connection", False):
            text_out_occl = text_out_occl + residual
        rescale = getattr(attn, "rescale_output_factor", 1.0)
        if rescale != 1.0:
            text_out_occl = text_out_occl / rescale

        # --- Phase-31 additive VCA depth-delta -------------------------------
        ctx = self.ctx.expand(BF, -1, -1).float()
        x = layer_norm(residual.float(), [INJECT_QUERY_DIM])
        vca_out = self.vca(x, ctx)                 # updates last_sigma
        delta_raw = vca_out - x
        text_mag = text_out_occl.float().abs().mean() + 1e-8
        delta_mag = delta_raw.abs().mean() + 1e-8
        vca_delta = delta_raw * (text_mag / delta_mag) * self.gamma_trained

        out_final = text_out_occl + vca_delta.to(text_out_occl.dtype)

        # --- cache sigma for NEXT step (lagged) ------------------------------
        if self.vca.last_sigma is not None:
            self.cached_sigma = self.vca.last_sigma.detach().clone()

        return out_final.to(in_dtype)


class OcclusionPropagator:
    """
    Attention-mask-only propagator for other cross-attention layers.
    Reuses the sigma cache from an OcclusionVCAProcessor (the "sigma provider")
    and bilinearly resamples it to this layer's spatial resolution.
    """
    def __init__(self, orig_processor: AttnProcessor2_0,
                 sigma_provider: "OcclusionVCAProcessor",
                 target_hw: int,
                 tok_e0: list[int], tok_e1: list[int],
                 suppression: float = 0.0):
        self.orig = orig_processor
        self.provider = sigma_provider
        self.target_hw = int(target_hw)
        self.tok_e0 = list(tok_e0)
        self.tok_e1 = list(tok_e1)
        self.suppression = float(suppression)
        self.call_count = 0

    # ------------------------------------------------------------------ utils
    def _resample_sigma(self, sig_src: torch.Tensor, target_S: int
                        ) -> Optional[torch.Tensor]:
        """
        sig_src: (BF, S_src, N, Z)  → (BF, target_S, N, Z) via bilinear upsample.
        """
        BF, S_src, N, Z = sig_src.shape
        src_hw = int(round(math.sqrt(S_src)))
        tgt_hw = int(round(math.sqrt(target_S)))
        if src_hw * src_hw != S_src or tgt_hw * tgt_hw != target_S:
            return None
        if src_hw == tgt_hw:
            return sig_src
        # (BF, S, N, Z) → (BF, N*Z, hw, hw)
        x = sig_src.permute(0, 2, 3, 1).reshape(BF, N * Z, src_hw, src_hw)
        x = F.interpolate(x, size=(tgt_hw, tgt_hw), mode="bilinear",
                          align_corners=False)
        x = x.reshape(BF, N, Z, tgt_hw * tgt_hw).permute(0, 3, 1, 2).contiguous()
        return x  # (BF, target_S, N, Z)

    # ------------------------------------------------------------------ call
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        self.call_count += 1

        # If we have no sigma cache, just pass through.
        sig_cache = self.provider.cached_sigma
        if sig_cache is None:
            return self.orig(attn, hidden_states, encoder_hidden_states,
                             attention_mask, temb, *args, **kwargs)

        BF, S, D = hidden_states.shape
        device = hidden_states.device
        in_dtype = hidden_states.dtype

        sig_layer = self._resample_sigma(sig_cache.to(device).float(), S)
        if sig_layer is None:
            return self.orig(attn, hidden_states, encoder_hidden_states,
                             attention_mask, temb, *args, **kwargs)

        # --- replicate AttnProcessor2_0 core plus pre-softmax suppression ----
        residual = hidden_states
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if getattr(attn, "group_norm", None) is not None:
            hs_gn = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        else:
            hs_gn = hidden_states

        if encoder_hidden_states is None:
            encoder_hidden_states = hs_gn
        if getattr(attn, "norm_cross", None) is not None:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        q = attn.head_to_batch_dim(attn.to_q(hs_gn))
        k = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
        v = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))
        nheads = attn.heads
        T = k.shape[1]

        qf, kf = q.float(), k.float()
        scores = torch.baddbmm(
            torch.empty(qf.shape[0], qf.shape[1], kf.shape[1],
                        device=device, dtype=torch.float32),
            qf, kf.transpose(-1, -2),
            beta=0.0, alpha=float(attn.scale),
        )

        # Build column scale using the provider's logic (same rules).
        sig_e0 = sig_layer[:, :, 0, 0]
        sig_e1 = sig_layer[:, :, 1, 0]
        e0_wins = sig_e0 > sig_e1
        e1_wins = ~e0_wins

        BFh = scores.shape[0]
        scale = torch.ones_like(scores)
        scale4 = scale.view(BF, nheads, S, T)

        tok_e0 = [t for t in self.tok_e0 if 0 <= t < T]
        tok_e1 = [t for t in self.tok_e1 if 0 <= t < T]

        if len(tok_e1) > 0:
            idx_e1 = torch.tensor(tok_e1, device=device, dtype=torch.long)
            col = torch.zeros(T, device=device, dtype=torch.bool)
            col[idx_e1] = True
            col = col.view(1, 1, 1, T)
            apply = e0_wins.unsqueeze(1).unsqueeze(-1) & col
            scale4 = torch.where(apply,
                                 torch.full_like(scale4, self.suppression),
                                 scale4)
        if len(tok_e0) > 0:
            idx_e0 = torch.tensor(tok_e0, device=device, dtype=torch.long)
            col = torch.zeros(T, device=device, dtype=torch.bool)
            col[idx_e0] = True
            col = col.view(1, 1, 1, T)
            apply = e1_wins.unsqueeze(1).unsqueeze(-1) & col
            scale4 = torch.where(apply,
                                 torch.full_like(scale4, self.suppression),
                                 scale4)

        col_scale = scale4.view(BFh, S, T)
        if self.suppression == 0.0:
            neg_inf = torch.finfo(scores.dtype).min
            scores = torch.where(col_scale == 0.0,
                                 torch.full_like(scores, neg_inf),
                                 scores)
        else:
            scores = scores * col_scale

        attn_probs = torch.softmax(scores, dim=-1).to(v.dtype)
        out = torch.bmm(attn_probs, v)
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if getattr(attn, "residual_connection", False):
            out = out + residual
        rescale = getattr(attn, "rescale_output_factor", 1.0)
        if rescale != 1.0:
            out = out / rescale
        return out.to(in_dtype)


# =============================================================================
# Injection utility
# =============================================================================
def inject_occlusion(pipe, vca_layer: VCALayer, entity_ctx: torch.Tensor,
                     tok_e0: list[int], tok_e1: list[int],
                     gamma_trained: float, suppression: float):
    """
    Set up OcclusionVCAProcessor at INJECT_KEY, and OcclusionPropagator at
    mid_block attn2 and up_blocks.3 attn2. Returns (procs_backup, occl_proc).
    """
    unet = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))
    new_procs = dict(orig_procs)

    occl = OcclusionVCAProcessor(
        vca_layer=vca_layer, entity_ctx=entity_ctx,
        orig_processor=orig_procs[INJECT_KEY],
        tok_e0=tok_e0, tok_e1=tok_e1,
        gamma_trained=gamma_trained,
        suppression=suppression,
        spatial_hw=16,
    )
    new_procs[INJECT_KEY] = occl

    propagator_keys = []
    for key in orig_procs.keys():
        if "attn2" not in key:
            continue
        if key == INJECT_KEY:
            continue
        if ("mid_block" in key) or ("up_blocks.3" in key):
            target_hw = 8 if "mid_block" in key else 32
            new_procs[key] = OcclusionPropagator(
                orig_processor=orig_procs[key],
                sigma_provider=occl,
                target_hw=target_hw,
                tok_e0=tok_e0, tok_e1=tok_e1,
                suppression=suppression,
            )
            propagator_keys.append(key)

    unet.set_attn_processor(new_procs)
    print(f"[phase32] occlusion injected at INJECT_KEY + {len(propagator_keys)} "
          f"propagator layers", flush=True)
    return orig_procs, occl


# =============================================================================
# Generation helper
# =============================================================================
def _generate(pipe, prompt: str, negative_prompt: str,
              num_frames: int, steps: int, cfg: float,
              height: int, width: int, seed: int) -> list[np.ndarray]:
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=cfg,
        height=height, width=width,
        generator=generator,
        output_type="pil",
    )
    return [np.array(f) for f in out.frames[0]]


# =============================================================================
# Per-prompt runner
# =============================================================================
def run_one_prompt(pipe, vca_layer, gamma_trained, prompt_cfg: dict,
                    args, debug_dir: Path):
    device = str(pipe.device)
    full_prompt = prompt_cfg["prompt"]
    neg_prompt = prompt_cfg["negative"]
    e0_text = prompt_cfg["entity0"]
    e1_text = prompt_cfg["entity1"]

    print(f"\n{'='*70}", flush=True)
    print(f"[phase32] prompt: {full_prompt}", flush=True)

    # Entity tokens
    tok_e0 = find_entity_tokens(pipe.tokenizer, full_prompt, e0_text)
    tok_e1 = find_entity_tokens(pipe.tokenizer, full_prompt, e1_text)
    print(f"  tok_e0={tok_e0}  tok_e1={tok_e1}", flush=True)

    # Entity context for VCA
    entity_ctx = get_entity_ctx_simple(pipe, e0_text, e1_text, device)

    best = {
        "seed": None, "score_base": None, "score_occl": None,
        "frames_base": None, "frames_occl": None,
    }

    n_seeds = min(args.n_seeds, len(SEEDS))
    for seed in SEEDS[:n_seeds]:
        # --- Baseline (no VCA / no occlusion) ----------------------------
        print(f"  [seed={seed}] baseline generation …", flush=True)
        frames_base = _generate(pipe, full_prompt, neg_prompt,
                                 args.n_frames, args.n_inference_steps,
                                 7.5, args.height, args.width, seed)
        score_base = chimera_score(frames_base)
        print(f"    baseline chimera_score = {score_base:.4f}", flush=True)

        # --- With occlusion VCA ------------------------------------------
        orig_procs, occl = inject_occlusion(
            pipe, vca_layer, entity_ctx, tok_e0, tok_e1,
            gamma_trained, args.suppression,
        )
        try:
            print(f"  [seed={seed}] occlusion generation …", flush=True)
            frames_occl = _generate(pipe, full_prompt, neg_prompt,
                                     args.n_frames, args.n_inference_steps,
                                     7.5, args.height, args.width, seed)
        finally:
            restore_procs(pipe, orig_procs)
        score_occl = chimera_score(frames_occl)
        occl.last_chimera_score = score_occl
        print(f"    occlusion chimera_score = {score_occl:.4f} "
              f"(calls={occl.call_count})", flush=True)

        if (best["score_occl"] is None) or (score_occl < best["score_occl"]):
            best = {
                "seed": seed,
                "score_base": score_base,
                "score_occl": score_occl,
                "frames_base": frames_base,
                "frames_occl": frames_occl,
                "sigma_cache": (occl.cached_sigma.detach().cpu().numpy()
                                if occl.cached_sigma is not None else None),
            }

    assert best["frames_base"] is not None, "no generations?!"
    print(f"  [phase32] best seed={best['seed']} "
          f"base={best['score_base']:.4f} occl={best['score_occl']:.4f}", flush=True)

    # --- Save GIFs -------------------------------------------------------
    label = f"{e0_text}__vs__{e1_text}".replace(" ", "_")
    base_gif = debug_dir / f"p32_{label}_baseline_frames.gif"
    occl_gif = debug_dir / f"p32_{label}_occlusion_frames.gif"
    chim_gif = debug_dir / f"p32_{label}_chimera_mask.gif"
    sig_gif = debug_dir / f"p32_{label}_sigma_overlay.gif"

    base_labeled = [
        add_label(f, f"BASE s{i} chim={best['score_base']:.2f}")
        for i, f in enumerate(best["frames_base"])
    ]
    occl_labeled = [
        add_label(f, f"OCCL s{i} chim={best['score_occl']:.2f}")
        for i, f in enumerate(best["frames_occl"])
    ]
    iio2.mimsave(str(base_gif), base_labeled, duration=120)
    iio2.mimsave(str(occl_gif), occl_labeled, duration=120)
    print(f"  wrote {base_gif.name}, {occl_gif.name}", flush=True)

    # Chimera mask overlay (occlusion run)
    chim_masks = chimera_masks(best["frames_occl"])
    H, W = best["frames_occl"][0].shape[:2]
    chim_frames = []
    for f, m in zip(best["frames_occl"], chim_masks):
        # resample mask if needed (shouldn't be)
        if m.shape[:2] != (H, W):
            m = np.array(Image.fromarray(m.astype(np.uint8) * 255)
                         .resize((W, H), Image.NEAREST)) > 128
        chim_frames.append(make_chimera_overlay(f, m))
    iio2.mimsave(str(chim_gif), chim_frames, duration=120)
    print(f"  wrote {chim_gif.name}", flush=True)

    # Sigma overlay (from the cached sigma at 16×16)
    sig_np = best["sigma_cache"]
    if sig_np is not None and sig_np.shape[1] == 16 * 16:
        BF = sig_np.shape[0]
        nF = len(best["frames_occl"])
        # Take the last nF frames (CFG → cond frames are trailing).
        take = sig_np[-nF:]  # (nF, 256, 2, 2)
        sigma_frames = []
        for fi in range(nF):
            se0 = take[fi, :, 0, 0]
            se1 = take[fi, :, 1, 0]
            sf = make_sigma_overlay(best["frames_occl"][fi], se0, se1,
                                    hw=16, P=H, alpha=0.4)
            sigma_frames.append(sf)
        iio2.mimsave(str(sig_gif), sigma_frames, duration=120)
        print(f"  wrote {sig_gif.name}", flush=True)
    else:
        print("  [warn] sigma cache unavailable, skipping sigma overlay",
              flush=True)

    return {
        "prompt": full_prompt,
        "best_seed": best["seed"],
        "score_base": best["score_base"],
        "score_occl": best["score_occl"],
    }


# =============================================================================
# Main
# =============================================================================
def run_phase32(args):
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility seeds
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pipe = load_pipeline(device=device, dtype=torch.float16)
    vca_layer, gamma_trained = load_vca_checkpoint(args.ckpt, pipe, device)

    summary = []
    for p_cfg in COLLISION_PROMPTS:
        stats = run_one_prompt(pipe, vca_layer, gamma_trained, p_cfg,
                                args, debug_dir)
        summary.append(stats)

    print("\n===== Phase 32 summary =====", flush=True)
    for s in summary:
        print(f"  {s['prompt'][:60]}…  seed={s['best_seed']}  "
              f"base={s['score_base']:.4f}  occl={s['score_occl']:.4f}",
              flush=True)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/phase31/best.pt")
    p.add_argument("--debug-dir", type=str, default="debug/chimera")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--suppression", type=float, default=0.0)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--n-inference-steps", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    run_phase32(_parse_args())
