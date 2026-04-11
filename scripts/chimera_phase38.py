"""
Phase 38 — Chimera 비교 (Entity-Slot Attention vs Phase 31 baseline)
=====================================================================

entity_score = survival × (1 - chimera) 로 평가.

두 단계 비교:
  1. Phase 31 baseline (표준 VCA, additive)
  2. Phase 38 entity-slot (Porter-Duff compositing)

각 pair × 3 seeds 로 생성 후 요약.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as iio2

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.entity_slot import (
    EntitySlotAttnProcessor,
    inject_entity_slot,
    entity_score as compute_entity_score,
    entity_survival_rate,
    chimera_rate,
)
from scripts.run_animatediff import load_pipeline
from scripts.train_phase31 import (
    VCA_ALPHA, INJECT_KEY, INJECT_QUERY_DIM,
    DEPTH_PE_INIT_SCALE,
    make_color_prompts, get_color_entity_context,
    restore_procs,
    inject_vca_p21_infer,
)
from scripts.train_phase35 import (
    DEFAULT_Z_BINS, get_entity_token_positions,
)


# =============================================================================
# Config
# =============================================================================

SEEDS = [42, 123, 456]

ENTITY_PAIRS = [
    ("cat",    "dog"),
    ("lion",   "bear"),
    ("tiger",  "wolf"),
    ("snake",  "alligator"),
    ("person", "snake"),
    ("cat",    "sword"),
]


# =============================================================================
# Helpers
# =============================================================================

def make_meta(kw0: str, kw1: str) -> dict:
    return {
        "keyword0": kw0,
        "keyword1": kw1,
        "color0": [0.85, 0.15, 0.1],
        "color1": [0.1,  0.25, 0.85],
    }


def load_p38_checkpoint(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    vca_layer = VCALayer(
        query_dim=INJECT_QUERY_DIM, context_dim=768,
        n_heads=8, n_entities=2, z_bins=DEFAULT_Z_BINS, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=DEPTH_PE_INIT_SCALE,
    ).to(device)
    vca_layer.load_state_dict(ckpt["vca_state_dict"])
    slot_blend_raw = ckpt.get("slot_blend_raw", None)
    slot_blend     = float(ckpt.get("slot_blend", 0.3))
    return vca_layer, slot_blend_raw, slot_blend


def save_gif(frames: np.ndarray, path: Path, fps: int = 8):
    path.parent.mkdir(parents=True, exist_ok=True)
    iio2.mimwrite(str(path), frames, fps=fps, loop=0)


def make_chimera_overlay(frames: np.ndarray) -> np.ndarray:
    out = []
    for frame in frames:
        f = frame.copy()
        r = f[:, :, 0].astype(np.float32)
        g = f[:, :, 1].astype(np.float32)
        b = f[:, :, 2].astype(np.float32)
        is_red  = (r > 80) & (g < 120) & (b < 120)
        is_blue = (b > 80) & (r < 120) & (g < 120)
        chimera = is_red & is_blue
        f[chimera] = [255, 255, 0]
        out.append(f)
    return np.stack(out)


# =============================================================================
# Generation
# =============================================================================

@torch.no_grad()
def generate_slot(pipe, proc, entity_ctx, toks_e0, toks_e1,
                  full_prompt, seed, n_frames, n_steps, height, width,
                  device) -> np.ndarray:
    proc.set_entity_ctx(entity_ctx.float())
    proc.set_entity_tokens(toks_e0, toks_e1)
    proc.reset_slot_store()

    generator = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt=full_prompt,
        num_frames=n_frames,
        num_inference_steps=n_steps,
        height=height, width=width,
        generator=generator,
        output_type="np",
    )
    return (out.frames[0] * 255).astype(np.uint8)


@torch.no_grad()
def generate_baseline(pipe, orig_procs, vca_layer, entity_ctx,
                      full_prompt, seed, n_frames, n_steps, height, width,
                      device) -> np.ndarray:
    inject_vca_p21_infer(pipe, vca_layer, entity_ctx, VCA_ALPHA)
    generator = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt=full_prompt,
        num_frames=n_frames,
        num_inference_steps=n_steps,
        height=height, width=width,
        generator=generator,
        output_type="np",
    )
    frames = (out.frames[0] * 255).astype(np.uint8)
    restore_procs(pipe, orig_procs)
    return frames


# =============================================================================
# Main
# =============================================================================

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"[p38] 파이프라인 로드...", flush=True)
    pipe = load_pipeline(device=device, dtype=torch.float16)

    print(f"[p38] 체크포인트 로드: {args.ckpt}", flush=True)
    vca_layer, slot_blend_raw, slot_blend = load_p38_checkpoint(args.ckpt, device)
    print(f"  slot_blend={slot_blend:.4f}", flush=True)

    # inject entity slot processor (dummy entity_ctx — overridden per pair)
    dummy_meta = make_meta("cat", "dog")
    dummy_ctx  = get_color_entity_context(pipe, dummy_meta, device)
    proc, orig_procs = inject_entity_slot(
        pipe, vca_layer, dummy_ctx,
        inject_key=INJECT_KEY,
        slot_blend_init=slot_blend,
    )
    proc = proc.to(device)

    # restore trained slot_blend_raw
    if slot_blend_raw is not None:
        proc.slot_blend_raw.data.copy_(slot_blend_raw.to(device))
    proc.slot_blend_raw.requires_grad_(False)

    vca_layer.eval()
    proc.eval()

    results = []

    for kw0, kw1 in ENTITY_PAIRS:
        meta       = make_meta(kw0, kw1)
        entity_ctx = get_color_entity_context(pipe, meta, device)
        toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)
        _, _, _, c0, c1 = make_color_prompts(meta)
        label = f"{c0}_{kw0}__{c1}_{kw1}"

        print(f"\n{'='*70}", flush=True)
        print(f"[p38] {full_prompt}", flush=True)

        all_scores = []
        best_slot_es = None
        best_seed    = None
        best_frames_slot = None
        best_frames_base = None

        for seed in SEEDS:
            print(f"  seed={seed}...", end=" ", flush=True)

            # ── Phase 38 slot ────────────────────────────────────────────
            # re-inject slot processor with correct entity_ctx
            restore_procs(pipe, orig_procs)
            proc2, _ = inject_entity_slot(
                pipe, vca_layer, entity_ctx,
                inject_key=INJECT_KEY,
                slot_blend_init=slot_blend,
            )
            proc2 = proc2.to(device)
            if slot_blend_raw is not None:
                proc2.slot_blend_raw.data.copy_(slot_blend_raw.to(device))
            proc2.slot_blend_raw.requires_grad_(False)
            proc2.eval()

            frames_slot = generate_slot(
                pipe, proc2, entity_ctx, toks_e0, toks_e1,
                full_prompt, seed, args.n_frames, args.n_steps,
                args.height, args.width, device)

            # ── Phase 31 baseline ────────────────────────────────────────
            restore_procs(pipe, orig_procs)
            frames_base = generate_baseline(
                pipe, orig_procs, vca_layer, entity_ctx,
                full_prompt, seed, args.n_frames, args.n_steps,
                args.height, args.width, device)

            # restore slot proc for next iter
            restore_procs(pipe, orig_procs)
            proc, _ = inject_entity_slot(
                pipe, vca_layer, entity_ctx,
                inject_key=INJECT_KEY,
                slot_blend_init=slot_blend,
            )
            proc = proc.to(device)
            if slot_blend_raw is not None:
                proc.slot_blend_raw.data.copy_(slot_blend_raw.to(device))
            proc.slot_blend_raw.requires_grad_(False)
            proc.eval()

            es_slot, sr_slot, cr_slot = compute_entity_score(frames_slot)
            es_base, sr_base, cr_base = compute_entity_score(frames_base)

            print(
                f"base es={es_base:.4f}(sr={sr_base:.2f},cr={cr_base:.4f})  "
                f"slot es={es_slot:.4f}(sr={sr_slot:.2f},cr={cr_slot:.4f})",
                flush=True)
            all_scores.append((seed, es_base, sr_base, cr_base,
                               es_slot, sr_slot, cr_slot,
                               frames_base, frames_slot))

            if best_slot_es is None or es_slot > best_slot_es:
                best_slot_es   = es_slot
                best_seed      = seed
                best_frames_slot = frames_slot
                best_frames_base = frames_base

        # save GIFs for best seed
        save_gif(best_frames_slot, debug_dir / f"p38_{label}_best_slot.gif")
        save_gif(best_frames_base, debug_dir / f"p38_{label}_best_base.gif")
        overlay_slot = make_chimera_overlay(best_frames_slot)
        overlay_base = make_chimera_overlay(best_frames_base)
        save_gif(overlay_slot, debug_dir / f"p38_{label}_best_slot_chimera.gif")
        save_gif(overlay_base, debug_dir / f"p38_{label}_best_base_chimera.gif")

        avg_es_base = float(np.mean([s[1] for s in all_scores]))
        avg_es_slot = float(np.mean([s[4] for s in all_scores]))
        better = "✓ BETTER" if avg_es_slot > avg_es_base else "✗ WORSE"
        print(f"\n  avg base es={avg_es_base:.4f}  slot es={avg_es_slot:.4f}  {better}",
              flush=True)

        results.append({
            "pair": f"{kw0}+{kw1}",
            "prompt": full_prompt,
            "best_seed": best_seed,
            "avg_es_base": avg_es_base,
            "avg_es_slot": avg_es_slot,
            "all": [
                {"seed": s, "es_base": eb, "sr_base": srb, "cr_base": crb,
                 "es_slot": es, "sr_slot": srs, "cr_slot": crs}
                for s, eb, srb, crb, es, srs, crs, _, _ in all_scores
            ],
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"[P38] Summary — entity_score (survival × (1-chimera))", flush=True)
    print(f"{'Pair':<22} {'Base':>8} {'Slot':>8} {'Δ':>9} {'%':>7}", flush=True)
    print("-" * 65, flush=True)
    total_base = 0.0
    total_slot = 0.0
    for r in results:
        b = r['avg_es_base']
        s = r['avg_es_slot']
        delta = s - b
        pct   = (s - b) / (b + 1e-9) * 100
        better = "✓" if s > b else "✗"
        print(f"{r['pair']:<22} {b:8.4f} {s:8.4f} {delta:+9.4f} {pct:+6.1f}%  {better}",
              flush=True)
        total_base += b
        total_slot += s
    n = len(results)
    print("-" * 65, flush=True)
    print(f"{'AVERAGE':<22} {total_base/n:8.4f} {total_slot/n:8.4f} "
          f"{(total_slot-total_base)/n:+9.4f}", flush=True)
    print(f"\nGIFs saved to: {debug_dir}/", flush=True)
    return results


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      type=str, default="checkpoints/phase38/best.pt")
    p.add_argument("--debug-dir", type=str, default="debug/chimera_phase38")
    p.add_argument("--n-frames",  type=int, default=16)
    p.add_argument("--n-steps",   type=int, default=20)
    p.add_argument("--height",    type=int, default=256)
    p.add_argument("--width",     type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
