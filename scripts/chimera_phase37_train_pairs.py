"""
Phase 37 — Training Pair 생성 및 Chimera 비교
================================================

실제 학습에 사용된 entity pair들로 Phase 37 volumetric VCA 와
Phase 31 standard VCA 를 비교한다.

사용 entity pairs (학습 데이터에서 선택):
  cat   + dog
  lion  + bear
  tiger + wolf
  snake + alligator
  person + snake
  cat   + sword

각 pair 에 대해:
  - Phase 31 baseline (표준 VCA)
  - Phase 37 vol VCA (z_pe 학습 완료)
  을 동일 seed 로 생성 후 chimera score 비교.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import imageio.v2 as iio2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.vca_volumetric import VolumetricTextCrossAttentionProcessor
from scripts.run_animatediff import load_pipeline
from scripts.train_phase31 import (
    VCA_ALPHA,
    INJECT_KEY,
    INJECT_QUERY_DIM,
    DEPTH_PE_INIT_SCALE,
    AdditiveVCAInferProcessor,
    inject_vca_p21_infer,
    restore_procs,
    make_color_prompts,
    get_color_entity_context,
)
from scripts.train_phase35 import (
    DEFAULT_Z_BINS,
    DEFAULT_GAMMA_INIT,
    inject_vca_p35,
    get_entity_token_positions,
)


# =============================================================================
# Config
# =============================================================================

SEEDS = [42, 123, 456]

# 학습 데이터에서 선택한 대표 entity pair
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
    """Entity pair → meta dict (색상 고정: red/blue)."""
    return {
        "keyword0": kw0,
        "keyword1": kw1,
        "color0": [0.85, 0.15, 0.1],   # red
        "color1": [0.1,  0.25, 0.85],   # blue
    }


def load_p37_checkpoint(ckpt_path: str, device: str):
    """Phase 37 체크포인트 로드."""
    ckpt = torch.load(ckpt_path, map_location=device)
    vca_layer = VCALayer(
        query_dim=INJECT_QUERY_DIM, context_dim=768,
        n_heads=8, n_entities=2, z_bins=DEFAULT_Z_BINS, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=DEPTH_PE_INIT_SCALE,
    ).to(device)
    vca_layer.load_state_dict(ckpt["vca_state_dict"])

    z_pe_state     = ckpt.get("z_pe")       # (Z, D) tensor
    gamma_val      = float(ckpt.get("gamma_trained", 1.0))
    return vca_layer, z_pe_state, gamma_val


def chimera_score(frames_rgb: np.ndarray) -> float:
    """
    Chimera score: 픽셀 중 R>80, G<80, B<80 (red) 이면서
    동시에 R<80, G<80, B>80 (blue) 인 것은 없으므로,
    per-frame 에서 (R>80 AND B>80) / (R>80 OR B>80) 비율.
    """
    scores = []
    for frame in frames_rgb:
        r = frame[:, :, 0].astype(np.float32)
        g = frame[:, :, 1].astype(np.float32)
        b = frame[:, :, 2].astype(np.float32)
        is_red  = (r > 80) & (g < 120) & (b < 120)
        is_blue = (b > 80) & (r < 120) & (g < 120)
        both = (is_red & is_blue).sum()
        either = (is_red | is_blue).sum()
        scores.append(float(both) / (float(either) + 1e-6))
    return float(np.mean(scores))


@torch.no_grad()
def generate_vol(pipe, proc, entity_ctx, toks_e0, toks_e1,
                 full_prompt, seed, n_frames, n_steps, height, width,
                 device) -> np.ndarray:
    """Phase 37 volumetric VCA 로 생성."""
    proc.set_entity_ctx(entity_ctx.float())
    proc.reset_sigma_acc()

    generator = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt=full_prompt,
        num_frames=n_frames,
        num_inference_steps=n_steps,
        height=height,
        width=width,
        generator=generator,
        output_type="np",
    )
    frames = (out.frames[0] * 255).astype(np.uint8)  # (T, H, W, 3)
    return frames


@torch.no_grad()
def generate_baseline(pipe, orig_procs, vca_layer, entity_ctx,
                      full_prompt, seed, n_frames, n_steps, height, width,
                      device) -> np.ndarray:
    """Phase 31 standard VCA (baseline) 로 생성."""
    # inject standard VCA
    proc31, _ = inject_vca_p21_infer(pipe, vca_layer, entity_ctx, VCA_ALPHA)
    generator = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt=full_prompt,
        num_frames=n_frames,
        num_inference_steps=n_steps,
        height=height,
        width=width,
        generator=generator,
        output_type="np",
    )
    frames = (out.frames[0] * 255).astype(np.uint8)
    # restore volumetric processor
    restore_procs(pipe, orig_procs)
    return frames


def save_gif(frames: np.ndarray, path: Path, fps: int = 8):
    path.parent.mkdir(parents=True, exist_ok=True)
    iio2.mimwrite(str(path), frames, fps=fps, loop=0)


def make_chimera_overlay(frames: np.ndarray) -> np.ndarray:
    """빨간/파란 픽셀이 겹치는 곳에 노란 오버레이."""
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
# Main
# =============================================================================

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"[p37-pairs] 파이프라인 로드...", flush=True)
    pipe = load_pipeline(device=device, dtype=torch.float16)

    print(f"[p37-pairs] 체크포인트 로드: {args.ckpt}", flush=True)
    vca_layer, z_pe_state, gamma_val = load_p37_checkpoint(args.ckpt, device)
    print(f"  gamma={gamma_val:.4f}  |z_pe|={z_pe_state.norm().item():.4f}", flush=True)

    # inject volumetric processor (dummy entity_ctx — overridden per pair)
    dummy_meta = make_meta("cat", "dog")
    dummy_ctx  = get_color_entity_context(pipe, dummy_meta, device)
    proc, orig_procs = inject_vca_p35(
        pipe, vca_layer, dummy_ctx,
        gamma_init=gamma_val, z_bins=DEFAULT_Z_BINS,
    )
    # load trained z_pe & gamma
    if z_pe_state is not None:
        proc.z_pe.data.copy_(z_pe_state.to(device))
    proc.gamma.data.fill_(gamma_val)
    proc.gamma.requires_grad_(False)
    proc.z_pe.requires_grad_(False)

    vca_layer.eval()
    proc.eval()

    results = []

    for kw0, kw1 in ENTITY_PAIRS:
        meta = make_meta(kw0, kw1)
        entity_ctx = get_color_entity_context(pipe, meta, device)
        toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)
        _, _, full_prompt_check, c0, c1 = make_color_prompts(meta)
        label = f"{c0}_{kw0}__{c1}_{kw1}"

        print(f"\n{'='*70}", flush=True)
        print(f"[p37] {full_prompt}", flush=True)
        print(f"  token positions: e0={toks_e0}, e1={toks_e1}", flush=True)
        print(f"  gamma={gamma_val:.4f}  |z_pe|={proc.z_pe.norm().item():.4f}", flush=True)

        best_base  = None   # lowest baseline (for "best seed")
        worst_base = None   # highest baseline (for "hardest seed" = most chimera)
        best_seed  = None
        worst_seed = None
        all_scores = []

        for seed in SEEDS:
            print(f"  seed={seed}...", end=" ", flush=True)

            # ── Phase 37 vol VCA ─────────────────────────────────────────
            proc.set_entity_ctx(entity_ctx.float())
            restore_procs(pipe, orig_procs)
            # re-inject volumetric
            proc2, _ = inject_vca_p35(pipe, vca_layer, entity_ctx,
                                      gamma_init=gamma_val, z_bins=DEFAULT_Z_BINS)
            proc2.z_pe.data.copy_(proc.z_pe.data)
            proc2.gamma.data.fill_(gamma_val)
            proc2.set_entity_ctx(entity_ctx.float())
            proc2.reset_sigma_acc()

            frames_vol = generate_vol(
                pipe, proc2, entity_ctx, toks_e0, toks_e1,
                full_prompt, seed, args.n_frames, args.n_steps,
                args.height, args.width, device)

            # ── Phase 31 baseline ─────────────────────────────────────────
            restore_procs(pipe, orig_procs)
            inject_vca_p21_infer(pipe, vca_layer, entity_ctx, VCA_ALPHA)
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                out31 = pipe(
                    prompt=full_prompt,
                    num_frames=args.n_frames, num_inference_steps=args.n_steps,
                    height=args.height, width=args.width,
                    generator=generator, output_type="np",
                )
            frames_base = (out31.frames[0] * 255).astype(np.uint8)
            restore_procs(pipe, orig_procs)

            # re-inject vol proc for next iter
            proc, _ = inject_vca_p35(pipe, vca_layer, entity_ctx,
                                     gamma_init=gamma_val, z_bins=DEFAULT_Z_BINS)
            proc.z_pe.data.copy_(z_pe_state.to(device))
            proc.gamma.data.fill_(gamma_val)
            proc.gamma.requires_grad_(False)
            proc.z_pe.requires_grad_(False)

            score_base = chimera_score(frames_base)
            score_vol  = chimera_score(frames_vol)

            print(f"base={score_base:.4f}  vol={score_vol:.4f}", flush=True)
            all_scores.append((seed, score_base, score_vol, frames_base, frames_vol))

            if best_base is None or score_base < best_base:
                best_base = score_base
                best_seed = seed
                best_frames_base = frames_base
                best_frames_vol  = frames_vol
                best_vol_score   = score_vol

            if worst_base is None or score_base > worst_base:
                worst_base = score_base
                worst_seed = seed
                worst_frames_base = frames_base
                worst_frames_vol  = frames_vol
                worst_vol_score   = score_vol

        # save GIFs for BOTH: best seed (lowest baseline) + worst seed (highest baseline)
        save_gif(best_frames_vol,   debug_dir / f"p37_{label}_best_seed_vol.gif")
        save_gif(best_frames_base,  debug_dir / f"p37_{label}_best_seed_baseline.gif")
        save_gif(worst_frames_vol,  debug_dir / f"p37_{label}_worst_seed_vol.gif")
        save_gif(worst_frames_base, debug_dir / f"p37_{label}_worst_seed_baseline.gif")
        overlay_worst_vol  = make_chimera_overlay(worst_frames_vol)
        overlay_worst_base = make_chimera_overlay(worst_frames_base)
        save_gif(overlay_worst_vol,  debug_dir / f"p37_{label}_worst_chimera_vol.gif")
        save_gif(overlay_worst_base, debug_dir / f"p37_{label}_worst_chimera_base.gif")

        better_worst = "✓ BETTER" if worst_vol_score < worst_base else "✗ WORSE"
        better_best  = "✓ BETTER" if best_vol_score  < best_base  else "✗ WORSE"
        print(f"  best seed={best_seed}   base={best_base:.4f}  vol={best_vol_score:.4f}  {better_best}", flush=True)
        print(f"  worst seed={worst_seed}  base={worst_base:.4f}  vol={worst_vol_score:.4f}  {better_worst}", flush=True)
        results.append({
            "pair": f"{kw0}+{kw1}",
            "prompt": full_prompt,
            "best_seed": best_seed,
            "base_best": best_base,
            "vol_best": best_vol_score,
            "worst_seed": worst_seed,
            "base_worst": worst_base,
            "vol_worst": worst_vol_score,
            "all": [(s, b, v) for s, b, v, _, _ in all_scores],
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"[P37 Training Pairs] Summary — HIGH CHIMERA SEED (worst baseline)", flush=True)
    print(f"{'Pair':<22} {'Seed':>5} {'Base':>8} {'Vol':>8} {'Δ':>9} {'%':>7}", flush=True)
    print("-" * 65, flush=True)
    total_base = 0.0
    total_vol  = 0.0
    for r in results:
        b = r['base_worst']
        v = r['vol_worst']
        delta = v - b
        pct = (v - b) / (b + 1e-9) * 100
        better = "✓" if v < b else "✗"
        print(f"{r['pair']:<22} {r['worst_seed']:>5} {b:8.4f} {v:8.4f} {delta:+9.4f} {pct:+6.1f}%  {better}", flush=True)
        total_base += b
        total_vol  += v
    n = len(results)
    print("-" * 65, flush=True)
    print(f"{'AVERAGE':<22} {'':>5} {total_base/n:8.4f} {total_vol/n:8.4f} {(total_vol-total_base)/n:+9.4f}", flush=True)

    print(f"\n[P37 Training Pairs] All seeds per pair:", flush=True)
    for r in results:
        print(f"\n  {r['pair']}  prompt='{r['prompt']}'", flush=True)
        for s, b, v in r['all']:
            delta = v - b
            better = "✓" if v < b else "✗"
            print(f"    seed={s}  base={b:.4f}  vol={v:.4f}  {delta:+.4f}  {better}", flush=True)
    print(f"\nGIFs saved to: {debug_dir}/", flush=True)

    return results


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      type=str, default="checkpoints/phase37/best.pt")
    p.add_argument("--debug-dir", type=str, default="debug/chimera_phase37_pairs")
    p.add_argument("--n-frames",  type=int, default=16)
    p.add_argument("--n-steps",   type=int, default=20)
    p.add_argument("--height",    type=int, default=256)
    p.add_argument("--width",     type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
