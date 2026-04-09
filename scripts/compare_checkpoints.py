"""
Phase 16 Part B: Phase 12 vs Phase 16 정량 비교

각 프롬프트에 3개 모델로 inference:
  baseline  — VCA 없는 표준 AnimateDiff
  phase12   — toy-only 학습 체크포인트
  phase16   — objaverse 학습 체크포인트

출력:
  debug/comparison_p16/{prompt_id}/
    baseline.gif, phase12.gif, phase16.gif
    threeway.gif          ← [baseline | phase12 | phase16] seed=42 동일 조건
    debug_phase12.gif     ← [RGB | E0σ | E1σ]
    debug_phase16.gif
    sigma_comparison.json ← {baseline, phase12, phase16, winner}

  summary/
    comparison_report.md
    all_sigma_stats.json
    best_threeway.gif     ← phase16 개선폭 최대인 프롬프트
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as iio2
import imageio.v3 as iio3
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from scripts.run_animatediff import load_pipeline, FixedContextVCAProcessor
from scripts.train_animatediff_vca import get_entity_embedding_mean
from scripts.make_figures import (
    _inject, _restore, generate, compute_stats,
    make_side_by_side, make_debug_sigma, extract_sigma_maps,
    _label, _resize_frame,
)


# ─── 비교 프롬프트 ────────────────────────────────────────────────────────────

COMPARISON_PROMPTS = [
    # In-distribution (Phase 16 학습 데이터와 유사)
    {
        "id": "cat_dog",
        "full": "A white cat and a black dog wrestling on the floor",
        "entity_0": "a white cat",
        "entity_1": "a black dog",
        "category": "in_dist_objaverse",
    },
    {
        "id": "swords",
        "full": "A golden sword and a silver sword crossing each other",
        "entity_0": "a golden sword",
        "entity_1": "a silver sword",
        "category": "in_dist_objaverse",
    },
    # In-distribution (Phase 12 학습 데이터와 유사)
    {
        "id": "chain",
        "full": "Two interlocked chain links, one red and one blue",
        "entity_0": "a red chain link",
        "entity_1": "a blue chain link",
        "category": "in_dist_toy",
    },
    {
        "id": "robot_arm",
        "full": "Two robotic arms crossing each other",
        "entity_0": "a red robotic arm",
        "entity_1": "a blue robotic arm",
        "category": "in_dist_toy",
    },
    # Zero-shot (둘 다 본 적 없는 것)
    {
        "id": "dancers",
        "full": "A man in red and a woman in blue dancing closely together",
        "entity_0": "a man in red",
        "entity_1": "a woman in blue",
        "category": "zero_shot",
    },
    {
        "id": "snakes",
        "full": "A red snake and a blue snake coiling around each other",
        "entity_0": "a red snake",
        "entity_1": "a blue snake",
        "category": "zero_shot",
    },
    {
        "id": "fighters",
        "full": "Two martial artists fighting, one in white and one in black",
        "entity_0": "a martial artist in white",
        "entity_1": "a martial artist in black",
        "category": "zero_shot",
    },
    {
        "id": "cables",
        "full": "An orange cable and a green cable tangled together",
        "entity_0": "an orange cable",
        "entity_1": "a green cable",
        "category": "zero_shot",
    },
]


# ─── VCA 로드 ─────────────────────────────────────────────────────────────────

def load_vca(ckpt_path: Path, device: str) -> VCALayer:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    vca = VCALayer(
        query_dim=1280, context_dim=768,
        n_heads=8, n_entities=2, z_bins=2, lora_rank=8,
        use_softmax=False,
    ).to(device)
    vca.load_state_dict(ckpt["vca_state_dict"])
    vca.eval()
    return vca


def build_entity_ctx(pipe, prompt_info: dict, device: str) -> torch.Tensor:
    """entity_0/1 텍스트 → (1, 2, 768) fp32"""
    e0 = get_entity_embedding_mean(pipe, prompt_info["entity_0"])
    e1 = get_entity_embedding_mean(pipe, prompt_info["entity_1"])
    return torch.cat([e0, e1], dim=1).float().to(device)


# ─── 3-panel GIF (threeway) ──────────────────────────────────────────────────

def make_threeway(baseline: list, p12: list, p16: list,
                  out_path: Path, panel_size: int = 256):
    """[Baseline | Phase12 | Phase16] 3-panel GIF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames = []
    for b, a, c in zip(baseline, p12, p16):
        p0 = _label(_resize_frame(b, panel_size), "Baseline")
        p1 = _label(_resize_frame(a, panel_size), "Phase12")
        p2 = _label(_resize_frame(c, panel_size), "Phase16")
        gif_frames.append(np.concatenate([p0, p1, p2], axis=1))
    iio2.mimsave(str(out_path), gif_frames, duration=250)


# ─── 단일 프롬프트 처리 ───────────────────────────────────────────────────────

def process_prompt(pipe, p12_vca: VCALayer, p16_vca: VCALayer,
                   prompt_info: dict, out_dir: Path, args) -> dict:
    pid = prompt_info["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pipe.device if hasattr(pipe, "device") else "cuda"
    entity_ctx = build_entity_ctx(pipe, prompt_info, device)
    prompt = prompt_info["full"]

    print(f"\n[{pid}] category={prompt_info['category']}", flush=True)

    kw = dict(num_frames=args.num_frames, steps=args.steps,
               height=args.height, width=args.width, seed=args.seed)

    # ── 1. baseline ────────────────────────────────────────────────────────
    print(f"  [{pid}] baseline...", flush=True)
    baseline_frames = generate(pipe, prompt, **kw)
    iio2.mimsave(str(out_dir / "baseline.gif"), baseline_frames, duration=250)

    # ── 2. phase12 ─────────────────────────────────────────────────────────
    print(f"  [{pid}] phase12...", flush=True)
    orig12 = _inject(pipe, p12_vca, entity_ctx)
    p12_frames = generate(pipe, prompt, **kw)
    p12_sigma_maps = extract_sigma_maps(p12_vca, args.num_frames, use_cfg=True)
    p12_stats = compute_stats(p12_vca, args.num_frames, use_cfg=True)
    _restore(pipe, orig12)
    iio2.mimsave(str(out_dir / "phase12.gif"), p12_frames, duration=250)

    # ── 3. phase16 ─────────────────────────────────────────────────────────
    print(f"  [{pid}] phase16...", flush=True)
    orig16 = _inject(pipe, p16_vca, entity_ctx)
    p16_frames = generate(pipe, prompt, **kw)
    p16_sigma_maps = extract_sigma_maps(p16_vca, args.num_frames, use_cfg=True)
    p16_stats = compute_stats(p16_vca, args.num_frames, use_cfg=True)
    _restore(pipe, orig16)
    iio2.mimsave(str(out_dir / "phase16.gif"), p16_frames, duration=250)

    # ── threeway GIF ───────────────────────────────────────────────────────
    make_threeway(baseline_frames, p12_frames, p16_frames,
                  out_dir / "threeway.gif", panel_size=args.height)

    # ── debug sigma GIFs ──────────────────────────────────────────────────
    if p12_sigma_maps:
        make_debug_sigma(p12_frames, p12_sigma_maps,
                         out_dir / "debug_phase12.gif",
                         label="Phase12", panel_size=args.height)
    if p16_sigma_maps:
        make_debug_sigma(p16_frames, p16_sigma_maps,
                         out_dir / "debug_phase16.gif",
                         label="Phase16", panel_size=args.height)

    # ── winner 판정 ────────────────────────────────────────────────────────
    b_sep  = 0.0
    p12_sep = p12_stats.get("sigma_separation", 0.0)
    p16_sep = p16_stats.get("sigma_separation", 0.0)

    if abs(p16_sep - p12_sep) < 0.005:
        winner = "COMPARABLE"
    elif p16_sep > p12_sep:
        winner = "phase16"
    else:
        winner = "phase12"

    comparison = {
        "prompt_id": pid,
        "category":  prompt_info["category"],
        "baseline":  {"sigma_separation": b_sep, "sigma_consistency": 0.5},
        "phase12":   p12_stats,
        "phase16":   p16_stats,
        "winner":    winner,
        "delta":     round(p16_sep - p12_sep, 4),
    }
    with open(out_dir / "sigma_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(
        f"  [{pid}] p12_sep={p12_sep:.4f}  p16_sep={p16_sep:.4f}  "
        f"winner={winner}",
        flush=True,
    )
    return comparison


# ─── 리포트 생성 ──────────────────────────────────────────────────────────────

def make_report(all_comparisons: list, out_path: Path,
                p12_ckpt: str, p16_ckpt: str, data_root: str):
    cats = {"in_dist_objaverse": [], "in_dist_toy": [], "zero_shot": []}
    for c in all_comparisons:
        cats[c["category"]].append(c)

    def avg_sep(lst, model):
        vals = [c[model].get("sigma_separation", 0) for c in lst]
        return sum(vals) / len(vals) if vals else 0.0

    # n_frames 추정
    n_frames_est = 0
    if Path(data_root).exists():
        n_frames_est = sum(1 for _ in Path(data_root).rglob("meta.json")) * 8

    p16_wins = sum(1 for c in all_comparisons if c["winner"] == "phase16")
    p12_wins = sum(1 for c in all_comparisons if c["winner"] == "phase12")
    comparable = sum(1 for c in all_comparisons if c["winner"] == "COMPARABLE")

    # 결론
    if p16_wins > p12_wins:
        conclusion = f"OBJAVERSE_BETTER — phase16 wins {p16_wins}/{len(all_comparisons)} prompts"
    elif p12_wins > p16_wins:
        conclusion = f"TOY_BETTER — phase12 wins {p12_wins}/{len(all_comparisons)} prompts"
    else:
        conclusion = f"COMPARABLE — tied {p16_wins}/{len(all_comparisons)} each"

    lines = [
        "# Phase 12 vs Phase 16 Comparison Report",
        "",
        "## Training Data Summary",
        "",
        "| | Phase 12 | Phase 16 |",
        "|---|---|---|",
        "| Dataset | ToyVCADataset | ObjaverseVCADataset |",
        f"| Frames | ~32 | ~{n_frames_est} |",
        "| Scenarios | 2 | 10+ |",
        "| CLIP context | dummy | real keyword CLIP |",
        f"| t_max | 1000 (Phase 12) | 200 (Phase 14 best) |",
        "",
        "## Sigma Separation Comparison",
        "",
        "| prompt_id | category | baseline | phase12 | phase16 | delta | winner |",
        "|-----------|----------|----------|---------|---------|-------|--------|",
    ]
    for c in all_comparisons:
        lines.append(
            f"| {c['prompt_id']:12s} | {c['category']:18s} | "
            f"{c['baseline']['sigma_separation']:.3f} | "
            f"{c['phase12'].get('sigma_separation',0):.3f} | "
            f"{c['phase16'].get('sigma_separation',0):.3f} | "
            f"{c['delta']:+.3f} | {c['winner']} |"
        )

    lines += [
        "",
        "## Generalization Analysis",
        "",
        f"- In-dist (objaverse) — phase12: {avg_sep(cats['in_dist_objaverse'],'phase12'):.4f}  "
        f"phase16: {avg_sep(cats['in_dist_objaverse'],'phase16'):.4f}",
        f"- In-dist (toy)       — phase12: {avg_sep(cats['in_dist_toy'],'phase12'):.4f}  "
        f"phase16: {avg_sep(cats['in_dist_toy'],'phase16'):.4f}",
        f"- Zero-shot           — phase12: {avg_sep(cats['zero_shot'],'phase12'):.4f}  "
        f"phase16: {avg_sep(cats['zero_shot'],'phase16'):.4f}",
        "",
        "## Key Finding",
        "",
        f"Phase 16이 Phase 12보다 나은 경우: {p16_wins}/{len(all_comparisons)} 프롬프트",
        f"Phase 12이 Phase 16보다 나은 경우: {p12_wins}/{len(all_comparisons)} 프롬프트",
        f"유사(diff < 0.005): {comparable}/{len(all_comparisons)} 프롬프트",
        "",
        "## Conclusion",
        "",
        f"**{conclusion}**",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[report] → {out_path}", flush=True)


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}", flush=True)

    p12_ckpt = Path(args.phase12_ckpt)
    p16_ckpt = Path(args.phase16_ckpt)
    if not p12_ckpt.exists():
        raise FileNotFoundError(f"Phase 12 checkpoint not found: {p12_ckpt}")
    if not p16_ckpt.exists():
        raise FileNotFoundError(f"Phase 16 checkpoint not found: {p16_ckpt}")

    pipe = load_pipeline(device=device, dtype=torch.float16)
    p12_vca = load_vca(p12_ckpt, device)
    p16_vca = load_vca(p16_ckpt, device)
    print(f"[init] loaded phase12 from {p12_ckpt}", flush=True)
    print(f"[init] loaded phase16 from {p16_ckpt}", flush=True)

    # 프롬프트 필터
    if args.prompts:
        ids = [p.strip() for p in args.prompts.split(",")]
        prompts_to_run = [p for p in COMPARISON_PROMPTS if p["id"] in ids]
    else:
        prompts_to_run = COMPARISON_PROMPTS

    out_root = Path(args.out_dir)
    all_comparisons = []

    for prompt_info in prompts_to_run:
        comp = process_prompt(pipe, p12_vca, p16_vca, prompt_info,
                              out_root / prompt_info["id"], args)
        all_comparisons.append(comp)

    # summary
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / "all_sigma_stats.json", "w") as f:
        json.dump(all_comparisons, f, indent=2)

    make_report(all_comparisons, summary_dir / "comparison_report.md",
                str(p12_ckpt), str(p16_ckpt), "toy/data_objaverse")

    # best threeway: phase16 delta 최대인 것
    if all_comparisons:
        best = max(all_comparisons, key=lambda c: c["delta"])
        src = out_root / best["prompt_id"] / "threeway.gif"
        dst = summary_dir / "best_threeway.gif"
        if src.exists():
            import shutil
            shutil.copy2(str(src), str(dst))
            print(f"[summary] best_threeway ← {best['prompt_id']} "
                  f"(delta={best['delta']:+.4f})", flush=True)

    print(f"\n[done] results → {out_root}", flush=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase12-ckpt", required=True, dest="phase12_ckpt")
    p.add_argument("--phase16-ckpt", required=True, dest="phase16_ckpt")
    p.add_argument("--out-dir",      default="debug/comparison_p16", dest="out_dir")
    p.add_argument("--prompts",      default="",
                   help="comma-separated prompt ids to run (default: all 8)")
    p.add_argument("--num-frames",   type=int, default=16, dest="num_frames")
    p.add_argument("--steps",        type=int, default=20)
    p.add_argument("--height",       type=int, default=256)
    p.add_argument("--width",        type=int, default=256)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
