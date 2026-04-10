"""
Phase 19: Phase 16 vs Phase 18 vs Phase 19 정량 비교

Phase 19 체크포인트는 multi_layer=True (6개 레이어 주입) →
  inference도 inject_vca_p19_infer() 로 6-layer 주입 필요.
Phase 16/18 체크포인트는 기존 _inject() (mid_block 단일 주입).

출력:
  debug/comparison_p19/{prompt_id}/
    baseline.gif, phase16.gif, phase18.gif, phase19.gif
    threeway.gif          ← [Phase16 | Phase18 | Phase19]
    debug_phase19.gif     ← [RGB | E0σ | E1σ]
    sigma_comparison.json

  debug/comparison_p19/summary/
    comparison_report.md
    all_sigma_stats.json
    best_threeway.gif
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as iio2

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from scripts.run_animatediff import load_pipeline, FixedContextVCAProcessor
from scripts.train_animatediff_vca import get_entity_embedding_mean
from scripts.make_figures import (
    _inject, _restore, generate, compute_stats,
    make_side_by_side, make_debug_sigma, extract_sigma_maps,
    _label, _resize_frame,
)
from scripts.train_phase19 import inject_vca_p19_infer, INJECT_KEYS_P19


# ─── 비교 프롬프트 ────────────────────────────────────────────────────────────

COMPARISON_PROMPTS = [
    {"id": "cat_dog",   "full": "A white cat and a black dog wrestling on the floor",
     "entity_0": "a white cat",           "entity_1": "a black dog",
     "category": "in_dist_objaverse"},
    {"id": "swords",    "full": "A golden sword and a silver sword crossing each other",
     "entity_0": "a golden sword",        "entity_1": "a silver sword",
     "category": "in_dist_objaverse"},
    {"id": "chain",     "full": "Two interlocked chain links, one red and one blue",
     "entity_0": "a red chain link",      "entity_1": "a blue chain link",
     "category": "in_dist_toy"},
    {"id": "robot_arm", "full": "Two robotic arms crossing each other",
     "entity_0": "a red robotic arm",     "entity_1": "a blue robotic arm",
     "category": "in_dist_toy"},
    {"id": "dancers",   "full": "A man in red and a woman in blue dancing closely together",
     "entity_0": "a man in red",          "entity_1": "a woman in blue",
     "category": "zero_shot"},
    {"id": "snakes",    "full": "A red snake and a blue snake coiling around each other",
     "entity_0": "a red snake",           "entity_1": "a blue snake",
     "category": "zero_shot"},
    {"id": "fighters",  "full": "Two martial artists fighting, one in white and one in black",
     "entity_0": "a martial artist in white", "entity_1": "a martial artist in black",
     "category": "zero_shot"},
    {"id": "cables",    "full": "An orange cable and a green cable tangled together",
     "entity_0": "an orange cable",       "entity_1": "a green cable",
     "category": "zero_shot"},
]


# ─── VCA 로드 ─────────────────────────────────────────────────────────────────

def load_vca_standard(ckpt_path: Path, device: str) -> VCALayer:
    """Phase 12/16/18: depth_pe_init_scale 기본값 0.02, single-layer 주입용."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    vca = VCALayer(
        query_dim=1280, context_dim=768,
        n_heads=8, n_entities=2, z_bins=2, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=0.02,
    ).to(device)
    vca.load_state_dict(ckpt["vca_state_dict"])
    vca.eval()
    return vca


def load_vca_p19(ckpt_path: Path, device: str) -> VCALayer:
    """Phase 19: depth_pe_init_scale=0.3, multi-layer 주입용."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    scale = ckpt.get("depth_pe_init_scale", 0.3)
    vca = VCALayer(
        query_dim=1280, context_dim=768,
        n_heads=8, n_entities=2, z_bins=2, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=scale,
    ).to(device)
    vca.load_state_dict(ckpt["vca_state_dict"])
    vca.eval()
    return vca


def build_entity_ctx(pipe, prompt_info: dict, device: str) -> torch.Tensor:
    """entity_0/1 텍스트 → (1, 2, 768) fp32"""
    e0 = get_entity_embedding_mean(pipe, prompt_info["entity_0"])
    e1 = get_entity_embedding_mean(pipe, prompt_info["entity_1"])
    return torch.cat([e0, e1], dim=1).float().to(device)


# ─── 4-panel GIF ─────────────────────────────────────────────────────────────

def make_fourway(baseline, p16, p18, p19, out_path: Path, panel_size=256):
    """[Baseline | Phase16 | Phase18 | Phase19] 4-panel GIF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for b, a, c, d in zip(baseline, p16, p18, p19):
        panels = [
            _label(_resize_frame(b, panel_size), "Baseline"),
            _label(_resize_frame(a, panel_size), "Phase16"),
            _label(_resize_frame(c, panel_size), "Phase18"),
            _label(_resize_frame(d, panel_size), "Phase19"),
        ]
        frames.append(np.concatenate(panels, axis=1))
    iio2.mimsave(str(out_path), frames, duration=250)


# ─── 단일 프롬프트 처리 ───────────────────────────────────────────────────────

def process_prompt(pipe, p16_vca, p18_vca, p19_vca,
                   prompt_info: dict, out_dir: Path, args) -> dict:
    pid = prompt_info["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = str(pipe.device)
    entity_ctx = build_entity_ctx(pipe, prompt_info, device)
    prompt = prompt_info["full"]
    kw = dict(num_frames=args.num_frames, steps=args.steps,
              height=args.height, width=args.width, seed=args.seed)

    print(f"\n[{pid}] category={prompt_info['category']}", flush=True)

    # 1. baseline
    print(f"  [{pid}] baseline...", flush=True)
    baseline_frames = generate(pipe, prompt, **kw)
    iio2.mimsave(str(out_dir / "baseline.gif"), baseline_frames, duration=250)

    # 2. phase16 (mid_block single inject)
    print(f"  [{pid}] phase16...", flush=True)
    orig16 = _inject(pipe, p16_vca, entity_ctx)
    p16_frames = generate(pipe, prompt, **kw)
    p16_sigma_maps = extract_sigma_maps(p16_vca, args.num_frames, use_cfg=True)
    p16_stats = compute_stats(p16_vca, args.num_frames, use_cfg=True)
    _restore(pipe, orig16)
    iio2.mimsave(str(out_dir / "phase16.gif"), p16_frames, duration=250)

    # 3. phase18 (mid_block single inject)
    print(f"  [{pid}] phase18...", flush=True)
    orig18 = _inject(pipe, p18_vca, entity_ctx)
    p18_frames = generate(pipe, prompt, **kw)
    p18_sigma_maps = extract_sigma_maps(p18_vca, args.num_frames, use_cfg=True)
    p18_stats = compute_stats(p18_vca, args.num_frames, use_cfg=True)
    _restore(pipe, orig18)
    iio2.mimsave(str(out_dir / "phase18.gif"), p18_frames, duration=250)

    # 4. phase19 (6-layer multi inject)
    print(f"  [{pid}] phase19 (6-layer)...", flush=True)
    orig19, injected19 = inject_vca_p19_infer(pipe, p19_vca, entity_ctx)
    p19_frames = generate(pipe, prompt, **kw)
    p19_sigma_maps = extract_sigma_maps(p19_vca, args.num_frames, use_cfg=True)
    p19_stats = compute_stats(p19_vca, args.num_frames, use_cfg=True)
    _restore(pipe, orig19)
    iio2.mimsave(str(out_dir / "phase19.gif"), p19_frames, duration=250)

    # 4-panel threeway
    make_fourway(baseline_frames, p16_frames, p18_frames, p19_frames,
                 out_dir / "threeway.gif", panel_size=args.height)

    # debug sigma GIFs
    if p19_sigma_maps:
        make_debug_sigma(p19_frames, p19_sigma_maps,
                         out_dir / "debug_phase19.gif",
                         label="Phase19", panel_size=args.height)
    if p16_sigma_maps:
        make_debug_sigma(p16_frames, p16_sigma_maps,
                         out_dir / "debug_phase16.gif",
                         label="Phase16", panel_size=args.height)

    # winner 판정 (p16 vs p19 기준)
    p16_sep = p16_stats.get("sigma_separation", 0.0)
    p18_sep = p18_stats.get("sigma_separation", 0.0)
    p19_sep = p19_stats.get("sigma_separation", 0.0)
    best_prev = max(p16_sep, p18_sep)

    if abs(p19_sep - best_prev) < 0.005:
        winner = "COMPARABLE"
    elif p19_sep > best_prev:
        winner = "phase19"
    else:
        winner = "phase16" if p16_sep >= p18_sep else "phase18"

    comparison = {
        "prompt_id": pid,
        "category":  prompt_info["category"],
        "phase16":   p16_stats,
        "phase18":   p18_stats,
        "phase19":   p19_stats,
        "winner":    winner,
        "delta_vs_p16": round(p19_sep - p16_sep, 4),
        "delta_vs_p18": round(p19_sep - p18_sep, 4),
    }
    with open(out_dir / "sigma_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(
        f"  [{pid}] p16_sep={p16_sep:.4f}  p18_sep={p18_sep:.4f}  "
        f"p19_sep={p19_sep:.4f}  winner={winner}",
        flush=True,
    )
    return comparison


# ─── 리포트 생성 ──────────────────────────────────────────────────────────────

def make_report(all_comparisons: list, out_path: Path):
    cats = {"in_dist_objaverse": [], "in_dist_toy": [], "zero_shot": []}
    for c in all_comparisons:
        cats[c["category"]].append(c)

    def avg_sep(lst, model):
        vals = [c[model].get("sigma_separation", 0) for c in lst]
        return sum(vals) / len(vals) if vals else 0.0

    p19_wins = sum(1 for c in all_comparisons if c["winner"] == "phase19")
    p16_wins = sum(1 for c in all_comparisons if c["winner"] == "phase16")
    p18_wins = sum(1 for c in all_comparisons if c["winner"] == "phase18")
    comparable = sum(1 for c in all_comparisons if c["winner"] == "COMPARABLE")
    n = len(all_comparisons)

    if p19_wins > max(p16_wins, p18_wins):
        conclusion = f"PHASE19_BEST — wins {p19_wins}/{n} prompts"
    elif p19_wins == 0 and p16_wins == 0 and comparable == n:
        conclusion = "ALL_COMPARABLE"
    else:
        best_prev = max(p16_wins, p18_wins)
        loser = "phase16" if p16_wins >= p18_wins else "phase18"
        conclusion = f"PREV_BETTER — phase19 wins {p19_wins}/{n}, {loser} wins {best_prev}/{n}"

    lines = [
        "# Phase 16 vs Phase 18 vs Phase 19 Comparison Report",
        "",
        "## Setup",
        "",
        "| | Phase 16 | Phase 18 | Phase 19 |",
        "|---|---|---|---|",
        "| Injection | mid_block (1 layer) | mid_block (1 layer) | 6 layers |",
        "| λ_depth | 1.0→adaptive | 0.3 | 0.3 |",
        "| depth_pe_init_scale | 0.02 | 0.02 | 0.3 |",
        "| depth loss | majority vote | majority vote | per-frame |",
        "| probe metric | random t | random t | fixed 5-t |",
        "",
        "## Sigma Separation Comparison",
        "",
        "| prompt_id | category | phase16 | phase18 | phase19 | Δvs16 | Δvs18 | winner |",
        "|-----------|----------|---------|---------|---------|-------|-------|--------|",
    ]
    for c in all_comparisons:
        lines.append(
            f"| {c['prompt_id']:12s} | {c['category']:18s} | "
            f"{c['phase16'].get('sigma_separation',0):.3f} | "
            f"{c['phase18'].get('sigma_separation',0):.3f} | "
            f"{c['phase19'].get('sigma_separation',0):.3f} | "
            f"{c['delta_vs_p16']:+.3f} | {c['delta_vs_p18']:+.3f} | {c['winner']} |"
        )

    lines += [
        "",
        "## Generalization Analysis",
        "",
        f"- In-dist (objaverse): p16={avg_sep(cats['in_dist_objaverse'],'phase16'):.4f}  "
        f"p18={avg_sep(cats['in_dist_objaverse'],'phase18'):.4f}  "
        f"p19={avg_sep(cats['in_dist_objaverse'],'phase19'):.4f}",
        f"- In-dist (toy):       p16={avg_sep(cats['in_dist_toy'],'phase16'):.4f}  "
        f"p18={avg_sep(cats['in_dist_toy'],'phase18'):.4f}  "
        f"p19={avg_sep(cats['in_dist_toy'],'phase19'):.4f}",
        f"- Zero-shot:           p16={avg_sep(cats['zero_shot'],'phase16'):.4f}  "
        f"p18={avg_sep(cats['zero_shot'],'phase18'):.4f}  "
        f"p19={avg_sep(cats['zero_shot'],'phase19'):.4f}",
        "",
        "## Key Finding",
        "",
        f"Phase 19 wins: {p19_wins}/{n}",
        f"Phase 16 wins: {p16_wins}/{n}",
        f"Phase 18 wins: {p18_wins}/{n}",
        f"Comparable:    {comparable}/{n}",
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

    p16_ckpt = Path(args.phase16_ckpt)
    p18_ckpt = Path(args.phase18_ckpt)
    p19_ckpt = Path(args.phase19_ckpt)
    for p in [p16_ckpt, p18_ckpt, p19_ckpt]:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    pipe = load_pipeline(device=device, dtype=torch.float16)
    p16_vca = load_vca_standard(p16_ckpt, device)
    p18_vca = load_vca_standard(p18_ckpt, device)
    p19_vca = load_vca_p19(p19_ckpt, device)
    print(f"[init] loaded p16 from {p16_ckpt}", flush=True)
    print(f"[init] loaded p18 from {p18_ckpt}", flush=True)
    print(f"[init] loaded p19 from {p19_ckpt} (multi_layer, 6-layer inject)", flush=True)

    if args.prompts:
        ids = [p.strip() for p in args.prompts.split(",")]
        prompts_to_run = [p for p in COMPARISON_PROMPTS if p["id"] in ids]
    else:
        prompts_to_run = COMPARISON_PROMPTS

    out_root = Path(args.out_dir)
    all_comparisons = []

    for prompt_info in prompts_to_run:
        comp = process_prompt(pipe, p16_vca, p18_vca, p19_vca, prompt_info,
                              out_root / prompt_info["id"], args)
        all_comparisons.append(comp)

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / "all_sigma_stats.json", "w") as f:
        json.dump(all_comparisons, f, indent=2)

    make_report(all_comparisons, summary_dir / "comparison_report.md")

    # best threeway: phase19 delta_vs_p16 최대
    if all_comparisons:
        best = max(all_comparisons, key=lambda c: c["delta_vs_p16"])
        src = out_root / best["prompt_id"] / "threeway.gif"
        dst = summary_dir / "best_threeway.gif"
        if src.exists():
            import shutil
            shutil.copy2(str(src), str(dst))
            print(f"[summary] best_threeway ← {best['prompt_id']} "
                  f"(Δp16={best['delta_vs_p16']:+.4f})", flush=True)

    print(f"\n[done] results → {out_root}", flush=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 16 vs 18 vs 19 inference comparison"
    )
    p.add_argument("--phase16-ckpt", required=True, dest="phase16_ckpt")
    p.add_argument("--phase18-ckpt", required=True, dest="phase18_ckpt")
    p.add_argument("--phase19-ckpt", required=True, dest="phase19_ckpt")
    p.add_argument("--out-dir",      default="debug/comparison_p19", dest="out_dir")
    p.add_argument("--prompts",      default="",
                   help="comma-separated prompt ids (default: all 8)")
    p.add_argument("--num-frames",   type=int, default=16, dest="num_frames")
    p.add_argument("--steps",        type=int, default=20)
    p.add_argument("--height",       type=int, default=256)
    p.add_argument("--width",        type=int, default=256)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
