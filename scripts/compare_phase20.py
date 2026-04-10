"""
Phase 20 비교: Phase 16 vs Phase 18 vs Phase 19 vs Phase 20

주입 방식:
  Phase 16/18/20 — _inject() (mid_block 단일 주입)
  Phase 19       — inject_vca_p19_infer() (6-layer 주입)

출력:
  debug/comparison_p20/{prompt_id}/
    baseline.gif, phase16.gif, phase18.gif, phase19.gif, phase20.gif
    fourway.gif      ← [Phase16 | Phase18 | Phase19 | Phase20]
    debug_phase20.gif
    sigma_comparison.json

  debug/comparison_p20/summary/
    comparison_report.md
    all_sigma_stats.json
    best_fourway.gif
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
from scripts.run_animatediff import load_pipeline
from scripts.train_animatediff_vca import get_entity_embedding_mean
from scripts.make_figures import (
    _inject, _restore, generate, compute_stats,
    make_debug_sigma, extract_sigma_maps,
    _label, _resize_frame,
)
from scripts.train_phase19 import inject_vca_p19_infer

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


def load_vca_standard(ckpt_path, device, init_scale=0.02):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    vca = VCALayer(query_dim=1280, context_dim=768,
                   n_heads=8, n_entities=2, z_bins=2, lora_rank=8,
                   use_softmax=False, depth_pe_init_scale=init_scale).to(device)
    vca.load_state_dict(ckpt["vca_state_dict"])
    vca.eval()
    return vca


def load_vca_from_ckpt(ckpt_path, device):
    """depth_pe_init_scale을 체크포인트에서 읽어 로드."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    scale = ckpt.get("depth_pe_init_scale", 0.02)
    vca = VCALayer(query_dim=1280, context_dim=768,
                   n_heads=8, n_entities=2, z_bins=2, lora_rank=8,
                   use_softmax=False, depth_pe_init_scale=scale).to(device)
    vca.load_state_dict(ckpt["vca_state_dict"])
    vca.eval()
    return vca


def build_entity_ctx(pipe, prompt_info, device):
    e0 = get_entity_embedding_mean(pipe, prompt_info["entity_0"])
    e1 = get_entity_embedding_mean(pipe, prompt_info["entity_1"])
    return torch.cat([e0, e1], dim=1).float().to(device)


def make_fourway(p16, p18, p19, p20, out_path, panel_size=256):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for a, b, c, d in zip(p16, p18, p19, p20):
        row = np.concatenate([
            _label(_resize_frame(a, panel_size), "Phase16"),
            _label(_resize_frame(b, panel_size), "Phase18"),
            _label(_resize_frame(c, panel_size), "Phase19"),
            _label(_resize_frame(d, panel_size), "Phase20"),
        ], axis=1)
        frames.append(row)
    iio2.mimsave(str(out_path), frames, duration=250)


def process_prompt(pipe, p16_vca, p18_vca, p19_vca, p20_vca,
                   prompt_info, out_dir, args):
    pid = prompt_info["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = str(pipe.device)
    entity_ctx = build_entity_ctx(pipe, prompt_info, device)
    prompt = prompt_info["full"]
    kw = dict(num_frames=args.num_frames, steps=args.steps,
              height=args.height, width=args.width, seed=args.seed)

    print(f"\n[{pid}] category={prompt_info['category']}", flush=True)

    # baseline
    print(f"  [{pid}] baseline...", flush=True)
    baseline_frames = generate(pipe, prompt, **kw)
    iio2.mimsave(str(out_dir / "baseline.gif"), baseline_frames, duration=250)

    # phase16
    print(f"  [{pid}] phase16...", flush=True)
    orig = _inject(pipe, p16_vca, entity_ctx)
    p16_frames = generate(pipe, prompt, **kw)
    p16_stats = compute_stats(p16_vca, args.num_frames, use_cfg=True)
    _restore(pipe, orig)
    iio2.mimsave(str(out_dir / "phase16.gif"), p16_frames, duration=250)

    # phase18
    print(f"  [{pid}] phase18...", flush=True)
    orig = _inject(pipe, p18_vca, entity_ctx)
    p18_frames = generate(pipe, prompt, **kw)
    p18_stats = compute_stats(p18_vca, args.num_frames, use_cfg=True)
    _restore(pipe, orig)
    iio2.mimsave(str(out_dir / "phase18.gif"), p18_frames, duration=250)

    # phase19 — 6-layer injection
    print(f"  [{pid}] phase19 (6-layer)...", flush=True)
    orig, _ = inject_vca_p19_infer(pipe, p19_vca, entity_ctx)
    p19_frames = generate(pipe, prompt, **kw)
    p19_stats = compute_stats(p19_vca, args.num_frames, use_cfg=True)
    _restore(pipe, orig)
    iio2.mimsave(str(out_dir / "phase19.gif"), p19_frames, duration=250)

    # phase20 — mid_block single injection
    print(f"  [{pid}] phase20...", flush=True)
    orig = _inject(pipe, p20_vca, entity_ctx)
    p20_frames = generate(pipe, prompt, **kw)
    p20_sigma_maps = extract_sigma_maps(p20_vca, args.num_frames, use_cfg=True)
    p20_stats = compute_stats(p20_vca, args.num_frames, use_cfg=True)
    _restore(pipe, orig)
    iio2.mimsave(str(out_dir / "phase20.gif"), p20_frames, duration=250)

    make_fourway(p16_frames, p18_frames, p19_frames, p20_frames,
                 out_dir / "fourway.gif", panel_size=args.height)

    if p20_sigma_maps:
        make_debug_sigma(p20_frames, p20_sigma_maps,
                         out_dir / "debug_phase20.gif",
                         label="Phase20", panel_size=args.height)

    # winner: phase20 vs best of prev
    seps = {
        "phase16": p16_stats.get("sigma_separation", 0.0),
        "phase18": p18_stats.get("sigma_separation", 0.0),
        "phase19": p19_stats.get("sigma_separation", 0.0),
        "phase20": p20_stats.get("sigma_separation", 0.0),
    }
    best_key = max(seps, key=lambda k: seps[k])
    # comparable threshold
    best_val = seps[best_key]
    p20_val  = seps["phase20"]
    if abs(p20_val - best_val) < 0.005 and best_key != "phase20":
        winner = "COMPARABLE"
    else:
        winner = best_key

    comparison = {
        "prompt_id":        pid,
        "category":         prompt_info["category"],
        "phase16":          p16_stats,
        "phase18":          p18_stats,
        "phase19":          p19_stats,
        "phase20":          p20_stats,
        "winner":           winner,
        "delta_p20_vs_p16": round(p20_val - seps["phase16"], 4),
        "delta_p20_vs_p18": round(p20_val - seps["phase18"], 4),
        "delta_p20_vs_p19": round(p20_val - seps["phase19"], 4),
    }
    with open(out_dir / "sigma_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(
        f"  [{pid}] p16={seps['phase16']:.4f} p18={seps['phase18']:.4f} "
        f"p19={seps['phase19']:.4f} p20={seps['phase20']:.4f}  winner={winner}",
        flush=True,
    )
    return comparison


def make_report(all_comparisons, out_path):
    cats = {"in_dist_objaverse": [], "in_dist_toy": [], "zero_shot": []}
    for c in all_comparisons:
        cats[c["category"]].append(c)

    def avg(lst, model):
        vals = [c[model].get("sigma_separation", 0) for c in lst]
        return sum(vals) / len(vals) if vals else 0.0

    wins = {k: sum(1 for c in all_comparisons if c["winner"] == k)
            for k in ("phase16", "phase18", "phase19", "phase20", "COMPARABLE")}
    n = len(all_comparisons)

    best_model = max(("phase16","phase18","phase19","phase20"), key=lambda k: wins[k])
    if wins["phase20"] > wins[best_model.replace("phase20","phase16")]:
        conclusion = f"PHASE20_BEST — wins {wins['phase20']}/{n} prompts"
    elif wins["phase20"] == max(wins["phase16"], wins["phase18"], wins["phase19"], wins["phase20"]):
        conclusion = f"PHASE20_BEST — wins {wins['phase20']}/{n} prompts"
    else:
        conclusion = (f"PREV_BETTER — p16:{wins['phase16']} p18:{wins['phase18']} "
                      f"p19:{wins['phase19']} p20:{wins['phase20']} / {n}")

    lines = [
        "# Phase 16 vs 18 vs 19 vs 20 Comparison Report",
        "",
        "## Setup",
        "",
        "| | Phase 16 | Phase 18 | Phase 19 | Phase 20 |",
        "|---|---|---|---|---|",
        "| Injection | mid_block×1 | mid_block×1 | 6 layers | mid_block×1 |",
        "| depth_pe_init_scale | 0.02 | 0.02 | 0.3 | 0.3 |",
        "| depth loss | majority vote | majority vote | per-frame | per-frame |",
        "| probe metric | random t | random t | fixed 5-t | fixed 5-t |",
        "| λ_depth | adaptive | 0.3 | 0.3 | 0.3 |",
        "",
        "## Sigma Separation",
        "",
        "| prompt_id | category | p16 | p18 | p19 | p20 | Δvs18 | winner |",
        "|-----------|----------|-----|-----|-----|-----|-------|--------|",
    ]
    for c in all_comparisons:
        lines.append(
            f"| {c['prompt_id']:12s} | {c['category']:18s} | "
            f"{c['phase16'].get('sigma_separation',0):.3f} | "
            f"{c['phase18'].get('sigma_separation',0):.3f} | "
            f"{c['phase19'].get('sigma_separation',0):.3f} | "
            f"{c['phase20'].get('sigma_separation',0):.3f} | "
            f"{c['delta_p20_vs_p18']:+.3f} | {c['winner']} |"
        )

    lines += [
        "",
        "## Generalization Analysis",
        "",
        f"- In-dist (objaverse): p16={avg(cats['in_dist_objaverse'],'phase16'):.4f}  "
        f"p18={avg(cats['in_dist_objaverse'],'phase18'):.4f}  "
        f"p19={avg(cats['in_dist_objaverse'],'phase19'):.4f}  "
        f"p20={avg(cats['in_dist_objaverse'],'phase20'):.4f}",
        f"- In-dist (toy):       p16={avg(cats['in_dist_toy'],'phase16'):.4f}  "
        f"p18={avg(cats['in_dist_toy'],'phase18'):.4f}  "
        f"p19={avg(cats['in_dist_toy'],'phase19'):.4f}  "
        f"p20={avg(cats['in_dist_toy'],'phase20'):.4f}",
        f"- Zero-shot:           p16={avg(cats['zero_shot'],'phase16'):.4f}  "
        f"p18={avg(cats['zero_shot'],'phase18'):.4f}  "
        f"p19={avg(cats['zero_shot'],'phase19'):.4f}  "
        f"p20={avg(cats['zero_shot'],'phase20'):.4f}",
        "",
        "## Win Count",
        "",
        f"| Phase16 | Phase18 | Phase19 | Phase20 | Comparable |",
        f"|---------|---------|---------|---------|------------|",
        f"| {wins['phase16']} | {wins['phase18']} | {wins['phase19']} "
        f"| {wins['phase20']} | {wins['COMPARABLE']} |",
        "",
        "## Conclusion",
        "",
        f"**{conclusion}**",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[report] → {out_path}", flush=True)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}", flush=True)

    for label, p in [("p16", args.phase16_ckpt), ("p18", args.phase18_ckpt),
                     ("p19", args.phase19_ckpt), ("p20", args.phase20_ckpt)]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} checkpoint not found: {p}")

    pipe = load_pipeline(device=device, dtype=torch.float16)
    p16_vca = load_vca_standard(args.phase16_ckpt, device, init_scale=0.02)
    p18_vca = load_vca_standard(args.phase18_ckpt, device, init_scale=0.02)
    p19_vca = load_vca_from_ckpt(args.phase19_ckpt, device)   # scale=0.3
    p20_vca = load_vca_from_ckpt(args.phase20_ckpt, device)   # scale=0.3

    print(f"[init] loaded p16={args.phase16_ckpt}", flush=True)
    print(f"[init] loaded p18={args.phase18_ckpt}", flush=True)
    print(f"[init] loaded p19={args.phase19_ckpt} (6-layer)", flush=True)
    print(f"[init] loaded p20={args.phase20_ckpt} (mid_block)", flush=True)

    prompts_to_run = COMPARISON_PROMPTS
    if args.prompts:
        ids = [x.strip() for x in args.prompts.split(",")]
        prompts_to_run = [p for p in COMPARISON_PROMPTS if p["id"] in ids]

    out_root = Path(args.out_dir)
    all_comparisons = []

    for prompt_info in prompts_to_run:
        comp = process_prompt(pipe, p16_vca, p18_vca, p19_vca, p20_vca,
                              prompt_info, out_root / prompt_info["id"], args)
        all_comparisons.append(comp)

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / "all_sigma_stats.json", "w") as f:
        json.dump(all_comparisons, f, indent=2)

    make_report(all_comparisons, summary_dir / "comparison_report.md")

    if all_comparisons:
        best = max(all_comparisons,
                   key=lambda c: c["phase20"].get("sigma_separation", 0))
        src = out_root / best["prompt_id"] / "fourway.gif"
        dst = summary_dir / "best_fourway.gif"
        if src.exists():
            import shutil
            shutil.copy2(str(src), str(dst))
            p20_sep = best["phase20"].get("sigma_separation", 0)
            print(f"[summary] best_fourway ← {best['prompt_id']} "
                  f"(p20_sep={p20_sep:.4f})", flush=True)

    print(f"\n[done] results → {out_root}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 16/18/19/20 inference comparison")
    p.add_argument("--phase16-ckpt", required=True, dest="phase16_ckpt")
    p.add_argument("--phase18-ckpt", required=True, dest="phase18_ckpt")
    p.add_argument("--phase19-ckpt", required=True, dest="phase19_ckpt")
    p.add_argument("--phase20-ckpt", required=True, dest="phase20_ckpt")
    p.add_argument("--out-dir",      default="debug/comparison_p20", dest="out_dir")
    p.add_argument("--prompts",      default="")
    p.add_argument("--num-frames",   type=int, default=16, dest="num_frames")
    p.add_argument("--steps",        type=int, default=20)
    p.add_argument("--height",       type=int, default=256)
    p.add_argument("--width",        type=int, default=256)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
