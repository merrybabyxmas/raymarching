"""
Phase 14 Part B: 8-prompt 비교 실험

BEST_EXP 체크포인트를 사용해 In-Distribution + Zero-Shot 프롬프트 비교.

프롬프트 구성:
  In-Distribution (2): chain_in, robot_in  — 학습 시나리오
  Zero-Shot       (6): wrestling, swords, dance, snakes, cables, fighters

출력:
  debug/comparison/{prompt_name}/
    baseline.gif          — VCA 없는 원본
    vca_generated.gif     — 학습 VCA 적용
    side_by_side.gif      — [Baseline | VCA] 2-panel
    debug_sigma.gif       — [RGB | E0σ | E1σ] 3-panel
    sigma_stats.json      — {sigma_separation, sigma_consistency, e0_z0, e1_z0}

  debug/comparison/summary/
    all_stats.json        — 전체 프롬프트 통계
    report.md             — In-Distribution vs Zero-Shot 분석 리포트
"""
import argparse
import copy
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import imageio.v2 as iio2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from scripts.run_animatediff import load_pipeline, FixedContextVCAProcessor
from scripts.train_animatediff_vca import get_entity_embedding_mean
from scripts.make_figures import (
    _inject, _restore, generate, compute_stats,
    make_side_by_side, make_debug_sigma, extract_sigma_maps,
    _label, _resize_frame, _sigma_to_heatmap,
)


# ─── 프롬프트 정의 ────────────────────────────────────────────────────────────

PROMPTS = {
    # In-Distribution
    'chain_in': {
        'category': 'in_dist',
        'entity_0': 'a red chain link rotating in the XZ plane',
        'entity_1': 'a blue chain link rotating in the YZ plane',
        'full': 'a red chain link rotating in the XZ plane and a blue chain link rotating in the YZ plane',
    },
    'robot_in': {
        'category': 'in_dist',
        'entity_0': 'a red robotic arm reaching toward center from the left',
        'entity_1': 'a blue robotic arm reaching toward center from the right',
        'full': 'a red robotic arm reaching toward center from the left and a blue robotic arm reaching toward center from the right',
    },
    # Zero-Shot
    'wrestling': {
        'category': 'zero_shot',
        'entity_0': 'a red wrestler performing a move',
        'entity_1': 'a blue wrestler defending',
        'full': 'a red wrestler performing a move and a blue wrestler defending',
    },
    'swords': {
        'category': 'zero_shot',
        'entity_0': 'a red swordsman attacking',
        'entity_1': 'a blue swordsman parrying',
        'full': 'a red swordsman attacking and a blue swordsman parrying',
    },
    'dance': {
        'category': 'zero_shot',
        'entity_0': 'a red dancer spinning',
        'entity_1': 'a blue dancer leaping',
        'full': 'a red dancer spinning and a blue dancer leaping',
    },
    'snakes': {
        'category': 'zero_shot',
        'entity_0': 'a red snake coiling',
        'entity_1': 'a blue snake uncoiling',
        'full': 'a red snake coiling and a blue snake uncoiling',
    },
    'cables': {
        'category': 'zero_shot',
        'entity_0': 'a red cable swinging from the left',
        'entity_1': 'a blue cable swinging from the right',
        'full': 'a red cable swinging from the left and a blue cable swinging from the right',
    },
    'fighters': {
        'category': 'zero_shot',
        'entity_0': 'a red fighter jet diving',
        'entity_1': 'a blue fighter jet climbing',
        'full': 'a red fighter jet diving and a blue fighter jet climbing',
    },
}


# ─── VCA 로드 ─────────────────────────────────────────────────────────────────

def load_vca(ckpt_path: Path, device: str) -> VCALayer:
    """체크포인트에서 VCALayer 로드."""
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    vca = VCALayer(
        query_dim=1280, context_dim=768,
        n_heads=8, n_entities=2, z_bins=2, lora_rank=8,
        use_softmax=False,
    ).to(device)
    vca.load_state_dict(ckpt['vca_state_dict'])
    vca.eval()
    return vca


# ─── entity context 빌드 ──────────────────────────────────────────────────────

def build_ctx_from_prompt(pipe, info: dict, device: str) -> torch.Tensor:
    """프롬프트 info dict → (1,2,768) entity_context fp32"""
    e0 = get_entity_embedding_mean(pipe, info['entity_0'])   # (1,1,768)
    e1 = get_entity_embedding_mean(pipe, info['entity_1'])   # (1,1,768)
    ctx = torch.cat([e0, e1], dim=1).float()                  # (1,2,768)
    diff = float((ctx[0, 0] - ctx[0, 1]).norm())
    print(f"  entity_ctx diff_norm={diff:.4f}", flush=True)
    return ctx.to(device)


# ─── 단일 프롬프트 처리 ───────────────────────────────────────────────────────

def process_prompt(pipe, vca_layer: VCALayer, name: str, info: dict,
                   out_dir: Path, args) -> dict:
    """한 프롬프트에 대해 baseline + VCA 생성 + 저장."""
    print(f"\n[{name}] category={info['category']}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pipe.device if hasattr(pipe, 'device') else 'cuda'
    ctx = build_ctx_from_prompt(pipe, info, device)

    prompt = info['full']

    # ── baseline 생성 ──────────────────────────────────────────────────────
    print(f"  [{name}] Generating baseline...", flush=True)
    baseline_frames = generate(
        pipe, prompt,
        num_frames=args.num_frames, steps=args.steps,
        height=args.height, width=args.width, seed=args.seed,
    )

    # ── VCA 주입 후 생성 ───────────────────────────────────────────────────
    print(f"  [{name}] Generating with VCA...", flush=True)
    orig = _inject(pipe, vca_layer, ctx)

    vca_frames = generate(
        pipe, prompt,
        num_frames=args.num_frames, steps=args.steps,
        height=args.height, width=args.width, seed=args.seed,
    )

    # sigma 추출 (마지막 forward pass에서)
    sigma_maps = extract_sigma_maps(vca_layer, num_frames=args.num_frames, use_cfg=True)
    stats = compute_stats(vca_layer, num_frames=args.num_frames, use_cfg=True)

    _restore(pipe, orig)

    # ── 저장 ──────────────────────────────────────────────────────────────
    # baseline.gif (FM-I2: imageio.v2)
    iio2.mimsave(str(out_dir / 'baseline.gif'), baseline_frames, duration=250)
    # vca_generated.gif
    iio2.mimsave(str(out_dir / 'vca_generated.gif'), vca_frames, duration=250)

    # side_by_side.gif
    make_side_by_side(baseline_frames, vca_frames,
                      out_dir / 'side_by_side.gif', panel_size=args.height)

    # debug_sigma.gif (sigma_maps가 있을 때만)
    if sigma_maps is not None:
        make_debug_sigma(vca_frames, sigma_maps,
                         out_dir / 'debug_sigma.gif',
                         label=name, panel_size=args.height)

    # sigma_stats.json
    with open(out_dir / 'sigma_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(
        f"  [{name}] sep={stats['sigma_separation']:.4f}  "
        f"cons={stats.get('sigma_consistency', 0):.4f}  "
        f"e0={stats.get('e0_z0', 0):.4f}  e1={stats.get('e1_z0', 0):.4f}",
        flush=True,
    )

    return {'name': name, 'category': info['category'], **stats}


# ─── report.md 생성 ──────────────────────────────────────────────────────────

def make_report(all_stats: list, out_path: Path, best_exp_info: dict):
    """In-Distribution vs Zero-Shot 분석 리포트 작성."""
    in_dist  = [s for s in all_stats if s['category'] == 'in_dist']
    zero_shot = [s for s in all_stats if s['category'] == 'zero_shot']

    def avg(lst, key):
        vals = [s[key] for s in lst if key in s]
        return sum(vals) / len(vals) if vals else 0.0

    in_sep  = avg(in_dist,   'sigma_separation')
    zs_sep  = avg(zero_shot, 'sigma_separation')
    in_cons = avg(in_dist,   'sigma_consistency')
    zs_cons = avg(zero_shot, 'sigma_consistency')

    best_name = best_exp_info.get('best_exp', 'unknown')
    ckpt_path = best_exp_info.get('checkpoint', 'unknown')
    train_sep = best_exp_info.get('train_sep', 0.0)
    inf_sep   = best_exp_info.get('inference_sep', 0.0)
    gap       = best_exp_info.get('gap', 0.0)

    lines = [
        f"# Phase 14 Experiment Report",
        f"",
        f"**Generated**: {date.today().isoformat()}",
        f"**Best experiment**: `{best_name}`",
        f"**Checkpoint**: `{ckpt_path}`",
        f"",
        f"## Best Experiment Config",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Train sigma_separation | {train_sep:.4f} |",
        f"| Inference sigma_separation | {inf_sep:.4f} |",
        f"| Gap (train - inference) | {gap:.4f} |",
        f"",
        f"## In-Distribution Results",
        f"",
        f"| Prompt | sigma_sep | consistency | e0_z0 | e1_z0 |",
        f"|--------|-----------|-------------|-------|-------|",
    ]
    for s in in_dist:
        lines.append(
            f"| {s['name']} | {s.get('sigma_separation',0):.4f} "
            f"| {s.get('sigma_consistency',0):.4f} "
            f"| {s.get('e0_z0',0):.4f} | {s.get('e1_z0',0):.4f} |"
        )
    lines += [
        f"",
        f"**Average**: sep={in_sep:.4f}, consistency={in_cons:.4f}",
        f"",
        f"## Zero-Shot Results",
        f"",
        f"| Prompt | sigma_sep | consistency | e0_z0 | e1_z0 |",
        f"|--------|-----------|-------------|-------|-------|",
    ]
    for s in zero_shot:
        lines.append(
            f"| {s['name']} | {s.get('sigma_separation',0):.4f} "
            f"| {s.get('sigma_consistency',0):.4f} "
            f"| {s.get('e0_z0',0):.4f} | {s.get('e1_z0',0):.4f} |"
        )
    lines += [
        f"",
        f"**Average**: sep={zs_sep:.4f}, consistency={zs_cons:.4f}",
        f"",
        f"## Analysis",
        f"",
        f"- In-Distribution avg sigma_separation: **{in_sep:.4f}**",
        f"- Zero-Shot avg sigma_separation: **{zs_sep:.4f}**",
        f"- Generalization ratio: **{zs_sep/max(in_sep,1e-6):.2f}x** "
        f"({'good' if zs_sep/max(in_sep,1e-6) > 0.5 else 'limited'})",
        f"",
        f"### Category Summary",
        f"",
        f"VCA sigma separation {'generalizes well' if zs_sep > 0.05 else 'is limited'} "
        f"to zero-shot prompts "
        f"(avg sep={zs_sep:.4f} vs in-dist sep={in_sep:.4f}).",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines))
    print(f"[report] → {out_path}", flush=True)


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[init] device={device}", flush=True)

    # best_exp.json 로드
    best_exp_path = Path(args.best_exp_json)
    if not best_exp_path.exists():
        raise FileNotFoundError(f"best_exp.json not found: {best_exp_path}\n"
                                f"Run scripts/train_experiments.py first.")
    with open(best_exp_path) as f:
        best_exp_info = json.load(f)

    ckpt_path = Path(best_exp_info['checkpoint'])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[init] best_exp={best_exp_info['best_exp']}", flush=True)
    print(f"[init] checkpoint={ckpt_path}", flush=True)

    # 파이프라인 로드
    pipe = load_pipeline(device=device, dtype=torch.float16)

    # VCA 로드
    vca_layer = load_vca(ckpt_path, device)
    print(f"[init] VCA loaded from {ckpt_path}", flush=True)

    # 프롬프트 필터링
    if args.prompts:
        prompt_names = [p.strip() for p in args.prompts.split(',')]
        prompts_to_run = {k: v for k, v in PROMPTS.items() if k in prompt_names}
    else:
        prompts_to_run = PROMPTS

    out_root = Path(args.out_dir)
    all_stats = []

    for name, info in prompts_to_run.items():
        prompt_dir = out_root / name
        stats = process_prompt(pipe, vca_layer, name, info, prompt_dir, args)
        all_stats.append(stats)

    # summary 저장
    summary_dir = out_root / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / 'all_stats.json', 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"[summary] all_stats.json → {summary_dir / 'all_stats.json'}", flush=True)

    # report.md 생성
    make_report(all_stats, summary_dir / 'report.md', best_exp_info)

    # best side_by_side (sigma_separation 최대)
    best_stat = max(all_stats, key=lambda s: s.get('sigma_separation', 0))
    best_sbs_src = out_root / best_stat['name'] / 'side_by_side.gif'
    best_sbs_dst = summary_dir / 'best_side_by_side.gif'
    if best_sbs_src.exists():
        import shutil
        shutil.copy2(str(best_sbs_src), str(best_sbs_dst))
        print(f"[summary] best_side_by_side.gif ← {best_stat['name']}", flush=True)

    print(f"\n[done] comparison results → {out_root}", flush=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--best-exp-json', default='debug/experiments/best_exp.json',
                   dest='best_exp_json')
    p.add_argument('--out-dir',       default='debug/comparison',    dest='out_dir')
    p.add_argument('--prompts',       default='',
                   help='comma-separated prompt names (default: all 8)')
    p.add_argument('--num-frames',    type=int, default=16, dest='num_frames')
    p.add_argument('--steps',         type=int, default=20)
    p.add_argument('--height',        type=int, default=256)
    p.add_argument('--width',         type=int, default=256)
    p.add_argument('--seed',          type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
