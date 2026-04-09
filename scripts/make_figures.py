"""
Phase 13: Paper figure generation

4가지 Figure 생성:
  Figure 1  side_by_side.gif     — baseline vs trained VCA 나란히
  Figure 2  debug_before.gif     — 미학습 VCA sigma 3-panel
            debug_after.gif      — 학습 후 VCA sigma 3-panel
  Figure 3  comparison.gif       — Sigmoid vs Softmax 4-panel (구조적 차이)
  Figure 4  training_progress.gif — before/after sigma 4-panel

FM-I2: 모든 GIF는 imageio.v2.mimsave() 사용
seed=42 고정: baseline과 generated 동일 조건 비교
debug_before: 체크포인트 로드 금지, 새 VCALayer 초기화 (sigma ≈ 0.5)
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
from scripts.run_animatediff import (
    load_pipeline, FixedContextVCAProcessor,
)
from scripts.train_animatediff_vca import (
    get_entity_embedding_mean, build_entity_context,
)

# ─── 프롬프트 ─────────────────────────────────────────────────────────────────

def load_prompt(scenario: str) -> dict:
    with open('toy/data/prompts.json') as f:
        return json.load(f)[scenario]


def full_prompt(scenario: str) -> str:
    info = load_prompt(scenario)
    return f"{info['entity_0']} and {info['entity_1']}"


# ─── VCA 주입 / 제거 ──────────────────────────────────────────────────────────

def _inject(pipe, vca_layer: VCALayer, entity_ctx: torch.Tensor):
    """mid_block attn2에 FixedContextVCAProcessor 주입. original_procs 반환."""
    proc = FixedContextVCAProcessor(vca_layer, entity_ctx.half().to(pipe.device))
    orig = copy.copy(dict(pipe.unet.attn_processors))
    new = {}
    for k, p in orig.items():
        new[k] = proc if ('mid_block' in k and 'attn2' in k) else p
    pipe.unet.set_attn_processor(new)
    return orig


def _restore(pipe, original_procs: dict):
    pipe.unet.set_attn_processor(original_procs)


# ─── 영상 생성 ────────────────────────────────────────────────────────────────

def generate(pipe, prompt: str, num_frames=16, steps=20,
             height=256, width=256, seed=42) -> list:
    """list of (H,W,3) uint8"""
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(
        prompt=prompt, num_frames=num_frames, num_inference_steps=steps,
        guidance_scale=7.5, height=height, width=width,
        generator=gen, output_type='pil',
    )
    return [np.array(f) for f in out.frames[0]]


# ─── sigma 추출 ───────────────────────────────────────────────────────────────

def extract_sigma_maps(vca_layer: VCALayer, num_frames: int,
                       use_cfg: bool = True) -> Optional[list]:
    """last_sigma → list of (N, hw, hw) float32 per frame (z=0 slice)"""
    sigma = vca_layer.last_sigma
    if sigma is None:
        return None
    sigma_np = sigma.detach().cpu().float().numpy()  # (BF, S, N, Z)
    BF, S, N, Z = sigma_np.shape
    if use_cfg:
        F = min(num_frames, BF // 2)
        cond = sigma_np[-F:]                       # (F, S, N, Z)
    else:
        F = min(num_frames, BF)
        cond = sigma_np[:F]
    hw = max(1, int(S ** 0.5))
    z0 = cond[:, :, :, 0]                          # (F, S, N)
    z0 = z0.transpose(0, 2, 1)                     # (F, N, S)
    z0 = z0.reshape(F, N, hw, hw)                  # (F, N, hw, hw)
    return [z0[fi] for fi in range(F)]             # list of (N, hw, hw)


def compute_stats(vca_layer: VCALayer, num_frames: int,
                  use_cfg: bool = True) -> dict:
    sigma = vca_layer.last_sigma
    if sigma is None:
        return {'sigma_separation': 0.0, 'e0_z0': 0.5, 'e1_z0': 0.5}
    sigma_np = sigma.detach().cpu().float().numpy()
    if use_cfg:
        F = min(num_frames, sigma_np.shape[0] // 2)
        cond = sigma_np[-F:]
    else:
        cond = sigma_np[:min(num_frames, sigma_np.shape[0])]
    e0 = float(cond[:, :, 0, 0].mean())
    e1 = float(cond[:, :, 1, 0].mean())
    return {
        'sigma_separation': abs(e0 - e1),
        'sigma_consistency': float(e0 > e1),
        'e0_z0': e0,
        'e1_z0': e1,
    }


# ─── 그리기 유틸 ─────────────────────────────────────────────────────────────

def _get_font(size=14):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _label(arr: np.ndarray, text: str, size=14) -> np.ndarray:
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.text((4, 4), text, fill=(255, 255, 255), font=_get_font(size))
    return np.array(img)


def _sigma_to_heatmap(sigma_hw: np.ndarray, panel_size: int) -> np.ndarray:
    s = sigma_hw.copy()
    lo, hi = s.min(), s.max()
    s = (s - lo) / (hi - lo + 1e-6)
    r = np.clip(s * 3.0,        0, 1)
    g = np.clip(s * 3.0 - 1.0, 0, 1)
    b = np.clip(s * 3.0 - 2.0, 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    return np.array(Image.fromarray(rgb).resize(
        (panel_size, panel_size), Image.NEAREST))


def _resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    return np.array(Image.fromarray(frame).resize((size, size), Image.BILINEAR))


# ─── Figure 1: side_by_side ───────────────────────────────────────────────────

def make_side_by_side(baseline_frames: list, generated_frames: list,
                      out_path: Path, panel_size: int = 256):
    """[Baseline | VCA] 2-panel GIF"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames = []
    for bf, gf in zip(baseline_frames, generated_frames):
        p0 = _label(_resize_frame(bf, panel_size), "Baseline")
        p1 = _label(_resize_frame(gf, panel_size), "VCA")
        gif_frames.append(np.concatenate([p0, p1], axis=1))
    iio2.mimsave(str(out_path), gif_frames, duration=250)


# ─── Figure 2: debug_before / debug_after ─────────────────────────────────────

def make_debug_sigma(frames_rgb: list, sigma_maps: list,
                     out_path: Path, label: str = "",
                     panel_size: int = 256):
    """[RGB | E0 σ | E1 σ] 3-panel GIF"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames = []
    for i, (rgb, sm) in enumerate(zip(frames_rgb, sigma_maps)):
        p0 = _label(_resize_frame(rgb, panel_size), f"{label} RGB")
        p1 = _label(_sigma_to_heatmap(sm[0], panel_size),
                    f"E0 σ={sm[0].mean():.3f}")
        p2 = _label(_sigma_to_heatmap(sm[1], panel_size),
                    f"E1 σ={sm[1].mean():.3f}")
        gif_frames.append(np.concatenate([p0, p1, p2], axis=1))
    iio2.mimsave(str(out_path), gif_frames, duration=250)


# ─── Figure 3: comparison (Sigmoid vs Softmax) ────────────────────────────────

def make_comparison(sigmoid_maps: list, softmax_maps: list,
                    out_path: Path, panel_size: int = 64):
    """[Sig-E0 | Sig-E1 | Sof-E0 | Sof-E1] 4-panel GIF"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames = []
    for sm_sig, sm_sof in zip(sigmoid_maps, softmax_maps):
        panels = [
            _label(_sigma_to_heatmap(sm_sig[0], panel_size),
                   f"Sig-E0 {sm_sig[0].mean():.2f}"),
            _label(_sigma_to_heatmap(sm_sig[1], panel_size),
                   f"Sig-E1 {sm_sig[1].mean():.2f}"),
            _label(_sigma_to_heatmap(sm_sof[0], panel_size),
                   f"Sof-E0 {sm_sof[0].mean():.2f}"),
            _label(_sigma_to_heatmap(sm_sof[1], panel_size),
                   f"Sof-E1 {sm_sof[1].mean():.2f}"),
        ]
        gif_frames.append(np.concatenate(panels, axis=1))
    iio2.mimsave(str(out_path), gif_frames, duration=250)


# ─── Figure 4: training_progress ─────────────────────────────────────────────

def make_training_progress(before_maps: list, after_maps: list,
                            out_path: Path, panel_size: int = 64):
    """[Bef-E0 | Bef-E1 | Aft-E0 | Aft-E1] 4-panel GIF"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames = []
    for bm, am in zip(before_maps, after_maps):
        panels = [
            _label(_sigma_to_heatmap(bm[0], panel_size),
                   f"Bef-E0 {bm[0].mean():.2f}"),
            _label(_sigma_to_heatmap(bm[1], panel_size),
                   f"Bef-E1 {bm[1].mean():.2f}"),
            _label(_sigma_to_heatmap(am[0], panel_size),
                   f"Aft-E0 {am[0].mean():.2f}"),
            _label(_sigma_to_heatmap(am[1], panel_size),
                   f"Aft-E1 {am[1].mean():.2f}"),
        ]
        gif_frames.append(np.concatenate(panels, axis=1))
    iio2.mimsave(str(out_path), gif_frames, duration=250)


# ─── 체크포인트 로드 ─────────────────────────────────────────────────────────

def load_vca_from_checkpoint(ckpt_path: Path, device: str) -> VCALayer:
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    vca = VCALayer(query_dim=1280, context_dim=768, n_heads=8,
                   n_entities=2, z_bins=2, lora_rank=8,
                   use_softmax=False).to(device)
    vca.load_state_dict(ckpt['vca_state_dict'])
    vca.eval()
    return vca


def make_fresh_vca(device: str, use_softmax: bool = False) -> VCALayer:
    vca = VCALayer(query_dim=1280, context_dim=768, n_heads=8,
                   n_entities=2, z_bins=2, lora_rank=8,
                   use_softmax=use_softmax).to(device)
    vca.eval()
    return vca


# ─── 시나리오별 figure 생성 ───────────────────────────────────────────────────

def process_scenario(
    scenario: str,
    ckpt_path: Path,
    pipe,
    entity_ctx: torch.Tensor,
    out_dir: Path,
    args,
) -> dict:
    """한 시나리오의 모든 figures 생성. sigma_stats dict 반환."""
    sc_dir = out_dir / scenario
    sc_dir.mkdir(parents=True, exist_ok=True)
    device = pipe.device if hasattr(pipe, 'device') else 'cuda'
    prompt = full_prompt(scenario)

    do_figs = set(args.figures) if args.figures else {1, 2, 3, 4}

    # ── 1. Baseline (VCA 없음) ─────────────────────────────────────────────
    print(f"[{scenario}] Generating baseline...", flush=True)
    baseline_frames = generate(pipe, prompt, num_frames=args.num_frames,
                               steps=args.steps, height=args.height,
                               width=args.width, seed=args.seed)

    # ── 2. debug_before: 미학습 VCA (sigma ≈ 0.5) ─────────────────────────
    print(f"[{scenario}] Generating with fresh VCA (before)...", flush=True)
    fresh_sigmoid_vca = make_fresh_vca(device, use_softmax=False)
    orig = _inject(pipe, fresh_sigmoid_vca, entity_ctx)
    before_frames = generate(pipe, prompt, num_frames=args.num_frames,
                             steps=args.steps, height=args.height,
                             width=args.width, seed=args.seed)
    before_stats  = compute_stats(fresh_sigmoid_vca, args.num_frames)
    before_maps   = extract_sigma_maps(fresh_sigmoid_vca, args.num_frames)
    _restore(pipe, orig)

    # ── 3. debug_after: 학습된 VCA ───────────────────────────────────────
    print(f"[{scenario}] Generating with trained VCA (after)...", flush=True)
    trained_vca = load_vca_from_checkpoint(ckpt_path, device)
    orig = _inject(pipe, trained_vca, entity_ctx)
    after_frames = generate(pipe, prompt, num_frames=args.num_frames,
                            steps=args.steps, height=args.height,
                            width=args.width, seed=args.seed)
    after_stats  = compute_stats(trained_vca, args.num_frames)
    after_maps   = extract_sigma_maps(trained_vca, args.num_frames)
    _restore(pipe, orig)

    # ── 4. Softmax VCA (구조적 비교용) ────────────────────────────────────
    print(f"[{scenario}] Generating with Softmax VCA...", flush=True)
    softmax_vca = make_fresh_vca(device, use_softmax=True)
    orig = _inject(pipe, softmax_vca, entity_ctx)
    _ = generate(pipe, prompt, num_frames=args.num_frames,
                 steps=args.steps, height=args.height,
                 width=args.width, seed=args.seed)
    softmax_maps = extract_sigma_maps(softmax_vca, args.num_frames)
    _restore(pipe, orig)

    # ── Figure 1: side_by_side ────────────────────────────────────────────
    if 1 in do_figs:
        p = sc_dir / 'side_by_side.gif'
        make_side_by_side(baseline_frames, after_frames, p, panel_size=args.height)
        print(f"[{scenario}] Figure 1 → {p}", flush=True)

    # ── Figure 2: debug_before + debug_after ─────────────────────────────
    if 2 in do_figs:
        if before_maps:
            p = sc_dir / 'debug_before.gif'
            make_debug_sigma(before_frames, before_maps, p,
                             label="Before", panel_size=args.height)
            print(f"[{scenario}] Figure 2a → {p}", flush=True)
        if after_maps:
            p = sc_dir / 'debug_after.gif'
            make_debug_sigma(after_frames, after_maps, p,
                             label="After", panel_size=args.height)
            print(f"[{scenario}] Figure 2b → {p}", flush=True)

    # ── Figure 3: comparison (Sigmoid vs Softmax) ─────────────────────────
    if 3 in do_figs and after_maps and softmax_maps:
        p = sc_dir / 'comparison.gif'
        make_comparison(after_maps, softmax_maps, p, panel_size=64)
        print(f"[{scenario}] Figure 3 → {p}", flush=True)

    # ── Figure 4: training_progress ───────────────────────────────────────
    if 4 in do_figs and before_maps and after_maps:
        p = sc_dir / 'training_progress.gif'
        make_training_progress(before_maps, after_maps, p, panel_size=64)
        print(f"[{scenario}] Figure 4 → {p}", flush=True)

    stats = {'before': before_stats, 'after': after_stats}
    print(
        f"[{scenario}] sigma_sep: "
        f"{before_stats['sigma_separation']:.4f} → {after_stats['sigma_separation']:.4f}  "
        f"consistency: {after_stats['sigma_consistency']:.2f}",
        flush=True,
    )
    return stats


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main(args):
    device_id = _pick_best_gpu()
    device = f'cuda:{device_id}' if device_id is not None else 'cpu'
    import os
    if device_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        device = 'cuda'
    print(f"[init] device={device}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}", flush=True)

    out_dir = Path(args.out_dir)
    ckpt_path = Path(args.checkpoint)

    # 시나리오 결정
    if args.all:
        with open('toy/data/prompts.json') as f:
            scenarios = list(json.load(f).keys())
    else:
        scenarios = [args.scenario]

    # 파이프라인 로드 (한 번만)
    print("[init] Loading pipeline...", flush=True)
    pipe = load_pipeline(device=device, dtype=torch.float16)

    sigma_stats_all = {}
    manifest_figures = []

    for scenario in scenarios:
        # 체크포인트: {dir}/{scenario}_best.pt 우선, 없으면 지정 파일
        sc_ckpt = ckpt_path.parent / f'{scenario}_best.pt'
        if not sc_ckpt.exists():
            sc_ckpt = ckpt_path
        if not sc_ckpt.exists():
            print(f"[{scenario}] SKIP: checkpoint not found ({sc_ckpt})", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"[scenario] {scenario}  checkpoint={sc_ckpt}", flush=True)

        entity_ctx = build_entity_context(pipe, scenario, device)
        stats = process_scenario(scenario, sc_ckpt, pipe, entity_ctx, out_dir, args)
        sigma_stats_all[scenario] = stats

        # manifest 항목
        for fig_id, fname, desc in [
            (f'fig1_{scenario}', f'{scenario}/side_by_side.gif',
             f'VCA vs baseline, {scenario}'),
            (f'fig2b_{scenario}', f'{scenario}/debug_after.gif',
             f'sigma after training, {scenario}'),
            (f'fig3_{scenario}', f'{scenario}/comparison.gif',
             f'Sigmoid vs Softmax, {scenario}'),
            (f'fig4_{scenario}', f'{scenario}/training_progress.gif',
             f'training progress, {scenario}'),
        ]:
            p = out_dir / fname
            if p.exists():
                manifest_figures.append({
                    'id': fig_id,
                    'path': fname,
                    'description': desc,
                    'sigma_separation_before': stats['before']['sigma_separation'],
                    'sigma_separation_after':  stats['after']['sigma_separation'],
                })

    # ── Summary ──────────────────────────────────────────────────────────
    summary_dir = out_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)

    # sigma_stats_all.json
    stats_path = summary_dir / 'sigma_stats_all.json'
    with open(stats_path, 'w') as f:
        json.dump(sigma_stats_all, f, indent=2)
    print(f"\n[summary] sigma_stats_all → {stats_path}", flush=True)

    # figure_manifest.json
    manifest = {
        'generated_at': str(date.today()),
        'figures': manifest_figures,
    }
    manifest_path = summary_dir / 'figure_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"[summary] figure_manifest → {manifest_path}", flush=True)

    # best_side_by_side.gif: sigma_separation 가장 높은 시나리오
    if sigma_stats_all:
        best_sc = max(sigma_stats_all,
                      key=lambda s: sigma_stats_all[s]['after']['sigma_separation'])
        src = out_dir / best_sc / 'side_by_side.gif'
        dst = summary_dir / 'best_side_by_side.gif'
        if src.exists():
            import shutil
            shutil.copy2(str(src), str(dst))
            print(f"[summary] best_side_by_side ({best_sc}) → {dst}", flush=True)

    print("\n[done] All figures generated.", flush=True)


# ─── GPU 선택 ─────────────────────────────────────────────────────────────────

def _pick_best_gpu():
    if not torch.cuda.is_available():
        return None
    best, best_free = 0, 0
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free, best = free, i
        except Exception:
            continue
    return best if best_free > 4 * 1024**3 else None


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--all', action='store_true')
    g.add_argument('--scenario', default=None)
    p.add_argument('--checkpoint', required=True,
                   help='Path to best.pt checkpoint (or {scenario}_best.pt in same dir)')
    p.add_argument('--out-dir', default='debug/figures')
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--num-frames', type=int, default=16, dest='num_frames')
    p.add_argument('--steps',      type=int, default=20)
    p.add_argument('--height',     type=int, default=256)
    p.add_argument('--width',      type=int, default=256)
    p.add_argument('--figures', type=lambda s: [int(x) for x in s.split(',')],
                   default=None, help='Comma-separated figure IDs, e.g. 1,2,3,4')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
