"""
Phase 14 Part A: t_max × λ_depth 실험

4가지 실험으로 train-inference sigma gap 최소화 최적 설정 탐색:
  baseline  : t_max=1000, λ_depth=0.1  (Phase 12 기본)
  exp1_late : t_max=200,  λ_depth=0.1  (late timestep only)
  exp2_high : t_max=1000, λ_depth=1.0  (high depth supervision)
  exp3_both : t_max=200,  λ_depth=1.0  (late + high)

BEST_EXP 선택 기준:
  1순위: inference_sep 최대
  2순위: gap (train_sep - inference_sep) 최소
  3순위: l_diff < 1.0 필터 (발산 제외)

출력 형식:
  [{name}] INFERENCE sigma_sep=X sigma_consistency=X
  [{name}] RESULT train_sep=X inference_sep=X gap=X
  BEST_EXP={name}
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as iio2
from torch.nn.functional import layer_norm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.losses import l_ortho as loss_ortho, l_depth_ranking, l_diff as loss_diff
from scripts.run_animatediff import load_pipeline, FixedContextVCAProcessor
from scripts.train_animatediff_vca import (
    load_frames, load_depths_and_masks, encode_frames_to_latents,
    inject_vca_train, build_entity_context,
    TrainVCAProcessor, compute_sigma_stats_train,
)


# ─── 실험 설정 ────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    {'name': 'baseline',   't_max': 1000, 'lambda_depth': 0.1},
    {'name': 'exp1_late',  't_max': 200,  'lambda_depth': 0.1},
    {'name': 'exp2_high',  't_max': 1000, 'lambda_depth': 1.0},
    {'name': 'exp3_both',  't_max': 200,  'lambda_depth': 1.0},
]


# ─── training_step (t_max 파라미터 추가) ─────────────────────────────────────

def training_step_tmax(
    pipe,
    vca_layer: VCALayer,
    latents: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    depth_orders: list,
    lambda_depth: float,
    lambda_ortho: float,
    device: str,
    t_max: int = 1000,
) -> dict:
    """t_max 범위의 timestep으로 학습 — t_max=200이면 후반 스텝만 사용 (inference와 유사)."""
    # ── noise 추가 ─────────────────────────────────────────────────────────
    noise = torch.randn_like(latents)
    t = torch.randint(0, t_max, (1,), device=device).long()
    noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

    # ── UNet forward (fp16, autocast) ──────────────────────────────────────
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        pred_noise = pipe.unet(
            noisy_latents, t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

    # ── L_diff ─────────────────────────────────────────────────────────────
    ld = loss_diff(pred_noise.float(), noise.float())

    # ── sigma_raw 수집 (FM-I6) ─────────────────────────────────────────────
    sigma_raw = vca_layer.last_sigma_raw
    if sigma_raw is None:
        l_depth = torch.tensor(0.0, device=device)
        l_ort   = torch.tensor(0.0, device=device)
    else:
        front_votes = sum(1 for d in depth_orders if d[0] == 0)
        order = [0, 1] if front_votes >= len(depth_orders) // 2 else [1, 0]
        l_depth = l_depth_ranking(sigma_raw, order)
        l_ort   = loss_ortho(vca_layer.depth_pe)

    loss = ld + lambda_depth * l_depth + lambda_ortho * l_ort

    return {
        'loss':        loss.detach().item(),
        'l_diff':      ld.detach().item(),
        'l_depth':     l_depth.detach().item(),
        'l_ortho':     l_ort.detach().item(),
        'loss_tensor': loss,
        'sigma_raw':   sigma_raw,
    }


# ─── inference sigma 측정 (clean latents, t=0 근처) ──────────────────────────

@torch.no_grad()
def measure_inference_sigma(pipe, vca_layer: VCALayer,
                             latents: torch.Tensor,
                             enc_hs: torch.Tensor,
                             device: str,
                             n_steps: int = 20) -> dict:
    """
    Denoising inference의 후반 스텝 (clean latents + 작은 t)에서
    sigma 측정 → train-inference 분포 갭 확인.

    pipeline.scheduler를 직접 써서 denoising trajectory의 마지막 스텝 시뮬레이션.
    """
    vca_layer.eval()
    # inference 조건: t를 [0, 50] 범위로 시뮬레이션
    t_inf = torch.tensor([25], device=device).long()
    noise_inf = torch.randn_like(latents) * 0.05  # very small noise
    noisy_inf = pipe.scheduler.add_noise(latents, noise_inf, t_inf)

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        _ = pipe.unet(
            noisy_inf, t_inf,
            encoder_hidden_states=enc_hs,
        ).sample

    stats = compute_sigma_stats_train(vca_layer.last_sigma)
    vca_layer.train()
    return stats


# ─── 실험 실행 ────────────────────────────────────────────────────────────────

def run_experiment(pipe, exp: dict, scenario: str,
                   frames_np: np.ndarray, latents: torch.Tensor,
                   depth_orders: list, enc_hs: torch.Tensor,
                   entity_ctx: torch.Tensor,
                   device: str, args) -> dict:
    """단일 실험 실행 → 결과 dict 반환."""
    name = exp['name']
    t_max = exp['t_max']
    lambda_depth = exp['lambda_depth']
    lambda_ortho = args.lambda_ortho

    print(f"\n{'='*60}", flush=True)
    print(f"[{name}] t_max={t_max} lambda_depth={lambda_depth}", flush=True)
    print(f"{'='*60}", flush=True)

    # VCA 주입 (항상 새로 초기화)
    vca_layer, injected_keys, original_procs = inject_vca_train(pipe, entity_ctx)
    print(f"[{name}] VCA injected → {injected_keys}", flush=True)

    trainable = [p for p in vca_layer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    # BEFORE 측정
    vca_layer.eval()
    with torch.no_grad():
        noise_b = torch.randn_like(latents)
        t_b = torch.randint(0, t_max, (1,), device=device).long()
        noisy_b = pipe.scheduler.add_noise(latents, noise_b, t_b)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _ = pipe.unet(noisy_b, t_b, encoder_hidden_states=enc_hs).sample
    before_stats = compute_sigma_stats_train(vca_layer.last_sigma)
    vca_layer.train()
    print(f"[{name}] BEFORE sep={before_stats['sigma_separation']:.6f}", flush=True)

    # 에폭 루프
    curve = []
    for epoch in range(args.epochs):
        vca_layer.train()
        optimizer.zero_grad()

        step_out = training_step_tmax(
            pipe, vca_layer, latents, enc_hs,
            depth_orders, lambda_depth, lambda_ortho, device, t_max,
        )

        step_out['loss_tensor'].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        stats = compute_sigma_stats_train(vca_layer.last_sigma)
        curve.append({
            'epoch': epoch,
            **{k: v for k, v in step_out.items() if k not in ('loss_tensor', 'sigma_raw')},
            **stats,
        })

        print(
            f"[{name}] epoch={epoch:3d}  loss={step_out['loss']:.4f}  "
            f"l_diff={step_out['l_diff']:.4f}  l_depth={step_out['l_depth']:.4f}  "
            f"sigma_sep={stats['sigma_separation']:.6f}",
            flush=True,
        )

    # AFTER (train 분포)
    vca_layer.eval()
    with torch.no_grad():
        noise_a = torch.randn_like(latents)
        t_a = torch.randint(0, t_max, (1,), device=device).long()
        noisy_a = pipe.scheduler.add_noise(latents, noise_a, t_a)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _ = pipe.unet(noisy_a, t_a, encoder_hidden_states=enc_hs).sample
    after_stats = compute_sigma_stats_train(vca_layer.last_sigma)

    train_sep = after_stats['sigma_separation']
    print(f"[{name}] TRAIN   sigma_sep={train_sep:.6f}", flush=True)

    # INFERENCE sigma 측정 (clean latents, t=25)
    inf_stats = measure_inference_sigma(pipe, vca_layer, latents, enc_hs, device)
    inf_sep  = inf_stats['sigma_separation']
    inf_cons = inf_stats.get('sigma_consistency', 0.0)
    gap = train_sep - inf_sep
    print(f"[{name}] INFERENCE sigma_sep={inf_sep:.6f} sigma_consistency={inf_cons:.4f}",
          flush=True)
    print(f"[{name}] RESULT train_sep={train_sep:.6f} inference_sep={inf_sep:.6f} gap={gap:.6f}",
          flush=True)

    # 체크포인트 저장
    ckpt_dir = Path(args.ckpt_root) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / 'best.pt'
    torch.save({
        'vca_state_dict': vca_layer.state_dict(),
        'epoch': args.epochs - 1,
        'sigma_stats': after_stats,
        'scenario': scenario,
        'exp_name': name,
        'train_sep': train_sep,
        'inference_sep': inf_sep,
        'gap': gap,
    }, best_path)
    print(f"[{name}] checkpoint → {best_path}", flush=True)

    # 원복
    pipe.unet.set_attn_processor(original_procs)

    # l_diff 최종값
    final_l_diff = curve[-1]['l_diff'] if curve else 1.0

    return {
        'name': name,
        't_max': t_max,
        'lambda_depth': lambda_depth,
        'train_sep': train_sep,
        'inference_sep': inf_sep,
        'gap': gap,
        'inf_consistency': inf_cons,
        'final_l_diff': final_l_diff,
        'curve': curve,
        'checkpoint': str(best_path),
    }


# ─── BEST_EXP 선택 ────────────────────────────────────────────────────────────

def pick_best(results: list) -> str:
    """
    1순위: inference_sep 최대
    2순위: gap (train-inference) 최소
    3순위: l_diff < 1.0 필터 (발산 제외)
    """
    # l_diff < 1.0인 것만 후보 (발산하지 않은 실험)
    valid = [r for r in results if r['final_l_diff'] < 1.0]
    if not valid:
        valid = results  # 모두 발산이면 전체

    # inference_sep 기준 정렬 (1순위), gap 기준 (2순위)
    valid.sort(key=lambda r: (-r['inference_sep'], r['gap']))
    return valid[0]['name']


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[init] device={device}", flush=True)

    # 파이프라인 로드
    pipe = load_pipeline(device=device, dtype=torch.float16)

    # UNet frozen
    for param in pipe.unet.parameters():
        param.requires_grad = False
    assert sum(1 for p in pipe.unet.parameters() if p.requires_grad) == 0
    print(f"[init] UNet frozen.", flush=True)

    scenario = args.scenario

    # 데이터 로드 (한 번만)
    frames_np    = load_frames(scenario, n_frames=args.n_frames,
                               height=args.height, width=args.width)
    depth_orders = load_depths_and_masks(scenario, n_frames=args.n_frames,
                                         height=args.height, width=args.width)
    print(f"[data] frames={frames_np.shape}", flush=True)

    # VAE 인코딩 (한 번만)
    latents = encode_frames_to_latents(pipe, frames_np, device)
    print(f"[vae] latents={latents.shape}", flush=True)

    # entity context (한 번만)
    entity_ctx = build_entity_context(pipe, scenario, device)

    # encoder_hidden_states
    with open('toy/data/prompts.json') as f:
        prompts_dict = json.load(f)
    info = prompts_dict[scenario]
    full_prompt = f"{info['entity_0']} and {info['entity_1']}"
    tokens = pipe.tokenizer(full_prompt, return_tensors='pt',
                            padding='max_length',
                            max_length=pipe.tokenizer.model_max_length,
                            truncation=True).to(device)
    with torch.no_grad():
        enc_hs = pipe.text_encoder(**tokens).last_hidden_state.half()

    # 실험 필터
    if args.experiments:
        exp_names = [e.strip() for e in args.experiments.split(',')]
        exps = [e for e in EXPERIMENTS if e['name'] in exp_names]
    else:
        exps = EXPERIMENTS

    # 실험 실행
    results = []
    for exp in exps:
        res = run_experiment(
            pipe, exp, scenario,
            frames_np, latents, depth_orders, enc_hs, entity_ctx,
            device, args,
        )
        results.append(res)

    # BEST_EXP 선택
    best_name = pick_best(results)
    print(f"\nBEST_EXP={best_name}", flush=True)

    # best_exp.json 저장
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    best_res = next(r for r in results if r['name'] == best_name)
    best_exp_data = {
        'best_exp': best_name,
        'scenario': scenario,
        'checkpoint': best_res['checkpoint'],
        'inference_sep': best_res['inference_sep'],
        'train_sep': best_res['train_sep'],
        'gap': best_res['gap'],
        'all_results': [
            {k: v for k, v in r.items() if k != 'curve'}
            for r in results
        ],
    }
    best_exp_path = debug_dir / 'best_exp.json'
    with open(best_exp_path, 'w') as f:
        json.dump(best_exp_data, f, indent=2)
    print(f"[done] best_exp.json → {best_exp_path}", flush=True)

    # experiment_results.json 저장
    all_curves = {r['name']: r['curve'] for r in results}
    curves_path = debug_dir / 'experiment_curves.json'
    with open(curves_path, 'w') as f:
        json.dump(all_curves, f, indent=2)
    print(f"[done] experiment_curves.json → {curves_path}", flush=True)

    # 요약 출력
    print(f"\n{'='*60}", flush=True)
    print(f"EXPERIMENT SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for r in results:
        marker = " ← BEST" if r['name'] == best_name else ""
        print(
            f"  {r['name']:12s}  train_sep={r['train_sep']:.4f}  "
            f"inf_sep={r['inference_sep']:.4f}  gap={r['gap']:.4f}{marker}",
            flush=True,
        )
    print(f"{'='*60}", flush=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--scenario',     default='chain')
    p.add_argument('--epochs',       type=int,   default=10)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--lambda-ortho', type=float, default=0.05, dest='lambda_ortho')
    p.add_argument('--n-frames',     type=int,   default=4, dest='n_frames')
    p.add_argument('--height',       type=int,   default=256)
    p.add_argument('--width',        type=int,   default=256)
    p.add_argument('--ckpt-root',    default='checkpoints/experiments', dest='ckpt_root')
    p.add_argument('--debug-dir',    default='debug/experiments',       dest='debug_dir')
    p.add_argument('--experiments',  default='',
                   help='comma-separated experiment names to run (default: all 4)')
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
