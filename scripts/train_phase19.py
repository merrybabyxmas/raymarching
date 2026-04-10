"""
Phase 19: 학습 근본 원인 4가지 수정

Fix 1 (HIGH)   — Per-frame depth ranking: majority vote 대신 프레임별 독립 손실
Fix 2 (HIGH)   — Fixed probe 측정: random t 1회 대신 고정 t 5회 평균 → 지표 안정화
Fix 3 (MEDIUM) — depth_pe init scale 0.02 → 0.3: z-bin 분리 학습 활성화
Fix 4 (MEDIUM-HIGH) — Multi-layer VCA 주입: mid_block 1개 → query_dim=1280 레이어 6개
                       (down_blocks.2 ×2, mid_block ×1, up_blocks.1 ×3)

FM-I7 교훈 유지: l_diff > l_depth_weighted × 10 (adaptive lambda)
Phase 18 winner: lambda_depth=0.3 유지
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.losses import l_ortho as loss_ortho, l_depth_ranking, l_diff as loss_diff
from scripts.run_animatediff import load_pipeline, FixedContextVCAProcessor
from scripts.train_animatediff_vca import (
    TrainVCAProcessor, compute_sigma_stats_train, save_sigma_gif,
    encode_frames_to_latents,
)
from scripts.train_objaverse_vca import (
    ObjaverseTrainDataset, check_dataset_quality, get_entity_context_from_meta,
)
from scripts.train_phase17 import (
    adaptive_lambda_depth, RATIO_WARNING_THRESH,
)
from torch.nn.functional import layer_norm

# ─── 기본값 ───────────────────────────────────────────────────────────────────
DEFAULT_LAMBDA_DEPTH  = 0.3     # Phase 18 winner
DEFAULT_LAMBDA_ORTHO  = 0.005
DEFAULT_LR            = 5e-5
DEFAULT_EPOCHS        = 60
DEFAULT_T_MAX         = 200
PROBE_T_VALUES        = [20, 60, 100, 140, 180]   # Fix 2: 고정 t 값들
DEPTH_PE_INIT_SCALE   = 0.3                         # Fix 3

# Fix 4: query_dim=1280인 모든 attn2 레이어
INJECT_KEYS_P19 = [
    'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor',
    'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor',
    'mid_block.attentions.0.transformer_blocks.0.attn2.processor',
    'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor',
    'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor',
]


# ─── Fix 1: 프레임별 depth ranking loss ─────────────────────────────────────

def l_depth_ranking_perframe(
    sigma_acc: list,
    depth_orders: list,
    margin: float = 0.05,
) -> torch.Tensor:
    """
    Fix 1: 각 레이어 × 각 프레임에 독립적으로 depth ranking 손실 적용.

    sigma_acc: list of (BF, S, N, Z) — 각 injection 레이어의 sigma (with grad)
    depth_orders: list of [front, back] per frame, len = BF

    Phase 16~18 문제: 8프레임 전체에 majority vote 단일 순서 → 절반 프레임 역방향 gradient
    Fix: 각 프레임의 실제 depth_order로 독립 계산 후 평균
    """
    if not sigma_acc:
        return torch.tensor(0.0, requires_grad=True)

    total = []
    for sigma_raw in sigma_acc:                  # 각 레이어
        BF = sigma_raw.shape[0]
        T  = min(BF, len(depth_orders))
        for fi in range(T):
            order       = depth_orders[fi]       # [front, back] for this frame
            frame_sigma = sigma_raw[fi:fi+1]     # (1, S, N, Z)
            total.append(l_depth_ranking(frame_sigma, order, margin))

    if not total:
        return torch.tensor(0.0, requires_grad=True)
    return torch.stack(total).mean()


# ─── Fix 2: 고정 probe sigma 측정 ───────────────────────────────────────────

def measure_probe_sep(pipe, vca_layer, probe_latents, probe_enc_hs, device):
    """
    Fix 2: 고정 probe × 여러 t 값 → 안정적인 sigma_separation 측정.

    Phase 16~18 문제: 매 epoch random t 1회 → t 노이즈가 지표 변동을 지배
    Fix: PROBE_T_VALUES 5개 t에서 forward → sigma_separation 평균
    """
    vca_layer.eval()
    noise = torch.randn_like(probe_latents)
    seps  = []
    with torch.no_grad():
        for t_val in PROBE_T_VALUES:
            t = torch.tensor([t_val], device=device)
            noisy = pipe.scheduler.add_noise(probe_latents, noise, t)
            vca_layer.reset_sigma_acc()
            pipe.unet(noisy, t, encoder_hidden_states=probe_enc_hs)
            if vca_layer.last_sigma is not None:
                stats = compute_sigma_stats_train(vca_layer.last_sigma)
                seps.append(stats['sigma_separation'])
    vca_layer.train()
    return float(sum(seps) / max(len(seps), 1))


# ─── Fix 4: multi-layer VCA 주입 ─────────────────────────────────────────────

def inject_vca_p19(pipe, entity_context: torch.Tensor):
    """
    Fix 3 + Fix 4: depth_pe_init_scale=0.3 + 모든 1280-dim attn2 레이어 주입.

    단일 VCALayer 공유 → 모든 injection point가 같은 파라미터 학습.
    각 forward 호출이 sigma_acc에 누적 → 레이어별 depth loss 계산 가능.
    """
    unet = pipe.unet
    vca_layer = VCALayer(
        query_dim=1280, context_dim=768,
        n_heads=8, n_entities=2, z_bins=2, lora_rank=8,
        use_softmax=False,
        depth_pe_init_scale=DEPTH_PE_INIT_SCALE,  # Fix 3: 0.02 → 0.3
    ).to(pipe.device)

    processor = TrainVCAProcessor(vca_layer, entity_context)
    original_procs = copy.copy(dict(unet.attn_processors))
    new_procs = dict(original_procs)

    injected = []
    for key in INJECT_KEYS_P19:
        if key in new_procs:
            new_procs[key] = processor
            injected.append(key)

    unet.set_attn_processor(new_procs)
    print(f"[inject_p19] {len(injected)} layers: {injected}", flush=True)
    return vca_layer, injected, original_procs


def inject_vca_p19_infer(pipe, vca_layer: VCALayer, entity_ctx: torch.Tensor):
    """추론 시 동일 레이어에 FixedContextVCAProcessor 주입."""
    unet = pipe.unet
    original_procs = copy.copy(dict(unet.attn_processors))
    proc = FixedContextVCAProcessor(vca_layer, entity_ctx.half().to(pipe.device))
    new_procs = dict(original_procs)
    injected = []
    for key in INJECT_KEYS_P19:
        if key in new_procs:
            new_procs[key] = proc
            injected.append(key)
    unet.set_attn_processor(new_procs)
    return original_procs, injected


# ─── training_step (Fix 1 + Fix 4 적용) ──────────────────────────────────────

def training_step_p19(pipe, vca_layer, latents, encoder_hidden_states,
                      depth_orders, lambda_depth, lambda_ortho, device,
                      t_max=200):
    """
    Fix 1: per-frame depth ranking via sigma_acc.
    Fix 4: vca_layer.reset_sigma_acc() 후 unet forward → 6개 레이어 sigma 누적.
    """
    noise = torch.randn_like(latents)
    t = torch.randint(0, t_max, (1,), device=device).long()
    noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

    vca_layer.reset_sigma_acc()   # Fix 4: 누적 초기화
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        pred_noise = pipe.unet(
            noisy_latents, t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

    ld = loss_diff(pred_noise.float(), noise.float())

    # Fix 1: per-frame loss across all accumulated layers
    sigma_acc = vca_layer.sigma_acc
    if sigma_acc:
        l_depth = l_depth_ranking_perframe(sigma_acc, depth_orders)
        l_ort   = loss_ortho(vca_layer.depth_pe)
    else:
        l_depth = torch.tensor(0.0, device=device)
        l_ort   = torch.tensor(0.0, device=device)

    loss = ld + lambda_depth * l_depth + lambda_ortho * l_ort
    return {
        "loss":        loss.detach().item(),
        "l_diff":      ld.detach().item(),
        "l_depth":     l_depth.detach().item(),
        "l_ortho":     l_ort.detach().item(),
        "loss_tensor": loss,
    }


# ─── 메인 학습 루프 ───────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}  lambda_depth={args.lambda_depth}", flush=True)

    # 데이터 품질 게이팅
    if args.stats_path and Path(args.stats_path).exists():
        if not check_dataset_quality(args.stats_path):
            return
    else:
        print("DATASET_OK: proceeding", flush=True)

    dataset = ObjaverseTrainDataset(
        data_root=args.data_root, max_samples=args.max_samples,
        n_frames=args.n_frames, height=args.height, width=args.width,
    )
    if len(dataset) == 0:
        print("DATASET_FAIL: no samples", flush=True)
        return

    n_samples = len(dataset)
    print(f"DATASET_OK: {n_samples} samples", flush=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=lambda x: x[0])

    pipe = load_pipeline(device=device, dtype=torch.float16)
    for p in pipe.unet.parameters():
        p.requires_grad = False

    save_dir  = Path(args.save_dir);  save_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(args.debug_dir); debug_dir.mkdir(parents=True, exist_ok=True)

    # probe 세트 (Fix 2용): 첫 3개 시퀀스 고정
    probe_samples = [dataset[i] for i in range(min(3, n_samples))]
    probe_frames, _, probe_orders, probe_meta = probe_samples[0]
    probe_entity_ctx = get_entity_context_from_meta(pipe, probe_meta, device)
    print(f"[probe] '{probe_meta.get('prompt_entity0')}' vs "
          f"'{probe_meta.get('prompt_entity1')}'", flush=True)

    # VCA 주입 (Fix 3 + Fix 4)
    vca_layer, injected_keys, original_procs = inject_vca_p19(pipe, probe_entity_ctx)

    # probe latents 미리 인코딩 (Fix 2)
    probe_latents = encode_frames_to_latents(pipe, probe_frames, device)
    full_probe_prompt = (f"{probe_meta.get('prompt_entity0','entity0')} and "
                         f"{probe_meta.get('prompt_entity1','entity1')}")
    probe_tokens = pipe.tokenizer(
        full_probe_prompt, return_tensors="pt", padding="max_length",
        max_length=pipe.tokenizer.model_max_length, truncation=True,
    ).to(device)
    with torch.no_grad():
        probe_enc_hs = pipe.text_encoder(**probe_tokens).last_hidden_state.half()

    trainable = [p for p in vca_layer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    print(f"[opt] trainable params: {sum(p.numel() for p in trainable):,}", flush=True)

    lambda_depth = args.lambda_depth
    lambda_ortho = args.lambda_ortho
    best_sep     = 0.0
    training_curve = []

    for epoch in range(args.epochs):
        vca_layer.train()
        epoch_losses = {"loss": 0., "l_diff": 0., "l_depth": 0., "l_ortho": 0.}
        epoch_steps  = 0
        last_frames_np = probe_frames

        for batch in loader:
            frames_np, depths_np, depth_orders, meta = batch
            last_frames_np = frames_np

            entity_ctx = get_entity_context_from_meta(pipe, meta, device)
            for key in injected_keys:
                proc = pipe.unet.attn_processors.get(key)
                if isinstance(proc, TrainVCAProcessor):
                    proc.entity_context = entity_ctx

            latents = encode_frames_to_latents(pipe, frames_np, device)
            full_prompt = (f"{meta.get('prompt_entity0','entity0')} and "
                           f"{meta.get('prompt_entity1','entity1')}")
            tokens = pipe.tokenizer(
                full_prompt, return_tensors="pt", padding="max_length",
                max_length=pipe.tokenizer.model_max_length, truncation=True,
            ).to(device)
            with torch.no_grad():
                enc_hs = pipe.text_encoder(**tokens).last_hidden_state.half()

            optimizer.zero_grad()
            step_out = training_step_p19(
                pipe, vca_layer, latents, enc_hs,
                depth_orders, lambda_depth, lambda_ortho, device, t_max=args.t_max,
            )
            step_out["loss_tensor"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += step_out[k]
            epoch_steps += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(epoch_steps, 1)

        # Fix 2: 고정 probe로 sep 측정
        probe_sep = measure_probe_sep(
            pipe, vca_layer, probe_latents, probe_enc_hs, device
        )

        l_diff_v  = epoch_losses["l_diff"]
        l_depth_w = epoch_losses["l_depth"] * lambda_depth
        ratio     = l_diff_v / max(l_depth_w, 1e-9)

        if l_depth_w > 0 and ratio < 1.0 / RATIO_WARNING_THRESH:
            lambda_depth = adaptive_lambda_depth(l_diff_v, l_depth_w, lambda_depth)

        print(
            f"epoch={epoch:3d} step={epoch_steps} "
            f"loss={epoch_losses['loss']:.4f} "
            f"l_diff={l_diff_v:.4f} "
            f"l_depth={l_depth_w:.4f} "
            f"l_ortho={epoch_losses['l_ortho']:.4f} "
            f"ratio={ratio:.1f}x "
            f"probe_sep={probe_sep:.4f}",
            flush=True,
        )

        training_curve.append({
            "epoch": epoch, "lambda_depth": lambda_depth,
            **epoch_losses, "l_depth_weighted": l_depth_w,
            "probe_sep": probe_sep,
        })

        if probe_sep > best_sep:
            best_sep = probe_sep
            torch.save({
                "vca_state_dict":      vca_layer.state_dict(),
                "epoch":               epoch,
                "probe_sep":           best_sep,
                "lambda_depth_final":  lambda_depth,
                "inject_keys":         INJECT_KEYS_P19,
                "depth_pe_init_scale": DEPTH_PE_INIT_SCALE,
                "multi_layer":         True,
            }, save_dir / "best.pt")
            print(f"[ckpt] best.pt (probe_sep={best_sep:.4f})", flush=True)

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            gif_path = debug_dir / f"sigma_epoch{epoch:03d}.gif"
            save_sigma_gif(last_frames_np, vca_layer.last_sigma, gif_path)
            print(f"[gif] → {gif_path}", flush=True)

    print(f"FINAL probe_sep={best_sep:.6f}", flush=True)
    if best_sep > 0.01:
        print("LEARNING=OK", flush=True)
    else:
        print("LEARNING=FAIL", flush=True)

    with open(debug_dir / "training_curve.json", "w") as f:
        json.dump(training_curve, f, indent=2)
    print(f"[done]", flush=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",    default="toy/data_objaverse",  dest="data_root")
    p.add_argument("--stats-path",   default="debug/dataset_stats/objaverse_stats.json",
                   dest="stats_path")
    p.add_argument("--lambda-depth", type=float, default=DEFAULT_LAMBDA_DEPTH, dest="lambda_depth")
    p.add_argument("--lambda-ortho", type=float, default=DEFAULT_LAMBDA_ORTHO, dest="lambda_ortho")
    p.add_argument("--epochs",       type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--lr",           type=float, default=DEFAULT_LR)
    p.add_argument("--t-max",        type=int,   default=DEFAULT_T_MAX, dest="t_max")
    p.add_argument("--save-dir",     default="checkpoints/phase19",    dest="save_dir")
    p.add_argument("--debug-dir",    default="debug/train_phase19",    dest="debug_dir")
    p.add_argument("--n-frames",     type=int,   default=8,  dest="n_frames")
    p.add_argument("--height",       type=int,   default=256)
    p.add_argument("--width",        type=int,   default=256)
    p.add_argument("--max-samples",  type=int,   default=None, dest="max_samples")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
