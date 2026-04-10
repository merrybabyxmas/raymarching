"""
Phase 20: Fix 1+2+3 단독 적용 (mid_block 단일 주입 복원)

Phase 19 교훈 (FM-I10):
  Fix 4 (6-layer injection)가 zero-shot generalization을 훼손.
  p19 zero-shot 평균 0.006 vs p16 0.028 — 6-layer inference injection 과부하.

Phase 20 전략:
  Fix 1 유지: per-frame depth ranking loss (majority vote → frame-wise)
  Fix 2 유지: fixed probe 측정 (random t → 5-t 평균)
  Fix 3 유지: depth_pe_init_scale 0.02 → 0.3 (z-bin 분리 학습)
  Fix 4 제거: multi-layer 6개 → mid_block 단일 주입 복원
    → inference도 _inject() 단일 주입과 일치 → zero-shot 안정화 기대

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
from scripts.train_phase17 import adaptive_lambda_depth, RATIO_WARNING_THRESH
from scripts.train_phase19 import l_depth_ranking_perframe, PROBE_T_VALUES

# ─── 기본값 ───────────────────────────────────────────────────────────────────
DEFAULT_LAMBDA_DEPTH  = 0.3     # Phase 18 winner
DEFAULT_LAMBDA_ORTHO  = 0.005
DEFAULT_LR            = 5e-5
DEFAULT_EPOCHS        = 60
DEFAULT_T_MAX         = 200
DEPTH_PE_INIT_SCALE   = 0.3    # Fix 3 유지

# Fix 4 제거: mid_block 단일 주입 복원
INJECT_KEY_P20 = 'mid_block.attentions.0.transformer_blocks.0.attn2.processor'


# ─── Fix 2: 고정 probe sigma 측정 ───────────────────────────────────────────

def measure_probe_sep(pipe, vca_layer, probe_latents, probe_enc_hs, device):
    """Fix 2: 고정 probe × 5 t 값 → 안정적인 sigma_separation 측정."""
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


# ─── single mid_block VCA 주입 ───────────────────────────────────────────────

def inject_vca_p20(pipe, entity_context: torch.Tensor):
    """
    Fix 3: depth_pe_init_scale=0.3.
    Fix 4 제거: mid_block 단일 주입.
    inference와 training이 동일한 injection scope → generalization 안정화.
    """
    unet = pipe.unet
    vca_layer = VCALayer(
        query_dim=1280, context_dim=768,
        n_heads=8, n_entities=2, z_bins=2, lora_rank=8,
        use_softmax=False,
        depth_pe_init_scale=DEPTH_PE_INIT_SCALE,  # Fix 3: 0.3
    ).to(pipe.device)

    processor = TrainVCAProcessor(vca_layer, entity_context)
    original_procs = copy.copy(dict(unet.attn_processors))
    new_procs = dict(original_procs)

    injected = []
    if INJECT_KEY_P20 in new_procs:
        new_procs[INJECT_KEY_P20] = processor
        injected.append(INJECT_KEY_P20)

    unet.set_attn_processor(new_procs)
    print(f"[inject_p20] {len(injected)} layer: {injected}", flush=True)
    return vca_layer, injected, original_procs


# ─── training_step (Fix 1: per-frame, single-layer) ──────────────────────────

def training_step_p20(pipe, vca_layer, latents, encoder_hidden_states,
                      depth_orders, lambda_depth, lambda_ortho, device,
                      t_max=200):
    """
    Fix 1: per-frame depth ranking via sigma_acc (single-layer).
    단일 주입이므로 sigma_acc는 forward 후 1개 원소.
    frame-wise loss는 여전히 majority vote보다 정확한 gradient 제공.
    """
    noise = torch.randn_like(latents)
    t = torch.randint(0, t_max, (1,), device=device).long()
    noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

    vca_layer.reset_sigma_acc()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        pred_noise = pipe.unet(
            noisy_latents, t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

    ld = loss_diff(pred_noise.float(), noise.float())

    sigma_acc = vca_layer.sigma_acc   # len=1 (single layer)
    if sigma_acc:
        # Fix 1: per-frame across BF frames (even with 1-layer acc)
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

    # probe 세트 (Fix 2용): 첫 번째 시퀀스 고정
    probe_frames, _, probe_orders, probe_meta = dataset[0]
    probe_entity_ctx = get_entity_context_from_meta(pipe, probe_meta, device)
    print(f"[probe] '{probe_meta.get('prompt_entity0')}' vs "
          f"'{probe_meta.get('prompt_entity1')}'", flush=True)

    # VCA 주입 (Fix 3, Fix 4 제거)
    vca_layer, injected_keys, original_procs = inject_vca_p20(pipe, probe_entity_ctx)

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
            proc = pipe.unet.attn_processors.get(INJECT_KEY_P20)
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
            step_out = training_step_p20(
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
                "inject_key":          INJECT_KEY_P20,
                "depth_pe_init_scale": DEPTH_PE_INIT_SCALE,
                "multi_layer":         False,
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
    p.add_argument("--save-dir",     default="checkpoints/phase20",    dest="save_dir")
    p.add_argument("--debug-dir",    default="debug/train_phase20",    dest="debug_dir")
    p.add_argument("--n-frames",     type=int,   default=8,  dest="n_frames")
    p.add_argument("--height",       type=int,   default=256)
    p.add_argument("--width",        type=int,   default=256)
    p.add_argument("--max-samples",  type=int,   default=None, dest="max_samples")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
