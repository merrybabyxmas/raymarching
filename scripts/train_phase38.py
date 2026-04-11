"""
Phase 38 — Entity-Slot Attention Training
==========================================

핵심 변경 (Phase 35~37 대비)
-----------------------------
Phase 35~37 (volumetric z_pe):
  - 단 하나의 attention stream에서 z_pe로 front/back을 분리하려 시도
  - K/V는 여전히 두 entity 토큰을 모두 포함 → z_pe가 아무리 커져도 chimera 불가피
  - entity 소멸이 chimera 감소로 보여지는 metric 왜곡

Phase 38 (Entity Slot Attention):
  - entity 전용 K/V stream으로 완전 분리 → chimera 구조적 불가능
  - Porter-Duff 합성 (VCA sigma depth 기반) → 물리적으로 올바른 occlusion
  - entity_score = survival × (1 - chimera) → metric 왜곡 방지

아키텍처
--------
  F_global = Attn(Q, K[all], V[all])          — 원본 품질 보존
  F_0      = Attn(Q, K[e0_toks], V[e0_toks])  — entity 0 전용
  F_1      = Attn(Q, K[e1_toks], V[e1_toks])  — entity 1 전용

  Porter-Duff compositing from VCA sigma:
    composed = w0*F_0 + w1*F_1 + w_bg*F_global
    output   = slot_blend*composed + (1-slot_blend)*F_global

손실 함수
---------
  L = λ_excl  × L_exclusive       ← entity i slot이 j exclusive 픽셀에서 F_global 재현
    + λ_ov    × L_overlap_ordering ← 겹침 영역에서 front entity w > back entity w
    + λ_depth × L_depth            ← VCA sigma depth ordering (Phase 31 동일)
    + λ_diff  × L_diff             ← 생성 품질 보존

학습 파라미터
-----------
  - vca_layer:   VCA sigma 계산 (Phase 31 체크포인트에서 초기화)
  - proc.slot_blend_raw: 합성 비율 (Phase 38 핵심 파라미터)
  - 나머지 UNet, text_encoder, VAE 동결

평가 기준
---------
  best checkpoint = entity_score 최고 (survival × (1 - chimera))
  Phase 37까지는 DRA / gen_diff로 평가 → metric 왜곡 방지를 위해 교체
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import imageio.v2 as iio2

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.entity_slot import (
    EntitySlotAttnProcessor,
    inject_entity_slot,
    l_entity_exclusive,
    l_overlap_ordering,
    entity_score as compute_entity_score,
    entity_survival_rate,
    chimera_rate,
)
from models.losses import l_diff as loss_diff
from scripts.run_animatediff import load_pipeline
from scripts.train_animatediff_vca import (
    compute_sigma_stats_train, encode_frames_to_latents,
)
from scripts.train_phase31 import (
    DEFAULT_LR,
    INJECT_KEY, INJECT_QUERY_DIM,
    DEPTH_PE_INIT_SCALE, VCA_ALPHA,
    ObjaverseDatasetWithMasks,
    make_color_prompts, get_color_entity_context,
    l_zorder_direct,
    restore_procs,
    measure_depth_rank_accuracy,
)
from scripts.train_phase35 import (
    DEFAULT_Z_BINS,
    get_entity_token_positions,
)


# ─── Phase 38 하이퍼파라미터 ──────────────────────────────────────────────────
DEFAULT_LAMBDA_EXCL   = 1.0    # L_exclusive: entity slot ↔ F_global in exclusive pixels
DEFAULT_LAMBDA_OV     = 2.0    # L_overlap_ordering: front entity dominance in overlap
DEFAULT_LAMBDA_DEPTH  = 3.0    # L_depth: VCA sigma depth ordering
DEFAULT_LAMBDA_DIFF   = 0.5    # L_diff: generation quality
DEFAULT_SLOT_BLEND    = 0.3    # initial slot blend ratio (30% composed, 70% global)
DEFAULT_SLOT_BLEND_MIN = 0.25  # minimum slot_blend to prevent collapse
DEFAULT_LAMBDA_BLEND_REG = 5.0 # regularizer weight to keep slot_blend >= min
DEFAULT_EPOCHS        = 100
DEFAULT_LR_SLOT       = 3e-4   # slot_blend_raw LR (higher — needs to move from ~0 to 1)
DEFAULT_LR_VCA        = 3e-5   # VCA layer LR (same as Phase 31)
DEFAULT_STEPS_PER_EPOCH = 20


# =============================================================================
# Inference: entity_score evaluation
# =============================================================================

@torch.no_grad()
def evaluate_entity_score(
    pipe,
    proc: EntitySlotAttnProcessor,
    vca_layer: VCALayer,
    orig_procs: dict,
    meta: dict,
    toks_e0: list,
    toks_e1: list,
    entity_ctx: torch.Tensor,
    device: str,
    seed: int = 42,
    n_frames: int = 16,
    n_steps: int = 20,
    height: int = 256,
    width: int = 256,
) -> tuple:
    """
    현재 proc으로 생성 후 entity_score 측정.
    Returns (entity_score, survival_rate, chimera_rate_, frames_rgb)
    """
    _, _, full_prompt, _, _ = make_color_prompts(meta)

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
    frames = (out.frames[0] * 255).astype(np.uint8)  # (T, H, W, 3)

    es, sr, cr = compute_entity_score(frames)
    return es, sr, cr, frames


# =============================================================================
# 훈련 루프
# =============================================================================

def train_phase38(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    debug_dir = Path(args.debug_dir)
    save_dir  = Path(args.save_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 파이프라인 로드 ──────────────────────────────────────────────────────
    print("[Phase 38] 파이프라인 로드 중...", flush=True)
    pipe = load_pipeline(device=device, dtype=torch.float16)

    # ── VCA 로드 (Phase 31 체크포인트) ───────────────────────────────────────
    print(f"[Phase 38] VCA 체크포인트 로드: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location=device)
    vca_layer = VCALayer(
        query_dim=INJECT_QUERY_DIM, context_dim=768,
        n_heads=8, n_entities=2, z_bins=DEFAULT_Z_BINS, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=DEPTH_PE_INIT_SCALE,
    ).to(device)
    vca_layer.load_state_dict(ckpt["vca_state_dict"])
    p31_gamma = float(ckpt.get("gamma_trained", VCA_ALPHA))
    print(f"  Phase 31 gamma_trained = {p31_gamma:.4f}", flush=True)

    # ── 데이터셋 ─────────────────────────────────────────────────────────────
    dataset = ObjaverseDatasetWithMasks(args.data_root, n_frames=args.n_frames)
    print(f"[Phase 38] 데이터셋: {len(dataset)} 시퀀스", flush=True)
    if len(dataset) == 0:
        raise RuntimeError(f"데이터셋 비어있음: {args.data_root}")

    # ── Probe 설정 ────────────────────────────────────────────────────────────
    probe_idx        = 0
    probe_sample     = dataset[probe_idx]
    probe_frames_np, _, probe_depth_orders, probe_meta, probe_entity_masks = probe_sample
    probe_entity_ctx = get_color_entity_context(pipe, probe_meta, device)
    probe_toks_e0, probe_toks_e1, probe_full_prompt = get_entity_token_positions(
        pipe, probe_meta)

    probe_latents = encode_frames_to_latents(pipe, probe_frames_np, device)
    probe_tok = pipe.tokenizer(
        probe_full_prompt, return_tensors="pt", padding="max_length",
        max_length=pipe.tokenizer.model_max_length, truncation=True,
    ).to(device)
    with torch.no_grad():
        probe_enc_hs = pipe.text_encoder(**probe_tok).last_hidden_state.half()

    # ── EntitySlotAttnProcessor 주입 ─────────────────────────────────────────
    proc, orig_procs = inject_entity_slot(
        pipe, vca_layer, probe_entity_ctx,
        inject_key=INJECT_KEY,
        slot_blend_init=args.slot_blend,
    )
    proc.set_entity_tokens(probe_toks_e0, probe_toks_e1)
    proc = proc.to(device)

    print(f"  slot_blend_init={args.slot_blend:.3f}  "
          f"(raw={float(proc.slot_blend_raw.item()):.4f})", flush=True)

    # ── 파라미터 동결 / 학습 대상 설정 ──────────────────────────────────────
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    for p in vca_layer.parameters():
        p.requires_grad_(True)
    proc.slot_blend_raw.requires_grad_(True)

    # ── 옵티마이저: slot_blend 는 더 높은 LR ────────────────────────────────
    param_groups = [
        {"params": list(vca_layer.parameters()), "lr": args.lr_vca,  "name": "vca"},
        {"params": [proc.slot_blend_raw],         "lr": args.lr_slot, "name": "slot"},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_vca * 0.05)

    # ── 메트릭 기록 ──────────────────────────────────────────────────────────
    history = []
    best_es    = -1.0
    best_epoch = -1

    print(f"\n[Phase 38] 훈련 시작: {args.epochs} epochs", flush=True)
    print(f"  λ_excl={args.lambda_excl}  λ_ov={args.lambda_ov}  "
          f"λ_depth={args.lambda_depth}  λ_diff={args.lambda_diff}", flush=True)
    print(f"  lr_vca={args.lr_vca:.2e}  lr_slot={args.lr_slot:.2e}", flush=True)

    for epoch in range(args.epochs):
        vca_layer.train()
        proc.train()

        epoch_losses = {
            "total": [], "excl": [], "ov": [], "depth": [], "diff": [],
        }

        indices = np.random.permutation(len(dataset)).tolist()

        for batch_idx, data_idx in enumerate(indices[:args.steps_per_epoch]):
            sample = dataset[data_idx]
            frames_np, _, depth_orders, meta, entity_masks = sample

            entity_ctx = get_color_entity_context(pipe, meta, device)
            toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)

            proc.set_entity_ctx(entity_ctx.float())
            proc.set_entity_tokens(toks_e0, toks_e1)
            proc.reset_slot_store()

            with torch.no_grad():
                latents = encode_frames_to_latents(pipe, frames_np, device)
            noise = torch.randn_like(latents)
            t     = torch.randint(0, args.t_max, (1,), device=device).long()
            noisy = pipe.scheduler.add_noise(latents, noise, t)

            tok = pipe.tokenizer(
                full_prompt, return_tensors="pt", padding="max_length",
                max_length=pipe.tokenizer.model_max_length, truncation=True,
            ).to(device)
            with torch.no_grad():
                enc_hs = pipe.text_encoder(**tok).last_hidden_state.half()

            T_frames       = min(frames_np.shape[0], entity_masks.shape[0])
            masks_t        = torch.from_numpy(
                entity_masks[:T_frames].astype(np.float32)).to(device)
            depth_orders_t = depth_orders[:T_frames]

            # ── UNet forward ──────────────────────────────────────────────
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = pipe.unet(noisy, t,
                                       encoder_hidden_states=enc_hs).sample

            # ── L_exclusive ───────────────────────────────────────────────
            l_excl = torch.tensor(0.0, device=device)
            if (proc.last_F0 is not None and proc.last_F1 is not None
                    and proc.last_Fg is not None):
                BF = proc.last_F0.shape[0]
                T_use = min(BF, T_frames)
                # masks: (B, 2, S) — stack frames as batch
                for fi in range(T_use):
                    F0_f  = proc.last_F0[fi:fi+1].float()   # (1, S, D)
                    F1_f  = proc.last_F1[fi:fi+1].float()
                    Fg_f  = proc.last_Fg[fi:fi+1].float()
                    m_f   = masks_t[fi:fi+1]                  # (1, 2, S)
                    l_excl = l_excl + l_entity_exclusive(F0_f, F1_f, Fg_f, m_f)
                l_excl = l_excl / max(T_use, 1)

            # ── L_overlap_ordering ────────────────────────────────────────
            l_ov = torch.tensor(0.0, device=device)
            if proc.last_w0 is not None and proc.last_w1 is not None:
                BF = proc.last_w0.shape[0]
                T_use = min(BF, T_frames)
                for fi in range(T_use):
                    w0_f = proc.last_w0[fi:fi+1].float()   # (1, S)
                    w1_f = proc.last_w1[fi:fi+1].float()
                    m_f  = masks_t[fi:fi+1]                 # (1, 2, S)
                    do   = [depth_orders_t[fi]] if fi < len(depth_orders_t) else [(0, 1)]
                    l_ov = l_ov + l_overlap_ordering(w0_f, w1_f, m_f, do)
                l_ov = l_ov / max(T_use, 1)

            # ── L_depth ───────────────────────────────────────────────────
            sigma_acc = list(proc.sigma_acc)
            l_depth   = l_zorder_direct(sigma_acc, depth_orders_t,
                                        entity_masks[:T_frames])

            # ── L_diff ────────────────────────────────────────────────────
            l_diff_val = loss_diff(noise_pred.float(), noise.float())

            # ── L_blend_reg: prevent slot_blend from collapsing to 0 ─────
            # L_diff gradient pushes slot_blend toward 0 (global = safe baseline).
            # This regularizer encourages slot_blend >= slot_blend_min.
            slot_blend_val = proc.slot_blend  # sigmoid(raw) ∈ (0,1)
            l_blend_reg = torch.relu(args.slot_blend_min - slot_blend_val).pow(2)

            # ── Total loss ────────────────────────────────────────────────
            loss = (args.lambda_excl      * l_excl
                  + args.lambda_ov        * l_ov
                  + args.lambda_depth     * l_depth
                  + args.lambda_diff      * l_diff_val
                  + args.lambda_blend_reg * l_blend_reg)

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss ep={epoch} step={batch_idx} "
                      f"excl={l_excl.item():.4f} ov={l_ov.item():.4f} "
                      f"depth={l_depth.item():.4f} → skip", flush=True)
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(list(vca_layer.parameters()), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([proc.slot_blend_raw], max_norm=5.0)

            optimizer.step()

            epoch_losses["total"].append(loss.item())
            epoch_losses["excl"].append(
                float(l_excl.item()) if isinstance(l_excl, torch.Tensor) else 0.0)
            epoch_losses["ov"].append(
                float(l_ov.item()) if isinstance(l_ov, torch.Tensor) else 0.0)
            epoch_losses["depth"].append(float(l_depth.item()))
            epoch_losses["diff"].append(float(l_diff_val.item()))

        scheduler.step()

        avg = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
        blend_val = float(proc.slot_blend.item())

        print(f"[Phase 38] epoch {epoch:03d}/{args.epochs-1}  "
              f"loss={avg['total']:.4f}  excl={avg['excl']:.4f}  "
              f"ov={avg['ov']:.4f}  depth={avg['depth']:.4f}  "
              f"blend={blend_val:.4f}", flush=True)

        # ── 검증 (entity_score) ───────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            vca_layer.eval()
            proc.eval()

            try:
                es, sr, cr, frames_eval = evaluate_entity_score(
                    pipe, proc, vca_layer, orig_procs,
                    probe_meta, probe_toks_e0, probe_toks_e1,
                    probe_entity_ctx, device,
                    seed=args.eval_seed,
                    n_frames=args.n_frames,
                    n_steps=args.n_steps,
                    height=args.height,
                    width=args.width,
                )
                print(f"  entity_score={es:.4f}  survival={sr:.4f}  "
                      f"chimera={cr:.4f}  blend={blend_val:.4f}", flush=True)
            except Exception as e:
                print(f"  [warn] entity_score 실패: {e}", flush=True)
                es, sr, cr = 0.0, 0.0, 1.0
                frames_eval = None

            try:
                dra, n_correct, n_total = measure_depth_rank_accuracy(
                    pipe, vca_layer, dataset, device,
                    n_samples=min(20, len(dataset)),
                )
                print(f"  DRA = {dra:.4f} ({n_correct}/{n_total})", flush=True)
            except Exception as e:
                print(f"  [warn] DRA 실패: {e}", flush=True)
                dra = 0.0

            history.append({
                "epoch": epoch,
                "entity_score": es,
                "survival": sr,
                "chimera": cr,
                "slot_blend": blend_val,
                "dra": dra,
                **avg,
            })

            # save GIF for this evaluation
            if frames_eval is not None:
                gif_path = debug_dir / f"eval_epoch{epoch:03d}.gif"
                iio2.mimwrite(str(gif_path), frames_eval, fps=8, loop=0)

            ckpt_data = {
                "epoch": epoch,
                "vca_state_dict": vca_layer.state_dict(),
                "slot_blend_raw": proc.slot_blend_raw.detach().cpu(),
                "slot_blend": blend_val,
                "entity_score": es,
                "survival": sr,
                "chimera": cr,
                "dra": dra,
                "lambda_excl":  args.lambda_excl,
                "lambda_ov":    args.lambda_ov,
                "lambda_depth": args.lambda_depth,
                "lambda_diff":  args.lambda_diff,
            }
            torch.save(ckpt_data, str(save_dir / "latest.pt"))

            if es > best_es:
                best_es    = es
                best_epoch = epoch
                torch.save(ckpt_data, str(save_dir / "best.pt"))
                print(f"  ★ best epoch={best_epoch}  entity_score={es:.4f} "
                      f"(survival={sr:.4f} chimera={cr:.4f})  "
                      f"→ {save_dir}/best.pt", flush=True)
                if frames_eval is not None:
                    iio2.mimwrite(str(debug_dir / "best.gif"), frames_eval,
                                  fps=8, loop=0)

            vca_layer.train()
            proc.train()

        # ── 주기적 시각화 ─────────────────────────────────────────────────
        if epoch % args.vis_every == 0 or epoch == args.epochs - 1:
            try:
                proc.set_entity_ctx(probe_entity_ctx.float())
                proc.set_entity_tokens(probe_toks_e0, probe_toks_e1)
                proc.reset_slot_store()
                _visualize_slot_weights(proc, probe_entity_ctx, vca_layer,
                                        debug_dir, epoch, device)
            except Exception as e:
                print(f"  [warn] slot 시각화 실패: {e}", flush=True)

    print(f"\n[Phase 38] 훈련 완료. best epoch={best_epoch} "
          f"entity_score={best_es:.4f}", flush=True)
    with open(debug_dir / "phase38_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  history → {debug_dir}/phase38_history.json", flush=True)
    return best_es


# =============================================================================
# 시각화 유틸
# =============================================================================

def _visualize_slot_weights(
    proc: EntitySlotAttnProcessor,
    entity_ctx: torch.Tensor,
    vca_layer: VCALayer,
    debug_dir: Path,
    epoch: int,
    device: str,
):
    """
    Porter-Duff 가중치 w0, w1 를 16×16 히트맵으로 저장.
    (훈련이 아닌 진단용 — entity slot이 올바른 영역에 집중하는지 확인)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blend = float(proc.slot_blend.item())
    sigma = proc.last_sigma

    if sigma is None:
        return  # forward 호출 없이는 nothing to visualize

    sigma_np = sigma[0].cpu().numpy()  # (S, N, Z) — first frame
    S = sigma_np.shape[0]
    HW = int(S ** 0.5)
    if HW * HW != S:
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Epoch {epoch}  slot_blend={blend:.3f}")

    for ni, title in enumerate(["sigma_e0_z0", "sigma_e0_z1",
                                 "sigma_e1_z0", "sigma_e1_z1"]):
        ei = ni // 2
        zi = ni % 2
        ax = axes[ni]
        ax.imshow(sigma_np[:, ei, zi].reshape(HW, HW), vmin=0, vmax=1,
                  cmap="hot", interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    out_path = debug_dir / f"slot_sigma_epoch{epoch:03d}.png"
    plt.savefig(str(out_path), dpi=80)
    plt.close()


# =============================================================================
# CLI
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Phase 38: Entity-Slot Attention + Porter-Duff Compositing")
    p.add_argument("--ckpt",              type=str,   default="checkpoints/phase31/best.pt")
    p.add_argument("--data-root",         type=str,   default="toy/data_objaverse")
    p.add_argument("--save-dir",          type=str,   default="checkpoints/phase38")
    p.add_argument("--debug-dir",         type=str,   default="debug/train_phase38")
    p.add_argument("--epochs",            type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--steps-per-epoch",   type=int,   default=DEFAULT_STEPS_PER_EPOCH)
    p.add_argument("--n-frames",          type=int,   default=8)
    p.add_argument("--n-steps",           type=int,   default=20)
    p.add_argument("--t-max",             type=int,   default=300)
    p.add_argument("--height",            type=int,   default=256)
    p.add_argument("--width",             type=int,   default=256)
    p.add_argument("--lr-vca",            type=float, default=DEFAULT_LR_VCA)
    p.add_argument("--lr-slot",           type=float, default=DEFAULT_LR_SLOT)
    p.add_argument("--lambda-excl",       type=float, default=DEFAULT_LAMBDA_EXCL)
    p.add_argument("--lambda-ov",         type=float, default=DEFAULT_LAMBDA_OV)
    p.add_argument("--lambda-depth",      type=float, default=DEFAULT_LAMBDA_DEPTH)
    p.add_argument("--lambda-diff",       type=float, default=DEFAULT_LAMBDA_DIFF)
    p.add_argument("--slot-blend",        type=float, default=DEFAULT_SLOT_BLEND)
    p.add_argument("--slot-blend-min",    type=float, default=DEFAULT_SLOT_BLEND_MIN)
    p.add_argument("--lambda-blend-reg",  type=float, default=DEFAULT_LAMBDA_BLEND_REG)
    p.add_argument("--eval-every",        type=int,   default=5)
    p.add_argument("--vis-every",         type=int,   default=10)
    p.add_argument("--eval-seed",         type=int,   default=42)
    p.add_argument("--seed",              type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_phase38(args)
