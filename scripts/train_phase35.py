"""
Phase 35 — Volumetric Text Cross-Attention Training
====================================================

아키텍처 변경
------------
Phase 31 (Additive VCA): text_out + gamma * vca_delta  → chimera 근본 해결 불가
Phase 35 (Volumetric):   text cross-attention Q 를 (S×Z) 볼류메트릭으로 확장

    Q_vol[s*Z+z] = Q_std[s] + z_pe[z]   (z_pe: 학습 파라미터)
    output = text_out + gamma * (vol_agg - text_out)

손실 함수
---------
  L = λ_vol   × l_vol_attn      ← NEW: text→volumetric attention GT supervision
    + λ_depth  × l_zorder_direct ← VCA sigma depth ordering
    + λ_diff   × l_diff          ← 생성 품질 보존

학습 파라미터
-----------
  - z_pe:       (Z, D) z-positional encoding (NEW)
  - gamma:      volumetric 혼합 비율 (NEW, phase31 gamma 대체)
  - vca_layer:  sigma 계산용 (phase31 체크포인트에서 초기화)

초기화
------
  - Phase 31 best.pt 체크포인트에서 VCA 가중치 로드
  - z_pe ≈ 0 (std=0.02) → 초기 volumetric ≈ standard (안정적 시작)
  - gamma = 0.5 (standard와 volumetric 50:50 혼합에서 시작)
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from diffusers.models.attention_processor import AttnProcessor2_0

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.vca_volumetric import (
    VolumetricTextCrossAttentionProcessor,
    find_entity_token_positions,
    l_vol_attn_loss,
)
from models.losses import l_diff as loss_diff
from scripts.run_animatediff import load_pipeline
from scripts.train_animatediff_vca import (
    compute_sigma_stats_train, encode_frames_to_latents,
)
from scripts.train_objaverse_vca import (
    ObjaverseTrainDataset, get_entity_context_from_meta,
)
from scripts.train_phase31 import (
    DEFAULT_LAMBDA_DEPTH, DEFAULT_LR,
    INJECT_KEY, INJECT_QUERY_DIM,
    DEPTH_PE_INIT_SCALE, VCA_ALPHA,
    ObjaverseDatasetWithMasks,
    make_color_prompts, get_color_entity_context,
    l_zorder_direct,
    restore_procs,
    measure_depth_rank_accuracy,
    measure_generation_diff,
    AdditiveVCAInferProcessor,       # inference 비교용
    debug_generation,
)

# ─── Phase 35 하이퍼파라미터 ──────────────────────────────────────────────────
DEFAULT_LAMBDA_VOL   = 2.0    # volumetric attention supervision 가중치
DEFAULT_LAMBDA_DEPTH = 3.0    # VCA sigma depth ordering (phase31보다 낮춤)
DEFAULT_LAMBDA_DIFF  = 0.5    # 생성 품질 (phase31과 동일)
DEFAULT_GAMMA_INIT   = 0.5    # volumetric 초기 혼합 비율
DEFAULT_Z_BINS       = 2
DEFAULT_EPOCHS       = 80
DEFAULT_LR           = 3e-5   # z_pe는 작은 LR로 시작


# =============================================================================
# VCA 주입 (Phase 35 전용)
# =============================================================================

def inject_vca_p35(pipe, vca_layer: VCALayer, entity_ctx: torch.Tensor,
                   gamma_init: float = DEFAULT_GAMMA_INIT,
                   z_bins: int = DEFAULT_Z_BINS):
    """
    VolumetricTextCrossAttentionProcessor 주입.

    Phase 31과 다르게:
    - 기존 text cross-attention을 완전히 대체 (additive가 아님)
    - volumetric Q 확장으로 depth-stratified attention 학습
    """
    unet = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))

    proc = VolumetricTextCrossAttentionProcessor(
        query_dim=INJECT_QUERY_DIM,
        z_bins=z_bins,
        vca_layer=vca_layer,
        entity_ctx=entity_ctx.float(),
        gamma_init=gamma_init,
    ).to(pipe.device)

    new_procs = dict(orig_procs)
    new_procs[INJECT_KEY] = proc
    unet.set_attn_processor(new_procs)

    print(f"[Phase 35] VolumetricTextCrossAttentionProcessor 주입 → {INJECT_KEY}", flush=True)
    print(f"           z_bins={z_bins}, gamma_init={gamma_init:.2f}", flush=True)
    return proc, orig_procs


# =============================================================================
# Entity token 위치 조회
# =============================================================================

def get_entity_token_positions(pipe, meta: dict) -> tuple[list[int], list[int], str]:
    """
    meta.json 의 entity text → full_prompt 내 token 위치 반환.
    Returns (e0_positions, e1_positions, full_prompt)
    """
    e0_text, e1_text, full_prompt, _, _ = make_color_prompts(meta)
    toks_e0 = find_entity_token_positions(pipe.tokenizer, full_prompt, e0_text)
    toks_e1 = find_entity_token_positions(pipe.tokenizer, full_prompt, e1_text)
    return toks_e0, toks_e1, full_prompt


# =============================================================================
# 메인 훈련 루프
# =============================================================================

def train_phase35(args):
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
    print("[Phase 35] 파이프라인 로드 중...", flush=True)
    pipe = load_pipeline(device=device, dtype=torch.float16)

    # ── VCA 로드 (Phase 31 체크포인트) ───────────────────────────────────────
    print(f"[Phase 35] VCA 체크포인트 로드: {args.ckpt}", flush=True)
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
    print(f"[Phase 35] 데이터셋: {len(dataset)} 시퀀스", flush=True)
    if len(dataset) == 0:
        raise RuntimeError(f"데이터셋 비어있음: {args.data_root}")

    # ── Probe 설정 (고정 검증 샘플) ──────────────────────────────────────────
    probe_idx       = 0
    probe_sample    = dataset[probe_idx]
    probe_frames_np, _, probe_depth_orders, probe_meta, probe_entity_masks = probe_sample
    probe_entity_ctx = get_color_entity_context(pipe, probe_meta, device)
    _, _, probe_full_prompt = get_entity_token_positions(pipe, probe_meta)

    # ── Probe latent + enc_hs ─────────────────────────────────────────────────
    probe_latents = encode_frames_to_latents(pipe, probe_frames_np, device)
    probe_tok = pipe.tokenizer(
        probe_full_prompt, return_tensors="pt", padding="max_length",
        max_length=pipe.tokenizer.model_max_length, truncation=True,
    ).to(device)
    with torch.no_grad():
        probe_enc_hs = pipe.text_encoder(**probe_tok).last_hidden_state.half()

    # ── VolumetricProcessor 주입 ─────────────────────────────────────────────
    proc, orig_procs = inject_vca_p35(
        pipe, vca_layer, probe_entity_ctx,
        gamma_init=DEFAULT_GAMMA_INIT, z_bins=DEFAULT_Z_BINS,
    )

    # ── 옵티마이저: z_pe, gamma, VCA 파라미터 ────────────────────────────────
    param_groups = [
        {"params": list(vca_layer.parameters()),    "lr": args.lr,       "name": "vca"},
        {"params": [proc.z_pe],                     "lr": args.lr * 0.3, "name": "z_pe"},
        {"params": [proc.gamma],                    "lr": args.lr * 0.1, "name": "gamma"},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    # ── UNet 나머지 파라미터 동결 ─────────────────────────────────────────────
    # proc 이 이미 UNet 서브모듈로 등록되어 있으므로 bulk freeze 후
    # 훈련 대상 파라미터의 requires_grad 를 명시적으로 복원한다.
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    # 훈련 파라미터 requires_grad 복원 (proc 이 UNet 에 submodule 로 등록되어
    # 위 freeze 루프에서 같이 꺼졌을 수 있음)
    for p in vca_layer.parameters():
        p.requires_grad_(True)
    proc.z_pe.requires_grad_(True)
    proc.gamma.requires_grad_(True)

    # ── 메트릭 기록 ──────────────────────────────────────────────────────────
    history = []
    best_dra = 0.0
    best_epoch = -1

    print(f"\n[Phase 35] 훈련 시작: {args.epochs} epochs", flush=True)
    print(f"  λ_vol={args.lambda_vol}, λ_depth={args.lambda_depth}, "
          f"λ_diff={args.lambda_diff}", flush=True)

    for epoch in range(args.epochs):
        vca_layer.train()
        proc.train()

        epoch_losses = {"total": [], "vol_attn": [], "depth": [], "diff": []}

        # 데이터 인덱스 셔플
        indices = np.random.permutation(len(dataset)).tolist()

        for batch_idx, data_idx in enumerate(indices[:args.steps_per_epoch]):
            sample = dataset[data_idx]
            frames_np, _, depth_orders, meta, entity_masks = sample

            # entity context
            entity_ctx = get_color_entity_context(pipe, meta, device)
            proc.set_entity_ctx(entity_ctx.float())
            proc.reset_sigma_acc()

            # entity token positions
            toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)

            # latent + noise
            with torch.no_grad():
                latents = encode_frames_to_latents(pipe, frames_np, device)
            noise = torch.randn_like(latents)
            t     = torch.randint(0, args.t_max, (1,), device=device).long()
            noisy = pipe.scheduler.add_noise(latents, noise, t)

            # text encoding
            tok = pipe.tokenizer(
                full_prompt, return_tensors="pt", padding="max_length",
                max_length=pipe.tokenizer.model_max_length, truncation=True,
            ).to(device)
            with torch.no_grad():
                enc_hs = pipe.text_encoder(**tok).last_hidden_state.half()

            # entity masks → tensor (1, 2, S)
            T_frames = min(frames_np.shape[0], entity_masks.shape[0])
            # UNet forward: BF = T_frames frames (no CFG during training)
            masks_t = torch.from_numpy(
                entity_masks[:T_frames].astype(np.float32)
            ).to(device)                                    # (T, 2, S)

            depth_orders_t = depth_orders[:T_frames]

            # ── UNet forward ──────────────────────────────────────────────
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = pipe.unet(noisy, t, encoder_hidden_states=enc_hs).sample

            # ── l_vol_attn: volumetric attention supervision ───────────────
            attn_w = proc.last_attn_weights   # (BF, S*Z, T_seq) or None
            l_vol  = torch.tensor(0.0, device=device)
            if attn_w is not None and len(toks_e0) > 0 and len(toks_e1) > 0:
                BF = attn_w.shape[0]
                T_use = min(BF, T_frames)
                # entity_masks → (BF, 2, S) matching UNet BF
                masks_bf = masks_t[:T_use]
                l_vol = l_vol_attn_loss(
                    attn_w[:T_use].float(),
                    toks_e0, toks_e1,
                    masks_bf,
                    depth_orders_t[:T_use],
                    z_bins=DEFAULT_Z_BINS,
                )
            else:
                if batch_idx == 0 and epoch == 0:
                    print(f"  [warn] vol_attn: attn_w={attn_w is not None}, "
                          f"toks_e0={toks_e0}, toks_e1={toks_e1}", flush=True)

            # ── l_zorder_direct: VCA sigma depth ordering ─────────────────
            sigma_acc = list(proc.sigma_acc)
            l_depth   = l_zorder_direct(sigma_acc, depth_orders_t, entity_masks[:T_frames])

            # ── l_diff: noise prediction quality ─────────────────────────
            l_diff_val = loss_diff(noise_pred.float(), noise.float())

            # ── total loss ────────────────────────────────────────────────
            loss = (args.lambda_vol   * l_vol
                  + args.lambda_depth * l_depth
                  + args.lambda_diff  * l_diff_val)

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss epoch={epoch} step={batch_idx} "
                      f"vol={l_vol.item():.4f} depth={l_depth.item():.4f} "
                      f"diff={l_diff_val.item():.4f} → skip", flush=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(vca_layer.parameters()) + [proc.z_pe, proc.gamma],
                max_norm=1.0
            )
            optimizer.step()

            epoch_losses["total"].append(loss.item())
            epoch_losses["vol_attn"].append(l_vol.item() if isinstance(l_vol, torch.Tensor) else 0.0)
            epoch_losses["depth"].append(l_depth.item())
            epoch_losses["diff"].append(l_diff_val.item())

        scheduler.step()

        # ── 에폭 요약 ─────────────────────────────────────────────────────
        avg = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
        gamma_val = float(proc.gamma.item())
        z_pe_norm = float(proc.z_pe.norm().item())

        print(f"[Phase 35] epoch {epoch:03d}/{args.epochs-1}  "
              f"loss={avg['total']:.4f}  vol={avg['vol_attn']:.4f}  "
              f"depth={avg['depth']:.4f}  diff={avg['diff']:.4f}  "
              f"gamma={gamma_val:.4f}  |z_pe|={z_pe_norm:.4f}", flush=True)

        # ── 검증: DRA ────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            vca_layer.eval()
            proc.eval()

            try:
                dra, n_correct, n_total = measure_depth_rank_accuracy(
                    pipe, vca_layer, dataset, device,
                    n_samples=min(20, len(dataset)),
                )
                print(f"  DRA = {dra:.4f} ({n_correct}/{n_total})", flush=True)
            except Exception as e:
                print(f"  [warn] DRA 계산 실패: {e}", flush=True)
                dra = 0.0

            # Generation diff (probe)
            try:
                proc.set_entity_ctx(probe_entity_ctx.float())
                gen_diff = measure_generation_diff(
                    pipe, vca_layer, orig_procs,
                    dict(pipe.unet.attn_processors),
                    probe_meta, probe_entity_ctx, device,
                )
            except Exception as e:
                print(f"  [warn] gen_diff 실패: {e}", flush=True)
                gen_diff = 0.0

            history.append({
                "epoch": epoch,
                "dra": dra,
                "gen_diff": gen_diff,
                "gamma": gamma_val,
                "z_pe_norm": z_pe_norm,
                **avg,
            })

            # 체크포인트 저장
            ckpt_data = {
                "epoch": epoch,
                "vca_state_dict": vca_layer.state_dict(),
                "z_pe": proc.z_pe.detach().cpu(),
                "gamma_trained": gamma_val,
                "dra": dra,
                "gen_diff": gen_diff,
            }
            latest_path = save_dir / "latest.pt"
            torch.save(ckpt_data, str(latest_path))

            if dra > best_dra:
                best_dra   = dra
                best_epoch = epoch
                best_path  = save_dir / "best.pt"
                torch.save(ckpt_data, str(best_path))
                print(f"  ★ best DRA={best_dra:.4f} → {best_path}", flush=True)

            vca_layer.train()
            proc.train()

        # ── 시각화 ───────────────────────────────────────────────────────
        if epoch % args.vis_every == 0 or epoch == args.epochs - 1:
            try:
                proc.set_entity_ctx(probe_entity_ctx.float())
                debug_generation(
                    pipe, vca_layer, orig_procs,
                    dict(pipe.unet.attn_processors),
                    probe_frames_np, probe_meta, probe_entity_ctx,
                    debug_dir, epoch,
                )
            except Exception as e:
                print(f"  [warn] debug_generation 실패: {e}", flush=True)

            # Volumetric attention map 시각화
            _visualize_vol_attn(
                proc, probe_full_prompt, vca_layer,
                debug_dir, epoch,
            )

    # ── 훈련 완료 요약 ────────────────────────────────────────────────────
    print(f"\n[Phase 35] 훈련 완료. best epoch={best_epoch} DRA={best_dra:.4f}", flush=True)
    with open(debug_dir / "phase35_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  history → {debug_dir}/phase35_history.json", flush=True)
    return best_dra


# =============================================================================
# 볼류메트릭 attention map 시각화
# =============================================================================

def _visualize_vol_attn(proc: VolumetricTextCrossAttentionProcessor,
                         full_prompt: str, vca_layer: VCALayer,
                         debug_dir: Path, epoch: int):
    """
    last_attn_weights (B, S*Z, T) 에서 z=0 (front) vs z=1 (back) attention 시각화.
    각 z-bin의 attention map이 분리되어 있는지 확인.
    """
    import imageio.v2 as iio2
    from PIL import Image, ImageDraw, ImageFont

    w = proc.last_attn_weights
    if w is None:
        return

    try:
        w_np = w.detach().cpu().float().numpy()  # (B, S*Z, T)
        B, SZ, T_seq = w_np.shape
        Z = proc.z_bins
        S = SZ // Z
        hw = int(S ** 0.5)
        if hw * hw != S:
            return

        panels = []
        for b in range(min(B, 2)):
            for z in range(Z):
                # z-bin의 attention map: (S, T) → sum over T → (S,) → (hw, hw)
                attn_sum = w_np[b, z::Z, :].sum(axis=-1).reshape(hw, hw)
                lo, hi = attn_sum.min(), attn_sum.max()
                attn_norm = ((attn_sum - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
                img = Image.fromarray(attn_norm, mode='L').convert('RGB')
                img = img.resize((128, 128), Image.NEAREST)
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
                except Exception:
                    font = ImageFont.load_default()
                draw.text((2, 2), f"b{b} z={z}", fill=(255, 255, 0), font=font)
                panels.append(np.array(img))

        if panels:
            row = np.concatenate(panels, axis=1)
            out_path = debug_dir / f"vol_attn_epoch{epoch:03d}.png"
            iio2.imwrite(str(out_path), row)
    except Exception as e:
        print(f"  [warn] vol_attn 시각화 실패: {e}", flush=True)


# =============================================================================
# CLI
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Phase 35: Volumetric Text Cross-Attention")
    p.add_argument("--ckpt",            type=str, default="checkpoints/phase31/best.pt",
                   help="Phase 31 VCA checkpoint (초기화용)")
    p.add_argument("--data-root",       type=str, default="data/objaverse_vca")
    p.add_argument("--save-dir",        type=str, default="checkpoints/phase35")
    p.add_argument("--debug-dir",       type=str, default="debug/train_phase35")
    p.add_argument("--epochs",          type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--steps-per-epoch", type=int, default=20)
    p.add_argument("--n-frames",        type=int, default=8)
    p.add_argument("--t-max",           type=int, default=300)
    p.add_argument("--lr",              type=float, default=DEFAULT_LR)
    p.add_argument("--lambda-vol",      type=float, default=DEFAULT_LAMBDA_VOL)
    p.add_argument("--lambda-depth",    type=float, default=DEFAULT_LAMBDA_DEPTH)
    p.add_argument("--lambda-diff",     type=float, default=DEFAULT_LAMBDA_DIFF)
    p.add_argument("--eval-every",      type=int, default=5)
    p.add_argument("--vis-every",       type=int, default=10)
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_phase35(args)
