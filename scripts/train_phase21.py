"""
Phase 21: 구조적 버그 수정 + 아이디어 유효성 측정

FM-I12 수정: VCA가 text cross-attention을 대체 → additive 방식으로 변경
  기존: output = VCA_output           (text attn 완전 대체 → 이미지 품질 파괴)
  수정: output = text_attn + α×VCA_delta  (text attn 유지 + depth bias 추가)

핵심 측정 지표:
  depth_rank_accuracy: sigma가 실제 depth 순서를 맞추는 비율 (%)
    - 이 값이 50% 초과 → 아이디어 작동
    - 80% 이상 → 실용적 수준
    - 50% 이하 → 아이디어 자체 문제

Fix 1/2/3 유지 (per-frame loss, fixed probe, depth_pe_init_scale=0.3)
Fix 4 제거 유지 (mid_block 단일 주입)
alpha=0.3 (VCA가 text attn 대비 30% 강도)
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import layer_norm
from torch.utils.data import DataLoader
from diffusers.models.attention_processor import AttnProcessor2_0

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.losses import l_ortho as loss_ortho, l_depth_ranking, l_diff as loss_diff
from scripts.run_animatediff import load_pipeline, FixedContextVCAProcessor
from scripts.train_animatediff_vca import (
    compute_sigma_stats_train, save_sigma_gif, encode_frames_to_latents,
)
from scripts.train_objaverse_vca import (
    ObjaverseTrainDataset, check_dataset_quality, get_entity_context_from_meta,
)
from scripts.train_phase17 import adaptive_lambda_depth, RATIO_WARNING_THRESH
from scripts.train_phase19 import l_depth_ranking_perframe, PROBE_T_VALUES

# ─── 기본값 ───────────────────────────────────────────────────────────────────
DEFAULT_LAMBDA_DEPTH  = 0.3
DEFAULT_LAMBDA_ORTHO  = 0.005
DEFAULT_LR            = 5e-5
DEFAULT_EPOCHS        = 60
DEFAULT_T_MAX         = 200
DEPTH_PE_INIT_SCALE   = 0.3
VCA_ALPHA             = 0.3   # VCA 기여 강도: text_attn + alpha*vca_delta

INJECT_KEY = 'mid_block.attentions.0.transformer_blocks.0.attn2.processor'


# ─── 핵심 수정: Additive VCA Processor ──────────────────────────────────────

class AdditiveVCAProcessor:
    """
    Phase 21: text cross-attention 유지 + VCA를 depth bias로 추가.

    기존 (Phase 1~20):
      return VCA_output   ← text attn 완전 대체, 이미지 품질 파괴

    Phase 21:
      text_out = original_AttnProcessor2_0(attn, hidden_states, encoder_hidden_states)
      vca_delta = VCA(layer_norm(hidden_states)) - layer_norm(hidden_states)
      return text_out + alpha * vca_delta

    gradient는 vca_delta(→VCALayer)에만 흐름. text_out은 detach.
    """
    def __init__(self, vca_layer: VCALayer, entity_ctx: torch.Tensor,
                 orig_processor: AttnProcessor2_0, alpha: float = VCA_ALPHA):
        self.vca   = vca_layer
        self.ctx   = entity_ctx      # (1, N, 768) fp32
        self.orig  = orig_processor  # 원본 AttnProcessor2_0
        self.alpha = alpha

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        BF = hidden_states.shape[0]

        # 1. 원본 text cross-attention (frozen, no grad)
        with torch.no_grad():
            text_out = self.orig(attn, hidden_states, encoder_hidden_states,
                                 attention_mask, temb, *args, **kwargs)

        # 2. VCA depth delta
        ctx = self.ctx.expand(BF, -1, -1).float()
        x   = layer_norm(hidden_states.float(), hidden_states.shape[-1:])
        vca_out   = self.vca(x, ctx)          # (BF, S, D), sets last_sigma_raw/acc
        delta_raw = vca_out - x               # LN-space delta, O(1) scale

        # FM-I13 수정: delta를 text_out magnitude에 비례 정규화
        # text_out은 fp16 attn projection (scale ≈ 0.05~0.1),
        # delta_raw는 LN-space (scale ≈ 1) → 초기 ratio ≈ 4~10x → manifold escape
        # 정규화: |vca_delta| = alpha * |text_out|  (alpha = 상대 강도)
        with torch.no_grad():
            text_mag  = text_out.float().abs().mean() + 1e-8
            delta_mag = delta_raw.abs().mean() + 1e-8
        vca_delta = delta_raw * (text_mag / delta_mag) * self.alpha

        # 3. text quality 유지 + depth bias 추가 (이제 비율이 정확히 alpha)
        return text_out + vca_delta.to(text_out.dtype)


class AdditiveVCAInferProcessor:
    """추론용: grad 불필요, text_out.detach() 생략."""
    def __init__(self, vca_layer: VCALayer, entity_ctx: torch.Tensor,
                 orig_processor: AttnProcessor2_0, alpha: float = VCA_ALPHA):
        self.vca   = vca_layer
        self.ctx   = entity_ctx
        self.orig  = orig_processor
        self.alpha = alpha
        # 진단용 (debug_vca_internals에서 읽음)
        self.last_text_out  = None   # (BF, S, D) fp32
        self.last_vca_delta = None   # (BF, S, D) fp32

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        BF = hidden_states.shape[0]
        text_out = self.orig(attn, hidden_states, encoder_hidden_states,
                             attention_mask, temb, *args, **kwargs)
        ctx = self.ctx.expand(BF, -1, -1).float()
        x   = layer_norm(hidden_states.float(), hidden_states.shape[-1:])
        vca_out   = self.vca(x, ctx)
        delta_raw = vca_out - x
        # FM-I13: text_out magnitude 기준 정규화 (ratio ≈ alpha 로 고정)
        text_mag  = text_out.float().abs().mean() + 1e-8
        delta_mag = delta_raw.abs().mean() + 1e-8
        vca_delta = delta_raw * (text_mag / delta_mag) * self.alpha
        # 마지막 호출 저장 (진단용)
        self.last_text_out  = text_out.detach().float()
        self.last_vca_delta = vca_delta.detach()
        return text_out + vca_delta.to(text_out.dtype)


# ─── VCA 주입 ────────────────────────────────────────────────────────────────

def inject_vca_p21(pipe, entity_ctx: torch.Tensor):
    """학습용 Additive VCA 주입."""
    unet = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))
    orig_proc  = orig_procs[INJECT_KEY]       # AttnProcessor2_0

    vca_layer = VCALayer(
        query_dim=1280, context_dim=768,
        n_heads=8, n_entities=2, z_bins=2, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=DEPTH_PE_INIT_SCALE,
    ).to(pipe.device)

    new_proc = AdditiveVCAProcessor(vca_layer, entity_ctx, orig_proc, alpha=VCA_ALPHA)
    new_procs = dict(orig_procs)
    new_procs[INJECT_KEY] = new_proc
    unet.set_attn_processor(new_procs)

    print(f"[inject_p21] additive VCA (alpha={VCA_ALPHA}) → {INJECT_KEY}", flush=True)
    return vca_layer, orig_procs


def inject_vca_p21_infer(pipe, vca_layer: VCALayer, entity_ctx: torch.Tensor):
    """추론용 Additive VCA 주입."""
    unet = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))
    orig_proc  = orig_procs[INJECT_KEY]
    new_proc   = AdditiveVCAInferProcessor(vca_layer, entity_ctx, orig_proc, alpha=VCA_ALPHA)
    new_procs  = dict(orig_procs)
    new_procs[INJECT_KEY] = new_proc
    unet.set_attn_processor(new_procs)
    return orig_procs


def restore_procs(pipe, orig_procs):
    pipe.unet.set_attn_processor(dict(orig_procs))  # copy: set_attn_processor pops from dict


# ─── Fix 2: 고정 probe 측정 ──────────────────────────────────────────────────

def measure_probe_sep(pipe, vca_layer, probe_latents, probe_enc_hs, device):
    """고정 probe × 5 t → 안정적 sigma_separation."""
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
                s = compute_sigma_stats_train(vca_layer.last_sigma)
                seps.append(s['sigma_separation'])
    vca_layer.train()
    return float(sum(seps) / max(len(seps), 1))


# ─── 핵심 지표: depth_rank_accuracy ─────────────────────────────────────────

@torch.no_grad()
def measure_depth_rank_accuracy(pipe, vca_layer, dataset, device,
                                 n_samples=20, t_val=100):
    """
    아이디어 유효성 직접 측정.

    학습 데이터 샘플에 대해:
      sigma[front_entity, z=0] > sigma[back_entity, z=0] 이면 정답

    Returns:
      accuracy: float (0~1)
      n_correct, n_total: int
    """
    vca_layer.eval()
    n_correct = 0
    n_total   = 0
    t = torch.tensor([t_val], device=device)

    indices = list(range(min(n_samples, len(dataset))))
    for idx in indices:
        frames_np, _, depth_orders, meta = dataset[idx]
        entity_ctx = get_entity_context_from_meta(pipe, meta, device)

        # 현재 주입된 processor entity_ctx 업데이트
        proc = pipe.unet.attn_processors.get(INJECT_KEY)
        if isinstance(proc, (AdditiveVCAProcessor, AdditiveVCAInferProcessor)):
            proc.ctx = entity_ctx.float()

        latents = encode_frames_to_latents(pipe, frames_np, device)
        noise   = torch.randn_like(latents)
        noisy   = pipe.scheduler.add_noise(latents, noise, t)

        # text encoding
        from scripts.train_objaverse_vca import get_entity_context_from_meta as _g
        full_prompt = (f"{meta.get('prompt_entity0','entity0')} and "
                       f"{meta.get('prompt_entity1','entity1')}")
        tok = pipe.tokenizer(full_prompt, return_tensors="pt", padding="max_length",
                             max_length=pipe.tokenizer.model_max_length,
                             truncation=True).to(device)
        enc_hs = pipe.text_encoder(**tok).last_hidden_state.half()

        vca_layer.reset_sigma_acc()
        pipe.unet(noisy, t, encoder_hidden_states=enc_hs)

        if vca_layer.last_sigma is None:
            continue

        # (BF, S, N, Z) → per-frame accuracy
        sigma_np = vca_layer.last_sigma.detach().cpu().float().numpy()
        BF = sigma_np.shape[0]
        T  = min(BF, len(depth_orders))

        for fi in range(T):
            order = depth_orders[fi]  # [front_idx, back_idx]
            front, back = order[0], order[1]
            # z=0 평균: 어떤 entity가 더 높은 sigma?
            e_front = float(sigma_np[fi, :, front, 0].mean())
            e_back  = float(sigma_np[fi, :, back,  0].mean())
            if e_front > e_back:
                n_correct += 1
            n_total += 1

    vca_layer.train()
    accuracy = n_correct / max(n_total, 1)
    return accuracy, n_correct, n_total


# ─── 학습 중간 recon 디버그 ──────────────────────────────────────────────────

def debug_generation(pipe, vca_layer, orig_procs, train_procs,
                     probe_frames_np, probe_meta, probe_entity_ctx,
                     debug_dir: Path, epoch: int, height=256, width=256):
    """
    매 N epoch마다 호출: [GT | Baseline | VCA] 3-panel GIF + sigma overlay 저장.
    학습 중간에 reconstruction 품질과 sigma 분리를 동시에 확인.

    orig_procs:  AttnProcessor2_0 (VCA 없음)
    train_procs: AdditiveVCAProcessor (학습용)
    """
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as iio2

    vca_layer.eval()
    # Save orig_proc reference BEFORE any set_attn_processor call depletes orig_procs.
    # diffusers' set_attn_processor() pops from the dict it receives — we must pass
    # dict(orig_procs) copies everywhere and never pass orig_procs directly.
    orig_proc_ref = orig_procs.get(INJECT_KEY)
    if orig_proc_ref is None:
        orig_proc_ref = AttnProcessor2_0()  # fallback (should not happen)

    prompt = (f"{probe_meta.get('prompt_entity0','entity0')} and "
              f"{probe_meta.get('prompt_entity1','entity1')}")
    kw = dict(num_frames=8, steps=20, height=height, width=width, seed=42)

    def _lbl(arr, text):
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        except Exception:
            font = ImageFont.load_default()
        draw.text((3, 3), text, fill=(255, 255, 255), font=font)
        return np.array(img)

    # 1. baseline: VCA 완전 제거 (copy: set_attn_processor pops from dict)
    pipe.unet.set_attn_processor(dict(orig_procs))
    gen = torch.Generator(device=pipe.device).manual_seed(42)
    out = pipe(prompt=prompt, num_frames=8, num_inference_steps=20,
               guidance_scale=7.5, height=height, width=width,
               generator=gen, output_type='pil')
    baseline_frames = [np.array(f) for f in out.frames[0]]

    # 2. VCA 생성: AdditiveVCAInferProcessor 임시 주입
    infer_proc = AdditiveVCAInferProcessor(
        vca_layer, probe_entity_ctx,
        orig_proc_ref, alpha=VCA_ALPHA,  # use pre-saved reference
    )
    infer_procs = dict(orig_procs)
    infer_procs[INJECT_KEY] = infer_proc
    pipe.unet.set_attn_processor(infer_procs)  # infer_procs is already a fresh dict
    gen2 = torch.Generator(device=pipe.device).manual_seed(42)
    out2 = pipe(prompt=prompt, num_frames=8, num_inference_steps=20,
                guidance_scale=7.5, height=height, width=width,
                generator=gen2, output_type='pil')
    vca_frames = [np.array(f) for f in out2.frames[0]]

    # 학습용 processor 복원 (copy to avoid depleting train_procs)
    pipe.unet.set_attn_processor(dict(train_procs))
    vca_layer.train()

    P = height

    # ── ① reconstruction.gif: [GT | Baseline | VCA] ──────────────────────
    recon_gif = []
    for fi in range(len(baseline_frames)):
        gt_arr = np.array(Image.fromarray(probe_frames_np[fi]).resize(
            (P, P), Image.BILINEAR)) if fi < len(probe_frames_np) else np.zeros((P,P,3),dtype=np.uint8)
        b_arr  = np.array(Image.fromarray(baseline_frames[fi]).resize((P, P), Image.BILINEAR))
        v_arr  = np.array(Image.fromarray(vca_frames[fi]).resize((P, P), Image.BILINEAR))
        row = np.concatenate([
            _lbl(gt_arr, f"GT  e{epoch:02d}"),
            _lbl(b_arr,  "Baseline"),
            _lbl(v_arr,  f"VCA a={VCA_ALPHA}"),
        ], axis=1)
        recon_gif.append(row)
    recon_path = debug_dir / f"recon_epoch{epoch:03d}.gif"
    iio2.mimsave(str(recon_path), recon_gif, duration=200)

    # ── ② sigma_overlay.gif: VCA 프레임 위에 E0(파)/E1(빨) 반투명 오버레이 ─
    if vca_layer.last_sigma is not None:
        sig_np = vca_layer.last_sigma.detach().cpu().float().numpy()
        BF, S, N, Z = sig_np.shape
        hw = max(1, int(S ** 0.5))
        overlay_gif = []
        for fi in range(min(len(vca_frames), BF)):
            s = sig_np[fi]                  # (S, N, Z)
            e0 = s[:, 0, 0].reshape(hw, hw)
            e1 = s[:, 1, 0].reshape(hw, hw)

            def _heat(m):
                lo, hi = m.min(), m.max()
                n = (m - lo) / (hi - lo + 1e-6)
                r = np.clip(n*3-2, 0, 1); g = np.clip(n*3-1, 0, 1); b = np.clip(n*3, 0, 1)
                return (np.stack([b, g, r], -1) * 255).astype(np.uint8)

            e0_heat = np.array(Image.fromarray(_heat(e0)).resize((P, P), Image.NEAREST))
            e1_heat = np.array(Image.fromarray(_heat(e1)).resize((P, P), Image.NEAREST))

            rgb = np.array(Image.fromarray(vca_frames[fi]).resize((P, P), Image.BILINEAR)).astype(float)
            e0n = np.array(Image.fromarray(((e0 - e0.min()) / (e0.max()-e0.min()+1e-6)*255).astype(np.uint8)).resize((P,P),Image.BILINEAR)) / 255.
            e1n = np.array(Image.fromarray(((e1 - e1.min()) / (e1.max()-e1.min()+1e-6)*255).astype(np.uint8)).resize((P,P),Image.BILINEAR)) / 255.
            overlay = np.clip(
                rgb/255 + 0.4*np.stack([np.zeros_like(e0n), np.zeros_like(e0n), e0n], -1)
                        + 0.4*np.stack([e1n, np.zeros_like(e1n), np.zeros_like(e1n)], -1), 0, 1)
            overlay_u8 = (overlay * 255).astype(np.uint8)

            row = np.concatenate([
                _lbl(np.array(Image.fromarray(vca_frames[fi]).resize((P,P),Image.BILINEAR)),
                     f"VCA e{epoch:02d}"),
                _lbl(e0_heat, f"E0σ sep={float(np.mean(e0)):.3f}"),
                _lbl(e1_heat, f"E1σ sep={float(np.mean(e1)):.3f}"),
                _lbl(overlay_u8, "Overlay"),
            ], axis=1)
            overlay_gif.append(row)
        sig_path = debug_dir / f"sigma_overlay_epoch{epoch:03d}.gif"
        iio2.mimsave(str(sig_path), overlay_gif, duration=200)

    e0_mean = float(sig_np[:, :, 0, 0].mean()) if vca_layer.last_sigma is not None else 0.
    e1_mean = float(sig_np[:, :, 1, 0].mean()) if vca_layer.last_sigma is not None else 0.
    print(f"  [debug] recon → {recon_path.name}  "
          f"E0σ={e0_mean:.3f} E1σ={e1_mean:.3f} sep={abs(e0_mean-e1_mean):.3f}",
          flush=True)


# ─── 다양한 카메라 뷰 시각화 ─────────────────────────────────────────────────

def debug_multiview(pipe, vca_layer, orig_procs, train_procs,
                    probe_meta, data_root: str, debug_dir: Path, epoch: int,
                    height=256, width=256, max_views=4):
    """
    같은 entity pair를 다양한 카메라 뷰(orbit/rotate × front/top/front_left 등)로 학습하는지 확인.

    probe_meta의 keyword0/keyword1로 같은 entity pair 시퀀스를 찾아서
    각 뷰(mode+camera 조합)의 GT 첫 프레임 + VCA 생성 결과를 나란히 시각화.

    출력: debug_dir/multiview_epoch{N}.gif
      행: 각 카메라 뷰
      열: [GT 첫 프레임 | VCA 생성 1프레임]
    """
    import imageio.v2 as iio2
    from PIL import Image, ImageDraw, ImageFont
    import glob, json as _json

    vca_layer.eval()
    # Save orig_proc reference before any set_attn_processor call depletes orig_procs
    orig_proc_ref_mv = orig_procs.get(INJECT_KEY)
    if orig_proc_ref_mv is None:
        orig_proc_ref_mv = AttnProcessor2_0()

    k0 = probe_meta.get("keyword0", "")
    k1 = probe_meta.get("keyword1", "")

    # 같은 keyword pair를 가진 시퀀스 찾기 (다른 카메라/모션)
    import os
    all_dirs = sorted(os.listdir(data_root))
    same_pair_dirs = []
    for d in all_dirs:
        meta_path = f"{data_root}/{d}/meta.json"
        if not os.path.exists(meta_path):
            continue
        m = _json.load(open(meta_path))
        if m.get("keyword0") == k0 and m.get("keyword1") == k1:
            same_pair_dirs.append((d, m))
        if len(same_pair_dirs) >= max_views:
            break

    if not same_pair_dirs:
        return

    def _lbl(arr, text):
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        except Exception:
            font = ImageFont.load_default()
        draw.text((2, 2), text, fill=(255,255,255), font=font)
        return np.array(img)

    P = height
    rows = []
    prompt = (f"{probe_meta.get('prompt_entity0','entity0')} and "
              f"{probe_meta.get('prompt_entity1','entity1')}")

    # VCA 생성 (probe entity ctx 사용, 여러 뷰 같은 프롬프트)
    infer_proc = AdditiveVCAInferProcessor(
        vca_layer, get_entity_context_from_meta(pipe, probe_meta,
                                                 str(pipe.device)),
        orig_proc_ref_mv, alpha=VCA_ALPHA,  # use pre-saved reference
    )
    infer_procs = dict(orig_procs)
    infer_procs[INJECT_KEY] = infer_proc
    pipe.unet.set_attn_processor(infer_procs)  # fresh dict, OK

    gen = torch.Generator(device=pipe.device).manual_seed(42)
    out = pipe(prompt=prompt, num_frames=8, num_inference_steps=20,
               guidance_scale=7.5, height=height, width=width,
               generator=gen, output_type='pil')
    vca_frames_shared = [np.array(f) for f in out.frames[0]]

    pipe.unet.set_attn_processor(dict(train_procs))  # copy to avoid depleting train_procs
    vca_layer.train()

    for dir_name, meta in same_pair_dirs:
        # GT 첫 프레임
        frame_paths = sorted(glob.glob(f"{data_root}/{dir_name}/frames/*.png"))
        if not frame_paths:
            continue
        gt_frame = np.array(Image.open(frame_paths[0]).convert("RGB").resize((P, P), Image.BILINEAR))
        vca_frame = np.array(Image.fromarray(vca_frames_shared[0]).resize((P, P), Image.BILINEAR))

        mode   = meta.get("mode", "?")
        camera = meta.get("camera", "?")
        view_label = f"{mode}/{camera}"

        # sigma: E0 vs E1 mean
        if vca_layer.last_sigma is not None:
            sig = vca_layer.last_sigma.detach().cpu().float().numpy()
            e0m = float(sig[:, :, 0, 0].mean())
            e1m = float(sig[:, :, 1, 0].mean())
            sig_str = f"E0={e0m:.2f} E1={e1m:.2f}"
        else:
            sig_str = ""

        row = np.concatenate([
            _lbl(gt_frame,   f"GT {view_label}"),
            _lbl(vca_frame,  f"VCA {sig_str}"),
        ], axis=1)
        rows.append(row)

    if rows:
        # 모든 뷰를 세로로 쌓음
        mosaic = np.concatenate(rows, axis=0)
        out_path = debug_dir / f"multiview_epoch{epoch:03d}.gif"
        iio2.mimsave(str(out_path), [mosaic], duration=2000)
        print(f"  [debug] multiview ({len(rows)} views) → {out_path.name}", flush=True)


# ─── 심층 VCA 진단 시각화 ────────────────────────────────────────────────────

def _heat_map(m, lo=None, hi=None):
    """(H, W) float → (H, W, 3) uint8 컬러 히트맵 (파랑→초록→빨강)."""
    lo = float(m.min()) if lo is None else lo
    hi = float(m.max()) if hi is None else hi
    n = (m - lo) / (hi - lo + 1e-8)
    r = np.clip(n * 3 - 2,       0, 1)
    g = np.clip(1 - np.abs(n * 3 - 1.5), 0, 1)
    b = np.clip(1 - n * 3,       0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def _decode_latents_safe(pipe, latents):
    """
    AnimateDiff latent (1, C, T, H, W) or (T, C, H, W) → (T, H, W, 3) uint8.
    VAE decode는 한 번에 한 frame씩 처리해 OOM 방지.
    """
    try:
        if latents.dim() == 5:
            lat = latents[0].permute(1, 0, 2, 3)  # (T, C, H, W)
        elif latents.dim() == 4:
            lat = latents                           # (T, C, H, W)
        else:
            return None
        lat = lat.float() / pipe.vae.config.scaling_factor
        frames = []
        for i in range(lat.shape[0]):
            dec = pipe.vae.decode(lat[i:i+1]).sample  # (1, 3, H, W)
            frames.append(dec[0].permute(1, 2, 0).cpu().numpy())
        arr = np.stack(frames)                          # (T, H, W, 3)
        return np.clip((arr + 1) / 2 * 255, 0, 255).astype(np.uint8)
    except Exception:
        return None


def debug_vca_internals(pipe, vca_layer, orig_procs, train_procs,
                        probe_frames_np, probe_meta, probe_entity_ctx,
                        debug_dir: Path, epoch: int, height=256, width=256):
    """
    Phase 21 심층 진단. 세 가지 GIF 저장:

    1. denoise_traj_epoch{N}.gif   — Baseline vs VCA 디노이징 경과
       step=0,4,9,14,19 × 8 frames.  어느 step에서 artifact가 발생하는지 포착.

    2. vca_delta_epoch{N}.gif      — VCA delta 강도 + weight collapse 진단
       패널: |Δ|/|text_out| 비율 / ΣWeight map / σ(E0) / σ(E1)
       mean ratio > 0.5 시 경고.  배경 collapse (ΣW→0) 시각화.

    3. ray3d_epoch{N}.gif          — ray marching 3D internals
       σ(E0,E1) × z=0,1 / Transmittance T × z=0,1 / ΣWeight / sep map
       3D module이 실제로 동작하는지 확인.
    """
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as iio2

    vca_layer.eval()
    orig_proc_ref = orig_procs.get(INJECT_KEY)
    if orig_proc_ref is None:
        orig_proc_ref = AttnProcessor2_0()

    prompt = (f"{probe_meta.get('prompt_entity0','entity0')} and "
              f"{probe_meta.get('prompt_entity1','entity1')}")
    P = height

    def _lbl(arr, text, fs=11):
        img = Image.fromarray(arr.astype(np.uint8))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
        except Exception:
            font = ImageFont.load_default()
        draw.text((2, 2), text, fill=(255, 255, 255), font=font)
        return np.array(img)

    def _to_panel(arr_hw, title, colormap=True):
        """1D spatial 배열 (S,) or 2D (H, W) → (P, P, 3) labeled panel."""
        m = arr_hw.reshape(int(len(arr_hw.flat) ** 0.5 + 0.5),
                           -1) if arr_hw.ndim == 1 else arr_hw
        hw_h, hw_w = m.shape
        if colormap:
            img = _heat_map(m)
        else:
            n = (m - m.min()) / (m.max() - m.min() + 1e-8)
            img = (np.stack([n, n, n], -1) * 255).astype(np.uint8)
        img_p = np.array(Image.fromarray(img).resize((P, P), Image.NEAREST))
        return _lbl(img_p, title)

    # ── Instrumented processor ────────────────────────────────────────────────
    instr_proc = AdditiveVCAInferProcessor(
        vca_layer, probe_entity_ctx, orig_proc_ref, alpha=VCA_ALPHA)

    # ── 1. Denoising trajectory ───────────────────────────────────────────────
    traj_baseline: dict = {}
    traj_vca:      dict = {}
    capture_steps  = {0, 4, 9, 14, 19}

    # Denoising trajectory: step count 변화로 근사 (callback 불필요)
    # step_counts = [3, 6, 10, 15, 20] → 점점 더 많이 denoising한 결과
    step_counts = [3, 6, 10, 15, 20]

    def _run_steps(procs, n_steps, gen_seed):
        pipe.unet.set_attn_processor(procs)
        g = torch.Generator(device=pipe.device).manual_seed(gen_seed)
        try:
            out = pipe(prompt=prompt, num_frames=8,
                       num_inference_steps=n_steps,
                       guidance_scale=7.5, height=height, width=width,
                       generator=g, output_type='pil')
            return np.stack([np.array(f) for f in out.frames[0]])
        except Exception:
            return None

    for n_steps in step_counts:
        arr_b = _run_steps(dict(orig_procs), n_steps, 99)
        if arr_b is not None:
            traj_baseline[n_steps] = arr_b

    infer_procs = dict(orig_procs)
    infer_procs[INJECT_KEY] = instr_proc
    for n_steps in step_counts:
        infer_procs_n = dict(orig_procs)
        infer_procs_n[INJECT_KEY] = instr_proc
        arr_v = _run_steps(infer_procs_n, n_steps, 99)
        if arr_v is not None:
            traj_vca[n_steps] = arr_v

    # 학습용 복원
    pipe.unet.set_attn_processor(dict(train_procs))
    vca_layer.train()

    # Build denoise_traj GIF: columns = step counts, rows = [Base | VCA]
    # GIF frame = (frame_index, step_count) — 가로: Base vs VCA, 세로: step 진행
    step_keys = sorted(traj_baseline.keys() | traj_vca.keys())
    if step_keys:
        traj_gif = []
        for fi in range(8):
            panels = []
            for n_steps in step_keys:
                b_arr = traj_baseline.get(n_steps)
                v_arr = traj_vca.get(n_steps)
                def _get(arr, i=fi):
                    if arr is None or i >= len(arr):
                        return np.zeros((P, P, 3), np.uint8)
                    return np.array(Image.fromarray(arr[i]).resize((P, P), Image.BILINEAR))
                col = np.concatenate([
                    _lbl(_get(b_arr), f"Base {n_steps}steps"),
                    _lbl(_get(v_arr), f"VCA  {n_steps}steps"),
                ], axis=0)  # [Base위 VCA아래]
                panels.append(col)
            traj_gif.append(np.concatenate(panels, axis=1))
        if traj_gif:
            iio2.mimsave(str(debug_dir / f"denoise_traj_epoch{epoch:03d}.gif"),
                         traj_gif, duration=250)
            print(f"  [debug] denoise_traj → denoise_traj_epoch{epoch:03d}.gif", flush=True)

    # ── 2. VCA delta magnitude + weight collapse ──────────────────────────────
    delta_gif = []
    mean_ratio = 0.0
    if (instr_proc.last_vca_delta is not None
            and instr_proc.last_text_out is not None):
        delta  = instr_proc.last_vca_delta.float()     # (BF, S, D)
        ttext  = instr_proc.last_text_out.float()
        BF, S, D = delta.shape
        hw = max(1, int(S ** 0.5))

        # |Δ|/|text| per token (manifold escape indicator)
        ratio    = delta.abs().mean(-1) / (ttext.abs().mean(-1) + 1e-8)   # (BF, S)
        mean_ratio = float(ratio.mean())
        max_ratio  = float(ratio.max())

        # weight sum from vca_layer
        weight_sum_np = None
        if (vca_layer.last_sigma is not None
                and vca_layer.last_transmittance is not None):
            sig_t = vca_layer.last_sigma.float()       # (BF, S, N, Z)
            T_t   = vca_layer.last_transmittance.float()  # (BF, S, Z)
            # w = T[z] * sigma[n,z], sum over N,Z
            ws = (T_t.unsqueeze(2) * sig_t).sum(dim=(2, 3))  # (BF, S)
            weight_sum_np = ws.cpu().numpy()

        for fi in range(BF):
            panels = []
            ratio_map = ratio[fi].cpu().numpy().reshape(hw, hw)
            panels.append(_to_panel(ratio_map,
                                    f"|Δ|/|txt| e{epoch:02d}f{fi} m={mean_ratio:.3f}"))

            if weight_sum_np is not None:
                ws_map = weight_sum_np[fi].reshape(hw, hw)
                panels.append(_to_panel(ws_map, f"ΣWeight (0=collapse) f{fi}"))

            if vca_layer.last_sigma is not None:
                sig_np = vca_layer.last_sigma[fi].cpu().numpy()  # (S, N, Z)
                panels.append(_to_panel(sig_np[:, 0, 0].reshape(hw, hw), "σ E0 z=0"))
                panels.append(_to_panel(sig_np[:, 1, 0].reshape(hw, hw), "σ E1 z=0"))

            delta_gif.append(np.concatenate(panels, axis=1))

        if delta_gif:
            iio2.mimsave(str(debug_dir / f"vca_delta_epoch{epoch:03d}.gif"),
                         delta_gif, duration=200)
            warn = "  ⚠ LARGE — manifold escape 위험" if mean_ratio > 0.5 else ""
            print(f"  [debug] vca_delta |Δ|/|txt| mean={mean_ratio:.3f} "
                  f"max={max_ratio:.3f}{warn}", flush=True)

    # ── 3. Ray marching 3D internals ─────────────────────────────────────────
    if vca_layer.last_sigma is not None:
        sig_np = vca_layer.last_sigma.float().cpu().numpy()        # (BF, S, N, Z)
        T_np   = (vca_layer.last_transmittance.float().cpu().numpy()
                  if vca_layer.last_transmittance is not None else None)
        BF, S, N, Z = sig_np.shape
        hw = max(1, int(S ** 0.5))
        n_per_row = 4
        ray_gif = []

        for fi in range(BF):
            panels = []
            # σ per entity × z-bin
            for n_e in range(N):
                for z in range(Z):
                    m = sig_np[fi, :, n_e, z].reshape(hw, hw)
                    panels.append(_to_panel(m, f"σ E{n_e} z={z}"))
            # Transmittance
            if T_np is not None:
                for z in range(Z):
                    m = T_np[fi, :, z].reshape(hw, hw)
                    panels.append(_to_panel(m, f"T z={z} (1=통과)"))
            # ΣWeight per token
            if T_np is not None:
                ws = (T_np[fi, :, :, None] * sig_np[fi]).sum(axis=(1, 2)).reshape(hw, hw)
                panels.append(_to_panel(ws, "ΣWeight (ray hit)"))
            # Separation map: σ(E0,z=0) - σ(E1,z=0) → 양수=E0 앞
            sep = (sig_np[fi, :, 0, 0] - sig_np[fi, :, 1, 0]).reshape(hw, hw)
            sep_col = _heat_map(sep, lo=-1.0, hi=1.0)
            sep_img  = np.array(Image.fromarray(sep_col).resize((P, P), Image.NEAREST))
            panels.append(_lbl(sep_img, "σ(E0)-σ(E1) [R=E0앞/B=E1앞]"))

            # 4-per-row 격자
            rows_out = []
            for i in range(0, len(panels), n_per_row):
                chunk = panels[i:i+n_per_row]
                while len(chunk) < n_per_row:
                    chunk.append(np.zeros((P, P, 3), np.uint8))
                rows_out.append(np.concatenate(chunk, axis=1))
            ray_gif.append(np.concatenate(rows_out, axis=0))

        if ray_gif:
            iio2.mimsave(str(debug_dir / f"ray3d_epoch{epoch:03d}.gif"),
                         ray_gif, duration=200)
            print(f"  [debug] ray3d internals → ray3d_epoch{epoch:03d}.gif", flush=True)

    return mean_ratio   # 호출부에서 경고 조건 판단용


# ─── 학습 denoising 품질 직접 시각화 ────────────────────────────────────────

@torch.no_grad()
def debug_train_denoising(pipe, vca_layer, probe_latents, probe_enc_hs,
                          probe_frames_np, out_dir: Path,
                          height=256, width=256,
                          t_values=(10, 50, 100, 150, 190)):
    """
    학습 중 denoising 품질 직접 시각화.

    고정 probe 훈련 샘플에 대해 여러 t 값별로:
      Row 1 — GT:        원본 영상 프레임 (첫 컬럼에만, 나머지는 blank)
      Row 2 — Noised:    x_t = add_noise(latents, t) → VAE decode
      Row 3 — Denoised:  x̂₀ pred = (x_t - √(1-ā)·ε̂) / √ā → VAE decode

    GIF 애니메이션 = video frame 순환
    각 column = t 값 (작은 noise → 큰 noise)

    저장: out_dir/train_denoise.gif
    """
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as iio2

    vca_layer.eval()
    P = height

    def _lbl(arr, text, fs=10):
        img = Image.fromarray(arr.astype(np.uint8))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
        except Exception:
            font = ImageFont.load_default()
        draw.text((2, 2), text, fill=(255, 255, 255), font=font)
        return np.array(img)

    # VAE dtype 확인 (pipeline이 fp16으로 로드되면 VAE도 fp16)
    _vae_dtype = next(pipe.vae.parameters()).dtype

    def _decode_frame(latents_5d, frame_idx):
        """(1, 4, T, lH, lW) → (H, W, 3) uint8 at frame_idx."""
        try:
            lat = latents_5d[0, :, frame_idx, :, :]       # (4, lH, lW)
            scaling = pipe.vae.config.scaling_factor
            # VAE dtype에 맞춰 입력 변환 (fp16 VAE에 fp32 넣으면 dtype 에러)
            lat_in = lat.unsqueeze(0).to(_vae_dtype) / scaling  # (1,4,lH,lW)
            dec = pipe.vae.decode(lat_in).sample[0]        # (3, H, W)
            arr = dec.float().permute(1, 2, 0).cpu().numpy()
            return np.clip((arr + 1) / 2 * 255, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"  [decode_frame err f={frame_idx}] {e}", flush=True)
            return np.zeros((P, P, 3), np.uint8)

    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(probe_latents.device)
    noise = torch.randn_like(probe_latents)
    T_frames = probe_latents.shape[2]  # video frame 수
    N_cols = len(t_values)

    gif_frames = []
    for fi in range(min(T_frames, len(probe_frames_np))):
        # GT row
        gt_raw = np.array(
            Image.fromarray(probe_frames_np[fi]).resize((P, P), Image.BILINEAR))
        gt_row = [_lbl(gt_raw, f"GT f={fi}")] + \
                 [np.zeros((P, P, 3), np.uint8)] * (N_cols - 1)

        noised_row   = []
        denoised_row = []

        for t_val in t_values:
            t = torch.tensor([t_val], device=probe_latents.device)

            # noised
            noisy = pipe.scheduler.add_noise(probe_latents, noise, t)
            n_frame = _decode_frame(noisy, fi)
            noised_row.append(_lbl(
                np.array(Image.fromarray(n_frame).resize((P, P), Image.BILINEAR)),
                f"t={t_val}"))

            # model prediction → x̂₀
            pred_eps = pipe.unet(
                noisy, t, encoder_hidden_states=probe_enc_hs).sample
            alpha_bar = alphas_cumprod[t_val]
            pred_x0 = (noisy - (1 - alpha_bar).sqrt() * pred_eps) / alpha_bar.sqrt()
            d_frame = _decode_frame(pred_x0.clamp(-4, 4), fi)
            denoised_row.append(_lbl(
                np.array(Image.fromarray(d_frame).resize((P, P), Image.BILINEAR)),
                f"Denoised t={t_val}"))

        gif_frames.append(np.concatenate([
            np.concatenate(gt_row,       axis=1),
            np.concatenate(noised_row,   axis=1),
            np.concatenate(denoised_row, axis=1),
        ], axis=0))

    if gif_frames:
        out_path = out_dir / "train_denoise.gif"
        iio2.mimsave(str(out_path), gif_frames, duration=300)
        print(f"  [debug] train_denoise → {out_path.parent.name}/{out_path.name}",
              flush=True)

    vca_layer.train()


# ─── training_step ───────────────────────────────────────────────────────────

def training_step_p21(pipe, vca_layer, latents, encoder_hidden_states,
                      depth_orders, lambda_depth, lambda_ortho, device,
                      t_max=200):
    noise = torch.randn_like(latents)
    t = torch.randint(0, t_max, (1,), device=device).long()
    noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

    vca_layer.reset_sigma_acc()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        pred_noise = pipe.unet(
            noisy_latents, t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

    # l_diff: additive 구조라 text attn이 diffusion을 주도 → VCA delta만 추가
    # 그러나 pred_noise에는 VCA delta가 포함되어 있어 gradient가 VCA로 흐름
    ld = loss_diff(pred_noise.float(), noise.float())

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
    print(f"[init] device={device}  lambda_depth={args.lambda_depth}  alpha={VCA_ALPHA}",
          flush=True)

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

    # probe 세트
    probe_frames, _, probe_orders, probe_meta = dataset[0]
    probe_entity_ctx = get_entity_context_from_meta(pipe, probe_meta, device)
    print(f"[probe] '{probe_meta.get('prompt_entity0')}' vs "
          f"'{probe_meta.get('prompt_entity1')}'", flush=True)

    # Additive VCA 주입 (핵심 수정)
    vca_layer, orig_procs = inject_vca_p21(pipe, probe_entity_ctx)
    # 학습용 processor dict 저장 (debug_generation 복원에 사용)
    train_procs = copy.copy(dict(pipe.unet.attn_processors))

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
    best_dra     = 0.0   # depth_rank_accuracy
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
            proc = pipe.unet.attn_processors.get(INJECT_KEY)
            if isinstance(proc, AdditiveVCAProcessor):
                proc.ctx = entity_ctx.float()

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
            step_out = training_step_p21(
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

        probe_sep = measure_probe_sep(
            pipe, vca_layer, probe_latents, probe_enc_hs, device
        )

        # 핵심 지표: depth_rank_accuracy
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
            dra, n_correct, n_total = measure_depth_rank_accuracy(
                pipe, vca_layer, dataset, device,
                n_samples=min(20, n_samples), t_val=100,
            )
            print(f"  [dra] depth_rank_accuracy={dra:.3f} ({n_correct}/{n_total})",
                  flush=True)
        else:
            dra = best_dra

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
            f"probe_sep={probe_sep:.4f} "
            f"dra={dra:.3f}",
            flush=True,
        )

        training_curve.append({
            "epoch": epoch, "lambda_depth": lambda_depth,
            **epoch_losses, "l_depth_weighted": l_depth_w,
            "probe_sep": probe_sep, "depth_rank_accuracy": dra,
        })

        if probe_sep > best_sep:
            best_sep = probe_sep
            best_dra = dra
            torch.save({
                "vca_state_dict":      vca_layer.state_dict(),
                "epoch":               epoch,
                "probe_sep":           best_sep,
                "depth_rank_accuracy": best_dra,
                "lambda_depth_final":  lambda_depth,
                "inject_key":          INJECT_KEY,
                "depth_pe_init_scale": DEPTH_PE_INIT_SCALE,
                "vca_alpha":           VCA_ALPHA,
                "multi_layer":         False,
                "additive":            True,   # 핵심 플래그
            }, save_dir / "best.pt")
            print(f"[ckpt] best.pt (probe_sep={best_sep:.4f} dra={best_dra:.3f})",
                  flush=True)

        # recon 디버그: 5 epoch마다 + 첫/마지막 epoch
        if (epoch + 1) % args.debug_every == 0 or epoch == 0 or epoch == args.epochs - 1:
            # epoch 전용 서브폴더
            epoch_dir = debug_dir / f"epoch_{epoch:03d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)

            # sigma gif → epoch 폴더
            gif_path = epoch_dir / "sigma.gif"
            save_sigma_gif(last_frames_np, vca_layer.last_sigma, gif_path)
            print(f"[gif] sigma → epoch_{epoch:03d}/sigma.gif", flush=True)

            # train_procs를 probe_entity_ctx 기준으로 업데이트
            proc = train_procs.get(INJECT_KEY)
            if isinstance(proc, AdditiveVCAProcessor):
                proc.ctx = probe_entity_ctx.float()

            debug_generation(
                pipe, vca_layer, orig_procs, train_procs,
                probe_frames, probe_meta, probe_entity_ctx,
                epoch_dir, epoch, height=args.height, width=args.width,
            )
            debug_multiview(
                pipe, vca_layer, orig_procs, train_procs,
                probe_meta, args.data_root, epoch_dir, epoch,
                height=args.height, width=args.width,
            )
            delta_ratio = debug_vca_internals(
                pipe, vca_layer, orig_procs, train_procs,
                probe_frames, probe_meta, probe_entity_ctx,
                epoch_dir, epoch, height=args.height, width=args.width,
            )
            if delta_ratio > 0.5:
                print(f"  ⚠ [epoch {epoch}] |Δ|/|text|={delta_ratio:.3f} > 0.5 "
                      f"— VCA가 diffusion manifold를 과도하게 수정 중", flush=True)

            # ★ 학습 denoising 품질: noised vs model denoised
            debug_train_denoising(
                pipe, vca_layer,
                probe_latents, probe_enc_hs,
                probe_frames, epoch_dir,
                height=args.height, width=args.width,
            )

    # ─── 최종 평가 ────────────────────────────────────────────────────────────
    final_dra, fc, ft = measure_depth_rank_accuracy(
        pipe, vca_layer, dataset, device,
        n_samples=min(50, n_samples), t_val=100,
    )
    print(f"\nFINAL probe_sep={best_sep:.6f}", flush=True)
    print(f"FINAL depth_rank_accuracy={final_dra:.4f} ({fc}/{ft})", flush=True)

    if final_dra >= 0.65:
        print("IDEA=WORKS", flush=True)
    elif final_dra >= 0.55:
        print("IDEA=PARTIAL", flush=True)
    else:
        print("IDEA=FAIL", flush=True)

    if best_sep > 0.01:
        print("LEARNING=OK", flush=True)
    else:
        print("LEARNING=FAIL", flush=True)

    with open(debug_dir / "training_curve.json", "w") as f:
        json.dump(training_curve, f, indent=2)
    print(f"[done]", flush=True)


# ─── CLI ─────────────────────────────────────────────────────────────────────

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
    p.add_argument("--save-dir",     default="checkpoints/phase21",    dest="save_dir")
    p.add_argument("--debug-dir",    default="debug/train_phase21",    dest="debug_dir")
    p.add_argument("--n-frames",     type=int,   default=8,  dest="n_frames")
    p.add_argument("--height",       type=int,   default=256)
    p.add_argument("--width",        type=int,   default=256)
    p.add_argument("--max-samples",  type=int,   default=None, dest="max_samples")
    p.add_argument("--debug-every",  type=int,   default=5,    dest="debug_every",
                   help="몇 epoch마다 recon/multiview 디버그 GIF 저장 (기본 5)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
