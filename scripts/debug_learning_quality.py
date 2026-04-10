"""
학습 품질 종합 진단: diffusion path 오염 여부 + 실제 학습 진행 확인

질문: VCA가 학습 데이터를 재구성할 수 있는가? diffusion이 망가지지 않는가?

출력 GIF 3종 (각 학습 샘플별):
  ① reconstruction.gif   — [GT(원본) | Baseline | P16 | P18 | P20]  5-panel
  ② denoising_traj.gif   — denoising 궤적 t=T→0: [Baseline | P20] 2-row × 10 컬럼
  ③ sigma_evolution.gif  — denoising 중 sigma 진화: [E0σ | E1σ | RGB] × 8 step

debug/learning_quality/{sample_id}/
  reconstruction.gif
  denoising_traj.gif
  sigma_evolution.gif
  sigma_final.gif        — 마지막 step sigma overlay (고해상도)
  stats.json

debug/learning_quality/summary.gif   — 4개 샘플 × reconstruction, 가장 대표적인 것
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as iio2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from scripts.run_animatediff import load_pipeline, FixedContextVCAProcessor
from scripts.train_animatediff_vca import (
    get_entity_embedding_mean,
    encode_frames_to_latents,
    compute_sigma_stats_train,
)
from scripts.train_objaverse_vca import (
    ObjaverseTrainDataset, get_entity_context_from_meta,
)
from scripts.make_figures import (
    _inject, _restore, _label, _resize_frame,
    _sigma_to_heatmap, extract_sigma_maps,
)
from scripts.train_phase19 import inject_vca_p19_infer

PANEL = 256   # 각 패널 크기


# ─── 유틸 ────────────────────────────────────────────────────────────────────

def _font(size=14):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _label2(arr, text, bg=(0,0,0), fg=(255,255,255), size=14):
    """라벨을 상단 바로 추가 (원본 크기 유지)."""
    H, W = arr.shape[:2]
    bar_h = 22
    canvas = np.zeros((H + bar_h, W, 3), dtype=np.uint8)
    canvas[:bar_h] = bg
    canvas[bar_h:] = arr
    img = Image.fromarray(canvas)
    ImageDraw.Draw(img).text((4, 3), text, fill=fg, font=_font(size))
    return np.array(img)


def decode_latents_to_frames(pipe, latents, height=256, width=256):
    """
    latents: (1, 4, T, H//8, W//8)
    → list of (H, W, 3) uint8
    """
    lat = latents / pipe.vae.config.scaling_factor
    B, C, T, h, w = lat.shape
    lat2d = lat[0].permute(1, 0, 2, 3)  # (T, C, h, w)
    frames = []
    with torch.no_grad():
        for i in range(T):
            dec = pipe.vae.decode(lat2d[i:i+1].to(pipe.device, dtype=torch.float16)).sample
            img = ((dec[0].permute(1,2,0).clamp(-1,1) + 1) * 127.5).byte().cpu().numpy()
            frames.append(img)
    return frames


def load_vca_standard(ckpt_path, device, init_scale=0.02):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    vca = VCALayer(query_dim=1280, context_dim=768,
                   n_heads=8, n_entities=2, z_bins=2, lora_rank=8,
                   use_softmax=False, depth_pe_init_scale=init_scale).to(device)
    vca.load_state_dict(ckpt["vca_state_dict"])
    vca.eval()
    return vca


def load_vca_from_ckpt(ckpt_path, device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    scale = ckpt.get("depth_pe_init_scale", 0.02)
    vca = VCALayer(query_dim=1280, context_dim=768,
                   n_heads=8, n_entities=2, z_bins=2, lora_rank=8,
                   use_softmax=False, depth_pe_init_scale=scale).to(device)
    vca.load_state_dict(ckpt["vca_state_dict"])
    vca.eval()
    return vca


# ─── ① 재구성 품질: [GT | Baseline | P16 | P18 | P20] ──────────────────────

def make_reconstruction_gif(gt_frames, baseline_frames, p16_frames, p18_frames,
                             p20_frames, out_path: Path, p19_frames=None):
    """
    행: 8개 프레임
    열: GT / Baseline / P16 / P18 / [P19] / P20
    각 패널에 라벨 포함.
    """
    panels_per_frame = 5 + (1 if p19_frames else 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames = []

    sets = [
        (gt_frames,       "GT (원본)"),
        (baseline_frames, "Baseline"),
        (p16_frames,      "Phase16"),
        (p18_frames,      "Phase18"),
    ]
    if p19_frames:
        sets.append((p19_frames, "Phase19"))
    sets.append((p20_frames, "Phase20"))

    n_frames = min(len(s[0]) for s in sets)
    for fi in range(n_frames):
        row = []
        for frames, label in sets:
            f = np.array(Image.fromarray(frames[fi]).resize(
                (PANEL, PANEL), Image.BILINEAR))
            row.append(_label2(f, label))
        gif_frames.append(np.concatenate(row, axis=1))

    iio2.mimsave(str(out_path), gif_frames, duration=300)
    print(f"  [gif] reconstruction → {out_path}", flush=True)


# ─── ② denoising 궤적 ─────────────────────────────────────────────────────

def run_denoising_with_capture(pipe, prompt, entity_ctx, vca_layer,
                                inject_fn, restore_fn,
                                num_frames=8, steps=20, height=256, width=256,
                                seed=42, capture_every=2, device="cuda"):
    """
    denoising loop를 수동으로 실행하며 중간 latent + sigma 캡처.

    Returns:
      step_frames: list of list-of-PIL (캡처된 step마다 frames)
      step_sigmas: list of np.ndarray (BF, S, N, Z) or None
      step_ts:     list of int (캡처된 t 값)
    """
    pipe.scheduler.set_timesteps(steps)
    gen = torch.Generator(device=device).manual_seed(seed)

    # 텍스트 인코딩
    tok = pipe.tokenizer(prompt, return_tensors="pt", padding="max_length",
                         max_length=pipe.tokenizer.model_max_length,
                         truncation=True).to(device)
    with torch.no_grad():
        enc_hs = pipe.text_encoder(**tok).last_hidden_state.half()
    # CFG: uncond + cond
    unc_tok = pipe.tokenizer("", return_tensors="pt", padding="max_length",
                              max_length=pipe.tokenizer.model_max_length,
                              truncation=True).to(device)
    with torch.no_grad():
        unc_hs = pipe.text_encoder(**unc_tok).last_hidden_state.half()
    enc_hs_cfg = torch.cat([unc_hs, enc_hs], dim=0)  # (2, seq, 768)

    # 초기 latent
    H8, W8 = height // 8, width // 8
    latents = torch.randn(1, 4, num_frames, H8, W8,
                          generator=gen, device=device, dtype=torch.float16)
    latents = latents * pipe.scheduler.init_noise_sigma

    step_frames = []
    step_sigmas = []
    step_ts = []

    orig_procs = inject_fn(pipe, vca_layer, entity_ctx)

    for step_idx, t in enumerate(pipe.scheduler.timesteps):
        latents_in = torch.cat([latents] * 2)  # CFG
        latents_in = pipe.scheduler.scale_model_input(latents_in, t)

        with torch.no_grad():
            vca_layer.reset_sigma_acc()
            noise_pred = pipe.unet(latents_in, t,
                                   encoder_hidden_states=enc_hs_cfg).sample

        # CFG
        noise_unc, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_unc + 7.5 * (noise_cond - noise_unc)

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # 캡처
        if step_idx % capture_every == 0 or step_idx == steps - 1:
            frames = decode_latents_to_frames(pipe, latents, height, width)
            step_frames.append(frames)
            if vca_layer.last_sigma is not None:
                step_sigmas.append(vca_layer.last_sigma.detach().cpu().float().numpy())
            else:
                step_sigmas.append(None)
            step_ts.append(int(t))

    restore_fn(pipe, orig_procs)
    return step_frames, step_sigmas, step_ts


def make_denoising_traj_gif(base_steps, p20_steps, step_ts, out_path: Path):
    """
    [Baseline denoising | Phase20 denoising] 2-row, N-col GIF.
    각 컬럼 = 캡처된 denoising step.
    프레임 하나 = 모든 step 한 번에 보이는 모자이크 (1장짜리 GIF가 아님).
    GIF 각 frame = video의 frame_idx번째 프레임에 대한 denoising 궤적.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # step_ts: [t_0, t_1, ...] (캡처 순서, 내림차순)
    # base_steps: list(len=n_capture) of list(n_frames) of (H,W,3)
    # GIF frame = 비디오 frame_idx 고정, denoising step가 column

    n_capture = len(base_steps)
    n_vid_frames = len(base_steps[0])
    gif_frames = []

    P = PANEL // 2  # 작게

    for fi in range(n_vid_frames):
        # 각 denoising step 열
        base_row = []
        p20_row  = []
        for si in range(n_capture):
            t_label = f"t={step_ts[si]}"
            b = np.array(Image.fromarray(base_steps[si][fi]).resize((P, P), Image.BILINEAR))
            p = np.array(Image.fromarray(p20_steps[si][fi]).resize((P, P), Image.BILINEAR))
            base_row.append(_label2(b, t_label, size=10))
            p20_row.append(_label2(p, t_label, size=10))

        # 행 라벨
        row_h = base_row[0].shape[0]
        row_w_total = P * n_capture
        base_label_col = np.zeros((row_h, 70, 3), dtype=np.uint8)
        p20_label_col  = np.zeros((row_h, 70, 3), dtype=np.uint8)
        lbl_img = Image.fromarray(base_label_col)
        ImageDraw.Draw(lbl_img).text((4, row_h//2 - 8), "Baseline", fill=(255,255,255), font=_font(11))
        base_label_col = np.array(lbl_img)
        lbl_img2 = Image.fromarray(p20_label_col)
        ImageDraw.Draw(lbl_img2).text((4, row_h//2 - 8), "Phase20", fill=(200,255,200), font=_font(11))
        p20_label_col = np.array(lbl_img2)

        base_full = np.concatenate([base_label_col] + base_row, axis=1)
        p20_full  = np.concatenate([p20_label_col]  + p20_row,  axis=1)

        frame = np.concatenate([base_full, p20_full], axis=0)
        gif_frames.append(frame)

    iio2.mimsave(str(out_path), gif_frames, duration=300)
    print(f"  [gif] denoising_traj → {out_path}", flush=True)


# ─── ③ sigma 진화 GIF ────────────────────────────────────────────────────────

def make_sigma_evolution_gif(p20_steps, step_sigmas, step_ts, out_path: Path):
    """
    각 video frame에 대해: 각 denoising step별로 [RGB | E0σ | E1σ | diff] 4-panel.
    GIF frame = video frame, 4col × n_step row 모자이크.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_capture = len(p20_steps)
    n_vid = len(p20_steps[0])
    P = PANEL // 2
    gif_frames = []

    for fi in range(n_vid):
        rows = []
        for si in range(n_capture):
            rgb_arr = np.array(Image.fromarray(p20_steps[si][fi]).resize((P, P), Image.BILINEAR))
            t_label = f"t={step_ts[si]}"

            if step_sigmas[si] is not None:
                sig = step_sigmas[si]    # (BF, S, N, Z)
                BF  = sig.shape[0]
                # CFG: 후반 BF//2가 cond
                cond_idx = BF // 2 + fi if (BF > n_vid) else fi
                cond_idx = min(cond_idx, BF - 1)
                s_frame  = sig[cond_idx]    # (S, N, Z)
                hw = max(1, int(s_frame.shape[0] ** 0.5))
                e0 = s_frame[:, 0, 0].reshape(hw, hw)
                e1 = s_frame[:, 1, 0].reshape(hw, hw)
                diff = np.abs(e0 - e1)

                e0_heat   = _sigma_to_heatmap(e0, P)
                e1_heat   = _sigma_to_heatmap(e1, P)
                diff_heat = _sigma_to_heatmap(diff, P)
            else:
                e0_heat = e1_heat = diff_heat = np.zeros((P, P, 3), dtype=np.uint8)

            row = np.concatenate([
                _label2(rgb_arr,   t_label,   size=10),
                _label2(e0_heat,   f"E0 t={step_ts[si]}", bg=(20,20,80),  size=10),
                _label2(e1_heat,   f"E1 t={step_ts[si]}", bg=(80,20,20),  size=10),
                _label2(diff_heat, f"|E0-E1|",             bg=(20,60,20),  size=10),
            ], axis=1)
            rows.append(row)

        gif_frames.append(np.concatenate(rows, axis=0))

    iio2.mimsave(str(out_path), gif_frames, duration=350)
    print(f"  [gif] sigma_evolution → {out_path}", flush=True)


# ─── ④ 최종 sigma overlay (고해상도) ─────────────────────────────────────────

def make_sigma_overlay_gif(frames, vca_layer, out_path: Path, alpha=0.5):
    """
    생성된 프레임 위에 E0σ (파란색), E1σ (빨간색) 반투명 오버레이.
    실제 sigma가 어디를 바라보는지 시각화.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if vca_layer.last_sigma is None:
        return

    sig_np = vca_layer.last_sigma.detach().cpu().float().numpy()
    BF, S, N, Z = sig_np.shape
    n = len(frames)
    hw = max(1, int(S ** 0.5))
    P = PANEL
    gif_frames = []

    for fi in range(n):
        cond_idx = BF // 2 + fi if BF > n else fi
        cond_idx = min(cond_idx, BF - 1)
        s = sig_np[cond_idx]   # (S, N, Z)

        e0 = s[:, 0, 0].reshape(hw, hw)
        e1 = s[:, 1, 0].reshape(hw, hw)

        rgb = np.array(Image.fromarray(frames[fi]).resize((P, P), Image.BILINEAR))

        # E0 → 파란 오버레이
        e0_norm = (e0 - e0.min()) / (e0.max() - e0.min() + 1e-6)
        e0_up = np.array(Image.fromarray((e0_norm * 255).astype(np.uint8)).resize(
            (P, P), Image.BILINEAR)) / 255.0
        blue_mask = np.stack([np.zeros_like(e0_up), np.zeros_like(e0_up), e0_up], axis=-1)

        # E1 → 빨간 오버레이
        e1_norm = (e1 - e1.min()) / (e1.max() - e1.min() + 1e-6)
        e1_up = np.array(Image.fromarray((e1_norm * 255).astype(np.uint8)).resize(
            (P, P), Image.BILINEAR)) / 255.0
        red_mask = np.stack([e1_up, np.zeros_like(e1_up), np.zeros_like(e1_up)], axis=-1)

        base = rgb.astype(np.float32) / 255.0
        overlay = np.clip(base + alpha * blue_mask + alpha * red_mask, 0, 1)
        overlay_u8 = (overlay * 255).astype(np.uint8)

        # 3-panel: [원본 | E0 heatmap | overlay]
        panel = np.concatenate([
            _label2(rgb,        "Generated"),
            _label2(_sigma_to_heatmap(e0, P), "E0 σ (blue)"),
            _label2(_sigma_to_heatmap(e1, P), "E1 σ (red)"),
            _label2(overlay_u8, "Overlay"),
        ], axis=1)
        gif_frames.append(panel)

    iio2.mimsave(str(out_path), gif_frames, duration=300)
    print(f"  [gif] sigma_overlay → {out_path}", flush=True)


# ─── null inject / restore (baseline용) ──────────────────────────────────────

def null_inject(pipe, vca_layer, entity_ctx):
    """baseline용: 아무것도 주입하지 않고 identity restore 반환."""
    return copy.copy(dict(pipe.unet.attn_processors))


def null_restore(pipe, orig):
    pipe.unet.set_attn_processor(orig)


def p16_inject(pipe, vca_layer, entity_ctx):
    orig = _inject(pipe, vca_layer, entity_ctx)
    return orig


def p16_restore(pipe, orig):
    _restore(pipe, orig)


def p19_inject_fn(pipe, vca_layer, entity_ctx):
    orig, _ = inject_vca_p19_infer(pipe, vca_layer, entity_ctx)
    return orig


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}", flush=True)

    # 체크포인트 로드
    p16_vca = load_vca_standard(args.phase16_ckpt, device, 0.02)
    p18_vca = load_vca_standard(args.phase18_ckpt, device, 0.02)
    p19_vca = load_vca_from_ckpt(args.phase19_ckpt, device)
    p20_vca = load_vca_from_ckpt(args.phase20_ckpt, device)
    print(f"[init] checkpoints loaded", flush=True)

    pipe = load_pipeline(device=device, dtype=torch.float16)
    for p in pipe.unet.parameters():
        p.requires_grad = False

    # 학습 데이터셋에서 샘플 선택
    dataset = ObjaverseTrainDataset(
        data_root=args.data_root, max_samples=None,
        n_frames=8, height=args.height, width=args.width,
    )
    print(f"[dataset] {len(dataset)} samples", flush=True)

    # 대표 샘플: 다양한 entity pair 선택
    import json as _json
    from pathlib import Path as _P
    import os

    all_dirs = sorted(os.listdir(args.data_root))
    selected = []
    seen_pairs = set()
    for d in all_dirs:
        m = _json.load(open(f"{args.data_root}/{d}/meta.json"))
        pair = (m.get("prompt_entity0"), m.get("prompt_entity1"))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            selected.append(d)
        if len(selected) >= args.n_samples:
            break
    print(f"[samples] selected {len(selected)} unique entity pairs", flush=True)

    out_root = _P(args.out_dir)
    summary_panels = []

    for sample_name in selected:
        print(f"\n{'='*60}", flush=True)
        print(f"[sample] {sample_name}", flush=True)

        # 실제 데이터 로드
        meta = _json.load(open(f"{args.data_root}/{sample_name}/meta.json"))
        from PIL import Image as _Img
        import glob
        frame_paths = sorted(glob.glob(f"{args.data_root}/{sample_name}/frames/*.png"))[:8]
        gt_frames = [np.array(_Img.open(p).convert("RGB").resize(
            (args.width, args.height), _Img.BILINEAR)) for p in frame_paths]
        if not gt_frames:
            print(f"  [skip] no frames found", flush=True)
            continue

        out_dir = out_root / sample_name
        out_dir.mkdir(parents=True, exist_ok=True)

        prompt    = meta.get("prompt_full", "two entities")
        entity_ctx = get_entity_context_from_meta(pipe, meta, device)

        print(f"  prompt: '{prompt}'", flush=True)
        print(f"  entity0: '{meta.get('prompt_entity0')}'  "
              f"entity1: '{meta.get('prompt_entity1')}'", flush=True)

        # ── 1. 각 모델로 생성 ─────────────────────────────────────────────
        kw = dict(num_frames=8, steps=args.steps,
                  height=args.height, width=args.width, seed=args.seed)

        def gen_with(vca, inject_fn, restore_fn):
            from scripts.make_figures import generate
            if vca is None:
                return generate(pipe, prompt, **kw)
            orig = inject_fn(pipe, vca, entity_ctx)
            frames = generate(pipe, prompt, **kw)
            restore_fn(pipe, orig)
            return frames

        print(f"  [gen] baseline...", flush=True)
        baseline_frames = gen_with(None, null_inject, null_restore)

        print(f"  [gen] phase16...", flush=True)
        p16_frames = gen_with(p16_vca, p16_inject, p16_restore)

        print(f"  [gen] phase18...", flush=True)
        p18_frames = gen_with(p18_vca, p16_inject, p16_restore)

        print(f"  [gen] phase19...", flush=True)
        p19_frames = gen_with(p19_vca, p19_inject_fn, p16_restore)

        print(f"  [gen] phase20...", flush=True)
        p20_frames = gen_with(p20_vca, p16_inject, p16_restore)

        # ① reconstruction GIF
        make_reconstruction_gif(
            gt_frames, baseline_frames, p16_frames, p18_frames, p20_frames,
            out_dir / "reconstruction.gif",
            p19_frames=p19_frames,
        )

        # ── 2. denoising 궤적 (baseline vs P20) ──────────────────────────
        if args.denoising_traj:
            print(f"  [traj] baseline denoising...", flush=True)
            base_dummy_vca = p20_vca   # dummy — null_inject이므로 무관
            base_steps, _, base_ts = run_denoising_with_capture(
                pipe, prompt, entity_ctx, p20_vca,
                null_inject, null_restore,
                num_frames=8, steps=args.steps,
                height=args.height, width=args.width, seed=args.seed,
                capture_every=args.capture_every, device=device,
            )
            print(f"  [traj] phase20 denoising...", flush=True)
            p20_steps, p20_sigs, p20_ts = run_denoising_with_capture(
                pipe, prompt, entity_ctx, p20_vca,
                p16_inject, p16_restore,
                num_frames=8, steps=args.steps,
                height=args.height, width=args.width, seed=args.seed,
                capture_every=args.capture_every, device=device,
            )
            make_denoising_traj_gif(base_steps, p20_steps, p20_ts,
                                    out_dir / "denoising_traj.gif")
            make_sigma_evolution_gif(p20_steps, p20_sigs, p20_ts,
                                     out_dir / "sigma_evolution.gif")

        # ③ sigma overlay (최종 step)
        # P20으로 한번 더 생성해서 last_sigma 확보
        orig = p16_inject(pipe, p20_vca, entity_ctx)
        from scripts.make_figures import generate
        final_p20 = generate(pipe, prompt, **kw)
        p16_restore(pipe, orig)
        make_sigma_overlay_gif(final_p20, p20_vca,
                               out_dir / "sigma_final.gif")

        # stats 저장
        def _get_sep(vca, inject_fn, restore_fn):
            orig = inject_fn(pipe, vca, entity_ctx)
            generate(pipe, prompt, **kw)
            from scripts.make_figures import compute_stats
            stats = compute_stats(vca, 8, use_cfg=True)
            restore_fn(pipe, orig)
            return stats

        stats = {
            "sample": sample_name,
            "prompt": prompt,
            "phase16": _get_sep(p16_vca, p16_inject, p16_restore),
            "phase18": _get_sep(p18_vca, p16_inject, p16_restore),
            "phase19": _get_sep(p19_vca, p19_inject_fn, p16_restore),
            "phase20": _get_sep(p20_vca, p16_inject, p16_restore),
        }
        with open(out_dir / "stats.json", "w") as f:
            _json.dump(stats, f, indent=2)
        print(f"  [stats] p16={stats['phase16']['sigma_separation']:.4f}  "
              f"p18={stats['phase18']['sigma_separation']:.4f}  "
              f"p19={stats['phase19']['sigma_separation']:.4f}  "
              f"p20={stats['phase20']['sigma_separation']:.4f}", flush=True)

        # summary 패널용: reconstruction 첫 프레임
        summary_panels.append({
            "name": sample_name,
            "gt": gt_frames[0],
            "baseline": baseline_frames[0],
            "p20": p20_frames[0],
            "sep_p20": stats["phase20"]["sigma_separation"],
        })

    # ─── summary.gif ─────────────────────────────────────────────────────
    if summary_panels:
        P = PANEL
        gif_row = []
        for s in summary_panels:
            col = np.concatenate([
                _label2(np.array(Image.fromarray(s["gt"]).resize((P, P), Image.BILINEAR)),
                        f"GT: {s['name'][:14]}"),
                _label2(np.array(Image.fromarray(s["baseline"]).resize((P, P), Image.BILINEAR)),
                        "Baseline"),
                _label2(np.array(Image.fromarray(s["p20"]).resize((P, P), Image.BILINEAR)),
                        f"P20 sep={s['sep_p20']:.3f}"),
            ], axis=0)
            gif_row.append(col)
        summary_frame = np.concatenate(gif_row, axis=1)
        iio2.mimsave(str(out_root / "summary.gif"), [summary_frame], duration=2000)
        print(f"\n[done] summary → {out_root / 'summary.gif'}", flush=True)

    print(f"\n[done] all results → {out_root}", flush=True)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase16-ckpt", required=True, dest="phase16_ckpt")
    p.add_argument("--phase18-ckpt", required=True, dest="phase18_ckpt")
    p.add_argument("--phase19-ckpt", required=True, dest="phase19_ckpt")
    p.add_argument("--phase20-ckpt", required=True, dest="phase20_ckpt")
    p.add_argument("--data-root",    default="toy/data_objaverse", dest="data_root")
    p.add_argument("--out-dir",      default="debug/learning_quality", dest="out_dir")
    p.add_argument("--n-samples",    type=int, default=4, dest="n_samples",
                   help="분석할 학습 샘플 수 (고유 entity pair 기준)")
    p.add_argument("--steps",        type=int, default=20)
    p.add_argument("--height",       type=int, default=256)
    p.add_argument("--width",        type=int, default=256)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--denoising-traj", action="store_true", dest="denoising_traj",
                   help="denoising 궤적 GIF 생성 (느림, ~3배 시간)")
    p.add_argument("--capture-every", type=int, default=4, dest="capture_every",
                   help="denoising step 몇 개마다 캡처 (기본 4 → 20step 중 5장)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
