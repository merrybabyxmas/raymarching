"""
Phase 39 — Chimera 비교 (Entity-Slot v2 vs Phase 38 baseline)
==============================================================

Phase 38 chimera_phase38.py와의 차이
--------------------------------------
Phase 38: entity_score (RGB threshold) 로 비교 → 배경색 오판 + entity 소멸 미감지
Phase 39: GT entity_masks 기반 정량 테이블 + 고정 프롬프트 qualitative GIF

두 가지 비교 산출물
-------------------
1. [정량] teacher-forced GT table
   - visible_iou, ordering_acc, wrong_slot_leak, val_slot_score
   - Phase 38 baseline vs Phase 39 improved
   - JSON + console 출력

2. [정성] 고정 프롬프트 qualitative GIF
   - 동일 seed, 동일 프롬프트로 baseline/p39 생성
   - debug/ 디렉토리에 저장

entity_score/chimera_rate는 debug 참고용으로 저장하되
정량 비교의 주 지표로 사용하지 않음.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as iio2

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.entity_slot import (
    EntitySlotAttnProcessor,
    inject_entity_slot,
    # GT mask 기반 metrics
    compute_visible_iou,
    compute_ordering_accuracy,
    compute_wrong_slot_leak,
    val_slot_score,
    compute_overlap_score,
    # debug only
    entity_score as compute_entity_score_debug,
)
from scripts.run_animatediff import load_pipeline
from scripts.train_animatediff_vca import encode_frames_to_latents
from scripts.train_phase31 import (
    VCA_ALPHA, INJECT_KEY, INJECT_QUERY_DIM,
    DEPTH_PE_INIT_SCALE,
    ObjaverseDatasetWithMasks,
    make_color_prompts, get_color_entity_context,
    restore_procs,
    inject_vca_p21_infer,
)
from scripts.train_phase35 import (
    DEFAULT_Z_BINS, get_entity_token_positions,
)


# =============================================================================
# Config
# =============================================================================

SEEDS = [42, 123, 456]

QUALITATIVE_PAIRS = [
    ("cat",    "dog"),
    ("lion",   "bear"),
    ("tiger",  "wolf"),
    ("snake",  "alligator"),
    ("person", "snake"),
    ("cat",    "sword"),
]


# =============================================================================
# Helpers
# =============================================================================

def make_meta(kw0: str, kw1: str) -> dict:
    return {
        "keyword0": kw0,
        "keyword1": kw1,
        "color0":   [0.85, 0.15, 0.1],
        "color1":   [0.1,  0.25, 0.85],
    }


def load_p39_checkpoint(ckpt_path: str, device: str,
                        adapter_rank: int = 64):
    """Phase 39 checkpoint 로드 (adapter + blend_head 포함)."""
    ckpt      = torch.load(ckpt_path, map_location=device)
    vca_layer = VCALayer(
        query_dim=INJECT_QUERY_DIM, context_dim=768,
        n_heads=8, n_entities=2, z_bins=DEFAULT_Z_BINS, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=DEPTH_PE_INIT_SCALE,
    ).to(device)
    vca_layer.load_state_dict(ckpt["vca_state_dict"])

    slot_blend_raw = ckpt.get("slot_blend_raw", None)
    slot_blend     = float(ckpt.get("slot_blend", 0.3))
    ckpt_adapter_rank = int(ckpt.get("adapter_rank", adapter_rank))

    return vca_layer, slot_blend_raw, slot_blend, ckpt, ckpt_adapter_rank


def restore_p39_proc(pipe, vca_layer, entity_ctx, slot_blend,
                     slot_blend_raw, ckpt, adapter_rank, device):
    """Phase 39 proc 재주입 (adapter + blend_head 복원)."""
    proc, orig_procs = inject_entity_slot(
        pipe, vca_layer, entity_ctx,
        inject_key=INJECT_KEY,
        slot_blend_init=slot_blend,
        adapter_rank=adapter_rank,
        use_blend_head=True,
    )
    proc = proc.to(device)
    if slot_blend_raw is not None:
        proc.slot_blend_raw.data.copy_(slot_blend_raw.to(device))
    if "slot0_adapter" in ckpt:
        proc.slot0_adapter.load_state_dict(ckpt["slot0_adapter"])
    if "slot1_adapter" in ckpt:
        proc.slot1_adapter.load_state_dict(ckpt["slot1_adapter"])
    if "blend_head" in ckpt:
        proc.blend_head.load_state_dict(ckpt["blend_head"])
    proc.eval()
    for p in proc.parameters():
        p.requires_grad_(False)
    return proc, orig_procs


def save_gif(frames: np.ndarray, path: Path, fps: int = 8):
    path.parent.mkdir(parents=True, exist_ok=True)
    iio2.mimwrite(str(path), frames, fps=fps, loop=0)


def make_chimera_overlay(frames: np.ndarray) -> np.ndarray:
    """chimera 영역을 노란색으로 하이라이트 (debug용)."""
    out = []
    for frame in frames:
        f = frame.copy()
        r = f[:, :, 0].astype(np.float32)
        g = f[:, :, 1].astype(np.float32)
        b = f[:, :, 2].astype(np.float32)
        is_red  = (r > 80) & (g < 120) & (b < 120)
        is_blue = (b > 80) & (r < 120) & (g < 120)
        chimera = is_red & is_blue
        f[chimera] = [255, 255, 0]
        out.append(f)
    return np.stack(out)


# =============================================================================
# Teacher-forced quantitative evaluation
# =============================================================================

@torch.no_grad()
def teacher_forced_eval(
    pipe,
    proc:       EntitySlotAttnProcessor,
    dataset,
    eval_idx:   list,
    device:     str,
    t_fixed:    int = 150,
) -> dict:
    """
    held-out 샘플에 대해 teacher-forced 평가.
    GT latents로 UNet forward → w0/w1 수집 → GT mask 기반 metrics.
    """
    vis_ious    = []
    ord_accs    = []
    wrong_leaks = []

    for vi in eval_idx:
        try:
            sample = dataset[vi]
            frames_np, _, depth_orders, meta, entity_masks = sample

            entity_ctx = get_color_entity_context(pipe, meta, device)
            toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)

            proc.set_entity_ctx(entity_ctx.float())
            proc.set_entity_tokens(toks_e0, toks_e1)
            proc.reset_slot_store()

            latents  = encode_frames_to_latents(pipe, frames_np, device)
            noise    = torch.randn_like(latents)
            t_tensor = torch.tensor([t_fixed], device=device).long()
            noisy    = pipe.scheduler.add_noise(latents, noise, t_tensor)

            tok = pipe.tokenizer(
                full_prompt, return_tensors="pt", padding="max_length",
                max_length=pipe.tokenizer.model_max_length, truncation=True,
            ).to(device)
            enc_hs = pipe.text_encoder(**tok).last_hidden_state.half()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = pipe.unet(noisy, t_tensor, encoder_hidden_states=enc_hs).sample

            if proc.last_w0 is None:
                continue

            T_frames = min(proc.last_w0.shape[0], entity_masks.shape[0])
            masks_t  = torch.from_numpy(
                entity_masks[:T_frames].astype(np.float32)).to(device)

            for fi in range(T_frames):
                w0_f = proc.last_w0[fi:fi+1].float()
                w1_f = proc.last_w1[fi:fi+1].float()
                m_f  = masks_t[fi:fi+1]
                do   = [depth_orders[fi]] if fi < len(depth_orders) else [(0, 1)]

                vis_ious.append(compute_visible_iou(w0_f, w1_f, m_f, do))
                ord_accs.append(compute_ordering_accuracy(w0_f, w1_f, m_f, do))
                wrong_leaks.append(compute_wrong_slot_leak(w0_f, w1_f, m_f))

        except Exception as e:
            print(f"  [warn] eval sample {vi} 실패: {e}", flush=True)

    if not vis_ious:
        return {"visible_iou": 0.0, "ordering_acc": 0.0,
                "wrong_slot_leak": 1.0, "val_score": 0.0}

    vi  = float(np.mean(vis_ious))
    oa  = float(np.mean(ord_accs))
    wl  = float(np.mean(wrong_leaks))
    vs  = val_slot_score(vi, oa, wl, 0.0)   # DRA=0 (별도 측정 필요)
    return {"visible_iou": vi, "ordering_acc": oa,
            "wrong_slot_leak": wl, "val_score": vs}


# =============================================================================
# Qualitative GIF generation
# =============================================================================

@torch.no_grad()
def generate_p39(pipe, proc, entity_ctx, toks_e0, toks_e1,
                 full_prompt, seed, n_frames, n_steps, height, width,
                 device) -> np.ndarray:
    proc.set_entity_ctx(entity_ctx.float())
    proc.set_entity_tokens(toks_e0, toks_e1)
    proc.reset_slot_store()
    gen = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt=full_prompt,
        num_frames=n_frames, num_inference_steps=n_steps,
        height=height, width=width,
        generator=gen, output_type="np",
    )
    return (out.frames[0] * 255).astype(np.uint8)


@torch.no_grad()
def generate_baseline(pipe, orig_procs, vca_layer, entity_ctx,
                      full_prompt, seed, n_frames, n_steps, height, width,
                      device) -> np.ndarray:
    inject_vca_p21_infer(pipe, vca_layer, entity_ctx, VCA_ALPHA)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt=full_prompt,
        num_frames=n_frames, num_inference_steps=n_steps,
        height=height, width=width,
        generator=gen, output_type="np",
    )
    frames = (out.frames[0] * 255).astype(np.uint8)
    restore_procs(pipe, orig_procs)
    return frames


# =============================================================================
# Main
# =============================================================================

def run(args):
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("[p39] 파이프라인 로드...", flush=True)
    pipe = load_pipeline(device=device, dtype=torch.float16)

    print(f"[p39] Phase 39 체크포인트 로드: {args.ckpt}", flush=True)
    vca_layer, slot_blend_raw, slot_blend, ckpt, adapter_rank = load_p39_checkpoint(
        args.ckpt, device, adapter_rank=args.adapter_rank)
    print(f"  slot_blend={slot_blend:.4f}  adapter_rank={adapter_rank}", flush=True)

    # ── 1. Teacher-forced quantitative table ─────────────────────────────────
    if args.data_root:
        print(f"\n[p39] Teacher-forced quantitative evaluation", flush=True)
        dataset = ObjaverseDatasetWithMasks(args.data_root, n_frames=args.n_frames)

        # overlap score로 어려운 샘플 선택
        from models.entity_slot import compute_overlap_score
        overlap_scores = []
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                overlap_scores.append(compute_overlap_score(sample[4]))
            except Exception:
                overlap_scores.append(0.0)
        overlap_scores = np.array(overlap_scores)
        n_eval = max(args.min_eval_samples, int(len(dataset) * args.eval_frac))
        eval_idx = np.argsort(overlap_scores)[::-1][:n_eval].tolist()
        print(f"  eval samples={len(eval_idx)} "
              f"(top {args.eval_frac:.0%} overlap-heavy)", flush=True)

        # Phase 39 proc 주입
        dummy_meta = make_meta("cat", "dog")
        dummy_ctx  = get_color_entity_context(pipe, dummy_meta, device)
        proc_p39, orig_procs = restore_p39_proc(
            pipe, vca_layer, dummy_ctx, slot_blend,
            slot_blend_raw, ckpt, adapter_rank, device)

        metrics_p39 = teacher_forced_eval(
            pipe, proc_p39, dataset, eval_idx, device,
            t_fixed=args.t_fixed)
        print(f"  [Phase 39] visible_iou={metrics_p39['visible_iou']:.4f}  "
              f"ordering_acc={metrics_p39['ordering_acc']:.4f}  "
              f"wrong_leak={metrics_p39['wrong_slot_leak']:.4f}  "
              f"val_score={metrics_p39['val_score']:.4f}", flush=True)

        # Phase 38 baseline (slot adapter 없이, scalar blend)
        restore_procs(pipe, orig_procs)
        p38_blend_init = float(ckpt.get("slot_blend", 0.3))
        proc_p38, orig_procs = inject_entity_slot(
            pipe, vca_layer, dummy_ctx,
            inject_key=INJECT_KEY,
            slot_blend_init=p38_blend_init,
            adapter_rank=0,        # adapter 없이
            use_blend_head=False,  # scalar blend
        )
        proc_p38 = proc_p38.to(device)
        if slot_blend_raw is not None:
            proc_p38.slot_blend_raw.data.copy_(slot_blend_raw.to(device))
        proc_p38.eval()
        for p in proc_p38.parameters():
            p.requires_grad_(False)

        metrics_p38 = teacher_forced_eval(
            pipe, proc_p38, dataset, eval_idx, device,
            t_fixed=args.t_fixed)
        print(f"  [Phase 38] visible_iou={metrics_p38['visible_iou']:.4f}  "
              f"ordering_acc={metrics_p38['ordering_acc']:.4f}  "
              f"wrong_leak={metrics_p38['wrong_slot_leak']:.4f}  "
              f"val_score={metrics_p38['val_score']:.4f}", flush=True)

        # Summary table
        print(f"\n{'='*70}", flush=True)
        print(f"{'Metric':<22} {'Phase38':>10} {'Phase39':>10} {'Δ':>10}", flush=True)
        print("-" * 60, flush=True)
        for key in ["visible_iou", "ordering_acc", "val_score"]:
            v38 = metrics_p38[key]
            v39 = metrics_p39[key]
            sign = "↑" if v39 > v38 else "↓"
            print(f"  {key:<20} {v38:10.4f} {v39:10.4f} {v39-v38:+10.4f} {sign}",
                  flush=True)
        key = "wrong_slot_leak"
        v38 = metrics_p38[key]; v39 = metrics_p39[key]
        sign = "↓" if v39 < v38 else "↑"
        print(f"  {key:<20} {v38:10.4f} {v39:10.4f} {v39-v38:+10.4f} {sign}", flush=True)
        print(f"{'='*70}", flush=True)

        quant_results = {
            "eval_samples": len(eval_idx),
            "phase38": metrics_p38,
            "phase39": metrics_p39,
        }
        with open(debug_dir / "quantitative_table.json", "w") as f:
            json.dump(quant_results, f, indent=2)
        print(f"  → {debug_dir}/quantitative_table.json", flush=True)

        restore_procs(pipe, orig_procs)

    # ── 2. Qualitative GIF (고정 프롬프트) ──────────────────────────────────
    print(f"\n[p39] Qualitative GIF 생성...", flush=True)

    dummy_meta = make_meta("cat", "dog")
    dummy_ctx  = get_color_entity_context(pipe, dummy_meta, device)
    proc_qual, orig_procs = restore_p39_proc(
        pipe, vca_layer, dummy_ctx, slot_blend,
        slot_blend_raw, ckpt, adapter_rank, device)

    qual_results = []

    for kw0, kw1 in QUALITATIVE_PAIRS:
        meta       = make_meta(kw0, kw1)
        entity_ctx = get_color_entity_context(pipe, meta, device)
        toks_e0, toks_e1, full_prompt = get_entity_token_positions(pipe, meta)
        _, _, _, c0, c1 = make_color_prompts(meta)
        label = f"{c0}_{kw0}__{c1}_{kw1}"

        print(f"\n  {full_prompt}", flush=True)

        best_seed = SEEDS[0]
        best_p39_frames = None
        best_base_frames = None

        # re-inject proc for each pair
        restore_procs(pipe, orig_procs)
        proc_qual, orig_procs = restore_p39_proc(
            pipe, vca_layer, entity_ctx, slot_blend,
            slot_blend_raw, ckpt, adapter_rank, device)

        for seed in SEEDS:
            print(f"    seed={seed}...", end=" ", flush=True)

            # Phase 39
            frames_p39 = generate_p39(
                pipe, proc_qual, entity_ctx, toks_e0, toks_e1,
                full_prompt, seed, args.n_frames, args.n_steps,
                args.height, args.width, device)

            # baseline
            restore_procs(pipe, orig_procs)
            frames_base = generate_baseline(
                pipe, orig_procs, vca_layer, entity_ctx,
                full_prompt, seed, args.n_frames, args.n_steps,
                args.height, args.width, device)
            # re-inject for next iter
            restore_procs(pipe, orig_procs)
            proc_qual, orig_procs = restore_p39_proc(
                pipe, vca_layer, entity_ctx, slot_blend,
                slot_blend_raw, ckpt, adapter_rank, device)

            # debug-only entity_score (참고용)
            es_p39,  sr_p39,  cr_p39  = compute_entity_score_debug(frames_p39)
            es_base, sr_base, cr_base = compute_entity_score_debug(frames_base)
            print(f"base es={es_base:.4f}  p39 es={es_p39:.4f} "
                  f"[debug-only, not used for comparison]", flush=True)

            if best_p39_frames is None:
                best_p39_frames  = frames_p39
                best_base_frames = frames_base
                best_seed        = seed

        # GIF 저장
        save_gif(best_p39_frames,  debug_dir / f"p39_{label}_seed{best_seed}.gif")
        save_gif(best_base_frames, debug_dir / f"base_{label}_seed{best_seed}.gif")
        overlay_p39  = make_chimera_overlay(best_p39_frames)
        overlay_base = make_chimera_overlay(best_base_frames)
        save_gif(overlay_p39,  debug_dir / f"p39_{label}_chimera_overlay.gif")
        save_gif(overlay_base, debug_dir / f"base_{label}_chimera_overlay.gif")
        print(f"    GIFs saved: p39/{label}", flush=True)

        qual_results.append({
            "pair":  f"{kw0}+{kw1}",
            "label": label,
            "seed":  best_seed,
        })

    print(f"\n[p39] 완료. 결과 → {debug_dir}/", flush=True)
    print("  NOTE: 정량 비교는 quantitative_table.json (GT-mask 기반) 참조", flush=True)
    print("  NOTE: GIF는 정성 참고용 — entity_score는 debug-only", flush=True)

    return qual_results


def _parse():
    p = argparse.ArgumentParser(
        description="Phase 39 chimera: GT-mask quantitative + qualitative GIF")
    p.add_argument("--ckpt",              type=str,   default="checkpoints/phase39/best.pt")
    p.add_argument("--data-root",         type=str,   default="toy/data_objaverse",
                   help="dataset root; 없으면 quantitative eval 생략")
    p.add_argument("--debug-dir",         type=str,   default="debug/chimera_phase39")
    p.add_argument("--n-frames",          type=int,   default=16)
    p.add_argument("--n-steps",           type=int,   default=20)
    p.add_argument("--height",            type=int,   default=256)
    p.add_argument("--width",             type=int,   default=256)
    p.add_argument("--adapter-rank",      type=int,   default=64)
    p.add_argument("--t-fixed",           type=int,   default=150)
    p.add_argument("--eval-frac",         type=float, default=0.3,
                   help="상위 overlap 비율 (eval set)")
    p.add_argument("--min-eval-samples",  type=int,   default=4)
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
