"""
Phase 35 — Chimera Reduction via Volumetric Text Cross-Attention
================================================================

Phase 32/33/34와 동일한 프롬프트/시드로 Phase 35 VCA 체크포인트를 사용해
생성 결과를 비교한다.

Phase 35는 compositing이나 guidance가 아닌 아키텍처 수정:
  - text cross-attention을 volumetric (S*Z) 공간으로 확장
  - z_pe (depth-bin PE) 학습 → 앞/뒤 entity가 각각 z=0, z=1 bin에 attend
  - 결과: chimera가 구조적으로 발생하기 어려운 attention 분리

Outputs (per prompt, for each seed)
-------------------------------------
  p35_{label}_vol_frames.gif       — Phase 35 volumetric VCA 생성
  p35_{label}_baseline_frames.gif  — 동일 seed, Phase 31 standard VCA 생성 (비교용)
  p35_{label}_chimera_mask.gif     — chimera yellow overlay (Phase 35)
  p35_{label}_attn_z0.gif          — z=0 (front) attention map
  p35_{label}_attn_z1.gif          — z=1 (back)  attention map
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import imageio.v2 as iio2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vca_attention import VCALayer
from models.vca_volumetric import VolumetricTextCrossAttentionProcessor
from scripts.run_animatediff import load_pipeline
from scripts.train_phase31 import (
    VCA_ALPHA,
    INJECT_KEY,
    INJECT_QUERY_DIM,
    DEPTH_PE_INIT_SCALE,
    AdditiveVCAInferProcessor,
    inject_vca_p21_infer,
    restore_procs,
)
from scripts.train_phase35 import (
    DEFAULT_Z_BINS,
    DEFAULT_GAMMA_INIT,
)


# =============================================================================
# Prompts / seeds — same as Phase 32/33/34
# =============================================================================
COLLISION_PROMPTS = [
    {
        "prompt": "a red ball and a blue ball rolling toward each other on a "
                  "wooden table, they collide in the center, cinematic lighting, "
                  "photorealistic, high quality",
        "negative": "low quality, blurry, deformed, watermark, text, chimera, "
                    "merged, fused, cartoon, painting",
        "entity0": "red ball",
        "entity1": "blue ball",
        "color0_rgb": (200, 50, 50),
        "color1_rgb": (50, 50, 200),
    },
    {
        "prompt": "a red cat and a blue cat running toward each other on a "
                  "grassy field, they meet in the middle, cinematic, "
                  "photorealistic, high detail",
        "negative": "low quality, blurry, deformed, watermark, text, chimera, "
                    "merged, fused, painting",
        "entity0": "red cat",
        "entity1": "blue cat",
        "color0_rgb": (200, 50, 50),
        "color1_rgb": (50, 50, 200),
    },
]

SEEDS = [42, 123, 456, 789, 1337]


# =============================================================================
# Metrics / GIF helpers
# =============================================================================
def chimera_score(frames: list[np.ndarray]) -> float:
    scores = []
    for f in frames:
        r = f[..., 0].astype(float)
        b = f[..., 2].astype(float)
        chim = (r > 80) & (b > 80)
        ovlp = (r > 80) | (b > 80)
        scores.append(float(chim.sum()) / float(ovlp.sum()) if ovlp.sum() > 0 else 0.0)
    return float(np.mean(scores))


def chimera_masks(frames: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for f in frames:
        r = f[..., 0].astype(np.int32)
        b = f[..., 2].astype(np.int32)
        mask = ((r > 80) & (b > 80)).astype(np.uint8) * 255
        out.append(np.stack([mask, mask, mask], axis=-1))
    return out


def make_chimera_overlay(frame: np.ndarray, mask: np.ndarray,
                          alpha: float = 0.6) -> np.ndarray:
    r = frame[..., 0].astype(np.int32)
    b = frame[..., 2].astype(np.int32)
    m = (r > 80) & (b > 80)
    out = frame.copy().astype(float)
    yellow = np.array([255, 220, 0], dtype=float)
    out[m] = out[m] * (1 - alpha) + yellow * alpha
    return out.clip(0, 255).astype(np.uint8)


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([(0, 0), (img.width, 16)], fill=(0, 0, 0))
    draw.text((3, 2), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def attn_map_to_rgb(attn: np.ndarray) -> np.ndarray:
    """(H, W) float [0,1] → (H, W, 3) uint8 heatmap."""
    norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-6)
    heat = np.zeros((*norm.shape, 3), dtype=np.uint8)
    heat[..., 0] = (norm * 255).astype(np.uint8)          # R channel (hot)
    heat[..., 2] = ((1 - norm) * 180).astype(np.uint8)    # B channel (cold)
    return heat


# =============================================================================
# Checkpoint loading
# =============================================================================
def load_p35_checkpoint(ckpt_path: str, device: str):
    """Load Phase 35 checkpoint: VCA layer + z_pe + gamma."""
    ckpt = torch.load(ckpt_path, map_location=device)

    vca_layer = VCALayer(
        query_dim=INJECT_QUERY_DIM, context_dim=768,
        n_heads=8, n_entities=2, z_bins=DEFAULT_Z_BINS, lora_rank=8,
        use_softmax=False, depth_pe_init_scale=DEPTH_PE_INIT_SCALE,
    ).to(device)
    vca_layer.load_state_dict(ckpt["vca_state_dict"])
    vca_layer.eval()

    gamma_trained = float(ckpt.get("gamma_trained", DEFAULT_GAMMA_INIT))
    z_pe_state    = ckpt.get("z_pe", None)            # (Z, D) tensor — saved as "z_pe"
    gamma_state   = None                               # gamma folded into gamma_trained

    print(f"[p35] loaded: {ckpt_path}  gamma={gamma_trained:.4f}", flush=True)
    if z_pe_state is not None:
        print(f"      z_pe norm={float(z_pe_state.norm()):.4f}", flush=True)
    return vca_layer, gamma_trained, z_pe_state, gamma_state


def get_entity_ctx(pipe, entity0_text: str, entity1_text: str,
                   device: str) -> torch.Tensor:
    embs = []
    for text in [entity0_text, entity1_text]:
        tokens = pipe.tokenizer(
            text, return_tensors="pt", padding="max_length",
            max_length=pipe.tokenizer.model_max_length, truncation=True,
        ).to(device)
        with torch.no_grad():
            out = pipe.text_encoder(**tokens).last_hidden_state
        ids = tokens.input_ids[0]
        mask = ((ids != pipe.tokenizer.pad_token_id) &
                (ids != pipe.tokenizer.eos_token_id))
        mask[0] = False
        embs.append(out[0][mask].mean(0))
    return torch.stack(embs, 0).unsqueeze(0).float().to(device)  # (1,2,768)


# =============================================================================
# Phase 35 processor injection
# =============================================================================
def inject_p35_infer(pipe, vca_layer: VCALayer, entity_ctx: torch.Tensor,
                     gamma_trained: float,
                     z_pe_state: Optional[torch.Tensor] = None,
                     gamma_state: Optional[torch.Tensor] = None,
                     z_bins: int = DEFAULT_Z_BINS):
    """Inject VolumetricTextCrossAttentionProcessor for inference."""
    unet = pipe.unet
    orig_procs = copy.copy(dict(unet.attn_processors))

    proc = VolumetricTextCrossAttentionProcessor(
        query_dim=INJECT_QUERY_DIM,
        z_bins=z_bins,
        vca_layer=vca_layer,
        entity_ctx=entity_ctx.float(),
        gamma_init=gamma_trained,
    ).to(pipe.device)

    # Restore trained z_pe / gamma from checkpoint
    if z_pe_state is not None:
        proc.z_pe.data.copy_(z_pe_state.to(pipe.device, dtype=proc.z_pe.dtype))
    if gamma_state is not None:
        proc.gamma.data.copy_(gamma_state.to(pipe.device, dtype=proc.gamma.dtype))

    proc.eval()

    new_procs = dict(orig_procs)
    new_procs[INJECT_KEY] = proc
    unet.set_attn_processor(new_procs)
    return proc, orig_procs


# =============================================================================
# Generation
# =============================================================================
def _generate_p35(pipe, proc, prompt, negative_prompt,
                  n_frames, n_steps, height, width, seed,
                  cfg_scale=7.5):
    """Generate with p35 processor already injected. Returns frames + attn weights."""
    proc.reset_sigma_acc()
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    with torch.no_grad():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=n_frames,
            num_inference_steps=n_steps,
            guidance_scale=cfg_scale,
            height=height,
            width=width,
            generator=g,
            output_type="pil",
        )
    frames = [np.array(f) for f in out.frames[0]]
    # last_attn_weights: (BF, S*Z, T) — mean over heads, last denoising step
    attn_w = proc.last_attn_weights
    return frames, attn_w


def _generate_baseline(pipe, vca_layer, gamma_trained, entity_ctx,
                        prompt, negative_prompt,
                        n_frames, n_steps, height, width, seed,
                        cfg_scale=7.5):
    """Generate with standard Phase-31 VCA (additive, no volumetric)."""
    orig_procs = inject_vca_p21_infer(pipe, vca_layer, entity_ctx,
                                       gamma_trained=gamma_trained)
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    with torch.no_grad():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=n_frames,
            num_inference_steps=n_steps,
            guidance_scale=cfg_scale,
            height=height,
            width=width,
            generator=g,
            output_type="pil",
        )
    frames = [np.array(f) for f in out.frames[0]]
    restore_procs(pipe, orig_procs)
    return frames


# =============================================================================
# Attention map visualization helpers
# =============================================================================
def extract_attn_maps(attn_w: Optional[torch.Tensor],
                      entity_tok_pos: list[int],
                      S: int, Z: int, H: int, W: int,
                      n_frames: int) -> list[np.ndarray]:
    """
    attn_w: (BF, S*Z, T_seq)  — BF = n_frames (or 2*n_frames with CFG)
    Returns list of (H, W) float32 maps, one per frame, for the given z-bin.
    """
    if attn_w is None or not entity_tok_pos:
        return [np.zeros((H, W), dtype=np.float32)] * n_frames

    attn_np = attn_w.float().cpu().numpy()  # (BF, S*Z, T)
    BF = attn_np.shape[0]
    # Use last n_frames (conditional half if CFG doubled)
    start = BF - n_frames if BF >= n_frames else 0
    attn_np = attn_np[start:]

    # Average attention over entity token positions
    tok_idx = [t for t in entity_tok_pos if t < attn_np.shape[2]]
    if not tok_idx:
        return [np.zeros((H, W), dtype=np.float32)] * n_frames

    maps = []
    for fi in range(len(attn_np)):
        frame_attn = attn_np[fi]          # (S*Z, T)
        a = frame_attn[:, tok_idx].mean(-1)  # (S*Z,)
        maps.append(a)

    return maps  # list of (S*Z,) arrays


def attn_to_rgb_frame(attn_sz: np.ndarray, Z: int, z_bin: int,
                       S: int, hw: int, H: int, W: int) -> np.ndarray:
    """
    Extract z_bin slice from (S*Z,) attention, reshape to (hw, hw), upsample to (H,W).
    """
    # Slice z_bin: positions z_bin, z_bin+Z, z_bin+2Z, ...
    a = attn_sz[z_bin::Z][:S]             # (S,)
    side = hw
    a_2d = a.reshape(side, side)
    a_norm = (a_2d - a_2d.min()) / (a_2d.max() - a_2d.min() + 1e-6)

    # Upsample
    img = Image.fromarray((a_norm * 255).astype(np.uint8), mode="L")
    img = img.resize((W, H), resample=Image.BILINEAR)
    a_up = np.array(img) / 255.0

    # Heatmap: red=high, blue=low
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = (a_up * 255).astype(np.uint8)
    rgb[..., 2] = ((1 - a_up) * 180).astype(np.uint8)
    return rgb


# =============================================================================
# Token position finder
# =============================================================================
def find_tok_pos(pipe, full_prompt: str, entity_text: str) -> list[int]:
    full_ids = pipe.tokenizer(
        full_prompt, padding=False, truncation=True,
        max_length=pipe.tokenizer.model_max_length,
    ).input_ids
    entity_ids = pipe.tokenizer(entity_text, add_special_tokens=False).input_ids
    positions = []
    for i in range(len(full_ids) - len(entity_ids) + 1):
        if full_ids[i:i + len(entity_ids)] == entity_ids:
            positions.extend(range(i, i + len(entity_ids)))
    return positions


# =============================================================================
# Per-prompt inference loop
# =============================================================================
def run_p35_for_prompt(pipe, vca_layer, gamma_trained, z_pe_state, gamma_state,
                        p_cfg: dict, args) -> dict:
    device = str(pipe.device)
    prompt   = p_cfg["prompt"]
    negative = p_cfg["negative"]
    e0_text  = p_cfg["entity0"]
    e1_text  = p_cfg["entity1"]

    entity_ctx = get_entity_ctx(pipe, e0_text, e1_text, device)

    # Token positions for attention visualization
    tok_e0 = find_tok_pos(pipe, prompt, e0_text)
    tok_e1 = find_tok_pos(pipe, prompt, e1_text)
    print(f"  token positions: e0={tok_e0}, e1={tok_e1}", flush=True)

    # Inject Phase 35 processor once (reuse across seeds)
    proc, orig_procs = inject_p35_infer(
        pipe, vca_layer, entity_ctx, gamma_trained,
        z_pe_state=z_pe_state, gamma_state=gamma_state,
    )
    print(f"  gamma={float(proc.gamma.item()):.4f}  "
          f"|z_pe|={float(proc.z_pe.norm().item()):.4f}", flush=True)

    best_vol = None
    all_runs = []

    for seed in SEEDS:
        print(f"  seed={seed}...", end=" ", flush=True)

        # Phase 35 generation
        frames_vol, attn_w = _generate_p35(
            pipe, proc, prompt, negative,
            args.n_frames, args.n_steps, args.height, args.width, seed,
        )
        score_vol = chimera_score(frames_vol)

        # Phase 31 baseline (same seed)
        restore_procs(pipe, orig_procs)
        frames_base = _generate_baseline(
            pipe, vca_layer, gamma_trained, entity_ctx,
            prompt, negative,
            args.n_frames, args.n_steps, args.height, args.width, seed,
        )
        score_base = chimera_score(frames_base)

        # Re-inject for next seed
        proc, orig_procs = inject_p35_infer(
            pipe, vca_layer, entity_ctx, gamma_trained,
            z_pe_state=z_pe_state, gamma_state=gamma_state,
        )

        print(f"base={score_base:.4f}  vol={score_vol:.4f}", flush=True)
        run = dict(seed=seed, frames_vol=frames_vol, frames_base=frames_base,
                   score_vol=score_vol, score_base=score_base, attn_w=attn_w)
        all_runs.append(run)
        if best_vol is None or score_vol < best_vol["score_vol"]:
            best_vol = run

    restore_procs(pipe, orig_procs)

    # Attention maps for best seed
    S    = args.height // 16 * args.width // 16   # 16×16 spatial
    Z    = DEFAULT_Z_BINS
    hw   = 16
    attn_maps = extract_attn_maps(best_vol["attn_w"], tok_e0 + tok_e1,
                                   S, Z, args.height, args.width, args.n_frames)

    return dict(
        best=best_vol,
        all_runs=all_runs,
        attn_maps=attn_maps,
        tok_e0=tok_e0,
        tok_e1=tok_e1,
        S=S, Z=Z, hw=hw,
    )


# =============================================================================
# Main
# =============================================================================
def run_phase35(args):
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("[p35] 파이프라인 로드...", flush=True)
    pipe = load_pipeline(device=device, dtype=torch.float16)

    print(f"[p35] 체크포인트 로드: {args.ckpt}", flush=True)
    vca_layer, gamma_trained, z_pe_state, gamma_state = load_p35_checkpoint(
        args.ckpt, device)

    summary = []
    for p_cfg in COLLISION_PROMPTS:
        label = f"{p_cfg['entity0']}__{p_cfg['entity1']}".replace(" ", "_")
        print(f"\n{'='*70}", flush=True)
        print(f"[p35] {p_cfg['prompt']}", flush=True)

        result = run_p35_for_prompt(
            pipe, vca_layer, gamma_trained, z_pe_state, gamma_state, p_cfg, args)

        best  = result["best"]
        S, Z, hw = result["S"], result["Z"], result["hw"]
        attn_maps = result["attn_maps"]

        vol_gif  = debug_dir / f"p35_{label}_vol_frames.gif"
        base_gif = debug_dir / f"p35_{label}_baseline_frames.gif"
        chim_gif = debug_dir / f"p35_{label}_chimera_mask.gif"
        z0_gif   = debug_dir / f"p35_{label}_attn_z0.gif"
        z1_gif   = debug_dir / f"p35_{label}_attn_z1.gif"

        # Labeled frames
        labeled_vol = [
            add_label(f, f"P35-VOL f{i} chim={best['score_vol']:.3f}")
            for i, f in enumerate(best["frames_vol"])
        ]
        labeled_base = [
            add_label(f, f"P31-BASE f{i} chim={best['score_base']:.3f}")
            for i, f in enumerate(best["frames_base"])
        ]

        # Chimera overlay on p35 frames
        masks = chimera_masks(best["frames_vol"])
        chim_overlay = [make_chimera_overlay(f, m)
                        for f, m in zip(best["frames_vol"], masks)]
        labeled_chim = [
            add_label(f, f"CHIM-OVERLAY f{i}")
            for i, f in enumerate(chim_overlay)
        ]

        # Attention map GIFs (z=0 and z=1)
        z0_frames = []
        z1_frames = []
        for a_sz in attn_maps:
            if isinstance(a_sz, np.ndarray) and a_sz.ndim == 1:
                z0_rgb = attn_to_rgb_frame(a_sz, Z, 0, S, hw,
                                            args.height, args.width)
                z1_rgb = attn_to_rgb_frame(a_sz, Z, 1, S, hw,
                                            args.height, args.width)
            else:
                z0_rgb = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                z1_rgb = z0_rgb.copy()
            z0_frames.append(add_label(z0_rgb, "ATTN z=0 (front)"))
            z1_frames.append(add_label(z1_rgb, "ATTN z=1 (back)"))

        iio2.mimsave(str(vol_gif),  labeled_vol,  duration=250)
        iio2.mimsave(str(base_gif), labeled_base, duration=250)
        iio2.mimsave(str(chim_gif), labeled_chim, duration=250)
        iio2.mimsave(str(z0_gif),   z0_frames,    duration=250)
        iio2.mimsave(str(z1_gif),   z1_frames,    duration=250)

        print(f"  best seed={best['seed']}  "
              f"base={best['score_base']:.4f}  vol={best['score_vol']:.4f}",
              flush=True)
        print(f"  wrote: {vol_gif.name}, {base_gif.name}, {chim_gif.name}, "
              f"{z0_gif.name}, {z1_gif.name}", flush=True)

        summary.append(dict(
            label=label,
            best_seed=best["seed"],
            score_base=best["score_base"],
            score_vol=best["score_vol"],
            all_runs=[(r["seed"], r["score_base"], r["score_vol"])
                       for r in result["all_runs"]],
        ))

    # Final summary
    print("\n===== Phase 35 Summary =====", flush=True)
    print("| Method | Chimera Score ↓ |", flush=True)
    print("|--------|----------------|", flush=True)
    for s in summary:
        print(f"  [{s['label']}]  best_seed={s['best_seed']}", flush=True)
        print(f"    P31 baseline: {s['score_base']:.4f}", flush=True)
        print(f"    P35 vol VCA:  {s['score_vol']:.4f}", flush=True)
        for seed, sb, sv in s["all_runs"]:
            print(f"      seed={seed}  base={sb:.4f}  vol={sv:.4f}", flush=True)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       type=str, default="checkpoints/phase35/best.pt")
    p.add_argument("--debug-dir",  type=str, default="debug/chimera")
    p.add_argument("--n-frames",   type=int, default=16)
    p.add_argument("--n-steps",    type=int, default=20)
    p.add_argument("--height",     type=int, default=256)
    p.add_argument("--width",      type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    run_phase35(_parse_args())
