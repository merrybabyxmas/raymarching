"""
Chimera Comparison — mega-GIF + bar chart + markdown table
==========================================================

Combines the outputs of

    phase32 (Entity-Aware Occlusion)
    phase33 (Score / Gradient Guidance)
    phase34 (Latent Compositing)

plus a "Baseline" column (full prompt, no mitigation) into a single 2-row ×
4-column comparison GIF:

    Row 0 (frames): [Baseline | Occlusion | Guidance | Compositing]
    Row 1 (masks):  [Baseline chim | Occlusion chim | Guidance chim | Compositing chim]

It also writes a chimera_scores_bar.png and prints a markdown summary table.

Expected files in --debug-dir
-----------------------------
    p32_*_baseline_frames.gif      → Baseline (shared across methods)
    p32_*_occlusion_frames.gif     → Occlusion frames
    p33_*_guided_frames.gif        → Guidance frames
    p34_*_composited_frames.gif    → Compositing frames

    p32_*_chimera_mask.gif         → Occlusion chimera overlay
    p33_*_chimera_mask.gif         → Guidance chimera overlay
    p34_*_chimera_mask.gif         → Compositing chimera overlay

(Baseline chimera overlay is recomputed on the fly from p32_*_baseline_frames.)

For each slot the script uses the FIRST matching GIF it finds. Missing inputs
are replaced with a blank dark panel containing "N/A".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import imageio.v2 as iio2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))


METHODS = ["Baseline", "Occlusion", "Guidance", "Compositing"]
BAR_COLORS = ["#888888", "#3f78c3", "#4caf50", "#ff9800"]


# =============================================================================
# Metrics
# =============================================================================
def chimera_score(frames: list[np.ndarray]) -> float:
    scores = []
    for f in frames:
        if f.ndim != 3 or f.shape[-1] < 3:
            scores.append(0.0)
            continue
        r = f[..., 0].astype(float)
        b = f[..., 2].astype(float)
        chimera = (r > 80) & (b > 80)
        overlap = (r > 80) | (b > 80)
        if overlap.sum() == 0:
            scores.append(0.0)
        else:
            scores.append(float(chimera.sum()) / float(overlap.sum()))
    return float(np.mean(scores)) if scores else 0.0


def chimera_overlay(frame: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[-1] < 3:
        return frame
    r = frame[..., 0].astype(np.int32)
    b = frame[..., 2].astype(np.int32)
    m = (r > 80) & (b > 80)
    out = frame.copy().astype(float)
    yellow = np.array([255, 220, 0], dtype=float)
    out[m] = out[m] * (1 - alpha) + yellow * alpha
    return out.clip(0, 255).astype(np.uint8)


# =============================================================================
# GIF I/O helpers
# =============================================================================
def _find_first(debug_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(debug_dir.glob(pattern))
    return matches[0] if matches else None


def _load_gif(path: Optional[Path]) -> Optional[list[np.ndarray]]:
    if path is None or not path.exists():
        return None
    try:
        frames = iio2.mimread(str(path))
    except Exception as e:
        print(f"[compare] WARN: failed to read {path}: {e}", flush=True)
        return None
    out = []
    for f in frames:
        arr = np.asarray(f)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        out.append(arr.astype(np.uint8))
    return out


def _resize_frame(frame: np.ndarray, h: int, w: int) -> np.ndarray:
    img = Image.fromarray(frame)
    img = img.resize((w, h), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _blank_panel(h: int, w: int, text: str) -> np.ndarray:
    img = Image.new("RGB", (w, h), color=(24, 24, 24))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = len(text) * 7, 12
    draw.text(((w - tw) // 2, (h - th) // 2), text,
              fill=(220, 220, 220), font=font,
              stroke_width=1, stroke_fill=(0, 0, 0))
    return np.asarray(img, dtype=np.uint8)


def _label_panel(frame: np.ndarray, text: str,
                 dark_bg: bool = True) -> np.ndarray:
    """
    Put `text` at the top of the panel with white text + black stroke,
    on a dark background strip.
    """
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    # dark strip behind the label
    if dark_bg:
        w = img.size[0]
        draw.rectangle([(0, 0), (w, 18)], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255), font=font,
              stroke_width=1, stroke_fill=(0, 0, 0))
    return np.asarray(img, dtype=np.uint8)


# =============================================================================
# Core comparison routine
# =============================================================================
def build_comparison(args):
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    H, W = args.height, args.width

    # ------------------------------------------------------------------- find
    frame_paths = {
        "Baseline":    _find_first(debug_dir, "p32_*_baseline_frames.gif"),
        "Occlusion":   _find_first(debug_dir, "p32_*_occlusion_frames.gif"),
        "Guidance":    _find_first(debug_dir, "p33_*_guided_frames.gif"),
        "Compositing": _find_first(debug_dir, "p34_*_composited_frames.gif"),
    }
    mask_paths = {
        "Baseline":    None,  # recomputed on the fly from baseline frames
        "Occlusion":   _find_first(debug_dir, "p32_*_chimera_mask.gif"),
        "Guidance":    _find_first(debug_dir, "p33_*_chimera_mask.gif"),
        "Compositing": _find_first(debug_dir, "p34_*_chimera_mask.gif"),
    }
    for k, p in frame_paths.items():
        print(f"[compare] {k:12s} frames : {p}", flush=True)
    for k, p in mask_paths.items():
        print(f"[compare] {k:12s} masks  : {p}", flush=True)

    # ------------------------------------------------------------------ load
    frame_videos = {k: _load_gif(p) for k, p in frame_paths.items()}
    mask_videos  = {k: _load_gif(p) for k, p in mask_paths.items()}

    # Baseline masks: compute from baseline frames (yellow-overlay)
    base_fr = frame_videos.get("Baseline")
    if base_fr is not None:
        mask_videos["Baseline"] = [chimera_overlay(f) for f in base_fr]

    # ------------------------------------------------------------- scores
    scores: dict[str, float] = {}
    for m in METHODS:
        fr = frame_videos.get(m)
        scores[m] = chimera_score(fr) if fr else float("nan")

    # -------------------------------------------------------- decide T (frames)
    lengths = [len(v) for v in list(frame_videos.values()) +
               list(mask_videos.values()) if v is not None]
    T = min(lengths) if lengths else 8
    if T <= 0:
        T = 8

    # --------------------------------------------------- build stacked frames
    mega_frames = []
    for fi in range(T):
        row0_cols = []
        row1_cols = []
        for mi, m in enumerate(METHODS):
            sc = scores[m]
            sc_txt = f"{sc:.2f}" if not np.isnan(sc) else "N/A"

            # --- Row 0 : frame panel ---
            fr_vid = frame_videos.get(m)
            if fr_vid and fi < len(fr_vid):
                panel = _resize_frame(fr_vid[fi], H, W)
                panel = _label_panel(panel, f"{m}  chim={sc_txt}")
            else:
                panel = _blank_panel(H, W, f"{m}\nN/A")
            row0_cols.append(panel)

            # --- Row 1 : chimera mask panel ---
            mk_vid = mask_videos.get(m)
            if mk_vid and fi < len(mk_vid):
                mpanel = _resize_frame(mk_vid[fi], H, W)
                mpanel = _label_panel(mpanel, f"{m} chim-mask")
            else:
                mpanel = _blank_panel(H, W, f"{m} mask\nN/A")
            row1_cols.append(mpanel)

        row0 = np.concatenate(row0_cols, axis=1)   # (H, 4W, 3)
        row1 = np.concatenate(row1_cols, axis=1)
        mega = np.concatenate([row0, row1], axis=0)  # (2H, 4W, 3)
        mega_frames.append(mega)

    out_gif = debug_dir / "chimera_comparison.gif"
    iio2.mimsave(str(out_gif), mega_frames, duration=250)
    print(f"[compare] wrote {out_gif}  ({len(mega_frames)} frames, "
          f"{mega_frames[0].shape[1]}×{mega_frames[0].shape[0]})", flush=True)

    # ---------------------------------------------------------- bar chart
    bar_path = debug_dir / "chimera_scores_bar.png"
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=140)
    display_scores = [0.0 if np.isnan(scores[m]) else scores[m] for m in METHODS]
    bars = ax.bar(METHODS, display_scores, color=BAR_COLORS, edgecolor="black")
    ax.set_ylabel("Chimera Score  (↓ better)")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0,
               label="perfect (0)")
    ax.set_title("Chimera Reduction — Method Comparison")
    for bar, m in zip(bars, METHODS):
        sc = scores[m]
        label = f"{sc:.3f}" if not np.isnan(sc) else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                label, ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(str(bar_path))
    plt.close(fig)
    print(f"[compare] wrote {bar_path}", flush=True)

    # ---------------------------------------------------------- markdown table
    print("", flush=True)
    print("| Method | Chimera Score ↓ |", flush=True)
    print("|--------|----------------|", flush=True)
    display_map = {
        "Baseline":    "Baseline",
        "Occlusion":   "Occlusion (P32)",
        "Guidance":    "Guidance (P33)",
        "Compositing": "Compositing (P34)",
    }
    for m in METHODS:
        sc = scores[m]
        val = f"{sc:.2f}" if not np.isnan(sc) else "N/A"
        print(f"| {display_map[m]} | {val} |", flush=True)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--debug-dir", type=str, default="debug/chimera")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    build_comparison(_parse_args())
