"""
Phase 62 — Training Debug Visualizer
=====================================

Produces rich PNG/GIF debug outputs **during training** (not just eval).
Saves to:
  debug_dir/training/stage1/   ← volume slices, projections, GT comparison
  debug_dir/training/stage2/   ← + guide gates, guide feature heatmaps
  debug_dir/training/stage3/   ← + diffusion noise comparison
  debug_dir/training/loss_curve.png   ← updated every epoch

Usage (called from AblationTrainer):
  viz = TrainingDebugViz(debug_dir, config)
  viz.save_volume_debug(vol_outputs, V_gt, gt_visible, gt_amodal, epoch, step, stage)
  viz.save_guide_debug(guides, assembler, epoch, stage)
  viz.save_diffusion_debug(noise_pred, noise_gt, epoch, step, stage)
  viz.update_loss_curve(history)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


# ─── colour palette ──────────────────────────────────────────────────────────
_E0_COLOR = np.array([1.0, 0.2, 0.2])   # red  → entity 0
_E1_COLOR = np.array([0.2, 0.4, 1.0])   # blue → entity 1
_BG_COLOR = np.array([0.12, 0.12, 0.12]) # dark background


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _norm01(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _heatmap_rgb(prob: np.ndarray, color: np.ndarray) -> np.ndarray:
    """prob (H,W) → RGB image with given entity colour."""
    rgb = np.ones((*prob.shape, 3)) * _BG_COLOR
    alpha = prob[..., None]
    return np.clip(rgb * (1 - alpha) + color * alpha, 0, 1)


class TrainingDebugViz:
    """Debug visualizer that writes training-phase PNG/GIF debug outputs."""

    def __init__(self, debug_dir: Path, config):
        self.debug_dir = Path(debug_dir)
        self.config = config
        self.depth_bins = getattr(config, "depth_bins", 8)
        self.spatial_h = getattr(config, "spatial_h", 16)
        self.spatial_w = getattr(config, "spatial_w", 16)

        for sub in ("stage1", "stage2", "stage3"):
            (self.debug_dir / "training" / sub).mkdir(parents=True, exist_ok=True)

    def _draw_contract_badge(self, fig, contract_metrics) -> None:
        _draw_contract_badge_fn(fig, contract_metrics)

    # ─── Volume debug ────────────────────────────────────────────────────────

    def save_volume_debug(
        self,
        vol_outputs,
        V_gt: torch.Tensor,         # (B, K, H, W)  long
        gt_visible: torch.Tensor,   # (B, 2, H, W)
        gt_amodal: torch.Tensor,    # (B, 2, H, W)
        epoch: int,
        step: int,
        stage: str,
        contract_metrics=None,
        frames_np=None,             # (T, H, W, 3) uint8 or float — actual video frames
    ) -> Optional[Path]:
        if not HAS_MPL:
            return None
        try:
            return self._volume_debug_impl(
                vol_outputs, V_gt, gt_visible, gt_amodal, epoch, step, stage,
                contract_metrics, frames_np)
        except Exception as e:
            print(f"  [debug_viz] volume debug failed: {e}", flush=True)
            return None

    def _volume_debug_impl(self, vol_outputs, V_gt, gt_visible, gt_amodal,
                            epoch, step, stage, contract_metrics=None,
                            frames_np=None) -> Path:
        b = 0  # always visualise first sample

        # ── 3D entity probs: (B, 2, K, H, W)
        ep = _to_np(vol_outputs.entity_probs[b].float())   # (2, K, H, W)
        K = ep.shape[1]

        # ── V_gt: (B, K, H, W) → binary per entity
        vgt = V_gt[b].cpu().numpy()  # (K, H, W)
        gt_e0 = (vgt == 1).astype(np.float32)
        gt_e1 = (vgt == 2).astype(np.float32)

        # ── 2D projections
        has_vis = vol_outputs.visible and "e0" in vol_outputs.visible
        vis_e0 = _to_np(vol_outputs.visible["e0"][b]) if has_vis else np.zeros((self.spatial_h, self.spatial_w))
        vis_e1 = _to_np(vol_outputs.visible["e1"][b]) if has_vis else np.zeros((self.spatial_h, self.spatial_w))
        amo_e0 = _to_np(vol_outputs.amodal["e0"][b]) if vol_outputs.amodal else np.zeros_like(vis_e0)
        amo_e1 = _to_np(vol_outputs.amodal["e1"][b]) if vol_outputs.amodal else np.zeros_like(vis_e1)

        gt_vis_e0 = _to_np(gt_visible[b, 0])
        gt_vis_e1 = _to_np(gt_visible[b, 1])
        gt_amo_e0 = _to_np(gt_amodal[b, 0])
        gt_amo_e1 = _to_np(gt_amodal[b, 1])

        # ── visible_class (predicted 2D label map)
        has_vc = vol_outputs.visible_class is not None
        if has_vc:
            vc = vol_outputs.visible_class[b].cpu().numpy()  # (H, W) int
        else:
            vc = np.zeros((self.spatial_h, self.spatial_w), dtype=np.int64)

        # ────────────────────────────────────────────────────────────────────
        # Layout:
        #   Row F: Actual video frames (top strip, shown if frames_np provided)
        #   Row 0: 3D depth slices — GT entity 0   (K columns)
        #   Row 1: 3D depth slices — Pred entity 0 (K columns)
        #   Row 2: 3D depth slices — GT entity 1
        #   Row 3: 3D depth slices — Pred entity 1
        #   Row 4: 2D projection panel (8 cells: vis/amo GT/pred for e0, e1)
        #   Row 5: visible_class pred vs GT overlay
        # ────────────────────────────────────────────────────────────────────
        n_slice_cols = K

        # Prepare frames strip if provided
        frames_strip = None
        n_frame_show = 0
        if frames_np is not None:
            try:
                frms = np.array(frames_np)  # (T, H, W, 3) or (T, H, W)
                if frms.ndim == 4 and frms.shape[-1] == 3:
                    if frms.dtype != np.float32:
                        frms = frms.astype(np.float32)
                    if frms.max() > 1.5:
                        frms = frms / 255.0
                    frms = np.clip(frms, 0, 1)
                    n_frame_show = min(frms.shape[0], n_slice_cols)
                    frames_strip = frms
            except Exception:
                frames_strip = None

        has_frames = frames_strip is not None and n_frame_show > 0
        fig_w = max(18, n_slice_cols * 1.4)
        fig_h = 16 if has_frames else 14

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1a1a2e")

        if has_frames:
            # Frames strip occupies top ~12% of figure
            gs_frames = gridspec.GridSpec(1, n_frame_show, figure=fig,
                                          top=0.97, bottom=0.86,
                                          hspace=0.02, wspace=0.03)
            gs_top = gridspec.GridSpec(4, n_slice_cols, figure=fig,
                                       top=0.84, bottom=0.32, hspace=0.05, wspace=0.03)
            gs_bot = gridspec.GridSpec(2, 8, figure=fig,
                                       top=0.28, bottom=0.02, hspace=0.08, wspace=0.04)
            # Draw frames strip
            T_total = frames_strip.shape[0]
            for fi in range(n_frame_show):
                # sample evenly from all T frames
                t_idx = int(fi * T_total / n_frame_show)
                ax_f = fig.add_subplot(gs_frames[0, fi])
                ax_f.imshow(frames_strip[t_idx], interpolation="bilinear")
                ax_f.axis("off")
                ax_f.set_title(f"t={t_idx}", color="#aaaaaa", fontsize=6, pad=1)
            # Label
            fig.text(0.005, 0.915, "Frames", color="#aaaacc", fontsize=7,
                     va="center", ha="left",
                     bbox=dict(facecolor="#111133", alpha=0.7, pad=2))
        else:
            gs_top = gridspec.GridSpec(4, n_slice_cols, figure=fig,
                                       top=0.96, bottom=0.38, hspace=0.05, wspace=0.03)
            gs_bot = gridspec.GridSpec(2, 8, figure=fig,
                                       top=0.34, bottom=0.02, hspace=0.08, wspace=0.04)

        row_labels = [
            ("GT  E0", gt_e0, _E0_COLOR),
            ("Pred E0", ep[0], _E0_COLOR),
            ("GT  E1", gt_e1, _E1_COLOR),
            ("Pred E1", ep[1], _E1_COLOR),
        ]

        for ri, (label, slices, color) in enumerate(row_labels):
            for ki in range(K):
                ax = fig.add_subplot(gs_top[ri, ki])
                img = _heatmap_rgb(np.clip(slices[ki], 0, 1), color)
                ax.imshow(img, interpolation="nearest")
                ax.axis("off")
                if ki == 0:
                    ax.set_ylabel(label, color="white", fontsize=7, rotation=0,
                                  labelpad=48, va="center")
                if ri == 0:
                    ax.set_title(f"z={ki}", color="#aaaaaa", fontsize=6, pad=1)

        # ── Bottom row: 2D projections
        def _proj_ax(row, col, title, img_arr, cmap="hot", vmin=0, vmax=1):
            ax = fig.add_subplot(gs_bot[row, col])
            ax.imshow(img_arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(title, color="white", fontsize=6, pad=2)
            ax.axis("off")
            return ax

        # Row 0: GT projections
        _proj_ax(0, 0, "GT Vis E0",  gt_vis_e0, cmap="Reds")
        _proj_ax(0, 1, "GT Vis E1",  gt_vis_e1, cmap="Blues")
        _proj_ax(0, 2, "GT Amo E0",  gt_amo_e0, cmap="Reds")
        _proj_ax(0, 3, "GT Amo E1",  gt_amo_e1, cmap="Blues")

        # Row 0 right half: 3D occupancy top-down view (max-proj along Z)
        maxproj_e0_gt  = gt_e0.max(axis=0)
        maxproj_e1_gt  = gt_e1.max(axis=0)
        maxproj_e0_pred = ep[0].max(axis=0)
        maxproj_e1_pred = ep[1].max(axis=0)
        _proj_ax(0, 4, "MaxZ GT E0",  maxproj_e0_gt,  cmap="Reds")
        _proj_ax(0, 5, "MaxZ GT E1",  maxproj_e1_gt,  cmap="Blues")
        _proj_ax(0, 6, "MaxZ Pd E0",  maxproj_e0_pred, cmap="Reds")
        _proj_ax(0, 7, "MaxZ Pd E1",  maxproj_e1_pred, cmap="Blues")

        # Row 1: Predicted projections + overlay
        _proj_ax(1, 0, "Pred Vis E0", vis_e0, cmap="Reds")
        _proj_ax(1, 1, "Pred Vis E1", vis_e1, cmap="Blues")
        _proj_ax(1, 2, "Pred Amo E0", amo_e0, cmap="Reds")
        _proj_ax(1, 3, "Pred Amo E1", amo_e1, cmap="Blues")

        # Predicted visible class overlay (RGB: red=e0, blue=e1, dark=bg)
        vc_rgb = np.ones((*vc.shape, 3)) * 0.1
        vc_rgb[vc == 1] = _E0_COLOR
        vc_rgb[vc == 2] = _E1_COLOR

        # GT visible class overlay
        gt_vc = np.zeros_like(vc)
        gt_vc[(gt_vis_e0 > 0.5)] = 1
        gt_vc[(gt_vis_e1 > 0.5)] = 2
        gt_vc_rgb = np.ones((*gt_vc.shape, 3)) * 0.1
        gt_vc_rgb[gt_vc == 1] = _E0_COLOR
        gt_vc_rgb[gt_vc == 2] = _E1_COLOR

        ax5 = fig.add_subplot(gs_bot[1, 4])
        ax5.imshow(gt_vc_rgb, interpolation="nearest")
        ax5.set_title("GT Class", color="white", fontsize=6, pad=2)
        ax5.axis("off")

        ax6 = fig.add_subplot(gs_bot[1, 5])
        ax6.imshow(vc_rgb, interpolation="nearest")
        ax6.set_title("Pred Class", color="white", fontsize=6, pad=2)
        ax6.axis("off")

        # Error map
        err_e0 = np.abs(vis_e0 - gt_vis_e0)
        err_e1 = np.abs(vis_e1 - gt_vis_e1)
        _proj_ax(1, 6, "Err E0", err_e0, cmap="RdYlGn_r")
        _proj_ax(1, 7, "Err E1", err_e1, cmap="RdYlGn_r")

        # IoU text overlay
        def _iou(p, g, thr=0.5):
            pm, gm = (p > thr), (g > thr)
            inter = (pm & gm).sum()
            union = (pm | gm).sum()
            return inter / max(union, 1)

        iou_e0 = _iou(vis_e0, gt_vis_e0)
        iou_e1 = _iou(vis_e1, gt_vis_e1)
        fig.text(0.01, 0.36,
                 f"Vis-IoU  E0={iou_e0:.3f}  E1={iou_e1:.3f}  min={min(iou_e0,iou_e1):.3f}",
                 color="white", fontsize=8, va="bottom",
                 bbox=dict(facecolor="#222244", alpha=0.8, pad=3))

        # Title
        fig.suptitle(
            f"[{stage.upper()}] ep={epoch:03d} step={step:04d} — 3D Volume Debug",
            color="white", fontsize=9, y=0.99)
        self._draw_contract_badge(fig, contract_metrics)

        save_dir = self.debug_dir / "training" / stage
        save_path = save_dir / f"ep{epoch:03d}_s{step:04d}_volume.png"
        fig.savefig(str(save_path), dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return save_path

    # ─── Guide debug ─────────────────────────────────────────────────────────

    def save_guide_debug(
        self,
        guides: Dict[str, torch.Tensor],
        assembler,
        epoch: int,
        stage: str,
        contract_metrics=None,
    ) -> Optional[Path]:
        if not HAS_MPL or not guides:
            return None
        try:
            return self._guide_debug_impl(guides, assembler, epoch, stage, contract_metrics)
        except Exception as e:
            print(f"  [debug_viz] guide debug failed: {e}", flush=True)
            return None

    def _guide_debug_impl(self, guides, assembler, epoch, stage, contract_metrics=None) -> Path:
        block_names = list(guides.keys())
        n_blocks = len(block_names)

        fig, axes = plt.subplots(
            n_blocks, 4, figsize=(14, 3.5 * n_blocks), facecolor="#1a1a2e")
        if n_blocks == 1:
            axes = axes[None, :]

        for ri, bn in enumerate(block_names):
            g = guides[bn]  # (B, C, H, W)
            b = 0
            g_np = _to_np(g[b])   # (C, H, W)
            C, H, W = g_np.shape

            # Gate value
            gate_val = float(torch.tanh(assembler.guide_gates[bn]).item()) if hasattr(assembler, 'guide_gates') else 1.0

            # Col 0: mean activation map
            mean_map = g_np.mean(axis=0)
            ax = axes[ri, 0]
            im = ax.imshow(mean_map, cmap="RdBu_r", interpolation="nearest")
            ax.set_title(f"{bn} | mean activation\ngate={gate_val:.4f}", color="white", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

            # Col 1: std map (feature diversity)
            std_map = g_np.std(axis=0)
            ax = axes[ri, 1]
            im = ax.imshow(std_map, cmap="viridis", vmin=0, interpolation="nearest")
            ax.set_title(f"{bn} | channel std", color="white", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

            # Col 2: per-channel magnitude histogram
            ax = axes[ri, 2]
            flat = g_np.reshape(C, -1)  # (C, H*W)
            channel_norms = np.linalg.norm(flat, axis=1)
            ax.bar(range(min(C, 64)), channel_norms[:64],
                   color=plt.cm.viridis(np.linspace(0, 1, min(C, 64))))
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white", labelsize=6)
            ax.set_title(f"{bn} | ch norms (first 64)", color="white", fontsize=8)

            # Col 3: top-2 channel spatial maps
            top2 = np.argsort(channel_norms)[-2:]
            rgb = np.zeros((H, W, 3))
            if len(top2) >= 2:
                rgb[:, :, 0] = _norm01(g_np[top2[0]])
                rgb[:, :, 2] = _norm01(g_np[top2[1]])
            ax = axes[ri, 3]
            ax.imshow(rgb, interpolation="nearest")
            ax.set_title(f"{bn} | top2 ch (R=ch{top2[0] if len(top2)>0 else '?'}, B=ch{top2[1] if len(top2)>1 else '?'})",
                        color="white", fontsize=8)
            ax.axis("off")

        # Gate summary bar
        if hasattr(assembler, 'guide_gates'):
            gate_vals = {bn: float(torch.tanh(assembler.guide_gates[bn]).item())
                         for bn in block_names}
            fig.text(0.01, 0.01,
                     "Guide gates: " + "  ".join(f"{k}={v:.4f}" for k, v in gate_vals.items()),
                     color="yellow", fontsize=8,
                     bbox=dict(facecolor="#222200", alpha=0.8, pad=3))

        fig.suptitle(f"[{stage.upper()}] ep={epoch:03d} — Guide Feature Debug",
                     color="white", fontsize=9)
        fig.patch.set_facecolor("#1a1a2e")
        self._draw_contract_badge(fig, contract_metrics)

        save_dir = self.debug_dir / "training" / stage
        save_path = save_dir / f"ep{epoch:03d}_guide.png"
        fig.savefig(str(save_path), dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return save_path

    # ─── Diffusion debug ─────────────────────────────────────────────────────

    def save_diffusion_debug(
        self,
        noise_pred: torch.Tensor,  # (B, C, T, H, W) or (B, C, H, W)
        noise_gt: torch.Tensor,
        epoch: int,
        step: int,
        stage: str,
        diff_weight: float = 1.0,
    ) -> Optional[Path]:
        if not HAS_MPL:
            return None
        try:
            return self._diffusion_debug_impl(
                noise_pred, noise_gt, epoch, step, stage, diff_weight)
        except Exception as e:
            print(f"  [debug_viz] diffusion debug failed: {e}", flush=True)
            return None

    def _diffusion_debug_impl(self, noise_pred, noise_gt, epoch, step, stage, diff_weight) -> Path:
        b = 0
        pred_np = _to_np(noise_pred[b])   # (C, [T,] H, W)
        gt_np   = _to_np(noise_gt[b])

        # If 4D (C, T, H, W): take first frame
        if pred_np.ndim == 4:
            pred_np = pred_np[:, 0]
            gt_np   = gt_np[:, 0]

        # Use first 3 channels as pseudo-RGB
        C = pred_np.shape[0]
        ch = min(C, 3)

        def _to_rgb(x):
            rgb = np.zeros((x.shape[-2], x.shape[-1], 3))
            for i in range(ch):
                rgb[:, :, i] = _norm01(x[i])
            return rgb

        pred_rgb = _to_rgb(pred_np)
        gt_rgb   = _to_rgb(gt_np)
        err_map  = np.abs(pred_np[:ch] - gt_np[:ch]).mean(axis=0)  # (H, W)
        mse_val  = float(((pred_np - gt_np) ** 2).mean())

        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), facecolor="#1a1a2e")

        axes[0].imshow(gt_rgb, interpolation="nearest")
        axes[0].set_title("GT Noise (ch0-2)", color="white", fontsize=8)
        axes[0].axis("off")

        axes[1].imshow(pred_rgb, interpolation="nearest")
        axes[1].set_title(f"Pred Noise (ch0-2)\ndiff_w={diff_weight:.2f}", color="white", fontsize=8)
        axes[1].axis("off")

        im = axes[2].imshow(err_map, cmap="hot", interpolation="nearest")
        axes[2].set_title(f"Error Map\nMSE={mse_val:.5f}", color="white", fontsize=8)
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046)

        # Noise power spectrum (channel-mean)
        from numpy.fft import fft2, fftshift
        fft_pred = np.abs(fftshift(fft2(pred_np.mean(axis=0))))
        fft_gt   = np.abs(fftshift(fft2(gt_np.mean(axis=0))))
        axes[3].plot(fft_gt.mean(axis=0), color="#4488ff", lw=1.5, label="GT")
        axes[3].plot(fft_pred.mean(axis=0), color="#ff4444", lw=1.5, label="Pred")
        axes[3].set_facecolor("#1a1a2e")
        axes[3].tick_params(colors="white", labelsize=6)
        axes[3].set_title("Noise Power (h-avg)", color="white", fontsize=8)
        axes[3].legend(fontsize=6, facecolor="#222244", labelcolor="white")

        fig.suptitle(f"[{stage.upper()}] ep={epoch:03d} step={step:04d} — Diffusion Noise Debug",
                     color="white", fontsize=9)
        fig.patch.set_facecolor("#1a1a2e")

        save_dir = self.debug_dir / "training" / stage
        save_path = save_dir / f"ep{epoch:03d}_s{step:04d}_diff.png"
        fig.savefig(str(save_path), dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return save_path

    # ─── Loss curve ──────────────────────────────────────────────────────────

    def update_loss_curve(
        self,
        train_history: List[Dict],
        val_history: List[Dict],
        stage_boundaries: Dict[str, int],
    ) -> Optional[Path]:
        if not HAS_MPL or not train_history:
            return None
        try:
            return self._loss_curve_impl(train_history, val_history, stage_boundaries)
        except Exception as e:
            print(f"  [debug_viz] loss curve failed: {e}", flush=True)
            return None

    def _loss_curve_impl(self, train_history, val_history, stage_boundaries) -> Path:
        epochs = [r["epoch"] for r in train_history]

        fig, axes = plt.subplots(2, 3, figsize=(15, 7), facecolor="#1a1a2e")

        def _plot_metric(ax, key, label, color, log_scale=False, secondary=None, sec_label=None, sec_color=None):
            vals = [r.get(key, np.nan) for r in train_history]
            vals_clean = [v if np.isfinite(v) else np.nan for v in vals]
            ax.plot(epochs, vals_clean, color=color, lw=1.5, label=label)
            if secondary:
                vals2 = [r.get(secondary, np.nan) for r in train_history]
                vals2_clean = [v if np.isfinite(v) else np.nan for v in vals2]
                ax2 = ax.twinx()
                ax2.plot(epochs, vals2_clean, color=sec_color, lw=1.0, alpha=0.7,
                         linestyle="--", label=sec_label)
                ax2.tick_params(colors=sec_color, labelsize=6)
                ax2.set_ylabel(sec_label, color=sec_color, fontsize=7)
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white", labelsize=7)
            ax.set_ylabel(label, color=color, fontsize=8)
            ax.set_xlabel("epoch", color="gray", fontsize=7)
            if log_scale:
                ax.set_yscale("symlog", linthresh=1e-4)
            # Stage boundaries
            for sname, sep in stage_boundaries.items():
                ax.axvline(x=sep, color="#ffaa00", lw=0.8, linestyle=":", alpha=0.7)
                ax.text(sep + 0.2, ax.get_ylim()[1] * 0.95, sname,
                        color="#ffaa00", fontsize=6, va="top")
            ax.legend(fontsize=7, facecolor="#222244", labelcolor="white", loc="upper right")

        _plot_metric(axes[0, 0], "loss", "Total Loss", "#ff6644",
                     secondary="l_struct", sec_label="Struct Loss", sec_color="#ffaa44")
        _plot_metric(axes[0, 1], "l_diff", "Diffusion Loss (MSE)", "#44aaff",
                     secondary="diff_weight", sec_label="diff_weight", sec_color="#88ffaa")
        _plot_metric(axes[0, 2], "acc_entity", "Entity Accuracy", "#aaffaa")

        # Val metrics
        if val_history:
            val_epochs = [r["epoch"] for r in val_history]

            def _val_plot(ax, key, label, color):
                vals = [r.get(key, np.nan) for r in val_history]
                ax.plot(val_epochs, vals, color=color, lw=2.0, marker="o", ms=3, label=label)

            ax = axes[1, 0]
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white", labelsize=7)
            _val_plot(ax, "val_iou_e0", "IoU E0 (val)", _rgb2hex(_E0_COLOR))
            _val_plot(ax, "val_iou_e1", "IoU E1 (val)", _rgb2hex(_E1_COLOR))
            _val_plot(ax, "val_iou_min", "IoU min (val)", "#ffffff")
            for sname, sep in stage_boundaries.items():
                ax.axvline(x=sep, color="#ffaa00", lw=0.8, linestyle=":", alpha=0.7)
            ax.legend(fontsize=7, facecolor="#222244", labelcolor="white")
            ax.set_xlabel("epoch", color="gray", fontsize=7)
            ax.set_ylabel("IoU", color="white", fontsize=8)
            ax.set_title("Projected IoU (val)", color="white", fontsize=8)

            ax = axes[1, 1]
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white", labelsize=7)
            _val_plot(ax, "val_score", "Val Score", "#ffffaa")
            _val_plot(ax, "val_acc_entity", "Entity Acc (val)", "#aaffaa")
            for sname, sep in stage_boundaries.items():
                ax.axvline(x=sep, color="#ffaa00", lw=0.8, linestyle=":", alpha=0.7)
            ax.legend(fontsize=7, facecolor="#222244", labelcolor="white")
            ax.set_xlabel("epoch", color="gray", fontsize=7)
            ax.set_title("Val Score / Acc", color="white", fontsize=8)

            ax = axes[1, 2]
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white", labelsize=7)
            _val_plot(ax, "val_diff_mse", "Val Diff MSE", "#44aaff")
            _val_plot(ax, "val_struct", "Val Struct", "#ff8844")
            for sname, sep in stage_boundaries.items():
                ax.axvline(x=sep, color="#ffaa00", lw=0.8, linestyle=":", alpha=0.7)
            ax.legend(fontsize=7, facecolor="#222244", labelcolor="white")
            ax.set_xlabel("epoch", color="gray", fontsize=7)
            ax.set_title("Val Loss Components", color="white", fontsize=8)
        else:
            for ax in axes[1]:
                ax.set_facecolor("#1a1a2e")
                ax.text(0.5, 0.5, "No val data yet", color="gray",
                        ha="center", va="center", transform=ax.transAxes)

        for ax in axes.flat:
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")

        current_ep = epochs[-1] if epochs else 0
        fig.suptitle(f"Training Curves — ep={current_ep}",
                     color="white", fontsize=10, y=1.01)
        fig.tight_layout()

        save_path = self.debug_dir / "training" / "loss_curve.png"
        fig.savefig(str(save_path), dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return save_path


def _rgb2hex(rgb):
    """(R,G,B) floats → '#rrggbb' string for matplotlib."""
    r, g, b = [int(x * 255) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


# ─── Contract badge helper ────────────────────────────────────────────────────
# Attached to TrainingDebugViz as a method so subclasses can use it too.

def _draw_contract_badge_fn(fig, contract_metrics) -> None:
    """Overlay pass/fail contract badge onto a figure."""
    if contract_metrics is None:
        return
    m = contract_metrics
    lines = [
        f"S1: {'✓ PASS' if m.stage1_pass else '✗ FAIL'}  "
        f"compact={m.vol_compactness:.3f}  "
        f"2color={'Y' if m.two_color_presence else 'N'}({m.two_color_e0_frac:.3f}/{m.two_color_e1_frac:.3f})  "
        f"iou_min={m.val_iou_min:.3f}",

        f"S2: {'✓ PASS' if m.stage2_pass else '✗ FAIL'}  "
        f"gate={m.gate_open:.4f}  "
        f"overlay={m.pred_overlay_match:.3f}  "
        f"winner={m.one_winner_ratio:.2f}",

        f"S3: {'✓ PASS' if m.stage3_pass else '✗ FAIL'}  "
        f"diff_mse={m.diffusion_mse:.4f}  "
        f"stable={'Y' if m.diffusion_stable else 'N'}  "
        f"contract_score={m.contract_score:.4f}",
    ]
    colors = [
        "#88ff88" if m.stage1_pass else "#ff6666",
        "#88ff88" if m.stage2_pass else "#ff6666",
        "#88ff88" if m.stage3_pass else "#ff6666",
    ]
    y_pos = 0.005
    for line, color in zip(reversed(lines), reversed(colors)):
        fig.text(0.50, y_pos, line, color=color, fontsize=7.5,
                 ha="center", va="bottom",
                 bbox=dict(facecolor="#111122", alpha=0.85, pad=2, boxstyle="round"))
        y_pos += 0.028
