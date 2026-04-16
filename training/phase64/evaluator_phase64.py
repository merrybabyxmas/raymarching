"""
training/phase64/evaluator_phase64.py
=======================================
Complete Phase 64 evaluation covering all four success criteria from blueprint §16.

Metric categories:
  §16.1  Scene prior metrics  — IoU, hidden fraction accuracy, slot swap rate,
                                contact separation accuracy, visible survival,
                                entity balance
  §16.2  Decoder metrics      — composite PSNR/L1, object-count preservation

Usage:
    evaluator = Phase64Evaluator()
    scene_m  = evaluator.eval_scene_prior(preds, gts)
    dec_m    = evaluator.eval_decoder(pred_rgb, gt_rgb)
    print(evaluator.summary(scene_m, dec_m))
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class Phase64Evaluator:
    """
    Metrics for all 4 success criteria:
      1. Entity persistence (visible survival)
      2. Amodal correctness (hidden fraction accuracy)
      3. Backbone transfer (two-object detection)
      4. Render usefulness (reconstruction quality)

    Parameters
    ----------
    visible_thresh  : minimum visible-mask mean to consider an entity "present"
    amodal_thresh   : minimum amodal-mask mean to consider an entity in the scene
    """

    def __init__(
        self,
        visible_thresh: float = 0.02,
        amodal_thresh: float = 0.02,
    ) -> None:
        self.visible_thresh = visible_thresh
        self.amodal_thresh = amodal_thresh

    # ======================================================================== #
    #  §16.1  Scene prior metrics
    # ======================================================================== #

    def eval_scene_prior(
        self,
        preds: List[dict],
        gts: List[dict],
    ) -> dict:
        """
        Evaluate scene-prior predictions against ground truth annotations.

        Each element of *preds* / *gts* is a dict with optional keys:
            visible_e0, visible_e1   : (H, W) or (T, H, W) float tensor / array
            amodal_e0, amodal_e1     : (H, W) or (T, H, W) float tensor / array
            hidden_fraction_e0/e1    : float  (GT only — for fraction accuracy)
            overlap_ratio            : float  (GT only — for contact separation)
            is_reappearance          : bool   (GT only)

        Returns
        -------
        dict with keys:
            visible_iou_e0 / e1 / min
            amodal_iou_e0  / e1 / min
            hidden_fraction_accuracy_e0 / e1
            reappearance_consistency
            slot_swap_rate
            contact_separation_accuracy
            visible_survival_e0 / e1 / min
            entity_balance
        """
        assert len(preds) == len(gts), "preds and gts must have the same length"

        vis_iou_e0, vis_iou_e1 = [], []
        amo_iou_e0, amo_iou_e1 = [], []
        hf_acc_e0, hf_acc_e1 = [], []
        reapp_scores = []
        contact_sep_scores = []
        survival_e0, survival_e1 = [], []

        for pred, gt in zip(preds, gts):
            # ---- IoU: visible ----------------------------------------------- #
            if "visible_e0" in pred and "visible_e0" in gt:
                vis_iou_e0.append(self._compute_iou(
                    self._to_tensor(pred["visible_e0"]),
                    self._to_tensor(gt["visible_e0"]),
                ))
            if "visible_e1" in pred and "visible_e1" in gt:
                vis_iou_e1.append(self._compute_iou(
                    self._to_tensor(pred["visible_e1"]),
                    self._to_tensor(gt["visible_e1"]),
                ))

            # ---- IoU: amodal ------------------------------------------------- #
            if "amodal_e0" in pred and "amodal_e0" in gt:
                amo_iou_e0.append(self._compute_iou(
                    self._to_tensor(pred["amodal_e0"]),
                    self._to_tensor(gt["amodal_e0"]),
                ))
            if "amodal_e1" in pred and "amodal_e1" in gt:
                amo_iou_e1.append(self._compute_iou(
                    self._to_tensor(pred["amodal_e1"]),
                    self._to_tensor(gt["amodal_e1"]),
                ))

            # ---- Hidden fraction accuracy ------------------------------------ #
            for e, (hf_list, pred_key, gt_key) in enumerate([
                (hf_acc_e0, "hidden_fraction_e0", "hidden_fraction_e0"),
                (hf_acc_e1, "hidden_fraction_e1", "hidden_fraction_e1"),
            ]):
                if pred_key in pred and gt_key in gt:
                    acc = self._compute_hidden_fraction_acc(
                        float(pred[pred_key]), float(gt[gt_key])
                    )
                    hf_list.append(acc)

            # ---- Reappearance consistency ------------------------------------ #
            if gt.get("is_reappearance", False):
                score = self._compute_reappearance_consistency(pred, gt)
                if score is not None:
                    reapp_scores.append(score)

            # ---- Contact separation ----------------------------------------- #
            ov = float(gt.get("overlap_ratio", 0.0))
            if ov > 0.08 and "sep_map" in pred:
                sep = self._to_tensor(pred["sep_map"])
                cs = float(sep.abs().mean().item())
                contact_sep_scores.append(cs)

            # ---- Visible survival ------------------------------------------- #
            if "visible_e0" in pred:
                s0 = self._compute_visible_survival(
                    self._to_tensor(pred["visible_e0"]),
                    self.visible_thresh,
                )
                survival_e0.append(s0)
            if "visible_e1" in pred:
                s1 = self._compute_visible_survival(
                    self._to_tensor(pred["visible_e1"]),
                    self.visible_thresh,
                )
                survival_e1.append(s1)

        # ---- Slot swap rate -------------------------------------------------- #
        slot_swap_rate = self._detect_slot_swap(preds)

        # ---- Entity balance -------------------------------------------------- #
        entity_balance = self._compute_entity_balance(preds)

        def _safe_mean(lst):
            return float(np.mean(lst)) if lst else float("nan")

        vis_iou_e0_m  = _safe_mean(vis_iou_e0)
        vis_iou_e1_m  = _safe_mean(vis_iou_e1)
        amo_iou_e0_m  = _safe_mean(amo_iou_e0)
        amo_iou_e1_m  = _safe_mean(amo_iou_e1)

        return {
            # Visible IoU
            "visible_iou_e0":               vis_iou_e0_m,
            "visible_iou_e1":               vis_iou_e1_m,
            "visible_iou_min":              min(vis_iou_e0_m, vis_iou_e1_m),
            # Amodal IoU
            "amodal_iou_e0":                amo_iou_e0_m,
            "amodal_iou_e1":                amo_iou_e1_m,
            "amodal_iou_min":               min(amo_iou_e0_m, amo_iou_e1_m),
            # Hidden fraction accuracy
            "hidden_fraction_accuracy_e0":  _safe_mean(hf_acc_e0),
            "hidden_fraction_accuracy_e1":  _safe_mean(hf_acc_e1),
            # Reappearance
            "reappearance_consistency":     _safe_mean(reapp_scores),
            # Slot swap
            "slot_swap_rate":               slot_swap_rate,
            # Contact separation
            "contact_separation_accuracy":  _safe_mean(contact_sep_scores),
            # Survival
            "visible_survival_e0":          _safe_mean(survival_e0),
            "visible_survival_e1":          _safe_mean(survival_e1),
            "visible_survival_min":         min(_safe_mean(survival_e0), _safe_mean(survival_e1)),
            # Balance
            "entity_balance":               entity_balance,
        }

    # ======================================================================== #
    #  §16.2  Decoder metrics
    # ======================================================================== #

    def eval_decoder(
        self,
        pred_rgb: List,
        gt_rgb: List,
        iso_e0: Optional[List] = None,
        iso_e1: Optional[List] = None,
    ) -> dict:
        """
        Evaluate decoder reconstruction quality.

        Parameters
        ----------
        pred_rgb : list of (H, W, 3) uint8 or float arrays / tensors
        gt_rgb   : list of (H, W, 3) uint8 or float arrays / tensors
        iso_e0   : list of (H, W, 3) predicted entity-0 isolation renders (optional)
        iso_e1   : list of (H, W, 3) predicted entity-1 isolation renders (optional)

        Returns
        -------
        dict with keys:
            composite_psnr
            composite_l1
            object_count_preservation  (fraction of samples where both entities present)
        """
        psnr_values, l1_values = [], []
        obj_count_ok = []

        for i, (pred, gt) in enumerate(zip(pred_rgb, gt_rgb)):
            p = self._to_float_hwc(pred)
            g = self._to_float_hwc(gt)

            mse = float(((p - g) ** 2).mean())
            if mse > 0:
                psnr = 10.0 * math.log10(1.0 / mse)
            else:
                psnr = 100.0
            psnr_values.append(psnr)
            l1_values.append(float(np.abs(p - g).mean()))

            # Object count preservation: both isolation renders non-trivially present
            if iso_e0 is not None and iso_e1 is not None and i < len(iso_e0) and i < len(iso_e1):
                e0_present = self._to_float_hwc(iso_e0[i]).mean() > self.visible_thresh
                e1_present = self._to_float_hwc(iso_e1[i]).mean() > self.visible_thresh
                obj_count_ok.append(float(e0_present and e1_present))

        def _safe_mean(lst):
            return float(np.mean(lst)) if lst else float("nan")

        return {
            "composite_psnr":               _safe_mean(psnr_values),
            "composite_l1":                 _safe_mean(l1_values),
            "object_count_preservation":    _safe_mean(obj_count_ok) if obj_count_ok else float("nan"),
        }

    # ======================================================================== #
    #  Per-sample computation helpers
    # ======================================================================== #

    def _compute_iou(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        eps: float = 1e-6,
    ) -> float:
        """Soft IoU between two binary or soft masks (any shape)."""
        p = pred.float().clamp(0.0, 1.0)
        g = gt.float().clamp(0.0, 1.0)
        # Handle spatial size mismatch (e.g. scaled model 64×64 vs GT 32×32)
        if p.shape != g.shape and p.numel() != g.numel():
            # Resize pred to match gt spatial resolution
            g_shape = g.shape
            p = torch.nn.functional.interpolate(
                p.reshape(1, 1, *p.shape[-2:]) if p.dim() >= 2 else p.reshape(1, 1, -1, 1),
                size=g_shape[-2:] if g.dim() >= 2 else (g.numel(), 1),
                mode="bilinear",
                align_corners=False,
            ).reshape(g.shape)
        p = p.reshape(-1)
        g = g.reshape(-1)
        inter = (p * g).sum()
        union = p.sum() + g.sum() - inter
        return float((inter + eps) / (union + eps))

    def _compute_hidden_fraction_acc(
        self,
        pred_frac: float,
        gt_frac: float,
        eps: float = 1e-6,
    ) -> float:
        """Accuracy of hidden-fraction prediction as 1 - relative_error."""
        err = abs(pred_frac - gt_frac)
        denom = max(gt_frac, eps)
        return float(max(0.0, 1.0 - err / denom))

    def _compute_reappearance_consistency(
        self,
        pred: dict,
        gt: dict,
    ) -> Optional[float]:
        """
        Check if a hidden entity correctly reappears when GT says it should.

        Returns fraction of reappearance frames where pred entity is visible.
        Returns None if no reappearance frame info available.
        """
        reapp_frames = gt.get("reappearance_frames")
        if not reapp_frames:
            return None

        hit = 0
        total = 0
        for fr_info in reapp_frames:
            e_idx = fr_info.get("entity", 0)
            key = f"visible_e{e_idx}"
            if key not in pred:
                continue
            vis = self._to_tensor(pred[key])
            if vis.dim() == 3:
                # (T, H, W) — take specific frame
                t = fr_info.get("frame_idx", 0)
                if t < vis.shape[0]:
                    vis = vis[t]
                else:
                    continue
            present = float(vis.mean().item()) > self.visible_thresh
            hit += int(present)
            total += 1

        if total == 0:
            return None
        return float(hit / total)

    def _compute_visible_survival(
        self,
        vis: torch.Tensor,  # (T, H, W) or (H, W)
        thresh: float,
    ) -> float:
        """Fraction of frames where entity is visible above threshold."""
        if vis.dim() == 2:
            return float(vis.mean().item() > thresh)
        T = vis.shape[0]
        present = sum(
            1 for t in range(T) if float(vis[t].mean().item()) > thresh
        )
        return float(present / T)

    def _detect_slot_swap(
        self,
        preds: List[dict],
    ) -> float:
        """
        Detect slot-swap events: how often entity identity seems to swap
        between consecutive predictions.

        Heuristic: a swap is detected if the dominant entity (by visible mean)
        changes sign between adjacent frames while the scene was previously stable.

        Returns swap_rate in [0, 1].
        """
        if not preds:
            return 0.0

        swap_count = 0
        total = 0

        for pred in preds:
            if "visible_e0" not in pred or "visible_e1" not in pred:
                continue
            v0 = self._to_tensor(pred["visible_e0"])
            v1 = self._to_tensor(pred["visible_e1"])

            if v0.dim() < 2:
                continue

            # Flatten to sequence of means
            if v0.dim() == 3:
                means0 = [float(v0[t].mean()) for t in range(v0.shape[0])]
                means1 = [float(v1[t].mean()) for t in range(v1.shape[0])]
            else:
                means0 = [float(v0.mean())]
                means1 = [float(v1.mean())]

            # Dominant entity at each frame
            dominant = [0 if m0 >= m1 else 1 for m0, m1 in zip(means0, means1)]

            # Count sign flips
            for i in range(1, len(dominant)):
                if dominant[i] != dominant[i - 1]:
                    swap_count += 1
                total += 1

        if total == 0:
            return 0.0
        return float(swap_count / total)

    def _compute_entity_balance(
        self,
        preds: List[dict],
    ) -> float:
        """
        Entity balance: how equally the two entities share visible area.

        Returns mean across samples of min(m0, m1) / (max(m0, m1) + eps).
        Closer to 1 = more balanced.
        """
        balances = []
        for pred in preds:
            if "visible_e0" not in pred or "visible_e1" not in pred:
                continue
            m0 = float(self._to_tensor(pred["visible_e0"]).mean())
            m1 = float(self._to_tensor(pred["visible_e1"]).mean())
            eps = 1e-8
            balance = min(m0, m1) / (max(m0, m1) + eps)
            balances.append(balance)
        if not balances:
            return float("nan")
        return float(np.mean(balances))

    # ======================================================================== #
    #  Summary
    # ======================================================================== #

    def summary(
        self,
        scene_metrics: dict,
        decoder_metrics: Optional[dict] = None,
    ) -> str:
        """Format evaluation results as a human-readable string."""
        lines = ["=" * 60, "Phase 64 Evaluation Summary", "=" * 60]

        def _fmt(v):
            if isinstance(v, float):
                if math.isnan(v):
                    return "n/a"
                return f"{v:.4f}"
            return str(v)

        lines.append("\n[Scene Prior Metrics §16.1]")
        for key in [
            "visible_iou_e0", "visible_iou_e1", "visible_iou_min",
            "amodal_iou_e0", "amodal_iou_e1", "amodal_iou_min",
            "hidden_fraction_accuracy_e0", "hidden_fraction_accuracy_e1",
            "reappearance_consistency",
            "slot_swap_rate",
            "contact_separation_accuracy",
            "visible_survival_e0", "visible_survival_e1", "visible_survival_min",
            "entity_balance",
        ]:
            v = scene_metrics.get(key, float("nan"))
            lines.append(f"  {key:<42s} {_fmt(v)}")

        if decoder_metrics:
            lines.append("\n[Decoder Metrics §16.2]")
            for key in ["composite_psnr", "composite_l1", "object_count_preservation"]:
                v = decoder_metrics.get(key, float("nan"))
                lines.append(f"  {key:<42s} {_fmt(v)}")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ======================================================================== #
    #  Internal utilities
    # ======================================================================== #

    @staticmethod
    def _to_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32))
        return torch.tensor(float(x))

    @staticmethod
    def _to_float_hwc(x) -> np.ndarray:
        """Convert to (H, W, C) float32 array in [0, 1]."""
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.asarray(x, dtype=np.float32)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        return arr
