"""
Phase 63 — Entity-aware sequence evaluator
============================================

Goal-aligned evaluation: a multi-entity video generation is considered good
only if *both* entities remain identifiable across the full sequence.  This
evaluator combines per-frame visible/amodal IoU with sequence-level
reappearance accuracy and entity-balance into a single scalar.

Metric definitions
------------------
visible_survival_e{0,1}   : fraction of frames where visible_area > threshold
visible_iou_{e0,e1,min}   : mean per-frame IoU of visible mask vs GT
amodal_iou_{e0,e1,min}    : mean per-frame IoU of amodal mask vs GT
reappearance_accuracy     : fraction of post-occlusion frames where the
                            correct entity identity is recovered
composite_isolated_consistency :
    SSIM between the isolated-entity render and a crop of the composite
    frame restricted to that entity's GT amodal mask
entity_balance            : min(area_e0, area_e1) / max(area_e0, area_e1)
                            — 1.0 means perfectly balanced
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Optional torchmetrics + kornia imports
# ---------------------------------------------------------------------------
try:
    from torchmetrics.classification import BinaryJaccardIndex                 # type: ignore
    _HAS_TM = True
except Exception:                                                              # pragma: no cover
    _HAS_TM = False

try:
    import kornia.metrics as _kornia_metrics                                    # type: ignore
    _HAS_KORNIA = True
except Exception:                                                              # pragma: no cover
    _HAS_KORNIA = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _binary_iou(
    pred: torch.Tensor,            # (B, H, W) or (H, W) probs in [0, 1]
    gt: torch.Tensor,              # same shape, {0, 1}
    thresh: float = 0.5,
) -> float:
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    p = (pred.float() > thresh)
    g = (gt.float() > 0.5)
    if _HAS_TM:
        jaccard = BinaryJaccardIndex().to(pred.device)
        return float(jaccard(p.int(), g.int()).item())
    # Fallback.
    inter = (p & g).float().sum(dim=(-1, -2))
    union = (p | g).float().sum(dim=(-1, -2)).clamp(min=1e-6)
    return float((inter / union).mean().item())


@torch.no_grad()
def _ssim(a: torch.Tensor, b: torch.Tensor, window: int = 11) -> float:
    """SSIM averaged over batch.  Inputs (B, C, H, W) in [0, 1]."""
    if a.dim() == 3:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    if _HAS_KORNIA:
        val = _kornia_metrics.ssim(a.float(), b.float(), window_size=window)
        # kornia returns per-pixel SSIM in [-1, 1]; reduce to mean.
        return float(val.mean().item())
    # Minimal fallback: 1 - MSE in [0, 1] (rough).
    return float(1.0 - (a.float() - b.float()).pow(2).mean().item())


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class EntityEvaluator:
    """
    Aggregate per-sequence entity-aware metrics.

    Parameters
    ----------
    visible_survival_thresh : float
        A frame is considered to "contain" an entity if the predicted visible
        mean area exceeds this fraction.
    reappearance_occlusion_thresh : float
        GT visible area below which a frame is labelled "occluded" when
        computing reappearance accuracy.
    """

    def __init__(
        self,
        visible_survival_thresh: float = 0.02,
        reappearance_occlusion_thresh: float = 0.01,
    ):
        self.visible_survival_thresh = visible_survival_thresh
        self.reappearance_occlusion_thresh = reappearance_occlusion_thresh

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_sequence(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        ground_truth: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Parameters
        ----------
        predictions : list of per-frame dicts with keys
            ``visible_e0, visible_e1``        : (H, W) in [0, 1]
            ``amodal_e0, amodal_e1``          : (H, W) in [0, 1]
            ``isolated_e0``, ``isolated_e1``  : optional, (C, H, W) renders
        ground_truth : list of per-frame dicts with the same shape keys, plus
            ``gt_visible_e0, gt_visible_e1``
            ``gt_amodal_e0, gt_amodal_e1``
            ``composite_frame``               : optional (C, H, W) — used for
                                                composite_isolated_consistency
        """
        assert len(predictions) == len(ground_truth), \
            "predictions and ground_truth must be the same length"
        if not predictions:
            return {}

        T = len(predictions)

        # --- Visible / amodal IoU per frame ------------------------------
        v_iou_e0, v_iou_e1 = [], []
        a_iou_e0, a_iou_e1 = [], []
        survive_e0, survive_e1 = [], []
        area_e0_list, area_e1_list = [], []
        gt_vis_e0_area, gt_vis_e1_area = [], []

        for pred, gt in zip(predictions, ground_truth):
            v0 = pred["visible_e0"]
            v1 = pred["visible_e1"]
            a0 = pred["amodal_e0"]
            a1 = pred["amodal_e1"]
            gv0 = gt.get("gt_visible_e0", gt.get("visible_e0"))
            gv1 = gt.get("gt_visible_e1", gt.get("visible_e1"))
            ga0 = gt.get("gt_amodal_e0", gt.get("amodal_e0"))
            ga1 = gt.get("gt_amodal_e1", gt.get("amodal_e1"))

            v_iou_e0.append(_binary_iou(v0, gv0))
            v_iou_e1.append(_binary_iou(v1, gv1))
            a_iou_e0.append(_binary_iou(a0, ga0))
            a_iou_e1.append(_binary_iou(a1, ga1))

            area_e0 = float(v0.float().mean().item())
            area_e1 = float(v1.float().mean().item())
            area_e0_list.append(area_e0)
            area_e1_list.append(area_e1)

            survive_e0.append(1.0 if area_e0 > self.visible_survival_thresh else 0.0)
            survive_e1.append(1.0 if area_e1 > self.visible_survival_thresh else 0.0)

            gt_vis_e0_area.append(float(gv0.float().mean().item()))
            gt_vis_e1_area.append(float(gv1.float().mean().item()))

        # --- Reappearance accuracy ----------------------------------------
        reappear_correct = 0
        reappear_total = 0
        for t in range(1, T):
            # If entity was occluded at t-1 and visible in GT at t, check
            # that the prediction also recovers it.
            for ei, (gt_area_prev, gt_area_curr, survive_curr) in enumerate([
                (gt_vis_e0_area[t - 1], gt_vis_e0_area[t], survive_e0[t]),
                (gt_vis_e1_area[t - 1], gt_vis_e1_area[t], survive_e1[t]),
            ]):
                was_occluded = gt_area_prev < self.reappearance_occlusion_thresh
                now_visible = gt_area_curr > self.visible_survival_thresh
                if was_occluded and now_visible:
                    reappear_total += 1
                    if survive_curr > 0.5:
                        reappear_correct += 1

        reappearance_acc = (
            reappear_correct / reappear_total if reappear_total > 0 else 1.0
        )

        # --- Composite / isolated SSIM ------------------------------------
        iso_ssim_vals: List[float] = []
        for pred, gt in zip(predictions, ground_truth):
            comp = gt.get("composite_frame", None)
            iso0 = pred.get("isolated_e0", None)
            iso1 = pred.get("isolated_e1", None)
            if comp is None or iso0 is None or iso1 is None:
                continue
            # Restrict composite to each entity's GT amodal mask.
            ga0 = gt["gt_amodal_e0"].float()
            ga1 = gt["gt_amodal_e1"].float()
            comp0 = comp.float() * ga0.unsqueeze(0)
            comp1 = comp.float() * ga1.unsqueeze(0)
            iso_ssim_vals.append(_ssim(iso0.float(), comp0))
            iso_ssim_vals.append(_ssim(iso1.float(), comp1))
        iso_ssim = float(sum(iso_ssim_vals) / len(iso_ssim_vals)) if iso_ssim_vals else 0.0

        # --- Entity balance -----------------------------------------------
        area_e0_mean = float(sum(area_e0_list) / max(len(area_e0_list), 1))
        area_e1_mean = float(sum(area_e1_list) / max(len(area_e1_list), 1))
        denom = max(area_e0_mean, area_e1_mean, 1e-6)
        entity_balance = min(area_e0_mean, area_e1_mean) / denom

        metrics = {
            "visible_iou_e0": float(sum(v_iou_e0) / T),
            "visible_iou_e1": float(sum(v_iou_e1) / T),
            "visible_iou_min": min(
                float(sum(v_iou_e0) / T), float(sum(v_iou_e1) / T),
            ),
            "amodal_iou_e0": float(sum(a_iou_e0) / T),
            "amodal_iou_e1": float(sum(a_iou_e1) / T),
            "amodal_iou_min": min(
                float(sum(a_iou_e0) / T), float(sum(a_iou_e1) / T),
            ),
            "visible_survival_e0": float(sum(survive_e0) / T),
            "visible_survival_e1": float(sum(survive_e1) / T),
            "visible_survival_min": min(
                float(sum(survive_e0) / T), float(sum(survive_e1) / T),
            ),
            "reappearance_accuracy": reappearance_acc,
            "composite_isolated_consistency": iso_ssim,
            "entity_balance": entity_balance,
        }
        return metrics

    # ------------------------------------------------------------------
    # Checkpoint selection score
    # ------------------------------------------------------------------
    def best_checkpoint_criteria(self, metrics: Dict[str, float]) -> float:
        """
        Scalar score used for "best checkpoint" selection.

        Hierarchy of priorities:
          1. visible_iou_min          — both entities must be visually correct
          2. visible_survival_min     — neither entity may die
          3. amodal_iou_min           — full-body understanding
          4. entity_balance           — no single-entity dominance

        We combine with a geometric mean: it rewards simultaneous improvements
        and strongly penalises any single metric falling near 0.
        """
        eps = 1e-3
        score_components = [
            metrics.get("visible_iou_min", 0.0) + eps,
            metrics.get("visible_survival_min", 0.0) + eps,
            metrics.get("amodal_iou_min", 0.0) + eps,
            metrics.get("entity_balance", 0.0) + eps,
        ]
        # Weights reflect priority order — heavier on visible_iou_min.
        weights = [0.4, 0.3, 0.2, 0.1]
        log_score = 0.0
        for w, s in zip(weights, score_components):
            log_score += w * float(torch.log(torch.tensor(s)).item())
        return float(torch.exp(torch.tensor(log_score)).item())


__all__ = ["EntityEvaluator"]
