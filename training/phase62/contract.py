"""
Phase 62 — Debug Contract
==========================

Formal pass/fail criteria for each training stage.
A stage is only "successful" when ALL of its contract conditions are met.

Stage 1 contract (volume):
  - vol_compactness > 0.20   (entity_probs concentrated in 1-2 depth slices)
  - two_color_presence = True  (both entities appear in pred_class)
  - val_iou_min > 0.10

Stage 2 contract (guide):
  - gate_open > 0.02         (guide gate has opened from 0)
  - pred_overlay_match > 0.05 (pred overlay spatially agrees with GT)
  - one_winner_ratio < 0.90  (not collapsed to one entity)

Stage 3 contract (diffusion):
  - diffusion_mse_stable = True  (no explosion over last N epochs)
  - two_color_presence = True    (still both entities in pred overlay)
  - gate_open > 0.05

Usage:
  contract = DebugContract()
  metrics = contract.compute(vol_outputs, guides, assembler, gt_visible, val_metrics)
  contract.log(metrics, epoch)
  if contract.stage1_pass(metrics):
      ...
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


# ─── Thresholds ──────────────────────────────────────────────────────────────

STAGE1_COMPACTNESS_MIN  = 0.20   # depth concentration (0=slab, 1=perfect blob)
STAGE1_IOU_MIN          = 0.10   # val_iou_min threshold
STAGE1_TWO_COLOR_MIN    = 0.01   # each entity must cover >1% of pixels

STAGE2_GATE_OPEN_MIN    = 0.02   # tanh(gate) > 0.02
STAGE2_OVERLAY_MATCH_MIN= 0.05   # pred/GT overlay IoU > 0.05
STAGE2_ONE_WINNER_MAX   = 0.90   # dominant entity fraction < 0.90

STAGE3_GATE_OPEN_MIN    = 0.05
STAGE3_DIFF_MSE_MAX     = 0.50   # diffusion MSE must stay below this
STAGE3_MSE_WINDOW       = 5      # stability window (epochs)


# ─── Metric dataclass ────────────────────────────────────────────────────────

@dataclass
class ContractMetrics:
    epoch: int = 0
    stage: str = "stage1"

    # Stage 1 — volume
    vol_compactness_e0: float = 0.0
    vol_compactness_e1: float = 0.0
    vol_compactness:    float = 0.0   # min(e0, e1)
    two_color_e0_frac:  float = 0.0
    two_color_e1_frac:  float = 0.0
    two_color_presence: bool  = False
    val_iou_min:        float = 0.0

    # Stage 2 — guide
    gate_open:          float = 0.0   # max gate across blocks
    gate_per_block:     Dict[str, float] = field(default_factory=dict)
    pred_overlay_match: float = 0.0
    one_winner_ratio:   float = 0.5

    # Stage 3 — diffusion
    diffusion_mse:      float = 0.0
    diffusion_stable:   bool  = True

    # Checkpoint score (multi-objective)
    contract_score:     float = 0.0

    # Pass/fail
    stage1_pass: bool = False
    stage2_pass: bool = False
    stage3_pass: bool = False


# ─── Contract engine ─────────────────────────────────────────────────────────

class DebugContract:
    """
    Computes and tracks debug contract metrics across epochs.
    Provides gate-adaptive diff_weight and multi-objective checkpoint score.
    """

    def __init__(self):
        self.history: List[ContractMetrics] = []

    # ── Volume compactness ────────────────────────────────────────────────

    @staticmethod
    def _depth_compactness(entity_probs_3d: torch.Tensor) -> float:
        """
        How concentrated is the entity_probs along the depth axis?
        entity_probs_3d: (K, H, W) for one entity.

        Uses 1 - normalised_entropy of depth-wise mass distribution.
        0 = uniform slab, 1 = single perfect slice.
        """
        K = entity_probs_3d.shape[0]
        if K <= 1:
            return 1.0
        depth_mass = entity_probs_3d.float().mean(dim=(1, 2))   # (K,)
        mass_sum = depth_mass.sum()
        if mass_sum < 1e-8:
            return 0.0
        p = (depth_mass / mass_sum).clamp(1e-9, 1.0)
        entropy = -(p * p.log()).sum().item()
        max_entropy = math.log(K)
        return float(1.0 - entropy / max_entropy)

    # ── Two-color presence ────────────────────────────────────────────────

    @staticmethod
    def _two_color(visible_class: torch.Tensor) -> tuple:
        """
        Returns (e0_frac, e1_frac, both_present).
        visible_class: (B, H, W) long tensor.
        """
        total = visible_class.numel()
        if total == 0:
            return 0.0, 0.0, False
        e0 = float((visible_class == 1).sum().item()) / total
        e1 = float((visible_class == 2).sum().item()) / total
        both = (e0 > STAGE1_TWO_COLOR_MIN) and (e1 > STAGE1_TWO_COLOR_MIN)
        return e0, e1, both

    # ── Guide gate ────────────────────────────────────────────────────────

    @staticmethod
    def _gate_open(assembler) -> tuple:
        """Returns (max_gate, per_block_dict)."""
        if not hasattr(assembler, "guide_gates"):
            return 0.0, {}
        per_block = {}
        for bn, param in assembler.guide_gates.items():
            per_block[bn] = float(torch.tanh(param).item())
        max_gate = max(per_block.values()) if per_block else 0.0
        return max_gate, per_block

    # ── Pred overlay match ────────────────────────────────────────────────

    @staticmethod
    def _overlay_match(visible_class: torch.Tensor,
                        gt_visible: torch.Tensor) -> float:
        """
        IoU between (pred_class == any_entity) and (GT_visible sum > 0).
        visible_class: (B, H, W), gt_visible: (B, 2, H, W)
        """
        pred_fg = (visible_class > 0).float()
        gt_fg   = (gt_visible.sum(dim=1) > 0.5).float()
        inter = (pred_fg * gt_fg).sum()
        union = (pred_fg + gt_fg).clamp(max=1).sum()
        if union < 1:
            return 0.0
        return float((inter / union).item())

    # ── One-winner ratio ──────────────────────────────────────────────────

    @staticmethod
    def _one_winner(vol_outputs) -> float:
        """
        How dominant is the winning entity?
        If max(vis_e0, vis_e1) / (vis_e0 + vis_e1) > 0.90 → collapse.
        """
        if not (vol_outputs.visible and "e0" in vol_outputs.visible):
            return 0.5
        vis_e0 = float(vol_outputs.visible["e0"].mean().item())
        vis_e1 = float(vol_outputs.visible["e1"].mean().item())
        total = vis_e0 + vis_e1
        if total < 1e-8:
            return 0.5
        return max(vis_e0, vis_e1) / total

    # ── Diffusion stability ───────────────────────────────────────────────

    def _diffusion_stable(self, current_mse: float) -> bool:
        """Check MSE hasn't exploded over last N epochs."""
        recent = [m.diffusion_mse for m in self.history[-STAGE3_MSE_WINDOW:]
                  if m.diffusion_mse > 0]
        if len(recent) < 2:
            return True
        return (current_mse < STAGE3_DIFF_MSE_MAX and
                current_mse < 3.0 * (sum(recent) / len(recent)))

    # ── Main compute ─────────────────────────────────────────────────────

    def compute(
        self,
        vol_outputs,
        assembler,
        gt_visible: torch.Tensor,    # (B, 2, H, W)
        val_metrics: Dict,
        epoch: int,
        stage: str,
    ) -> ContractMetrics:
        m = ContractMetrics(epoch=epoch, stage=stage)

        # ── Volume compactness
        if vol_outputs.entity_probs is not None:
            ep = vol_outputs.entity_probs  # (B, 2, K, H, W)
            b = 0
            m.vol_compactness_e0 = self._depth_compactness(ep[b, 0])
            m.vol_compactness_e1 = self._depth_compactness(ep[b, 1])
            m.vol_compactness    = min(m.vol_compactness_e0, m.vol_compactness_e1)

        # ── Two-color presence
        if vol_outputs.visible_class is not None:
            m.two_color_e0_frac, m.two_color_e1_frac, m.two_color_presence = \
                self._two_color(vol_outputs.visible_class)

        # ── Val IoU
        m.val_iou_min = float(val_metrics.get("val_iou_min", 0.0))

        # ── Guide gate
        m.gate_open, m.gate_per_block = self._gate_open(assembler)

        # ── Overlay match
        if vol_outputs.visible_class is not None:
            m.pred_overlay_match = self._overlay_match(vol_outputs.visible_class, gt_visible)

        # ── One-winner
        m.one_winner_ratio = self._one_winner(vol_outputs)

        # ── Diffusion
        m.diffusion_mse     = float(val_metrics.get("val_diff_mse", 0.0))
        m.diffusion_stable  = self._diffusion_stable(m.diffusion_mse)

        # ── Pass/fail
        m.stage1_pass = (
            m.vol_compactness    >= STAGE1_COMPACTNESS_MIN and
            m.val_iou_min        >= STAGE1_IOU_MIN         and
            m.two_color_presence
        )
        m.stage2_pass = (
            m.stage1_pass                              and
            m.gate_open          >= STAGE2_GATE_OPEN_MIN   and
            m.pred_overlay_match >= STAGE2_OVERLAY_MATCH_MIN and
            m.one_winner_ratio   <  STAGE2_ONE_WINNER_MAX
        )
        m.stage3_pass = (
            m.stage2_pass                             and
            m.gate_open          >= STAGE3_GATE_OPEN_MIN   and
            m.diffusion_stable
        )

        # ── Multi-objective checkpoint score
        gate_score        = min(1.0, m.gate_open / 0.20)
        compactness_score = min(1.0, m.vol_compactness / 0.40)
        two_color_score   = float(m.two_color_presence)
        diff_stable_score = float(m.diffusion_stable)
        m.contract_score = (
            0.30 * m.val_iou_min           +
            0.20 * two_color_score         +
            0.20 * compactness_score       +
            0.15 * gate_score              +
            0.15 * diff_stable_score
        )

        self.history.append(m)
        return m

    # ── Gate-adaptive diff weight ─────────────────────────────────────────

    def adaptive_diff_weight(
        self,
        base_diff_weight: float,
        gate_open: float,
    ) -> float:
        """
        If gate hasn't opened, slow down diffusion loss ramp.
        Prevents diffusion from dominating before guide is ready.
        """
        if gate_open < STAGE2_GATE_OPEN_MIN:
            # Gate still closed: cap diff_weight at 0.2 to give guide time to open
            return min(base_diff_weight, 0.20)
        if gate_open < 0.05:
            # Gate partially open: scale proportionally
            scale = gate_open / 0.05
            return min(base_diff_weight, 0.20 + 0.80 * scale)
        return base_diff_weight

    # ── Logging ──────────────────────────────────────────────────────────

    def log(self, m: ContractMetrics) -> None:
        s1 = "PASS" if m.stage1_pass else "FAIL"
        s2 = "PASS" if m.stage2_pass else "----"
        s3 = "PASS" if m.stage3_pass else "----"

        print(
            f"  [contract ep{m.epoch:03d}] "
            f"S1:{s1}  S2:{s2}  S3:{s3}  "
            f"compact={m.vol_compactness:.3f}  "
            f"2color={'Y' if m.two_color_presence else 'N'}({m.two_color_e0_frac:.3f}/{m.two_color_e1_frac:.3f})  "
            f"iou_min={m.val_iou_min:.3f}  "
            f"gate={m.gate_open:.4f}  "
            f"overlay={m.pred_overlay_match:.3f}  "
            f"winner={m.one_winner_ratio:.2f}  "
            f"diff_mse={m.diffusion_mse:.4f}  "
            f"contract_score={m.contract_score:.4f}",
            flush=True,
        )

    def summary(self) -> str:
        """One-line summary of contract state across all epochs."""
        if not self.history:
            return "[contract] no data"
        last = self.history[-1]
        s1_epochs = [m.epoch for m in self.history if m.stage1_pass]
        s2_epochs = [m.epoch for m in self.history if m.stage2_pass]
        s3_epochs = [m.epoch for m in self.history if m.stage3_pass]
        return (
            f"[contract summary] "
            f"S1 first_pass={s1_epochs[0] if s1_epochs else 'never'}  "
            f"S2 first_pass={s2_epochs[0] if s2_epochs else 'never'}  "
            f"S3 first_pass={s3_epochs[0] if s3_epochs else 'never'}  "
            f"last_contract_score={last.contract_score:.4f}"
        )
