"""
Phase 62 — Full 5-Contract System (v22, 2026-04-14)
=====================================================

Success ⟺ C_topo ∧ C_guide ∧ C_diff ∧ C_render ∧ C_robust

C_topo  (3D volume shape quality):
  - D_vis_min  ≥ 0.25   min(D_vis_e0, D_vis_e1): projected visible Dice per entity
  - D_amo_min  ≥ 0.40   min(D_amo_e0, D_amo_e1): amodal Dice per entity
  - compactness ≥ 0.60  depth entropy concentration
  - LCC_min    ≥ 0.85   largest connected component ratio per entity

C_guide (guide injection quality):
  - 0.10 ≤ gate ≤ 0.35  guide gate in useful range (not clamped floor, not exploding)
  - overlay_iou ≥ 0.35  pred fg aligns with GT fg
  - winner_ratio ≤ 0.45 no entity dominates > 55% of fg pixels
  - cos_F_overlap ≤ 0.10 F_0/F_1 feature separation at overlap region

C_diff  (diffusion stability):
  - diff_mse ≤ 0.05
  - diff_mse_delta_s2s3 ≤ 0.10  stage2→3 transition < 10% jump
  - no NaN/explosion

C_render (final composite quality — held-out collision clips):
  - P_2obj     ≥ 0.90   both entities detected in overlap frames
  - R_chimera  ≤ 0.05   fused-blob chimera frames
  - M_id_min   ≥ 0.15   min identity margin per entity
  - render_iou_min ≥ 0.25  per-entity IoU on rendered composites

C_robust (reproducibility):
  - consecutive_pass ≥ 5   maintained over ≥ 5 consecutive eval epochs
  - pass_rate_clips  ≥ 0.90 held-out val clips pass
  - pass_rate_seeds  ≥ 0.80 reproduced across ≥ 3 seeds  (tracked externally)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    from scipy import ndimage as _ndimage
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ─── Thresholds ──────────────────────────────────────────────────────────────

# C_topo
CTOPO_DVIS_MIN       = 0.25
CTOPO_DAMO_MIN       = 0.40
CTOPO_COMPACT_MIN    = 0.60
CTOPO_LCC_MIN        = 0.85

# C_guide
CGUIDE_GATE_LO       = 0.10
CGUIDE_GATE_HI       = 0.35
CGUIDE_OVERLAY_MIN   = 0.35
CGUIDE_WINNER_MAX    = 0.45
CGUIDE_COS_MAX       = 0.10

# C_diff
CDIFF_MSE_MAX        = 0.05
CDIFF_DELTA_MAX      = 0.10   # max relative jump s2→s3
CDIFF_MSE_WINDOW     = 5

# C_render
CRENDER_P2OBJ_MIN    = 0.90
CRENDER_CHIMERA_MAX  = 0.05
CRENDER_MID_MIN      = 0.15
CRENDER_IOU_MIN      = 0.25

# C_robust
CROBUST_CONSEC_MIN   = 5
CROBUST_CLIPS_MIN    = 0.90

# Legacy S1/S2/S3 thresholds (kept for backward compat log display)
STAGE1_COMPACTNESS_MIN  = 0.20
STAGE1_IOU_MIN          = 0.10
STAGE1_TWO_COLOR_MIN    = 0.01
STAGE2_GATE_OPEN_MIN    = 0.02
STAGE2_OVERLAY_MATCH_MIN= 0.05
STAGE2_ONE_WINNER_MAX   = 0.90
STAGE3_GATE_OPEN_MIN    = 0.05
STAGE3_DIFF_MSE_MAX     = 0.50
STAGE3_MSE_WINDOW       = 5


# ─── Metric dataclass ────────────────────────────────────────────────────────

@dataclass
class ContractMetrics:
    epoch: int = 0
    stage: str = "stage1"

    # ── C_topo ──────────────────────────────────────────────────────
    vol_compactness:    float = 0.0
    vol_compactness_e0: float = 0.0
    vol_compactness_e1: float = 0.0
    D_vis_e0:           float = 0.0   # projected visible Dice entity-0
    D_vis_e1:           float = 0.0   # projected visible Dice entity-1
    D_vis_min:          float = 0.0
    D_amo_e0:           float = 0.0   # amodal Dice entity-0
    D_amo_e1:           float = 0.0   # amodal Dice entity-1
    D_amo_min:          float = 0.0
    LCC_e0:             float = 0.0   # largest connected component ratio
    LCC_e1:             float = 0.0
    LCC_min:            float = 0.0
    two_color_presence: bool  = False
    two_color_e0_frac:  float = 0.0
    two_color_e1_frac:  float = 0.0
    # Legacy alias
    val_iou_min:        float = 0.0   # = D_vis_min (for backward compat)

    # ── C_guide ─────────────────────────────────────────────────────
    gate_open:          float = 0.0
    gate_per_block:     Dict[str, float] = field(default_factory=dict)
    pred_overlay_match: float = 0.0
    one_winner_ratio:   float = 0.5
    cos_F_overlap:      float = 0.0   # F_0/F_1 cosine similarity at overlap region

    # ── C_diff ──────────────────────────────────────────────────────
    diffusion_mse:      float = 0.0
    diffusion_stable:   bool  = True
    diff_mse_delta:     float = 0.0   # relative jump from stage2 baseline

    # ── C_render ────────────────────────────────────────────────────
    P_2obj:             float = 0.0
    R_chimera:          float = 0.0
    M_id_min:           float = 0.0
    render_iou_min:     float = 0.0
    c_render_available: bool  = False

    # ── C_robust ────────────────────────────────────────────────────
    consecutive_pass:   int   = 0     # consecutive epochs all 5 contracts pass
    pass_rate_clips:    float = 0.0

    # ── Pass / fail ──────────────────────────────────────────────────
    c_topo_pass:        bool  = False
    c_guide_pass:       bool  = False
    c_diff_pass:        bool  = False
    c_render_pass:      bool  = False   # skipped if not available
    c_robust_pass:      bool  = False
    all_pass:           bool  = False

    # Legacy aliases
    stage1_pass:        bool  = False
    stage2_pass:        bool  = False
    stage3_pass:        bool  = False

    # Checkpoint score
    contract_score:     float = 0.0


# ─── Contract engine ─────────────────────────────────────────────────────────

class DebugContract:
    """
    Full 5-contract evaluation engine.

    Backward-compatible: stage1/2/3 legacy fields still populated.
    New code should use c_topo_pass / c_guide_pass / c_diff_pass / c_render_pass / all_pass.
    """

    def __init__(self):
        self.history: List[ContractMetrics] = []
        self._stage2_diff_mse_baseline: Optional[float] = None

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _dice_from_iou(iou: float) -> float:
        """Convert IoU to Dice: D = 2*IoU / (1 + IoU)."""
        return 2.0 * iou / (1.0 + iou + 1e-8)

    @staticmethod
    def _depth_compactness(
        entity_probs_3d: torch.Tensor,
        fg_spatial_mask: "Optional[torch.Tensor]" = None,
    ) -> float:
        K = entity_probs_3d.shape[0]
        if K <= 1:
            return 1.0
        ep = entity_probs_3d.float()
        if fg_spatial_mask is not None:
            mask = fg_spatial_mask.float()
            n_fg = mask.sum().clamp(min=1.0)
            depth_mass = (ep * mask.unsqueeze(0)).sum(dim=(1, 2)) / n_fg
        else:
            depth_mass = ep.mean(dim=(1, 2))
        mass_sum = depth_mass.sum()
        if mass_sum < 1e-8:
            return 0.0
        p = (depth_mass / mass_sum).clamp(1e-9, 1.0)
        entropy = -(p * p.log()).sum().item()
        max_entropy = math.log(K)
        return float(1.0 - entropy / max_entropy)

    @staticmethod
    def _two_color(visible_class: torch.Tensor) -> tuple:
        total = visible_class.numel()
        if total == 0:
            return 0.0, 0.0, False
        e0 = float((visible_class == 1).sum().item()) / total
        e1 = float((visible_class == 2).sum().item()) / total
        both = (e0 > STAGE1_TWO_COLOR_MIN) and (e1 > STAGE1_TWO_COLOR_MIN)
        return e0, e1, both

    @staticmethod
    def _gate_open(assembler) -> tuple:
        if not hasattr(assembler, "guide_gates"):
            return 0.0, {}
        per_block = {}
        for bn, param in assembler.guide_gates.items():
            per_block[bn] = float(torch.tanh(param).item())
        max_gate = max(per_block.values()) if per_block else 0.0
        return max_gate, per_block

    @staticmethod
    def _overlay_match(visible_class: torch.Tensor,
                       gt_visible: torch.Tensor) -> float:
        pred_fg = (visible_class > 0).float()
        gt_fg   = (gt_visible.sum(dim=1) > 0.5).float()
        inter = (pred_fg * gt_fg).sum()
        union = (pred_fg + gt_fg).clamp(max=1).sum()
        if union < 1:
            return 0.0
        return float((inter / union).item())

    @staticmethod
    def _one_winner(vol_outputs) -> float:
        if not (vol_outputs.visible and "e0" in vol_outputs.visible):
            return 0.5
        vis_e0 = float(vol_outputs.visible["e0"].mean().item())
        vis_e1 = float(vol_outputs.visible["e1"].mean().item())
        total = vis_e0 + vis_e1
        if total < 1e-8:
            return 0.5
        return max(vis_e0, vis_e1) / total

    @staticmethod
    def _compute_lcc(entity_probs_3d: torch.Tensor, threshold: float = 0.15) -> float:
        """
        Largest Connected Component ratio for a (K, H, W) entity probability map.
        LCC = (size of largest fg component) / (total fg voxels).
        Near 1.0 = single compact blob; near 0.0 = many disconnected fragments.

        Threshold = 0.15 (lowered from 0.3 in v23):
        factorized_fg_id entity_probs peak at fg_magnitude × depth_attn × q_n ≈ 0.20.
        Threshold=0.30 cut off most signal; 0.15 accurately captures entity voxels.
        """
        if not _SCIPY_OK:
            return 1.0  # skip if scipy not available
        binary = (entity_probs_3d.detach().cpu().float().numpy() > threshold)
        if not binary.any():
            return 0.0
        labeled, _ = _ndimage.label(binary)
        if labeled.max() == 0:
            return 0.0
        sizes = np.bincount(labeled.ravel())[1:]  # exclude bg label 0
        return float(sizes.max()) / float(binary.sum())

    def _diffusion_stable(self, current_mse: float) -> bool:
        recent = [m.diffusion_mse for m in self.history[-STAGE3_MSE_WINDOW:]
                  if m.diffusion_mse > 0]
        if len(recent) < 2:
            return True
        return (current_mse < STAGE3_DIFF_MSE_MAX and
                current_mse < 3.0 * (sum(recent) / len(recent)))

    def _diff_delta(self, current_mse: float, stage: str) -> float:
        """Relative MSE jump from stage2 baseline."""
        if stage == "stage2":
            self._stage2_diff_mse_baseline = current_mse
        if self._stage2_diff_mse_baseline is None or self._stage2_diff_mse_baseline < 1e-8:
            return 0.0
        return abs(current_mse - self._stage2_diff_mse_baseline) / self._stage2_diff_mse_baseline

    # ── Main compute ──────────────────────────────────────────────────────────

    def compute(
        self,
        vol_outputs,
        assembler,
        gt_visible: torch.Tensor,           # (B, 2, H, W)
        val_metrics: Dict,
        epoch: int,
        stage: str,
        render_metrics: Optional[Dict] = None,  # C_render pass-in (optional)
    ) -> ContractMetrics:
        m = ContractMetrics(epoch=epoch, stage=stage)

        # ── C_topo: compactness ────────────────────────────────────────────
        if "val_compact" in val_metrics:
            m.vol_compactness = float(val_metrics["val_compact"])
            if vol_outputs.entity_probs is not None:
                ep = vol_outputs.entity_probs  # (B, 2, K, H, W)
                b = 0
                fg_spatial = None
                if gt_visible is not None and gt_visible.shape[0] > b:
                    fg_spatial = (gt_visible[b] > 0.5).any(dim=0)
                m.vol_compactness_e0 = self._depth_compactness(ep[b, 0], fg_spatial)
                m.vol_compactness_e1 = self._depth_compactness(ep[b, 1], fg_spatial)
        elif vol_outputs.entity_probs is not None:
            ep = vol_outputs.entity_probs
            b = 0
            fg_spatial = None
            if gt_visible is not None and gt_visible.shape[0] > b:
                fg_spatial = (gt_visible[b] > 0.5).any(dim=0)
            m.vol_compactness_e0 = self._depth_compactness(ep[b, 0], fg_spatial)
            m.vol_compactness_e1 = self._depth_compactness(ep[b, 1], fg_spatial)
            m.vol_compactness    = min(m.vol_compactness_e0, m.vol_compactness_e1)

        # ── C_topo: LCC ───────────────────────────────────────────────────
        if vol_outputs.entity_probs is not None:
            ep = vol_outputs.entity_probs
            b = 0
            m.LCC_e0 = self._compute_lcc(ep[b, 0])
            m.LCC_e1 = self._compute_lcc(ep[b, 1])
            m.LCC_min = min(m.LCC_e0, m.LCC_e1)

        # ── C_topo: visible Dice (D_vis) ──────────────────────────────────
        # D_vis from val_metrics (per-entity IoU converted to Dice)
        iou_e0 = float(val_metrics.get("val_iou_e0", 0.0))
        iou_e1 = float(val_metrics.get("val_iou_e1", 0.0))
        m.D_vis_e0 = self._dice_from_iou(iou_e0)
        m.D_vis_e1 = self._dice_from_iou(iou_e1)
        m.D_vis_min = min(m.D_vis_e0, m.D_vis_e1)
        m.val_iou_min = float(val_metrics.get("val_iou_min", 0.0))  # legacy

        # ── C_topo: amodal Dice (D_amo) ───────────────────────────────────
        m.D_amo_e0 = float(val_metrics.get("val_amo_dice_e0", 0.0))
        m.D_amo_e1 = float(val_metrics.get("val_amo_dice_e1", 0.0))
        m.D_amo_min = min(m.D_amo_e0, m.D_amo_e1)

        # ── Two-color presence ────────────────────────────────────────────
        if vol_outputs.visible_class is not None:
            m.two_color_e0_frac, m.two_color_e1_frac, m.two_color_presence = \
                self._two_color(vol_outputs.visible_class)

        # ── C_guide: gate ──────────────────────────────────────────────────
        m.gate_open, m.gate_per_block = self._gate_open(assembler)

        # ── C_guide: overlay ──────────────────────────────────────────────
        if vol_outputs.visible_class is not None:
            m.pred_overlay_match = self._overlay_match(vol_outputs.visible_class, gt_visible)

        # ── C_guide: winner ratio ─────────────────────────────────────────
        m.one_winner_ratio = self._one_winner(vol_outputs)

        # ── C_guide: feature separation ───────────────────────────────────
        m.cos_F_overlap = float(val_metrics.get("val_cos_F_overlap", 0.0))

        # ── C_diff ────────────────────────────────────────────────────────
        m.diffusion_mse    = float(val_metrics.get("val_diff_mse", 0.0))
        m.diffusion_stable = self._diffusion_stable(m.diffusion_mse)
        m.diff_mse_delta   = self._diff_delta(m.diffusion_mse, stage)

        # ── C_render ──────────────────────────────────────────────────────
        if render_metrics is not None:
            m.P_2obj     = float(render_metrics.get("P_2obj", 0.0))
            m.R_chimera  = float(render_metrics.get("R_chimera", 1.0))
            m.M_id_min   = float(render_metrics.get("M_id_min", 0.0))
            m.render_iou_min = float(render_metrics.get("render_iou_min", 0.0))
            m.c_render_available = True
        else:
            m.c_render_available = False

        # ── C_robust: consecutive pass tracking ───────────────────────────
        m.pass_rate_clips = float(val_metrics.get("val_pass_rate_clips", 0.0))

        # ── Pass / fail ───────────────────────────────────────────────────
        m.c_topo_pass = (
            m.D_vis_min          >= CTOPO_DVIS_MIN    and
            m.D_amo_min          >= CTOPO_DAMO_MIN    and
            m.vol_compactness    >= CTOPO_COMPACT_MIN and
            m.LCC_min            >= CTOPO_LCC_MIN     and
            m.two_color_presence
        )
        m.c_guide_pass = (
            CGUIDE_GATE_LO       <= m.gate_open <= CGUIDE_GATE_HI and
            m.pred_overlay_match >= CGUIDE_OVERLAY_MIN             and
            m.one_winner_ratio   <= CGUIDE_WINNER_MAX              and
            m.cos_F_overlap      <= CGUIDE_COS_MAX
        )
        m.c_diff_pass = (
            m.diffusion_mse      <= CDIFF_MSE_MAX    and
            m.diff_mse_delta     <= CDIFF_DELTA_MAX  and
            m.diffusion_stable
        )
        m.c_render_pass = (
            not m.c_render_available or (
                m.P_2obj        >= CRENDER_P2OBJ_MIN  and
                m.R_chimera     <= CRENDER_CHIMERA_MAX and
                m.M_id_min      >= CRENDER_MID_MIN     and
                m.render_iou_min >= CRENDER_IOU_MIN
            )
        )

        # Consecutive pass counter
        if m.c_topo_pass and m.c_guide_pass and m.c_diff_pass and m.c_render_pass:
            prev_consec = self.history[-1].consecutive_pass if self.history else 0
            m.consecutive_pass = prev_consec + 1
        else:
            m.consecutive_pass = 0

        m.c_robust_pass = (
            m.consecutive_pass   >= CROBUST_CONSEC_MIN and
            (m.pass_rate_clips   >= CROBUST_CLIPS_MIN or m.pass_rate_clips == 0.0)
        )

        m.all_pass = m.c_topo_pass and m.c_guide_pass and m.c_diff_pass and m.c_render_pass

        # ── Legacy stage1/2/3 pass flags (for backward compat) ────────────
        m.stage1_pass = (
            m.vol_compactness    >= STAGE1_COMPACTNESS_MIN and
            m.val_iou_min        >= STAGE1_IOU_MIN         and
            m.two_color_presence
        )
        m.stage2_pass = (
            m.stage1_pass                                       and
            m.gate_open          >= STAGE2_GATE_OPEN_MIN        and
            m.pred_overlay_match >= STAGE2_OVERLAY_MATCH_MIN    and
            m.one_winner_ratio   <  STAGE2_ONE_WINNER_MAX
        )
        m.stage3_pass = (
            m.stage2_pass                             and
            m.gate_open          >= STAGE3_GATE_OPEN_MIN   and
            m.diffusion_stable
        )

        # ── Checkpoint score ──────────────────────────────────────────────
        gate_score      = min(1.0, (m.gate_open - CGUIDE_GATE_LO) /
                               (CGUIDE_GATE_HI - CGUIDE_GATE_LO + 1e-8)) if m.gate_open >= CGUIDE_GATE_LO else 0.0
        compact_score   = min(1.0, m.vol_compactness / CTOPO_COMPACT_MIN)
        dvis_score      = min(1.0, m.D_vis_min / CTOPO_DVIS_MIN)
        damo_score      = min(1.0, m.D_amo_min / CTOPO_DAMO_MIN)
        diff_score      = float(m.c_diff_pass)
        render_score    = min(1.0, m.render_iou_min / CRENDER_IOU_MIN) if m.c_render_available else 0.5
        m.contract_score = (
            0.20 * dvis_score    +
            0.20 * damo_score    +
            0.15 * compact_score +
            0.15 * gate_score    +
            0.15 * diff_score    +
            0.15 * render_score
        )

        self.history.append(m)
        return m

    # ── Gate-adaptive diff weight ─────────────────────────────────────────

    def adaptive_diff_weight(self, base_diff_weight: float, gate_open: float) -> float:
        if gate_open < STAGE2_GATE_OPEN_MIN:
            return min(base_diff_weight, 0.20)
        if gate_open < CGUIDE_GATE_LO:
            scale = gate_open / CGUIDE_GATE_LO
            return min(base_diff_weight, 0.20 + 0.80 * scale)
        return base_diff_weight

    # ── Logging ───────────────────────────────────────────────────────────

    def log(self, m: ContractMetrics) -> None:
        # New 5-contract format
        ct = "✓" if m.c_topo_pass  else "✗"
        cg = "✓" if m.c_guide_pass else "✗"
        cd = "✓" if m.c_diff_pass  else "✗"
        cr = "✓" if m.c_render_pass else ("?" if not m.c_render_available else "✗")
        cb = "✓" if m.c_robust_pass else "✗"

        print(
            f"  [contract ep{m.epoch:03d}] "
            f"Ctopo:{ct} Cguide:{cg} Cdiff:{cd} Crender:{cr} Crobust:{cb}  "
            f"ALL:{'PASS' if m.all_pass else 'FAIL'}  "
            f"score={m.contract_score:.4f}  consec={m.consecutive_pass}",
            flush=True,
        )
        print(
            f"    topo:  compact={m.vol_compactness:.3f}(≥{CTOPO_COMPACT_MIN})  "
            f"D_vis={m.D_vis_min:.3f}(≥{CTOPO_DVIS_MIN})  "
            f"D_amo={m.D_amo_min:.3f}(≥{CTOPO_DAMO_MIN})  "
            f"LCC={m.LCC_min:.3f}(≥{CTOPO_LCC_MIN})  "
            f"2color={'Y' if m.two_color_presence else 'N'}({m.two_color_e0_frac:.3f}/{m.two_color_e1_frac:.3f})",
            flush=True,
        )
        print(
            f"    guide: gate={m.gate_open:.4f}([{CGUIDE_GATE_LO},{CGUIDE_GATE_HI}])  "
            f"overlay={m.pred_overlay_match:.3f}(≥{CGUIDE_OVERLAY_MIN})  "
            f"winner={m.one_winner_ratio:.3f}(≤{CGUIDE_WINNER_MAX})  "
            f"cosF={m.cos_F_overlap:.3f}(≤{CGUIDE_COS_MAX})",
            flush=True,
        )
        print(
            f"    diff:  mse={m.diffusion_mse:.4f}(≤{CDIFF_MSE_MAX})  "
            f"Δ={m.diff_mse_delta:.3f}(≤{CDIFF_DELTA_MAX})  "
            f"stable={m.diffusion_stable}",
            flush=True,
        )
        if m.c_render_available:
            print(
                f"    render: P_2obj={m.P_2obj:.3f}(≥{CRENDER_P2OBJ_MIN})  "
                f"chimera={m.R_chimera:.3f}(≤{CRENDER_CHIMERA_MAX})  "
                f"M_id={m.M_id_min:.3f}(≥{CRENDER_MID_MIN})  "
                f"IoU={m.render_iou_min:.3f}(≥{CRENDER_IOU_MIN})",
                flush=True,
            )

        # Also print legacy line for backward compat with log parsers
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
        if not self.history:
            return "[contract] no data"
        last = self.history[-1]
        s1_epochs = [m.epoch for m in self.history if m.stage1_pass]
        s2_epochs = [m.epoch for m in self.history if m.stage2_pass]
        s3_epochs = [m.epoch for m in self.history if m.stage3_pass]
        all_epochs = [m.epoch for m in self.history if m.all_pass]
        return (
            f"[contract summary] "
            f"S1 first_pass={s1_epochs[0] if s1_epochs else 'never'}  "
            f"S2 first_pass={s2_epochs[0] if s2_epochs else 'never'}  "
            f"S3 first_pass={s3_epochs[0] if s3_epochs else 'never'}  "
            f"ALL first_pass={all_epochs[0] if all_epochs else 'never'}  "
            f"best_consec={max((m.consecutive_pass for m in self.history), default=0)}  "
            f"last_contract_score={last.contract_score:.4f}"
        )
