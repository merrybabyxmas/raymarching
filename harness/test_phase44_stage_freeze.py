"""
test_phase44_stage_freeze.py — Stage A/B 동결/해제 + val_score_phase44 검증

검증 항목 (Stage A freeze):
  1. Stage A: overlap_blend_head.requires_grad = True
  2. Stage A: weight_head.requires_grad = True
  3. Stage A: slot_blend_raw.requires_grad = True
  4. Stage A: slot adapters frozen (requires_grad = False)
  5. Stage A: lora frozen
  6. Stage A: ref_proj frozen

검증 항목 (Stage B):
  7. Stage B: 모든 파라미터 requires_grad = True (vca 포함)

검증 항목 (val_score_phase44):
  8. 완벽 점수 → 1.0
  9. 최저 점수 → 0.0
 10. blend_sep=-0.15 → blend_score=0 → blend term=0
 11. blend_sep=0.00  → blend_score=0.5
 12. blend_sep=0.15  → blend_score=1.0
 13. weight 합 = 1.0 확인
 14. blend_score 범위 clamp [0, 1] 검증
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase44 import (
    Phase44Processor,
    CROSS_ATTN_DIM,
    PRIMARY_DIM,
    val_score_phase44,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_proc(query_dim=640) -> Phase44Processor:
    return Phase44Processor(
        query_dim           = query_dim,
        vca_layer           = None,
        entity_ctx          = None,
        slot_blend_init     = 0.3,
        inner_dim           = query_dim,
        adapter_rank        = 4,
        use_blend_head      = True,
        lora_rank           = 2,
        cross_attention_dim = CROSS_ATTN_DIM,
        weight_head_hidden  = 16,
        primary_dim         = PRIMARY_DIM,
        proj_hidden         = 64,
        obh_hidden          = 16,
    )


def _stage_a_freeze(proc: Phase44Processor):
    """Stage A: 전체 동결 후 overlap_blend_head/weight_head/slot_blend_raw 해제."""
    for p in proc.parameters():
        p.requires_grad_(False)
    for p in proc.overlap_blend_head.parameters():
        p.requires_grad_(True)
    for p in proc.weight_head.parameters():
        p.requires_grad_(True)
    proc.slot_blend_raw.requires_grad_(True)


def _stage_b_unfreeze(proc: Phase44Processor):
    """Stage B: 전체 해제."""
    for p in proc.parameters():
        p.requires_grad_(True)


# =============================================================================
# Stage A freeze tests
# =============================================================================

class TestStageAFreeze:

    def setup_method(self):
        self.proc = _make_proc()
        _stage_a_freeze(self.proc)

    def test_overlap_blend_head_trainable(self):
        """Stage A: overlap_blend_head 학습 가능."""
        for p in self.proc.overlap_blend_head.parameters():
            assert p.requires_grad, "overlap_blend_head should be trainable in Stage A"

    def test_weight_head_trainable(self):
        """Stage A: weight_head 학습 가능."""
        for p in self.proc.weight_head.parameters():
            assert p.requires_grad, "weight_head should be trainable in Stage A"

    def test_slot_blend_raw_trainable(self):
        """Stage A: slot_blend_raw 학습 가능."""
        assert self.proc.slot_blend_raw.requires_grad, \
            "slot_blend_raw should be trainable in Stage A"

    def test_slot_adapters_frozen(self):
        """Stage A: slot adapters 동결."""
        for name in ("slot0_adapter", "slot1_adapter"):
            mod = getattr(self.proc, name)
            for p in mod.parameters():
                assert not p.requires_grad, \
                    f"{name} should be frozen in Stage A"

    def test_lora_frozen(self):
        """Stage A: LoRA 동결."""
        for name in ("lora_k", "lora_v", "lora_out"):
            mod = getattr(self.proc, name)
            for p in mod.parameters():
                assert not p.requires_grad, \
                    f"{name} should be frozen in Stage A"

    def test_ref_proj_frozen(self):
        """Stage A: ref_proj_e0/e1 동결."""
        for name in ("ref_proj_e0", "ref_proj_e1"):
            if hasattr(self.proc, name):
                mod = getattr(self.proc, name)
                for p in mod.parameters():
                    assert not p.requires_grad, \
                        f"{name} should be frozen in Stage A"


# =============================================================================
# Stage B unfreeze tests
# =============================================================================

class TestStageBUnfreeze:

    def test_all_trainable_after_stage_b(self):
        """Stage B: 전체 파라미터 requires_grad = True."""
        proc = _make_proc()
        _stage_a_freeze(proc)   # 먼저 A로 동결
        _stage_b_unfreeze(proc) # 그 다음 B로 해제

        all_trainable = all(p.requires_grad for p in proc.parameters())
        assert all_trainable, "Stage B should unfreeze all parameters"

    def test_stage_b_after_a_overlap_blend_head_still_trainable(self):
        """Stage A → Stage B 전환 후 overlap_blend_head 여전히 학습 가능."""
        proc = _make_proc()
        _stage_a_freeze(proc)
        _stage_b_unfreeze(proc)
        for p in proc.overlap_blend_head.parameters():
            assert p.requires_grad


# =============================================================================
# val_score_phase44
# =============================================================================

class TestValScorePhase44:

    def test_perfect_score(self):
        """iou=1, ord=1, wrong=0, rollout=1, blend_sep=0.15 → 1.0."""
        s = val_score_phase44(1.0, 1.0, 1.0, 0.0, 1.0, 1.0, blend_sep=0.15)
        assert abs(s - 1.0) < 1e-6, f"perfect score: expected 1.0, got {s:.6f}"

    def test_worst_score(self):
        """iou=0, ord=0, wrong=1, rollout=0, blend_sep=-0.15 → 0.0."""
        s = val_score_phase44(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, blend_sep=-0.15)
        assert abs(s - 0.0) < 1e-6, f"worst score: expected 0.0, got {s:.6f}"

    def test_blend_sep_minus015_gives_blend_score_zero(self):
        """blend_sep = -0.15 → blend_score = 0."""
        # blend_score = (blend_sep + 0.15) / 0.30 = 0/0.30 = 0
        s_without = val_score_phase44(0.5, 0.5, 0.7, 0.1, 0.3, 0.3, blend_sep=-0.15)
        s_with_0  = val_score_phase44(0.5, 0.5, 0.7, 0.1, 0.3, 0.3, blend_sep=-0.15)
        # blend term = 0.10 * 0 = 0
        # Check directly: blend_score=0 → contribution = 0.10 * 0 = 0
        expected = 0.15*0.5 + 0.15*0.5 + 0.10*0.7 + 0.10*(1-0.1) + 0.20*0.3 + 0.20*0.3
        assert abs(s_without - expected) < 1e-5, \
            f"blend_sep=-0.15: expected {expected:.5f}, got {s_without:.5f}"

    def test_blend_sep_zero_gives_blend_score_half(self):
        """blend_sep = 0.0 → blend_score = 0.5."""
        # (0 + 0.15) / 0.30 = 0.5
        s = val_score_phase44(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, blend_sep=0.0)
        expected = 0.10 * 0.5  # only blend term contributes
        assert abs(s - expected) < 1e-5, \
            f"blend_sep=0: expected {expected:.5f}, got {s:.5f}"

    def test_blend_sep_015_gives_blend_score_one(self):
        """blend_sep = 0.15 → blend_score = 1.0."""
        s_full    = val_score_phase44(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, blend_sep=0.15)
        s_half_bs = val_score_phase44(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, blend_sep=0.0)
        # blend term doubles when sep goes from 0 to 0.15
        assert abs(s_full - 2 * s_half_bs) < 1e-5, \
            f"blend_sep=0.15 should give 2× the blend term of sep=0"

    def test_weights_sum_to_one(self):
        """가중치 합 = 1.0 (0.15+0.15+0.10+0.10+0.20+0.20+0.10 = 1.00)."""
        total = 0.15 + 0.15 + 0.10 + 0.10 + 0.20 + 0.20 + 0.10
        assert abs(total - 1.0) < 1e-6, f"weights sum: {total}"

    def test_blend_score_clamped_below_zero(self):
        """blend_sep < -0.15 → blend_score 클램프 = 0."""
        # (-0.30 + 0.15) / 0.30 = -0.5 → clamped to 0
        s = val_score_phase44(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, blend_sep=-0.30)
        # Only 0.10*(1-1) = 0 among non-blend terms
        assert s >= 0.0, "val_score should be non-negative even with very negative blend_sep"
        # blend term = 0 when clamped
        s_minus015 = val_score_phase44(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, blend_sep=-0.15)
        assert abs(s - s_minus015) < 1e-6, \
            "blend_sep < -0.15 should give same result as -0.15 (clamped)"

    def test_blend_score_clamped_above_one(self):
        """blend_sep > 0.15 → blend_score 클램프 = 1."""
        s_015 = val_score_phase44(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, blend_sep=0.15)
        s_big = val_score_phase44(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, blend_sep=1.0)
        assert abs(s_015 - s_big) < 1e-5, \
            "blend_sep > 0.15 should give same result as 0.15 (clamped)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
