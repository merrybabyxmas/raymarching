"""
test_phase45_stage_freeze.py — Stage A/B 동결/해제 + val_score 검증

검증 항목 (Stage A freeze):
  1. Stage A: occ_head_e0 파라미터 requires_grad = True
  2. Stage A: occ_head_e1 파라미터 requires_grad = True
  3. Stage A: overlap_blend_head 파라미터 requires_grad = True
  4. Stage A: weight_head 파라미터 requires_grad = True
  5. Stage A: slot_blend_raw requires_grad = True
  6. Stage A: slot0_adapter/slot1_adapter frozen
  7. Stage A: lora_k/v/out frozen
  8. Stage A: ref_proj_e0/e1 frozen

검증 항목 (Stage B):
  9. Stage B: 전체 파라미터 requires_grad = True
 10. Stage A → Stage B 전환 후 occ_heads 여전히 학습 가능

검증 항목 (occupancy_head_params):
 11. occupancy_head_params()가 occ_head_e0/e1 파라미터만 반환
 12. 파라미터 수 일치

검증 항목 (val_score_phase45):
 13. 완벽 점수 → 1.0
 14. 최저 점수 → 0.0
 15. blend_sep=-0.15 → blend_score=0
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase45 import (
    Phase45Processor,
    MultiBlockSlotManagerP45,
    CROSS_ATTN_DIM,
    PRIMARY_DIM,
    val_score_phase45,
)


# =============================================================================
# Helpers
# =============================================================================

SMALL_DIM = 64

def _make_proc(query_dim=SMALL_DIM) -> Phase45Processor:
    return Phase45Processor(
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
        occ_hidden          = 32,
        use_occ_head        = True,
    )


def _make_manager(n_procs=2) -> MultiBlockSlotManagerP45:
    keys  = [f"up_blocks.{2-i}.attentions.0.transformer_blocks.0.attn2.processor"
             for i in range(n_procs)]
    procs = [_make_proc() for _ in range(n_procs)]
    return MultiBlockSlotManagerP45(procs, keys, primary_idx=1)


# Mimic train_phase45._set_requires_grad_stage_a
def _stage_a_freeze(procs):
    """Stage A: 전체 동결 후 occ_heads + OBH + weight_head + slot_blend_raw 해제."""
    for proc in procs:
        for p in proc.parameters():
            p.requires_grad_(False)
    for proc in procs:
        if hasattr(proc, 'occ_head_e0') and proc.occ_head_e0 is not None:
            for p in proc.occ_head_e0.parameters():
                p.requires_grad_(True)
        if hasattr(proc, 'occ_head_e1') and proc.occ_head_e1 is not None:
            for p in proc.occ_head_e1.parameters():
                p.requires_grad_(True)
        for p in proc.overlap_blend_head.parameters():
            p.requires_grad_(True)
        for p in proc.weight_head.parameters():
            p.requires_grad_(True)
        proc.slot_blend_raw.requires_grad_(True)


# Mimic train_phase45._set_requires_grad_stage_b
def _stage_b_unfreeze(procs):
    """Stage B: 전체 해제."""
    for proc in procs:
        for p in proc.parameters():
            p.requires_grad_(True)


# =============================================================================
# Stage A freeze tests
# =============================================================================

class TestStageAFreeze:

    def setup_method(self):
        self.manager = _make_manager(n_procs=2)
        _stage_a_freeze(self.manager.procs)
        self.proc = self.manager.procs[0]

    def test_occ_head_e0_trainable(self):
        """Stage A: occ_head_e0 파라미터 학습 가능."""
        assert self.proc.occ_head_e0 is not None
        for p in self.proc.occ_head_e0.parameters():
            assert p.requires_grad, "occ_head_e0 should be trainable in Stage A"

    def test_occ_head_e1_trainable(self):
        """Stage A: occ_head_e1 파라미터 학습 가능."""
        assert self.proc.occ_head_e1 is not None
        for p in self.proc.occ_head_e1.parameters():
            assert p.requires_grad, "occ_head_e1 should be trainable in Stage A"

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
        """Stage A: slot0_adapter/slot1_adapter 동결."""
        for name in ("slot0_adapter", "slot1_adapter"):
            mod = getattr(self.proc, name)
            for p in mod.parameters():
                assert not p.requires_grad, \
                    f"{name} should be frozen in Stage A"

    def test_lora_frozen(self):
        """Stage A: lora_k/v/out 동결."""
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
        manager = _make_manager()
        _stage_a_freeze(manager.procs)
        _stage_b_unfreeze(manager.procs)

        for proc in manager.procs:
            all_trainable = all(p.requires_grad for p in proc.parameters())
            assert all_trainable, "Stage B should unfreeze all parameters"

    def test_occ_heads_still_trainable_after_stage_b(self):
        """Stage A → Stage B 전환 후 occ_heads 여전히 학습 가능."""
        manager = _make_manager()
        _stage_a_freeze(manager.procs)
        _stage_b_unfreeze(manager.procs)

        for proc in manager.procs:
            for head in (proc.occ_head_e0, proc.occ_head_e1):
                if head is not None:
                    for p in head.parameters():
                        assert p.requires_grad, \
                            "occ_heads should remain trainable after Stage B"


# =============================================================================
# occupancy_head_params tests
# =============================================================================

class TestOccupancyHeadParams:

    def test_returns_correct_param_count(self):
        """occupancy_head_params()가 occ_head_e0/e1 파라미터만 반환."""
        manager = _make_manager(n_procs=2)

        # Manually count occ_head params
        expected_count = 0
        for proc in manager.procs:
            for head in (proc.occ_head_e0, proc.occ_head_e1):
                if head is not None:
                    expected_count += sum(1 for _ in head.parameters())

        actual_params = manager.occupancy_head_params()
        assert len(actual_params) == expected_count, \
            f"expected {expected_count} params, got {len(actual_params)}"

    def test_occ_params_not_in_other_groups(self):
        """occupancy_head_params가 adapter/lora 파라미터와 겹치지 않음."""
        manager = _make_manager(n_procs=1)
        proc = manager.procs[0]

        occ_param_ids = {id(p) for p in manager.occupancy_head_params()}

        # Check no overlap with adapter params
        adapter_param_ids = set()
        for name in ("slot0_adapter", "slot1_adapter"):
            for p in getattr(proc, name).parameters():
                adapter_param_ids.add(id(p))

        overlap = occ_param_ids & adapter_param_ids
        assert len(overlap) == 0, \
            "occupancy_head_params should not contain adapter parameters"

    def test_occupancy_params_have_grad_after_stage_a(self):
        """Stage A 설정 후 occ params에 gradient 흐름 가능."""
        manager = _make_manager()
        _stage_a_freeze(manager.procs)

        occ_params = manager.occupancy_head_params()
        assert all(p.requires_grad for p in occ_params), \
            "all occupancy_head_params should require grad after Stage A setup"


# =============================================================================
# val_score_phase45 tests
# =============================================================================

class TestValScorePhase45:
    """val_score_phase45는 val_score_phase44 alias — 동일 검증."""

    def test_perfect_score(self):
        """완벽 → 1.0."""
        s = val_score_phase45(1.0, 1.0, 1.0, 0.0, 1.0, 1.0, blend_sep=0.15)
        assert abs(s - 1.0) < 1e-6, f"perfect: expected 1.0, got {s:.6f}"

    def test_worst_score(self):
        """최저 → 0.0."""
        s = val_score_phase45(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, blend_sep=-0.15)
        assert abs(s - 0.0) < 1e-6, f"worst: expected 0.0, got {s:.6f}"

    def test_blend_sep_minus015_gives_blend_score_zero(self):
        """blend_sep = -0.15 → blend_score = 0."""
        s = val_score_phase45(0.5, 0.5, 0.7, 0.1, 0.3, 0.3, blend_sep=-0.15)
        expected = 0.15*0.5 + 0.15*0.5 + 0.10*0.7 + 0.10*(1-0.1) + 0.20*0.3 + 0.20*0.3
        assert abs(s - expected) < 1e-5, \
            f"blend_sep=-0.15: expected {expected:.5f}, got {s:.5f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
