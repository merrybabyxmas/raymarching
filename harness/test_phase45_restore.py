"""
test_phase45_restore.py — restore_multiblock_state_p45 검증

검증 항목:
  1. Phase44 checkpoint (occ_head 없음) → occ_head zero-init 유지
  2. Phase44 checkpoint (occ_head 없음) → overlap_blend_head 정상 복원
  3. Phase44 checkpoint (occ_head 없음) → slot0/1_adapter 정상 복원
  4. Phase44 checkpoint (occ_head 없음) → weight_head 정상 복원
  5. Phase45 checkpoint (occ_head 있음) → occ_head 정상 복원
  6. procs_state 개수 불일치 → RuntimeError
  7. procs_state 없음 → RuntimeError
  8. 복원 후 slot_blend 값 일치 (slot_blend_raw → sigmoid → blend)
  9. 복원 후 lora_k/v/out state 일치
 10. occ_head 없는 ckpt → sigmoid(0)=0.5 everywhere (zero-init 보장)
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
    restore_multiblock_state_p45,
    OccupancyHead,
    CROSS_ATTN_DIM,
    PRIMARY_DIM,
)
from models.entity_slot_phase44 import BLOCK_INNER_DIMS, DEFAULT_INJECT_KEYS


# =============================================================================
# Helpers
# =============================================================================

SMALL_DIM = 64   # test에서 사용하는 경량 dim

def _make_proc(query_dim=SMALL_DIM, use_occ_head=True) -> Phase45Processor:
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
        use_occ_head        = use_occ_head,
    )


def _make_manager(n_procs=2) -> MultiBlockSlotManagerP45:
    keys  = [f"up_blocks.{2-i}.attentions.0.transformer_blocks.0.attn2.processor"
             for i in range(n_procs)]
    procs = [_make_proc() for _ in range(n_procs)]
    return MultiBlockSlotManagerP45(procs, keys, primary_idx=1)


def _build_proc_state(proc: Phase45Processor, include_occ: bool = True) -> dict:
    """proc의 state_dict를 Phase45 checkpoint 형식으로 직렬화."""
    state = {
        "slot_blend_raw":     proc.slot_blend_raw.data.clone(),
        "slot0_adapter":      proc.slot0_adapter.state_dict(),
        "slot1_adapter":      proc.slot1_adapter.state_dict(),
        "blend_head":         proc.blend_head.state_dict()
                              if hasattr(proc, 'blend_head') else {},
        "lora_k":             proc.lora_k.state_dict(),
        "lora_v":             proc.lora_v.state_dict(),
        "lora_out":           proc.lora_out.state_dict(),
        "weight_head":        proc.weight_head.state_dict(),
        "ref_proj_e0":        proc.ref_proj_e0.state_dict(),
        "ref_proj_e1":        proc.ref_proj_e1.state_dict(),
        "overlap_blend_head": proc.overlap_blend_head.state_dict(),
    }
    if include_occ:
        state["occ_head_e0"] = (proc.occ_head_e0.state_dict()
                                 if proc.occ_head_e0 is not None else {})
        state["occ_head_e1"] = (proc.occ_head_e1.state_dict()
                                 if proc.occ_head_e1 is not None else {})
    return state


def _build_fake_ckpt(manager: MultiBlockSlotManagerP45,
                     include_occ: bool = True) -> dict:
    """현재 manager procs에서 가짜 checkpoint 생성."""
    return {
        "procs_state": [
            _build_proc_state(p, include_occ=include_occ)
            for p in manager.procs
        ]
    }


# =============================================================================
# restore tests
# =============================================================================

class TestRestoreFromPhase44Ckpt:
    """occ_head 없는 Phase44 checkpoint에서 복원."""

    def test_occ_head_zero_init_when_not_in_ckpt(self):
        """Phase44 ckpt (occ_head 없음) → occ_heads zero-init 유지 (sigmoid=0.5)."""
        manager = _make_manager(n_procs=2)
        # Build ckpt WITHOUT occ_head
        ckpt = _build_fake_ckpt(manager, include_occ=False)

        # Modify occ_heads to non-zero to verify restore resets them
        # (we DON'T modify; restore should keep them as-is when absent)
        restore_multiblock_state_p45(manager, ckpt)

        for proc in manager.procs:
            if proc.occ_head_e0 is not None:
                feat = torch.randn(1, 8, SMALL_DIM)
                with torch.no_grad():
                    o0 = proc.occ_head_e0(feat)
                assert o0.allclose(torch.full_like(o0, 0.5), atol=1e-5), \
                    "occ_head not in ckpt → should stay at zero-init (0.5)"

    def test_overlap_blend_head_restored(self):
        """Phase44 ckpt → overlap_blend_head state 정상 복원."""
        manager_src = _make_manager()
        manager_dst = _make_manager()

        # Modify src overlap_blend_head
        with torch.no_grad():
            manager_src.procs[0].overlap_blend_head.net[2].weight.fill_(0.42)

        ckpt = _build_fake_ckpt(manager_src, include_occ=False)
        restore_multiblock_state_p45(manager_dst, ckpt)

        w_src = manager_src.procs[0].overlap_blend_head.net[2].weight
        w_dst = manager_dst.procs[0].overlap_blend_head.net[2].weight
        assert w_dst.allclose(w_src, atol=1e-6), \
            f"overlap_blend_head not restored: src={w_src[0,0].item():.4f}, dst={w_dst[0,0].item():.4f}"

    def test_slot_adapter_restored(self):
        """Phase44 ckpt → slot0_adapter state 정상 복원."""
        manager_src = _make_manager()
        manager_dst = _make_manager()

        # Modify src slot0_adapter first layer bias
        with torch.no_grad():
            list(manager_src.procs[0].slot0_adapter.parameters())[0].fill_(0.77)

        ckpt = _build_fake_ckpt(manager_src, include_occ=False)
        restore_multiblock_state_p45(manager_dst, ckpt)

        p_src = list(manager_src.procs[0].slot0_adapter.parameters())[0]
        p_dst = list(manager_dst.procs[0].slot0_adapter.parameters())[0]
        assert p_dst.allclose(p_src, atol=1e-6), \
            "slot0_adapter not correctly restored"

    def test_weight_head_restored(self):
        """Phase44 ckpt → weight_head state 정상 복원."""
        manager_src = _make_manager()
        manager_dst = _make_manager()

        with torch.no_grad():
            list(manager_src.procs[0].weight_head.parameters())[0].fill_(0.33)

        ckpt = _build_fake_ckpt(manager_src, include_occ=False)
        restore_multiblock_state_p45(manager_dst, ckpt)

        p_src = list(manager_src.procs[0].weight_head.parameters())[0]
        p_dst = list(manager_dst.procs[0].weight_head.parameters())[0]
        assert p_dst.allclose(p_src, atol=1e-6), "weight_head not correctly restored"

    def test_slot_blend_raw_restored(self):
        """복원 후 slot_blend_raw 값 일치."""
        manager_src = _make_manager()
        manager_dst = _make_manager()

        with torch.no_grad():
            manager_src.procs[0].slot_blend_raw.data.fill_(1.5)

        ckpt = _build_fake_ckpt(manager_src, include_occ=False)
        restore_multiblock_state_p45(manager_dst, ckpt)

        sbr_src = manager_src.procs[0].slot_blend_raw.item()
        sbr_dst = manager_dst.procs[0].slot_blend_raw.item()
        assert abs(sbr_src - sbr_dst) < 1e-6, \
            f"slot_blend_raw mismatch: {sbr_src} vs {sbr_dst}"

    def test_lora_restored(self):
        """복원 후 lora_k/v/out state 일치."""
        manager_src = _make_manager()
        manager_dst = _make_manager()

        for name in ("lora_k", "lora_v", "lora_out"):
            with torch.no_grad():
                list(getattr(manager_src.procs[0], name).parameters())[0].fill_(0.11)

        ckpt = _build_fake_ckpt(manager_src, include_occ=False)
        restore_multiblock_state_p45(manager_dst, ckpt)

        for name in ("lora_k", "lora_v", "lora_out"):
            p_src = list(getattr(manager_src.procs[0], name).parameters())[0]
            p_dst = list(getattr(manager_dst.procs[0], name).parameters())[0]
            assert p_dst.allclose(p_src, atol=1e-6), f"{name} not correctly restored"


class TestRestoreFromPhase45Ckpt:
    """occ_head 있는 Phase45 checkpoint에서 복원."""

    def test_occ_head_loaded_from_phase45_ckpt(self):
        """Phase45 ckpt (occ_head 있음) → occ_head state 정상 복원."""
        manager_src = _make_manager()
        manager_dst = _make_manager()

        # Set occ_head weights to known value
        with torch.no_grad():
            manager_src.procs[0].occ_head_e0.net[2].weight.fill_(0.55)
            manager_src.procs[0].occ_head_e1.net[2].weight.fill_(-0.22)

        ckpt = _build_fake_ckpt(manager_src, include_occ=True)
        restore_multiblock_state_p45(manager_dst, ckpt)

        w_e0_src = manager_src.procs[0].occ_head_e0.net[2].weight
        w_e0_dst = manager_dst.procs[0].occ_head_e0.net[2].weight
        assert w_e0_dst.allclose(w_e0_src, atol=1e-6), \
            "occ_head_e0 not correctly restored from Phase45 ckpt"

        w_e1_src = manager_src.procs[0].occ_head_e1.net[2].weight
        w_e1_dst = manager_dst.procs[0].occ_head_e1.net[2].weight
        assert w_e1_dst.allclose(w_e1_src, atol=1e-6), \
            "occ_head_e1 not correctly restored from Phase45 ckpt"


class TestRestoreErrors:
    """에러 케이스."""

    def test_missing_procs_state_raises(self):
        """procs_state 없음 → RuntimeError."""
        manager = _make_manager()
        with pytest.raises(RuntimeError, match="procs_state"):
            restore_multiblock_state_p45(manager, {"other_key": {}})

    def test_procs_state_count_mismatch_raises(self):
        """procs_state 개수 불일치 → RuntimeError."""
        manager = _make_manager(n_procs=2)
        # Build ckpt with 3 procs
        proc = _make_proc()
        ckpt = {
            "procs_state": [
                _build_proc_state(proc, include_occ=False)
                for _ in range(3)
            ]
        }
        with pytest.raises(RuntimeError, match="불일치"):
            restore_multiblock_state_p45(manager, ckpt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
