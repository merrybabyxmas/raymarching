"""
test_phase43_restore.py — Phase 43 checkpoint 복원 테스트

검증 항목:
  1. phase42 ckpt에서 weight_head 복원 (ref_proj는 zero-init 유지)
  2. phase43 ckpt에서 weight_head + ref_proj 모두 복원
  3. phase40 ckpt에서 weight_head/ref_proj 없어도 OK (zero-init)
  4. procs_state 없는 ckpt → RuntimeError
  5. 개수 불일치 → RuntimeError
  6. MultiBlockSlotManagerP43.projector_params()가 비지 않음
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase40 import (
    DEFAULT_INJECT_KEYS, BLOCK_INNER_DIMS,
)
from models.entity_slot_phase43 import (
    Phase43Processor,
    MultiBlockSlotManagerP43,
    restore_multiblock_state_p43,
    PRIMARY_DIM,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_proc(inner_dim: int,
               adapter_rank: int = 32,
               lora_rank: int    = 2) -> Phase43Processor:
    return Phase43Processor(
        query_dim    = inner_dim,
        inner_dim    = inner_dim,
        adapter_rank = adapter_rank,
        lora_rank    = lora_rank,
        primary_dim  = PRIMARY_DIM,
        proj_hidden  = 64,
    )


def _make_manager(n_blocks: int = 3,
                  adapter_rank: int = 32,
                  lora_rank: int    = 2) -> MultiBlockSlotManagerP43:
    keys  = DEFAULT_INJECT_KEYS[:n_blocks]
    procs = [_make_proc(BLOCK_INNER_DIMS[k], adapter_rank, lora_rank) for k in keys]
    return MultiBlockSlotManagerP43(procs, keys, primary_idx=1)


def _fill_nonzero(module: nn.Module):
    for p in module.parameters():
        nn.init.uniform_(p, -0.1, 0.1)


def _make_phase42_ckpt(n_blocks: int = 3,
                       adapter_rank: int = 32,
                       lora_rank: int    = 2) -> dict:
    """Phase42 형식 ckpt: weight_head 있음, ref_proj 없음."""
    keys = DEFAULT_INJECT_KEYS[:n_blocks]
    procs_state = []
    for key in keys:
        inner_dim = BLOCK_INNER_DIMS[key]
        p = _make_proc(inner_dim, adapter_rank, lora_rank)
        _fill_nonzero(p)
        state = {
            "slot_blend_raw": p.slot_blend_raw.detach().cpu(),
            "slot0_adapter":  p.slot0_adapter.state_dict(),
            "slot1_adapter":  p.slot1_adapter.state_dict(),
            "blend_head":     p.blend_head.state_dict(),
            "lora_k":         p.lora_k.state_dict(),
            "lora_v":         p.lora_v.state_dict(),
            "lora_out":       p.lora_out.state_dict(),
            "weight_head":    p.weight_head.state_dict(),
            # ref_proj NOT present (phase42 ckpt)
        }
        procs_state.append(state)
    return {
        "inject_keys":   keys,
        "procs_state":   procs_state,
        "adapter_rank":  adapter_rank,
        "lora_rank":     lora_rank,
        "vca_state_dict": {},
    }


def _make_phase43_ckpt(n_blocks: int = 3,
                       adapter_rank: int = 32,
                       lora_rank: int    = 2) -> dict:
    """Phase43 형식 ckpt: weight_head + ref_proj 모두 있음."""
    keys = DEFAULT_INJECT_KEYS[:n_blocks]
    procs_state = []
    for key in keys:
        inner_dim = BLOCK_INNER_DIMS[key]
        p = _make_proc(inner_dim, adapter_rank, lora_rank)
        _fill_nonzero(p)
        state = {
            "slot_blend_raw": p.slot_blend_raw.detach().cpu(),
            "slot0_adapter":  p.slot0_adapter.state_dict(),
            "slot1_adapter":  p.slot1_adapter.state_dict(),
            "blend_head":     p.blend_head.state_dict(),
            "lora_k":         p.lora_k.state_dict(),
            "lora_v":         p.lora_v.state_dict(),
            "lora_out":       p.lora_out.state_dict(),
            "weight_head":    p.weight_head.state_dict(),
            "ref_proj_e0":    p.ref_proj_e0.state_dict(),
            "ref_proj_e1":    p.ref_proj_e1.state_dict(),
        }
        procs_state.append(state)
    return {
        "inject_keys":   keys,
        "procs_state":   procs_state,
        "adapter_rank":  adapter_rank,
        "lora_rank":     lora_rank,
        "vca_state_dict": {},
    }


def _make_phase40_ckpt(n_blocks: int = 3,
                       adapter_rank: int = 32,
                       lora_rank: int    = 2) -> dict:
    """Phase40 형식 ckpt: weight_head/ref_proj 없음."""
    keys = DEFAULT_INJECT_KEYS[:n_blocks]
    procs_state = []
    for key in keys:
        inner_dim = BLOCK_INNER_DIMS[key]
        p = _make_proc(inner_dim, adapter_rank, lora_rank)
        _fill_nonzero(p)
        state = {
            "slot_blend_raw": p.slot_blend_raw.detach().cpu(),
            "slot0_adapter":  p.slot0_adapter.state_dict(),
            "slot1_adapter":  p.slot1_adapter.state_dict(),
            "blend_head":     p.blend_head.state_dict(),
            "lora_k":         p.lora_k.state_dict(),
            "lora_v":         p.lora_v.state_dict(),
            "lora_out":       p.lora_out.state_dict(),
            # weight_head NOT present (phase40 ckpt)
        }
        procs_state.append(state)
    return {
        "inject_keys":   keys,
        "procs_state":   procs_state,
        "adapter_rank":  adapter_rank,
        "lora_rank":     lora_rank,
        "vca_state_dict": {},
    }


# =============================================================================
# Tests
# =============================================================================

class TestRestoreFromPhase42Ckpt:

    def test_adapters_restored(self):
        """phase42 ckpt에서 slot0_adapter 복원."""
        manager = _make_manager()
        ckpt    = _make_phase42_ckpt()

        for proc in manager.procs:
            nn.init.zeros_(proc.slot0_adapter.up.weight)

        restore_multiblock_state_p43(manager, ckpt, device="cpu")

        for i, proc in enumerate(manager.procs):
            norm = proc.slot0_adapter.up.weight.norm().item()
            assert norm != 0.0, f"block[{i}] slot0_adapter not restored"

    def test_weight_head_restored(self):
        """phase42 ckpt: weight_head 복원 (non-zero)."""
        manager = _make_manager()
        ckpt    = _make_phase42_ckpt()

        # Set ckpt weight_head to non-zero
        for state in ckpt["procs_state"]:
            for k, v in state["weight_head"].items():
                if isinstance(v, torch.Tensor) and v.numel() > 1:
                    state["weight_head"][k] = torch.ones_like(v) * 0.5

        restore_multiblock_state_p43(manager, ckpt, device="cpu")

        for i, proc in enumerate(manager.procs):
            # weight_head last layer (net[2])
            wh_norm = proc.weight_head.net[2].weight.norm().item()
            assert wh_norm > 0.0, \
                f"block[{i}] weight_head not restored from phase42 ckpt"

    def test_ref_proj_stays_zero_init_from_phase42_ckpt(self):
        """phase42 ckpt에 ref_proj 없음 → zero-init 유지."""
        manager = _make_manager()
        ckpt    = _make_phase42_ckpt()

        restore_multiblock_state_p43(manager, ckpt, device="cpu")

        for i, proc in enumerate(manager.procs):
            # ref_proj_e0 last layer should be zero (zero-init preserved)
            norm_e0 = proc.ref_proj_e0.net[-1].weight.norm().item()
            norm_e1 = proc.ref_proj_e1.net[-1].weight.norm().item()
            assert norm_e0 == 0.0, \
                f"block[{i}] ref_proj_e0 should remain zero-init, got norm={norm_e0:.6f}"
            assert norm_e1 == 0.0, \
                f"block[{i}] ref_proj_e1 should remain zero-init, got norm={norm_e1:.6f}"


class TestRestoreFromPhase43Ckpt:

    def test_ref_proj_e0_restored(self):
        """phase43 ckpt: ref_proj_e0 복원."""
        manager = _make_manager()
        ckpt    = _make_phase43_ckpt()

        # Set ckpt ref_proj to a known value
        for state in ckpt["procs_state"]:
            for k, v in state["ref_proj_e0"].items():
                if isinstance(v, torch.Tensor) and v.numel() > 1:
                    state["ref_proj_e0"][k] = torch.ones_like(v) * 0.3

        restore_multiblock_state_p43(manager, ckpt, device="cpu")

        for i, proc in enumerate(manager.procs):
            norm = proc.ref_proj_e0.net[-1].weight.norm().item()
            assert norm > 0.0, \
                f"block[{i}] ref_proj_e0 not restored from phase43 ckpt"

    def test_ref_proj_e1_restored(self):
        """phase43 ckpt: ref_proj_e1 복원."""
        manager = _make_manager()
        ckpt    = _make_phase43_ckpt()

        for state in ckpt["procs_state"]:
            for k, v in state["ref_proj_e1"].items():
                if isinstance(v, torch.Tensor) and v.numel() > 1:
                    state["ref_proj_e1"][k] = torch.ones_like(v) * 0.7

        restore_multiblock_state_p43(manager, ckpt, device="cpu")

        for i, proc in enumerate(manager.procs):
            norm = proc.ref_proj_e1.net[-1].weight.norm().item()
            assert norm > 0.0, \
                f"block[{i}] ref_proj_e1 not restored from phase43 ckpt"


class TestRestoreFromPhase40Ckpt:

    def test_adapters_restored_no_weight_head_no_proj(self):
        """phase40 ckpt: adapters 복원, weight_head/ref_proj zero-init."""
        manager = _make_manager()
        ckpt    = _make_phase40_ckpt()

        for proc in manager.procs:
            nn.init.zeros_(proc.slot0_adapter.up.weight)

        restore_multiblock_state_p43(manager, ckpt, device="cpu")

        for i, proc in enumerate(manager.procs):
            # adapters restored
            norm_adapter = proc.slot0_adapter.up.weight.norm().item()
            assert norm_adapter != 0.0, f"block[{i}] adapter not restored"
            # weight_head stays zero
            norm_wh = proc.weight_head.net[2].weight.norm().item()
            assert norm_wh == 0.0, f"block[{i}] weight_head should be zero-init"
            # ref_proj stays zero
            norm_proj = proc.ref_proj_e0.net[-1].weight.norm().item()
            assert norm_proj == 0.0, f"block[{i}] ref_proj_e0 should be zero-init"


class TestRestoreErrors:

    def test_missing_procs_state_raises(self):
        """procs_state 없는 ckpt → RuntimeError."""
        manager  = _make_manager()
        bad_ckpt = {"vca_state_dict": {}}
        with pytest.raises(RuntimeError, match="procs_state"):
            restore_multiblock_state_p43(manager, bad_ckpt)

    def test_block_count_mismatch_raises(self):
        """procs 개수 불일치 → RuntimeError."""
        manager = _make_manager(n_blocks=3)
        ckpt    = _make_phase42_ckpt(n_blocks=2)
        with pytest.raises(RuntimeError, match="불일치"):
            restore_multiblock_state_p43(manager, ckpt)


class TestMultiBlockSlotManagerP43:

    def test_projector_params_non_empty(self):
        """MultiBlockSlotManagerP43.projector_params()가 비지 않음."""
        manager = _make_manager()
        params  = manager.projector_params()
        assert len(params) > 0, "projector_params() is empty"

    def test_projector_params_trainable(self):
        """projector params requires_grad=True."""
        manager = _make_manager()
        for p in manager.projector_params():
            assert p.requires_grad, "projector param requires_grad=False"

    def test_projector_params_distinct_from_adapter_lora(self):
        """projector_params ∩ adapter_params = ∅."""
        manager = _make_manager()
        proj_ids    = {id(p) for p in manager.projector_params()}
        adapter_ids = {id(p) for p in manager.adapter_params()}
        lora_ids    = {id(p) for p in manager.lora_params()}
        overlap_a = proj_ids & adapter_ids
        overlap_l = proj_ids & lora_ids
        assert len(overlap_a) == 0, f"projector/adapter overlap: {overlap_a}"
        assert len(overlap_l) == 0, f"projector/lora overlap: {overlap_l}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
