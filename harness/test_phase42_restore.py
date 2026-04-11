"""
test_phase42_restore.py — Phase 42 checkpoint 복원 테스트

검증 항목:
  1. restore_multiblock_state가 모든 block의 adapter/lora norm을 0이 아니게 복원하는지
  2. procs_state 불일치 시 즉시 RuntimeError 발생하는지
  3. procs_state 없는 checkpoint에서 RuntimeError 발생하는지
  4. primary-only 복원(phase41 버그)이 secondary block을 랜덤 init으로 방치하는지 검증
"""
import sys
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase40 import (
    Phase40Processor, MultiBlockSlotManager, DEFAULT_INJECT_KEYS, BLOCK_INNER_DIMS,
)
from models.entity_slot_phase42 import (
    Phase42Processor, MultiBlockSlotManagerP42, restore_multiblock_state,
)


# =============================================================================
# Helpers: 합성 checkpoint 생성 (실제 파이프라인 없이)
# =============================================================================

def _make_proc(inner_dim: int, adapter_rank: int = 64, lora_rank: int = 4) -> Phase42Processor:
    return Phase42Processor(
        query_dim    = inner_dim,
        vca_layer    = None,
        entity_ctx   = None,
        inner_dim    = inner_dim,
        adapter_rank = adapter_rank,
        lora_rank    = lora_rank,
        use_blend_head = True,
    )


def _make_fake_ckpt(n_blocks: int = 3, adapter_rank: int = 64, lora_rank: int = 4) -> dict:
    """phase40 형식의 가짜 checkpoint 생성 (non-zero weights)."""
    inject_keys = DEFAULT_INJECT_KEYS[:n_blocks]
    procs_state = []
    for key in inject_keys:
        inner_dim = BLOCK_INNER_DIMS[key]
        p = _make_proc(inner_dim, adapter_rank, lora_rank)
        # fill with non-zero values to detect if restored
        for param in p.parameters():
            nn.init.uniform_(param, -0.1, 0.1)
        procs_state.append({
            "slot_blend_raw": p.slot_blend_raw.detach().cpu(),
            "slot0_adapter":  p.slot0_adapter.state_dict(),
            "slot1_adapter":  p.slot1_adapter.state_dict(),
            "blend_head":     p.blend_head.state_dict(),
            "lora_k":         p.lora_k.state_dict(),
            "lora_v":         p.lora_v.state_dict(),
            "lora_out":       p.lora_out.state_dict(),
        })
    return {
        "inject_keys": inject_keys,
        "procs_state": procs_state,
        "adapter_rank": adapter_rank,
        "lora_rank": lora_rank,
        "vca_state_dict": {},
    }


def _make_manager(n_blocks: int = 3,
                  adapter_rank: int = 64,
                  lora_rank: int = 4) -> MultiBlockSlotManagerP42:
    inject_keys = DEFAULT_INJECT_KEYS[:n_blocks]
    procs = []
    for key in inject_keys:
        inner_dim = BLOCK_INNER_DIMS[key]
        procs.append(_make_proc(inner_dim, adapter_rank, lora_rank))
    return MultiBlockSlotManagerP42(procs, inject_keys, primary_idx=1)


# =============================================================================
# Tests
# =============================================================================

class TestRestoreAllBlocks:

    def test_all_blocks_restored_non_zero(self):
        """복원 후 모든 block의 adapter/lora norm이 0이 아닌지."""
        manager = _make_manager()
        ckpt    = _make_fake_ckpt()

        # 초기화: lora_B는 zero-init이 맞으므로 slot0_adapter의 up.weight를 확인
        for proc in manager.procs:
            nn.init.zeros_(proc.slot0_adapter.up.weight)

        restore_multiblock_state(manager, ckpt, device="cpu", strict=False)

        for i, proc in enumerate(manager.procs):
            # slot0_adapter.up.weight가 0이면 복원 안 된 것
            norm = proc.slot0_adapter.up.weight.norm().item()
            assert norm != 0.0, \
                f"block[{i}] slot0_adapter.up.weight는 0 — 복원 실패"

    def test_lora_k_restored_all_blocks(self):
        """lora_k.lora_A.weight가 모든 block에 복원되는지."""
        manager = _make_manager()
        ckpt    = _make_fake_ckpt()

        for proc in manager.procs:
            nn.init.zeros_(proc.lora_k.lora_A.weight)

        restore_multiblock_state(manager, ckpt, device="cpu", strict=False)

        for i, proc in enumerate(manager.procs):
            norm = proc.lora_k.lora_A.weight.norm().item()
            assert norm != 0.0, f"block[{i}] lora_k.lora_A.weight는 0"

    def test_slot_blend_raw_restored(self):
        """slot_blend_raw scalar이 올바르게 복원되는지."""
        manager = _make_manager()
        ckpt    = _make_fake_ckpt()

        # ckpt의 slot_blend_raw 값을 알 수 있게 특정 값으로 설정
        for i, state in enumerate(ckpt["procs_state"]):
            state["slot_blend_raw"] = torch.tensor(float(i + 0.5))

        restore_multiblock_state(manager, ckpt, device="cpu", strict=False)

        for i, proc in enumerate(manager.procs):
            expected = float(i + 0.5)
            actual   = proc.slot_blend_raw.item()
            assert abs(actual - expected) < 1e-5, \
                f"block[{i}] slot_blend_raw: expected={expected:.4f}, actual={actual:.4f}"

    def test_secondary_blocks_restored_different_from_fresh(self):
        """
        복원 전 secondary block과 복원 후 secondary block이 다른지.
        (phase41 버그: secondary가 랜덤 init으로 방치됨을 검출)
        """
        manager = _make_manager()
        ckpt    = _make_fake_ckpt()

        # 복원 전 secondary block weight 스냅샷
        sec_before = manager.procs[0].slot0_adapter.down.weight.clone()

        # ckpt에 특정 값 주입
        for state in ckpt["procs_state"]:
            # slot0_adapter.down.weight에 큰 값 설정
            d = state["slot0_adapter"]
            for k in d:
                if isinstance(d[k], torch.Tensor) and d[k].numel() > 1:
                    d[k] = torch.ones_like(d[k]) * 2.0
            state["slot0_adapter"] = d

        restore_multiblock_state(manager, ckpt, device="cpu", strict=False)

        sec_after = manager.procs[0].slot0_adapter.down.weight
        # 복원 후에는 2.0으로 채워져야 함
        assert (sec_after == 2.0).all(), \
            "secondary block[0] slot0_adapter.down.weight가 올바르게 복원되지 않음"

    def test_weight_head_stays_zero_init(self):
        """
        WeightHead는 phase40 ckpt에 없으므로 복원 후에도 zero-init 상태여야 함.
        """
        manager = _make_manager()
        ckpt    = _make_fake_ckpt()  # weight_head 없음

        restore_multiblock_state(manager, ckpt, device="cpu", strict=False)

        for i, proc in enumerate(manager.procs):
            wh_last = proc.weight_head.net[2]
            assert wh_last.weight.norm().item() == 0.0, \
                f"block[{i}] weight_head last layer가 zero-init이 아님"
            assert wh_last.bias.norm().item() == 0.0, \
                f"block[{i}] weight_head last bias가 zero-init이 아님"


class TestRestoreErrors:

    def test_missing_procs_state_raises(self):
        """procs_state 없는 ckpt → RuntimeError."""
        manager = _make_manager()
        bad_ckpt = {"vca_state_dict": {}}  # procs_state 없음

        with pytest.raises(RuntimeError, match="procs_state"):
            restore_multiblock_state(manager, bad_ckpt)

    def test_block_count_mismatch_raises(self):
        """procs_state 개수 불일치 → RuntimeError."""
        manager = _make_manager(n_blocks=3)
        ckpt    = _make_fake_ckpt(n_blocks=2)  # 2 blocks, manager expects 3

        with pytest.raises(RuntimeError, match="불일치"):
            restore_multiblock_state(manager, ckpt)

    def test_missing_module_key_raises(self):
        """procs_state에 필수 key 없음 → RuntimeError."""
        manager = _make_manager()
        ckpt    = _make_fake_ckpt()
        del ckpt["procs_state"][0]["lora_v"]  # 필수 key 삭제

        with pytest.raises(RuntimeError, match="lora_v"):
            restore_multiblock_state(manager, ckpt, strict=True)


class TestPhase41Bug:
    """
    Phase41 버그 재현: primary-only 복원이 secondary blocks를 random init으로 방치.
    """

    def test_primary_only_restore_leaves_secondary_random(self):
        """
        Phase41 방식으로 primary만 복원하면 secondary block은 랜덤 init 상태.
        이 테스트는 버그를 문서화한다 (pass가 버그 존재, fail이 수정됨 의미).
        """
        manager = _make_manager()
        ckpt    = _make_fake_ckpt()

        # Phase41 버그 방식: primary만 복원
        primary = manager.primary
        for mod_name in ("slot0_adapter", "slot1_adapter", "blend_head"):
            if mod_name in ckpt:
                getattr(primary, mod_name).load_state_dict(ckpt[mod_name], strict=False)
        # secondary block[0]과 block[2]는 복원 안 됨

        # 검증: secondary block이 ckpt와 다른지 (복원 안 됨 = 버그 확인)
        sec_proc = manager.procs[0]  # index 0 = secondary
        sec_state = ckpt["procs_state"][0]["slot0_adapter"]

        # secondary가 올바르게 복원됐으면 이 assertion이 실패해야 함
        # (이 테스트 자체는 phase42 restore_multiblock_state를 쓰지 않아 버그 상태)
        up_key = [k for k in sec_state if "up" in k and "weight" in k]
        if up_key:
            sec_ckpt_w = sec_state[up_key[0]]
            sec_proc_w = sec_proc.slot0_adapter.up.weight.detach()
            # Phase41 버그: secondary는 복원되지 않았으므로 값이 다름
            assert not torch.allclose(sec_proc_w, sec_ckpt_w, atol=1e-4), \
                "Phase41 버그 재현 실패 — secondary가 복원됨 (예상: 복원 안 됨)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
