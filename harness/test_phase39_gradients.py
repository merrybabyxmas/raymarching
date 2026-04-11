"""
Phase 39 — Test: gradient flow through new parameters
=======================================================

slot0_adapter, slot1_adapter, blend_head에 실제 gradient가 흐르는지 검증.

Phase 38의 근본 문제: slot_blend_raw 하나만 학습되고 F_0/F_1이 frozen q/k/v에서
나오므로 L_exclusive gradient가 사실상 흐르지 않았음.

Phase 39: adapter와 blend_head가 optimizer에 등록되고 실제로 학습되는지 확인.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot import (
    EntitySlotAttnProcessor,
    SlotAdapter,
    BlendHead,
    l_visible_weights,
    l_wrong_slot_suppression,
    l_entity_exclusive,
    build_visible_targets,
)

pytestmark = pytest.mark.phase39

# =============================================================================
# Fixtures
# =============================================================================

DIM = 64
RANK = 16
BATCH = 2
SEQ = 8


@pytest.fixture
def adapter():
    return SlotAdapter(dim=DIM, r=RANK)


@pytest.fixture
def blend_head():
    return BlendHead(hidden=8, init_bias=-0.847)


@pytest.fixture
def simple_entity_masks():
    """B=2, S=8 마스크."""
    m = torch.zeros(BATCH, 2, SEQ)
    m[0, 0, :3] = 1.0   # e0 exclusive
    m[0, 1, 3:6] = 1.0  # e1 exclusive
    m[0, 0, 6] = 1.0    # overlap
    m[0, 1, 6] = 1.0
    m[1, 0, :4] = 1.0
    m[1, 1, 2:7] = 1.0
    return m


# =============================================================================
# SlotAdapter gradient tests
# =============================================================================

class TestSlotAdapterGradients:

    def test_zero_init_output_equals_input(self, adapter):
        """zero-init → 초기 출력이 입력과 동일해야 함."""
        x = torch.randn(BATCH, SEQ, DIM)
        y = adapter(x)
        assert torch.allclose(y, x, atol=1e-6), \
            "zero-init adapter should return input unchanged"

    def test_gradient_flows_to_up_weight(self, adapter):
        """loss.backward() 후 up.weight에 gradient가 흐르는지."""
        x = torch.randn(BATCH, SEQ, DIM, requires_grad=True)
        y = adapter(x)
        loss = y.pow(2).mean()
        loss.backward()
        assert adapter.up.weight.grad is not None
        assert adapter.down.weight.grad is not None
        assert adapter.norm.weight.grad is not None

    def test_gradient_flows_to_input(self, adapter):
        """adapter를 통해 input에도 gradient가 흐르는지."""
        x = torch.randn(BATCH, SEQ, DIM, requires_grad=True)
        y = adapter(x)
        loss = y.pow(2).mean()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_gradient_nonzero_after_step(self, adapter):
        """옵티마이저 step 후 up.weight에 gradient가 흐르고 값이 변하는지.

        Note: zero-init 구조 특성상 up.weight만 처음에 gradient를 받음.
        (down.weight.grad는 초기에 0 — up.weight가 zeros라 역전파가 여기서 막힘)
        up.weight.grad가 non-zero → opt.step() → up.weight가 zeros에서 변함.
        """
        opt = optim.SGD(adapter.parameters(), lr=0.01)
        orig_up_w = adapter.up.weight.data.clone()   # all zeros
        x = torch.randn(BATCH, SEQ, DIM)
        target = torch.randn_like(x)
        y = adapter(x)
        loss = (y - target).pow(2).mean()
        loss.backward()
        # up.weight가 유일하게 gradient를 받음 (zero-init이라 역전파 경로가 여기서 시작)
        assert adapter.up.weight.grad is not None
        assert not torch.all(adapter.up.weight.grad == 0), \
            "up.weight.grad should be non-zero"
        opt.step()
        # step 후 up.weight가 zeros에서 벗어나야 함
        assert not torch.allclose(adapter.up.weight, orig_up_w), \
            "up.weight should change from zeros after optimizer step"

    def test_adapter_rank_reduces_parameters(self):
        """rank가 작으면 파라미터가 적어야 함."""
        a_large = SlotAdapter(dim=DIM, r=DIM // 2)
        a_small = SlotAdapter(dim=DIM, r=8)
        n_large = sum(p.numel() for p in a_large.parameters())
        n_small = sum(p.numel() for p in a_small.parameters())
        assert n_small < n_large

    def test_residual_path_preserved(self, adapter):
        """adapter(x) = x + residual 구조 — residual이 점점 커지면서 학습."""
        opt = optim.SGD(adapter.parameters(), lr=1.0)
        x = torch.randn(BATCH, SEQ, DIM)
        target = x * 2   # x + x를 목표로
        for _ in range(50):
            opt.zero_grad()
            y = adapter(x.detach())
            loss = (y - target).pow(2).mean()
            loss.backward()
            opt.step()
        # 충분히 학습하면 residual이 x에 가까워져야 함
        final = adapter(x.detach())
        assert float((final - x).abs().mean().item()) > 0.01, \
            "adapter should learn non-zero residual"


# =============================================================================
# BlendHead gradient tests
# =============================================================================

class TestBlendHeadGradients:

    def test_output_range(self, blend_head):
        """출력이 [0, 1] 사이인지."""
        alpha0   = torch.rand(BATCH, SEQ)
        alpha1   = torch.rand(BATCH, SEQ)
        e0_front = torch.rand(BATCH, SEQ)
        out = blend_head(alpha0, alpha1, e0_front)
        assert out.shape == (BATCH, SEQ, 1)
        assert float(out.min().item()) >= 0.0
        assert float(out.max().item()) <= 1.0

    def test_init_bias_approx_030(self, blend_head):
        """init_bias=-0.847 → sigmoid(-0.847) ≈ 0.3."""
        alpha0   = torch.zeros(BATCH, SEQ)
        alpha1   = torch.zeros(BATCH, SEQ)
        e0_front = torch.zeros(BATCH, SEQ)
        out = blend_head(alpha0, alpha1, e0_front)
        # When linear weights are zero, net output = bias → sigmoid(-0.847) ≈ 0.3
        assert float(out.mean().item()) == pytest.approx(0.3, abs=0.05)

    def test_gradient_flows_to_all_params(self, blend_head):
        """모든 파라미터에 gradient가 흐르는지."""
        alpha0   = torch.rand(BATCH, SEQ)
        alpha1   = torch.rand(BATCH, SEQ)
        e0_front = torch.rand(BATCH, SEQ)
        out = blend_head(alpha0, alpha1, e0_front)
        loss = out.mean()
        loss.backward()
        for name, p in blend_head.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_gradient_flows_to_inputs(self, blend_head):
        """alpha0, alpha1, e0_front에도 gradient가 흐르는지."""
        alpha0   = torch.rand(BATCH, SEQ, requires_grad=True)
        alpha1   = torch.rand(BATCH, SEQ, requires_grad=True)
        e0_front = torch.rand(BATCH, SEQ, requires_grad=True)
        out = blend_head(alpha0, alpha1, e0_front)
        loss = out.mean()
        loss.backward()
        assert alpha0.grad is not None
        assert alpha1.grad is not None
        assert e0_front.grad is not None

    def test_overlap_channel_affects_output(self, blend_head):
        """alpha0*alpha1 채널이 출력에 영향을 주는지."""
        # alpha0=alpha1=0이면 overlap channel도 0
        # alpha0=alpha1=1이면 overlap channel=1
        alpha0_lo = torch.zeros(1, 1)
        alpha1_lo = torch.zeros(1, 1)
        alpha0_hi = torch.ones(1, 1)
        alpha1_hi = torch.ones(1, 1)
        e0_front  = torch.zeros(1, 1)

        # Train blend_head to produce higher output for high overlap
        opt = optim.Adam(blend_head.parameters(), lr=0.01)
        for _ in range(100):
            opt.zero_grad()
            out_lo = blend_head(alpha0_lo, alpha1_lo, e0_front)
            out_hi = blend_head(alpha0_hi, alpha1_hi, e0_front)
            # maximize: out_hi > out_lo
            loss = out_lo - out_hi + 0.1
            loss = loss.relu().mean()
            if float(loss.item()) < 1e-4:
                break
            loss.backward()
            opt.step()
        out_lo_f = blend_head(alpha0_lo, alpha1_lo, e0_front)
        out_hi_f = blend_head(alpha0_hi, alpha1_hi, e0_front)
        # 학습 후 overlap 높으면 blend 높아야 함
        assert float(out_hi_f.item()) > float(out_lo_f.item()), \
            "blend_head should learn higher blend for high overlap"


# =============================================================================
# End-to-end: l_visible_weights gradient flows through w0/w1
# =============================================================================

class TestVisibleWeightsGradientE2E:

    def test_visible_loss_gradient_wrt_w0_w1(self, simple_entity_masks):
        """l_visible_weights가 w0, w1로 gradient를 잘 전달하는지."""
        B, _, S = simple_entity_masks.shape
        w0 = torch.rand(B, S, requires_grad=True)
        w1 = torch.rand(B, S, requires_grad=True)
        depth_orders = [(0, 1), (1, 0)]

        loss = l_visible_weights(w0, w1, simple_entity_masks, depth_orders)
        loss.backward()

        assert w0.grad is not None
        assert w1.grad is not None
        # gradient should be nonzero for entity-masked positions
        entity_any = (simple_entity_masks[:, 0] + simple_entity_masks[:, 1]).clamp(0, 1)
        assert float((w0.grad.abs() * entity_any).sum().item()) > 0.0

    def test_visible_loss_drives_w0_toward_target(self, simple_entity_masks):
        """Gradient descent로 w0이 GT target으로 수렴하는지."""
        B, _, S = simple_entity_masks.shape
        depth_orders = [(0, 1)] * B
        w0 = nn.Parameter(torch.full((B, S), 0.5))
        w1 = nn.Parameter(torch.full((B, S), 0.5))
        opt = optim.Adam([w0, w1], lr=0.1)

        w0_target, w1_target = build_visible_targets(simple_entity_masks, depth_orders)

        init_loss = float(
            l_visible_weights(w0, w1, simple_entity_masks, depth_orders).item())

        for _ in range(100):
            opt.zero_grad()
            loss = l_visible_weights(w0, w1, simple_entity_masks, depth_orders)
            loss.backward()
            opt.step()

        final_loss = float(
            l_visible_weights(w0, w1, simple_entity_masks, depth_orders).item())
        assert final_loss < init_loss * 0.1, \
            f"Loss should decrease: {init_loss:.4f} → {final_loss:.4f}"

    def test_wrong_slot_gradient_wrt_w1_in_e0_exclusive(self, simple_entity_masks):
        """l_wrong_slot: e0 exclusive에서 w1에 gradient가 흐르는지."""
        B, _, S = simple_entity_masks.shape
        w0 = torch.zeros(B, S)
        w1 = torch.rand(B, S, requires_grad=True)

        loss = l_wrong_slot_suppression(w0, w1, simple_entity_masks)
        loss.backward()

        assert w1.grad is not None
        # e0 exclusive 위치에만 gradient
        excl_0 = simple_entity_masks[:, 0, :] * (1 - simple_entity_masks[:, 1, :])
        grad_in_excl = float((w1.grad.abs() * excl_0).sum().item())
        assert grad_in_excl > 0.0, "wrong slot grad should be in e0 exclusive region"


# =============================================================================
# Phase 39 optimizer parameter group test
# =============================================================================

class TestOptimizerParameterGroups:

    def test_adapter_params_in_optimizer(self):
        """slot adapter 파라미터가 optimizer에 등록되는지."""
        proc = EntitySlotAttnProcessor(
            query_dim=DIM,
            inner_dim=DIM,
            adapter_rank=RANK,
            use_blend_head=True,
        )

        # Phase 39 파라미터 그룹 구성 (train_phase39.py 패턴과 동일)
        param_groups = [
            {"params": [proc.slot_blend_raw], "name": "slot_blend"},
            {"params": list(proc.slot0_adapter.parameters())
                     + list(proc.slot1_adapter.parameters()), "name": "adapters"},
            {"params": list(proc.blend_head.parameters()), "name": "blend_head"},
        ]
        opt = optim.AdamW(param_groups, lr=1e-4)

        # 각 그룹의 파라미터 수 확인
        group_names = {g["name"]: g["params"] for g in opt.param_groups}
        assert "adapters"   in group_names
        assert "blend_head" in group_names
        assert "slot_blend" in group_names

        n_adapter = sum(p.numel() for p in group_names["adapters"])
        n_blend   = sum(p.numel() for p in group_names["blend_head"])
        assert n_adapter > 0, "adapters should have parameters"
        assert n_blend > 0, "blend_head should have parameters"

    def test_requires_grad_on_new_params(self):
        """새 파라미터들이 requires_grad=True인지."""
        proc = EntitySlotAttnProcessor(
            query_dim=DIM, inner_dim=DIM,
            adapter_rank=RANK, use_blend_head=True)

        for name, p in proc.slot0_adapter.named_parameters():
            assert p.requires_grad, f"slot0_adapter.{name} should require grad"
        for name, p in proc.slot1_adapter.named_parameters():
            assert p.requires_grad, f"slot1_adapter.{name} should require grad"
        for name, p in proc.blend_head.named_parameters():
            assert p.requires_grad, f"blend_head.{name} should require grad"
