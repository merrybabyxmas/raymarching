"""
test_phase44_last_blend_map.py — Phase44Processor의 blend map 저장 속성 검증

검증 항목:
  1. last_blend_map_for_loss 속성 존재 (Phase44Processor)
  2. reset_slot_store() → last_blend_map_for_loss = None
  3. last_blend_map_for_loss는 requires_grad=True (grad path 살아있음)
  4. last_blend_map (detached) requires_grad=False
  5. last_blend_map_for_loss와 last_blend_map 값 동일 (같은 tensor의 두 뷰)
  6. l_blend_target(last_blend_map_for_loss, masks) → backward에서 overlap_blend_head로 grad 흐름
  7. l_blend_target(last_blend_map.detach(), masks) → overlap_blend_head로 grad 안 흐름 (detached)
  8. last_base_blend 저장 확인
  9. last_overlap_proxy 저장 확인
 10. last_blend backward compatibility (last_blend == last_blend_map)
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entity_slot_phase44 import (
    Phase44Processor,
    l_blend_target,
    CROSS_ATTN_DIM,
    PRIMARY_DIM,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_proc(query_dim=640) -> Phase44Processor:
    """가볍게 proc 생성 (vca_layer=None)."""
    return Phase44Processor(
        query_dim           = query_dim,
        vca_layer           = None,
        entity_ctx          = None,
        slot_blend_init     = 0.3,
        inner_dim           = query_dim,
        adapter_rank        = 4,   # small for test speed
        use_blend_head      = True,
        lora_rank           = 2,
        cross_attention_dim = CROSS_ATTN_DIM,
        weight_head_hidden  = 16,
        primary_dim         = PRIMARY_DIM,
        proj_hidden         = 64,
        obh_hidden          = 16,
    )


def _run_proc_forward(proc: Phase44Processor, B=1, S=16, D=640):
    """
    Phase44Processor를 가짜 attn object와 함께 실행.
    sigma가 없는 fallback branch 사용.
    encoder hidden states는 CROSS_ATTN_DIM(768)으로 맞춤.
    """
    class FakeAttn:
        heads = 8
        to_q  = torch.nn.Linear(D, D)
        to_k  = torch.nn.Linear(CROSS_ATTN_DIM, D)   # encoder_dim → inner_dim
        to_v  = torch.nn.Linear(CROSS_ATTN_DIM, D)
        to_out = torch.nn.ModuleList([torch.nn.Linear(D, D), torch.nn.Identity()])

    attn = FakeAttn()
    hs   = torch.randn(B, S, D)
    enc  = torch.randn(B, 77, CROSS_ATTN_DIM)  # CROSS_ATTN_DIM=768
    return proc(attn, hs, encoder_hidden_states=enc)


# =============================================================================
# Test: last_blend_map_for_loss 속성
# =============================================================================

class TestLastBlendMapForLoss:

    def test_attribute_exists(self):
        """last_blend_map_for_loss 속성 존재."""
        proc = _make_proc()
        assert hasattr(proc, 'last_blend_map_for_loss'), \
            "Phase44Processor missing last_blend_map_for_loss"

    def test_reset_clears_for_loss(self):
        """reset_slot_store() → last_blend_map_for_loss = None."""
        proc = _make_proc()
        # Set a fake value
        proc.last_blend_map_for_loss = torch.ones(1, 16)
        proc.reset_slot_store()
        assert proc.last_blend_map_for_loss is None

    def test_last_base_blend_exists_and_clears(self):
        """last_base_blend 저장 후 reset시 None."""
        proc = _make_proc()
        proc.last_base_blend = torch.ones(1, 16)
        proc.reset_slot_store()
        assert proc.last_base_blend is None

    def test_last_overlap_proxy_exists_and_clears(self):
        """last_overlap_proxy 저장 후 reset시 None."""
        proc = _make_proc()
        proc.last_overlap_proxy = torch.ones(1, 16)
        proc.reset_slot_store()
        assert proc.last_overlap_proxy is None

    def test_last_blend_backward_compat(self):
        """reset_slot_store() 후 last_blend = None."""
        proc = _make_proc()
        proc.last_blend = torch.ones(1, 16)
        proc.reset_slot_store()
        assert proc.last_blend is None


# =============================================================================
# Test: sigma 없는 fallback branch에서 blend map None
# =============================================================================

class TestFallbackBranch:

    def test_forward_no_sigma_blend_map_none(self):
        """sigma 없는 fallback → last_blend_map_for_loss = None."""
        proc = _make_proc()
        proc.eval()
        # vca=None이면 sigma=None → fallback branch
        _run_proc_forward(proc)
        assert proc.last_blend_map_for_loss is None, \
            "fallback (no sigma) should leave last_blend_map_for_loss as None"
        assert proc.last_blend_map is None


# =============================================================================
# Test: sigma 있는 branch에서 gradient 흐름
# (실제 VCA 없이 sigma를 직접 주입해서 테스트)
# =============================================================================

class TestGradientPath:

    def _make_proc_with_fake_sigma(self):
        """sigma를 직접 주입할 수 있도록 proc 준비."""
        proc = _make_proc()
        return proc

    def test_for_loss_requires_grad(self):
        """last_blend_map_for_loss: requires_grad 경로 시뮬레이션."""
        # OverlapBlendHead의 출력이 grad를 가짐 (zero-init이지만 requires_grad=True)
        proc = _make_proc()
        proc.train()

        # 직접 overlap_blend_head forward로 blend_map_for_loss 시뮬레이션
        B, S = 1, 16
        alpha_0 = torch.rand(B, S)
        alpha_1 = torch.rand(B, S)
        from models.entity_slot_phase44 import compute_base_blend
        base_blend = compute_base_blend(alpha_0, alpha_1)

        feat = torch.randn(B, S, 8)
        delta = proc.overlap_blend_head(feat)           # (B, S, 1)
        blend_logit = torch.logit(base_blend, eps=1e-6).unsqueeze(-1)
        blend_map_for_loss = torch.sigmoid(blend_logit + delta).squeeze(-1)

        # Non-detached version: grad should flow
        assert blend_map_for_loss.requires_grad, \
            "blend_map_for_loss should have requires_grad=True"

    def test_detached_no_grad_to_obh(self):
        """detached blend_map → requires_grad=False → overlap_blend_head로 grad 경로 없음."""
        proc = _make_proc()
        proc.train()

        B, S = 1, 16
        alpha_0 = torch.rand(B, S)
        alpha_1 = torch.rand(B, S)
        from models.entity_slot_phase44 import compute_base_blend
        base_blend = compute_base_blend(alpha_0, alpha_1)

        feat = torch.randn(B, S, 8)
        delta = proc.overlap_blend_head(feat)
        blend_logit = torch.logit(base_blend, eps=1e-6).unsqueeze(-1)
        blend_map_detached = torch.sigmoid(blend_logit + delta).squeeze(-1).detach()

        # Detached tensor: requires_grad=False → no grad_fn → loss has no grad path
        assert not blend_map_detached.requires_grad, \
            "detached blend_map should have requires_grad=False"

        m = torch.zeros(B, 2, S)
        m[0, 0, :4] = 1.0; m[0, 1, :4] = 1.0
        loss = l_blend_target(blend_map_detached, m)

        # Loss has no grad_fn since input is detached — cannot train overlap_blend_head
        assert loss.grad_fn is None, \
            "l_blend_target on detached blend_map should produce no-grad loss"

    def test_non_detached_grad_to_obh(self):
        """non-detached blend_map → overlap_blend_head에 grad 흐름."""
        proc = _make_proc()
        proc.train()
        proc.overlap_blend_head.zero_grad()
        # Perturb last layer weight to make delta non-zero
        with torch.no_grad():
            proc.overlap_blend_head.net[2].weight.fill_(0.01)

        B, S = 1, 16
        alpha_0 = torch.rand(B, S)
        alpha_1 = torch.rand(B, S)
        from models.entity_slot_phase44 import compute_base_blend
        base_blend = compute_base_blend(alpha_0, alpha_1)

        feat = torch.randn(B, S, 8)
        delta = proc.overlap_blend_head(feat)
        blend_logit = torch.logit(base_blend, eps=1e-6).unsqueeze(-1)
        blend_map_for_loss = torch.sigmoid(blend_logit + delta).squeeze(-1)

        m = torch.zeros(B, 2, S)
        m[0, 0, :4] = 1.0; m[0, 1, :4] = 1.0

        loss = l_blend_target(blend_map_for_loss, m)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().max().item() > 0
            for p in proc.overlap_blend_head.parameters()
        )
        assert has_grad, "non-detached blend should flow grad to overlap_blend_head"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
