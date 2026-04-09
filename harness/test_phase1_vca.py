import pytest
import torch

pytestmark = pytest.mark.phase1

@pytest.fixture(scope="module")
def layer():
    from models.vca_attention import VCALayer
    # context_dim=768: SD 1.5 기준 단위 테스트용 (SDXL이면 2048로 바꿔라)
    return VCALayer(query_dim=320, context_dim=768, n_heads=8, n_entities=2, z_bins=2)

@pytest.fixture(scope="module")
def fwd(layer):
    x   = torch.randn(4, 1024, 320)
    ctx = torch.randn(4, 2, 768)
    out = layer(x, ctx)
    return x, ctx, out, layer

# Shape
def test_output_shape(fwd):
    x, _, out, _ = fwd
    assert out.shape == x.shape

def test_sigma_shape(fwd):
    assert fwd[3].last_sigma.shape == (4, 1024, 2, 2)

def test_transmittance_shape(fwd):
    assert fwd[3].last_transmittance.shape == (4, 1024, 2)

# Physics
def test_sigma_range(fwd):
    s = fwd[3].last_sigma
    assert s.min() >= 0.0 and s.max() <= 1.0

def test_T_z0_is_one(fwd):
    T = fwd[3].last_transmittance
    assert torch.allclose(T[:, :, 0], torch.ones_like(T[:, :, 0]))

def test_T_monotone(fwd):
    T = fwd[3].last_transmittance
    assert (T[:, :, 0] >= T[:, :, 1] - 1e-5).all()

# Gradient
def test_grad_flows_to_lora():
    from models.vca_attention import VCALayer
    l = VCALayer(query_dim=320, context_dim=768, n_heads=8, n_entities=2, z_bins=2)
    l(torch.randn(2, 64, 320), torch.randn(2, 2, 768)).mean().backward()
    assert l.to_k.lora_A.grad is not None
    assert l.depth_pe.grad    is not None

def test_base_weight_frozen():
    from models.vca_attention import VCALayer
    l = VCALayer(query_dim=320, context_dim=768, n_heads=8, n_entities=2, z_bins=2)
    l(torch.randn(2, 64, 320), torch.randn(2, 2, 768)).mean().backward()
    assert l.to_k.weight.grad is None

# Sigmoid vs Softmax 검증
def test_entities_independent():
    """
    Sigmoid는 entity 간 경쟁이 없음을 검증.

    설계 원칙 (FM-I4):
      ctx = [+5, -5] 설정은 K_entity1 = -K_entity0을 만들어
      sigmoid(x)+sigmoid(-x)=1 항등식으로 diff ≈ 0이 된다 — 잘못된 테스트.

      올바른 방법: ctx = [+5, +5] (동일 방향) → K_entity0 = K_entity1
      → σ_entity0[s] = σ_entity1[s] (항상 같음)
      → 두 entity가 동시에 sigma > 0.5인 픽셀이 ~50% 존재
      → Softmax였다면 zero-sum → both_high ≈ 0%
    """
    from models.vca_attention import VCALayer
    CD = 768
    l = VCALayer(query_dim=320, context_dim=CD, n_heads=8, n_entities=2, z_bins=2)
    ctx = torch.zeros(2, 2, CD)
    ctx[:, 0, :] = 5.0   # entity 0: large positive
    ctx[:, 1, :] = 5.0   # entity 1: same direction — 반대(±5)가 아님!
    l(torch.randn(2, 64, 320), ctx)
    s = l.last_sigma  # (BF, S, N, Z)
    # 두 entity가 동시에 0.5 초과인 spatial position 비율
    # Sigmoid: ~50% (entity 간 경쟁 없음)
    # Softmax: ~0%  (zero-sum이라 한 entity가 높으면 다른 entity는 낮아짐)
    both_high = ((s[:, :, 0, :] > 0.5) & (s[:, :, 1, :] > 0.5)).float().mean()
    assert both_high.item() > 0.1, \
        f"Sigmoid should allow both entities to be high simultaneously. " \
        f"both_high={both_high:.4f} (expected >0.1, Softmax would give ~0)"
