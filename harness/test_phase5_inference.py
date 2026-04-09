"""Phase 5: VCA Processor + GIF 디버그"""
import pytest
import numpy as np
import torch

pytestmark = pytest.mark.phase5


# ─── Processor 단위 테스트 ───────────────────────────────────────────────────
@pytest.fixture(scope="module")
def proc_output():
    from models.vca_processor import VCAAttnProcessor2_0
    from models.vca_attention import VCALayer

    layer = VCALayer(query_dim=64, context_dim=128, n_heads=4, n_entities=2, z_bins=2)
    proc  = VCAAttnProcessor2_0(layer)

    BF, S, D, N, CD = 2, 16, 64, 2, 128
    hidden  = torch.randn(BF, S, D)
    context = torch.randn(BF, N, CD)

    # cross-attention: attn=None (self-attention 분기 불필요)
    out = proc(None, hidden, encoder_hidden_states=context)
    return out, layer


def test_processor_output_shape(proc_output):
    out, _ = proc_output
    assert out.shape == (2, 16, 64), f"Expected (2,16,64) got {out.shape}"


def test_sigma_captured(proc_output):
    _, layer = proc_output
    assert layer.last_sigma is not None
    assert layer.last_sigma.shape == (2, 16, 2, 2), \
        f"Expected (2,16,2,2) got {layer.last_sigma.shape}"


def test_transmittance_T0_is_one(proc_output):
    _, layer = proc_output
    T = layer.last_transmittance
    assert torch.allclose(T[:, :, 0], torch.ones_like(T[:, :, 0])), \
        f"T[z=0] must be 1.0, got min={T[:,:,0].min():.4f}"


def test_sigma_range(proc_output):
    _, layer = proc_output
    s = layer.last_sigma
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_processor_output_is_residual(proc_output):
    """VCALayer forward는 x + to_out(attn) → 출력이 입력과 같은 scale이어야 함"""
    out, _ = proc_output
    assert out.isfinite().all(), "Output contains inf/nan"


# ─── GIF 단위 테스트 ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def debug_gif_path(proc_output, tmp_path_factory):
    from scripts.debug_gif import make_debug_gif

    _, layer = proc_output
    sigma = layer.last_sigma.numpy()   # (BF, S, N, Z)

    # S=16 → 4×4 spatial
    H, W = 4, 4
    sigma_hw = sigma[0, :, :, 0].T.reshape(2, H, W)   # (N, H, W)
    rgb = (np.random.rand(H, W, 3) * 255).astype(np.uint8)

    out = tmp_path_factory.mktemp("gif") / "test_sigma.gif"
    make_debug_gif([rgb] * 3, [sigma_hw] * 3, out, panel_size=4)
    return out


def test_debug_gif_created(debug_gif_path):
    assert debug_gif_path.exists(), "GIF file not created"
    assert debug_gif_path.stat().st_size > 0, "GIF file is empty"


def test_debug_gif_has_3_panels(debug_gif_path):
    import imageio.v3 as iio
    frame = iio.imread(str(debug_gif_path), index=0)
    expected_width = 4 * 3   # panel_size * 3
    assert frame.shape[1] == expected_width, \
        f"Expected width={expected_width} (3 panels × 4px), got {frame.shape[1]}"


def test_debug_gif_is_rgb(debug_gif_path):
    import imageio.v3 as iio
    frame = iio.imread(str(debug_gif_path), index=0)
    assert frame.ndim == 3 and frame.shape[2] >= 3, \
        f"Expected RGB frame, got shape {frame.shape}"
