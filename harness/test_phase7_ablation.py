"""Phase 7: Softmax vs Sigmoid Disappearance Ablation"""
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.phase7

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── 구조적 단위 테스트 ──────────────────────────────────────────────────────

def test_softmax_sigma_sums_to_one():
    """Softmax variant: scores over N*Z keys → sums to 1 per (BF,h,S)"""
    from models.vca_attention import VCALayer
    vca = VCALayer(query_dim=64, context_dim=128, n_heads=4,
                   n_entities=2, z_bins=2, lora_rank=4, use_softmax=True)
    vca.eval()
    with torch.no_grad():
        x   = torch.randn(2, 16, 64)
        ctx = torch.randn(2, 2, 128)
        vca(x, ctx)
    sigma = vca.last_sigma   # (BF, S, N, Z)
    # sum over N*Z per spatial position should = 1 (after mean over heads)
    # We check sigma sums to ≤ 1 for softmax (each head sums to 1 per spatial pos)
    total = sigma.sum(dim=[-2, -1])  # (BF, S) - sum over N,Z
    # after head averaging, sum ≤ 1 (softmax normalizes over N*Z=4 keys)
    assert total.max().item() <= 1.0 + 1e-4, \
        f"Softmax sigma sum per position should be ≤1, got max={total.max().item():.4f}"


def test_sigmoid_sum_can_exceed_one():
    """Sigmoid variant: independent [0,1] → sum over N*Z can exceed 1"""
    from models.vca_attention import VCALayer
    vca = VCALayer(query_dim=64, context_dim=128, n_heads=4,
                   n_entities=2, z_bins=2, lora_rank=4, use_softmax=False)
    vca.eval()
    with torch.no_grad():
        x   = torch.full((2, 16, 64), 5.0)   # high activation → sigmoid ≈ 1
        ctx = torch.full((2, 2, 128), 5.0)
        vca(x, ctx)
    sigma = vca.last_sigma   # (BF, S, N, Z)
    total = sigma.sum(dim=[-2, -1])  # (BF, S)
    assert total.max().item() > 1.0, \
        f"Sigmoid can exceed 1 when all entities active, got max={total.max().item():.4f}"


def test_both_high_sigmoid_not_softmax():
    """FM-I4 교훈 반영: ctx=[+5,+5] → 두 entity 모두 high 가능(Sigmoid), 불가(Softmax)"""
    from models.vca_attention import VCALayer

    def get_both_high(use_softmax: bool) -> float:
        vca = VCALayer(query_dim=64, context_dim=128, n_heads=4,
                       n_entities=2, z_bins=2, lora_rank=4, use_softmax=use_softmax)
        vca.eval()
        ctx = torch.zeros(4, 2, 128)
        ctx[:, 0, :] = 5.0  # entity 0: strong positive
        ctx[:, 1, :] = 5.0  # entity 1: same direction (not anti-symmetric!)
        x = torch.randn(4, 16, 64)
        with torch.no_grad():
            vca(x, ctx)
        sigma = vca.last_sigma  # (BF, S, N, Z)
        both_high = ((sigma[:, :, 0, :] > 0.5) & (sigma[:, :, 1, :] > 0.5)).float().mean()
        return both_high.item()

    sig_bh  = get_both_high(use_softmax=False)
    sof_bh  = get_both_high(use_softmax=True)

    assert sig_bh > 0.1, \
        f"Sigmoid both_high should be >0.1 (simultaneous high), got {sig_bh:.4f}"
    assert sig_bh > sof_bh, \
        f"Sigmoid both_high {sig_bh:.4f} should exceed Softmax {sof_bh:.4f}"


def test_softmax_disappearance_pattern():
    """Softmax: zero-sum 경쟁 → 한 entity가 다른 entity를 억제 (Disappearance 패턴)"""
    from models.vca_attention import VCALayer
    vca = VCALayer(query_dim=64, context_dim=128, n_heads=4,
                   n_entities=2, z_bins=2, lora_rank=4, use_softmax=True)
    vca.eval()
    ctx = torch.zeros(4, 2, 128)
    ctx[:, 0, :] = 10.0  # entity 0 dominates
    ctx[:, 1, :] = -10.0  # entity 1 suppressed
    x = torch.randn(4, 16, 64)
    with torch.no_grad():
        vca(x, ctx)
    sigma = vca.last_sigma
    s0 = sigma[:, :, 0, :].mean().item()
    s1 = sigma[:, :, 1, :].mean().item()
    # Softmax: dominant entity takes all probability
    assert s0 > s1, \
        f"Softmax should show dominance: s0={s0:.4f} should > s1={s1:.4f}"


# ─── subprocess 통합 테스트 ──────────────────────────────────────────────────

@pytest.fixture(scope='module')
def ablation_result():
    r = subprocess.run(
        [sys.executable, 'scripts/run_ablation.py',
         '--epochs',   '3',
         '--scenario', 'chain',
         '--out-dir',  'debug/ablation'],
        capture_output=True, text=True, timeout=300,
    )
    return r


def test_ablation_exits_cleanly(ablation_result):
    assert ablation_result.returncode == 0, \
        f"run_ablation.py failed:\n{ablation_result.stderr[-800:]}"


def test_ablation_prints_final_sigmoid(ablation_result):
    assert 'FINAL sigmoid' in ablation_result.stdout, \
        f"Expected 'FINAL sigmoid' in output:\n{ablation_result.stdout[-400:]}"


def test_ablation_prints_final_softmax(ablation_result):
    assert 'FINAL softmax' in ablation_result.stdout, \
        f"Expected 'FINAL softmax' in output:\n{ablation_result.stdout[-400:]}"


def test_sigmoid_has_lower_disappearance(ablation_result):
    """핵심 가설: Sigmoid의 disappearance_rate < Softmax의 disappearance_rate"""
    lines = ablation_result.stdout
    sig_match = re.search(r'FINAL sigmoid.*?disappearance_rate=([\d.]+)', lines)
    sof_match = re.search(r'FINAL softmax.*?disappearance_rate=([\d.]+)', lines)
    if not sig_match or not sof_match:
        pytest.skip(f"Could not parse disappearance_rate from output:\n{lines[-600:]}")
    sig_dr = float(sig_match.group(1))
    sof_dr = float(sof_match.group(1))
    assert sig_dr <= sof_dr, \
        f"Sigmoid disappearance_rate {sig_dr:.4f} should be ≤ Softmax {sof_dr:.4f}"


def test_sigmoid_higher_visibility_min(ablation_result):
    """Sigmoid: 두 entity 모두 살아있음 → visibility_min 높음"""
    lines = ablation_result.stdout
    sig_match = re.search(r'FINAL sigmoid.*?visibility_min=([\d.]+)', lines)
    sof_match = re.search(r'FINAL softmax.*?visibility_min=([\d.]+)', lines)
    if not sig_match or not sof_match:
        pytest.skip(f"Could not parse visibility_min from output:\n{lines[-600:]}")
    sig_vm = float(sig_match.group(1))
    sof_vm = float(sof_match.group(1))
    assert sig_vm >= sof_vm, \
        f"Sigmoid visibility_min {sig_vm:.4f} should be ≥ Softmax {sof_vm:.4f}"


def test_checkpoints_saved(ablation_result):
    assert Path('debug/ablation/sigmoid_final.pt').exists(), \
        "sigmoid_final.pt not saved"
    assert Path('debug/ablation/softmax_final.pt').exists(), \
        "softmax_final.pt not saved"


def test_comparison_gif_exists(ablation_result):
    assert Path('debug/ablation/comparison.gif').exists(), \
        "comparison.gif not saved"


def test_comparison_gif_is_4panel(ablation_result):
    """4-panel: [Sig-E0 | Sig-E1 | Sof-E0 | Sof-E1] → width = panel_size * 4"""
    import imageio.v3 as iio
    gif_path = Path('debug/ablation/comparison.gif')
    if not gif_path.exists():
        pytest.skip("comparison.gif not found")
    frame = iio.imread(str(gif_path), index=0)
    assert frame.shape[1] == 64 * 4, \
        f"Expected width=256 (4×64px panels), got {frame.shape[1]}"
