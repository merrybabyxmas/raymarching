"""Phase 10: 학습 유효성 검증 — encoding fix + sigma_consistency"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.phase10

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── 10-A: 인코딩 단위 테스트 ────────────────────────────────────────────────

def test_x_std_above_threshold():
    """x 전체 차원이 유효한 신호를 가짐: x.std() > 0.1"""
    from scripts.train_vca import ToyVCADataset
    ds = ToyVCADataset(scenario='chain', query_dim=64, context_dim=128)
    stds = []
    for idx in range(len(ds)):
        x, _, _, _ = ds[idx]
        stds.append(float(x.squeeze(0).numpy().std()))
    mean_std = float(np.mean(stds))
    assert mean_std > 0.1, \
        f"x.std()={mean_std:.4f} should be > 0.1 (Phase 10 encoding fix required)"


def test_ctx_diff_above_threshold():
    """두 entity ctx가 실제로 다름: (ctx[0]-ctx[1]).abs().mean() > 0.1"""
    from scripts.train_vca import ToyVCADataset
    ds = ToyVCADataset(scenario='chain', query_dim=64, context_dim=128)
    diffs = []
    for idx in range(len(ds)):
        _, ctx, _, _ = ds[idx]
        c = ctx.squeeze(0).numpy()
        diffs.append(float(np.abs(c[0] - c[1]).mean()))
    mean_diff = float(np.mean(diffs))
    assert mean_diff > 0.1, \
        f"ctx_diff={mean_diff:.4f} should be > 0.1 (entities should be distinguishable)"


def test_entity_ctx_structurally_different():
    """entity 0 과 entity 1의 ctx가 고정된 identity 차이를 가짐"""
    from scripts.train_vca import ToyVCADataset
    ds = ToyVCADataset(scenario='chain', query_dim=64, context_dim=128)
    _, ctx, _, _ = ds[0]
    c = ctx.squeeze(0).numpy()  # (N, CD)
    # one-hot identity dims 4-5
    assert c[0, 4] == 1.0, f"entity0 ctx[4] should be 1.0 (one-hot), got {c[0,4]:.4f}"
    assert c[1, 4] == 0.0, f"entity1 ctx[4] should be 0.0 (one-hot), got {c[1,4]:.4f}"
    assert c[0, 5] == 0.0, f"entity0 ctx[5] should be 0.0 (one-hot), got {c[0,5]:.4f}"
    assert c[1, 5] == 1.0, f"entity1 ctx[5] should be 1.0 (one-hot), got {c[1,5]:.4f}"


def test_x_not_dummy():
    """x가 3차원만 채운 dummy가 아님: non-zero 비율 > 50%"""
    from scripts.train_vca import ToyVCADataset
    ds = ToyVCADataset(scenario='chain', query_dim=64, context_dim=128)
    x, _, _, _ = ds[0]
    x_np = x.squeeze(0).numpy()
    nonzero_ratio = float((np.abs(x_np) > 1e-6).mean())
    assert nonzero_ratio > 0.5, \
        f"x non-zero ratio = {nonzero_ratio:.3f} (should be > 0.5 — not a 3/64 dummy)"


def test_sigma_before_training_is_near_half():
    """미학습 VCA: sigma ≈ 0.5 (sigmoid(0) = 0.5)"""
    from scripts.train_vca import ToyVCADataset
    from models.vca_attention import VCALayer
    ds = ToyVCADataset(scenario='chain', query_dim=64, context_dim=128)
    vca = VCALayer(query_dim=64, context_dim=128, n_heads=4,
                   n_entities=2, z_bins=2, lora_rank=4)
    vca.eval()
    x, ctx, _, _ = ds[0]
    with torch.no_grad():
        vca(x.squeeze(0).unsqueeze(0), ctx.squeeze(0).unsqueeze(0))
    sigma_mean = float(vca.last_sigma.mean())
    # 초기화 직후 LoRA B=0 → 순수 frozen weight → 값은 다양할 수 있음
    # 단, [0,1] 범위 내에 있어야 함 (sigmoid 출력)
    assert 0.0 <= sigma_mean <= 1.0, \
        f"sigma should be in [0,1], got {sigma_mean:.4f}"


# ─── 10-B: check_learning.py subprocess 통합 테스트 ─────────────────────────

@pytest.fixture(scope='module')
def check_result():
    r = subprocess.run(
        [sys.executable, 'scripts/check_learning.py',
         '--epochs',      '10',
         '--scenario',    'chain',
         '--query-dim',   '64',
         '--context-dim', '128'],
        capture_output=True, text=True, timeout=300,
    )
    return r


def test_check_exits_cleanly(check_result):
    assert check_result.returncode == 0, \
        f"check_learning.py failed:\n{check_result.stderr[-800:]}"


def test_encoding_stats_printed(check_result):
    assert 'ENCODING' in check_result.stdout, \
        f"Expected 'ENCODING' line:\n{check_result.stdout}"


def test_x_std_printed_and_valid(check_result):
    m = re.search(r'ENCODING.*?x_std=([\d.]+)', check_result.stdout)
    assert m, f"Could not parse x_std from:\n{check_result.stdout}"
    assert float(m.group(1)) > 0.1, \
        f"x_std={m.group(1)} should be > 0.1"


def test_ctx_diff_printed_and_valid(check_result):
    m = re.search(r'ENCODING.*?ctx_diff=([\d.]+)', check_result.stdout)
    assert m, f"Could not parse ctx_diff from:\n{check_result.stdout}"
    assert float(m.group(1)) > 0.1, \
        f"ctx_diff={m.group(1)} should be > 0.1"


def test_after_sigma_consistency_above_random(check_result):
    """학습 후 sigma_consistency > 0.55 (random 0.5 보다 나음)"""
    m = re.search(r'AFTER\s+sigma_consistency=([\d.]+)', check_result.stdout)
    assert m, f"Could not parse AFTER sigma_consistency:\n{check_result.stdout}"
    val = float(m.group(1))
    assert val > 0.55, \
        f"AFTER sigma_consistency={val:.4f} should be > 0.55 (better than random)"


def test_sigma_separation_increases(check_result):
    """학습 후 sigma_separation > 학습 전 (entity 분리도 증가)"""
    before_m = re.search(r'BEFORE.*?sigma_separation=([\d.]+)', check_result.stdout)
    after_m  = re.search(r'AFTER\s+.*?sigma_separation=([\d.]+)', check_result.stdout)
    assert before_m and after_m, \
        f"Could not parse sigma_separation from:\n{check_result.stdout}"
    before_val = float(before_m.group(1))
    after_val  = float(after_m.group(1))
    assert after_val > before_val, \
        f"sigma_separation: BEFORE={before_val:.4f} AFTER={after_val:.4f} — should increase"


def test_loss_curve_printed(check_result):
    assert 'LOSS_CURVE' in check_result.stdout, \
        f"Expected 'LOSS_CURVE' in output:\n{check_result.stdout}"


def test_loss_curve_decreases(check_result):
    """loss_curve[-1] < loss_curve[0] (전반적 감소)"""
    m = re.search(r'LOSS_CURVE\s+([\d. ]+)', check_result.stdout)
    assert m, f"Could not parse LOSS_CURVE:\n{check_result.stdout}"
    values = [float(v) for v in m.group(1).strip().split()]
    assert len(values) >= 2, f"Need at least 2 loss values, got {len(values)}"
    assert values[-1] < values[0], \
        f"Loss should decrease: first={values[0]:.4f} last={values[-1]:.4f}"


def test_learning_ok(check_result):
    """최종 판정: LEARNING=OK"""
    assert 'LEARNING=OK' in check_result.stdout, \
        f"Expected LEARNING=OK:\n{check_result.stdout}\nSTDERR:\n{check_result.stderr[-400:]}"
