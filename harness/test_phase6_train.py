"""Phase 6: VCA 학습 루프 + sigma GIF"""
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.phase6


# ─── 손실 함수 단위 테스트 ──────────────────────────────────────────────────
def test_l_ortho_zero_for_orthogonal():
    from models.losses import l_ortho
    # 2D 직교: [1,0,...] 과 [0,1,...] 은 gram=I → loss=0
    pe = torch.zeros(2, 64)
    pe[0, 0] = 1.0
    pe[1, 1] = 1.0
    loss = l_ortho(pe)
    assert loss.item() < 1e-5, f"Expected ~0, got {loss.item():.2e}"


def test_l_ortho_nonzero_for_parallel():
    from models.losses import l_ortho
    # 동일한 벡터 두 개: gram ≠ I → loss > 0
    pe = torch.zeros(2, 64)
    pe[0, 0] = 1.0
    pe[1, 0] = 1.0   # same direction
    loss = l_ortho(pe)
    assert loss.item() > 0.1, f"Expected >0.1 for parallel vectors, got {loss.item():.4f}"


def test_l_depth_ranking_penalizes_wrong_order():
    from models.losses import l_depth_ranking
    sigma = torch.zeros(1, 4, 2, 2)
    sigma[:, :, 0, 0] = 0.2   # entity 0 (front): low sigma — WRONG
    sigma[:, :, 1, 0] = 0.8   # entity 1 (back):  high sigma — WRONG
    loss = l_depth_ranking(sigma, [0, 1], margin=0.05)
    assert loss.item() > 0.0, f"Expected penalty, got loss={loss.item():.4f}"


def test_l_depth_ranking_zero_for_correct_order():
    from models.losses import l_depth_ranking
    sigma = torch.zeros(1, 4, 2, 2)
    sigma[:, :, 0, 0] = 0.9   # entity 0 (front): high sigma — CORRECT
    sigma[:, :, 1, 0] = 0.1   # entity 1 (back):  low sigma — CORRECT
    loss = l_depth_ranking(sigma, [0, 1], margin=0.05)
    assert loss.item() == 0.0, f"Expected 0 for correct order, got {loss.item():.4f}"


def test_l_depth_ranking_margin():
    """margin 경계: front-back 차이가 margin보다 작으면 패널티 발생"""
    from models.losses import l_depth_ranking
    sigma = torch.zeros(1, 4, 2, 2)
    sigma[:, :, 0, 0] = 0.55   # front
    sigma[:, :, 1, 0] = 0.50   # back — 차이 0.05 = margin → loss=0
    loss = l_depth_ranking(sigma, [0, 1], margin=0.05)
    assert loss.item() == 0.0


# ─── 학습 루프 통합 테스트 ──────────────────────────────────────────────────
@pytest.fixture(scope="module")
def train_result():
    r = subprocess.run(
        [sys.executable, 'scripts/train_vca.py',
         '--epochs',     '2',
         '--scenario',   'chain',
         '--batch-size', '2',
         '--query-dim',  '64',
         '--context-dim','128',
         '--out-dir',    'debug/train_test'],
        capture_output=True, text=True, timeout=120,
    )
    return r


def test_train_exits_cleanly(train_result):
    assert train_result.returncode == 0, \
        f"Training failed:\n{train_result.stderr[-600:]}"


def test_sigma_gifs_saved(train_result):
    gifs = sorted(Path('debug/train_test').glob('sigma_epoch*.gif'))
    assert len(gifs) >= 2, f"Expected ≥2 GIFs (one per epoch), got {len(gifs)}"


def test_loss_finite(train_result):
    losses = re.findall(r'loss=([\d.eE+\-]+)', train_result.stdout)
    assert len(losses) >= 2, f"Expected ≥2 loss values in stdout: {train_result.stdout}"
    assert all(math.isfinite(float(l)) for l in losses), \
        f"Non-finite loss: {losses}"


def test_depth_loss_logged(train_result):
    assert 'l_depth' in train_result.stdout, \
        f"l_depth not found in output:\n{train_result.stdout}"


def test_ortho_loss_logged(train_result):
    assert 'l_ortho' in train_result.stdout


def test_sigma_gif_has_3_panels(train_result):
    """학습 후 저장된 GIF가 3-panel 구조인지 확인"""
    import imageio.v3 as iio
    gifs = sorted(Path('debug/train_test').glob('sigma_epoch*.gif'))
    if not gifs:
        pytest.skip("No GIF found")
    frame = iio.imread(str(gifs[0]), index=0)
    # panel_size=64 (train_vca.py _save_sigma_gif) → width=192
    assert frame.shape[1] == 64 * 3, \
        f"Expected width=192 (3×64px panels), got {frame.shape[1]}"
