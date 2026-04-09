"""Phase 13: Paper figure generation 테스트"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import imageio.v3 as iio

pytestmark = pytest.mark.phase13

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT_DIR  = Path('debug/figures_test')
CKPT     = Path('checkpoints/animatediff_test/chain_best.pt')
CHAIN    = OUT_DIR / 'chain'
SUMMARY  = OUT_DIR / 'summary'


def _pick_best_gpu_env() -> dict:
    """여유 있는 GPU 선택 후 env dict 반환."""
    if not torch.cuda.is_available():
        return {}
    best, best_free = 0, 0
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free, best = free, i
        except Exception:
            continue
    if best_free < 7 * 1024**3:
        pytest.skip(f"All GPUs < 7 GB free. Best: {best_free/1024**3:.1f} GB")
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': str(best)}


@pytest.fixture(scope='module')
def figures_result():
    if not CKPT.exists():
        pytest.skip(f"Checkpoint not found: {CKPT}")
    env = _pick_best_gpu_env()
    r = subprocess.run(
        [sys.executable, 'scripts/make_figures.py',
         '--scenario', 'chain',
         '--checkpoint', str(CKPT),
         '--out-dir',   str(OUT_DIR),
         '--seed',      '42',
         '--num-frames', '16',
         '--steps',     '20',
         '--height',    '256',
         '--width',     '256'],
        capture_output=True, text=True, timeout=600,
        env=env if env else None,
    )
    return r


# ─── 기본 ────────────────────────────────────────────────────────────────────

def test_exits_cleanly(figures_result):
    assert figures_result.returncode == 0, \
        f"make_figures.py failed:\n{figures_result.stderr[-1000:]}"


# ─── Figure 1: side_by_side ───────────────────────────────────────────────────

def test_side_by_side_created(figures_result):
    assert (CHAIN / 'side_by_side.gif').exists()


def test_side_by_side_is_double_width(figures_result):
    p = CHAIN / 'side_by_side.gif'
    if not p.exists():
        pytest.skip("side_by_side.gif not found")
    frame = iio.imread(str(p), index=0)
    assert frame.shape[1] >= 256 * 2, \
        f"side_by_side width={frame.shape[1]} should be ≥ 512"


def test_vca_changes_output(figures_result):
    """baseline ≠ generated (VCA가 출력에 실제로 영향을 줌)"""
    p = CHAIN / 'side_by_side.gif'
    if not p.exists():
        pytest.skip("side_by_side.gif not found")
    frame = iio.imread(str(p), index=0)
    w = frame.shape[1] // 2
    left  = frame[:, :w, :].astype(np.float32)
    right = frame[:, w:, :].astype(np.float32)
    diff  = float(np.abs(left - right).mean())
    assert diff > 0.5, f"baseline vs VCA diff={diff:.3f} — VCA has no effect?"


# ─── Figure 2: debug_before / debug_after ─────────────────────────────────────

def test_debug_before_created(figures_result):
    assert (CHAIN / 'debug_before.gif').exists()


def test_debug_after_created(figures_result):
    assert (CHAIN / 'debug_after.gif').exists()


def test_sigma_separates_after_training(figures_result):
    """sigma_stats_all.json: after sigma_separation > before"""
    p = SUMMARY / 'sigma_stats_all.json'
    if not p.exists():
        pytest.skip("sigma_stats_all.json not found")
    data = json.loads(p.read_text())
    assert 'chain' in data
    before_sep = data['chain']['before']['sigma_separation']
    after_sep  = data['chain']['after']['sigma_separation']
    assert after_sep > before_sep, \
        f"after sep={after_sep:.4f} should > before sep={before_sep:.4f}"


def test_before_sigma_near_uniform(figures_result):
    """debug_before: E0 sigma ≈ E1 sigma (미학습 → 0.5±0.1)"""
    p = SUMMARY / 'sigma_stats_all.json'
    if not p.exists():
        pytest.skip("sigma_stats_all.json not found")
    data = json.loads(p.read_text())
    b = data['chain']['before']
    diff = abs(b['e0_z0'] - b['e1_z0'])
    assert diff < 0.15, \
        f"Before training: E0={b['e0_z0']:.4f} E1={b['e1_z0']:.4f} diff={diff:.4f} " \
        f"(미학습 VCA는 균일해야 함)"


def test_after_sigma_separated(figures_result):
    """debug_after: E0 sigma ≠ E1 sigma (학습 후 분리)"""
    p = SUMMARY / 'sigma_stats_all.json'
    if not p.exists():
        pytest.skip("sigma_stats_all.json not found")
    data = json.loads(p.read_text())
    a = data['chain']['after']
    assert a['sigma_separation'] > 0.05, \
        f"After training separation={a['sigma_separation']:.4f} should be > 0.05"


# ─── Figure 3: comparison ─────────────────────────────────────────────────────

def test_comparison_gif_created(figures_result):
    assert (CHAIN / 'comparison.gif').exists()


def test_comparison_is_4panel(figures_result):
    p = CHAIN / 'comparison.gif'
    if not p.exists():
        pytest.skip("comparison.gif not found")
    frame = iio.imread(str(p), index=0)
    assert frame.shape[1] == 64 * 4, \
        f"comparison width={frame.shape[1]} should be {64*4} (4-panel × 64px)"


def test_sigmoid_e0_greater_than_softmax(figures_result):
    """Sigmoid 학습 후 E0 > Softmax E0 (구조적 이점)"""
    p = SUMMARY / 'sigma_stats_all.json'
    if not p.exists():
        pytest.skip("sigma_stats_all.json not found")
    data = json.loads(p.read_text())
    after_e0 = data['chain']['after']['e0_z0']
    # Softmax untrained: ~0.25 (1/(N*Z)=0.25), Sigmoid trained: ~0.667
    assert after_e0 > 0.4, \
        f"Sigmoid trained E0={after_e0:.4f} should be > 0.4 (Softmax untrained ≈ 0.25)"


# ─── Figure 4: training_progress ─────────────────────────────────────────────

def test_training_progress_created(figures_result):
    assert (CHAIN / 'training_progress.gif').exists()


def test_training_progress_is_4panel(figures_result):
    p = CHAIN / 'training_progress.gif'
    if not p.exists():
        pytest.skip("training_progress.gif not found")
    frame = iio.imread(str(p), index=0)
    assert frame.shape[1] == 64 * 4, \
        f"training_progress width={frame.shape[1]} should be {64*4}"


# ─── Summary ─────────────────────────────────────────────────────────────────

def test_manifest_created(figures_result):
    assert (SUMMARY / 'figure_manifest.json').exists()


def test_manifest_valid(figures_result):
    p = SUMMARY / 'figure_manifest.json'
    if not p.exists():
        pytest.skip("manifest not found")
    data = json.loads(p.read_text())
    assert 'figures' in data
    assert 'generated_at' in data
    assert len(data['figures']) > 0
    fig = data['figures'][0]
    assert 'id' in fig and 'path' in fig
    assert 'sigma_separation_before' in fig
    assert 'sigma_separation_after'  in fig


def test_sigma_stats_all_json(figures_result):
    p = SUMMARY / 'sigma_stats_all.json'
    assert p.exists()
    data = json.loads(p.read_text())
    assert 'chain' in data
    assert data['chain']['after']['sigma_separation'] > \
           data['chain']['before']['sigma_separation']


def test_best_side_by_side_created(figures_result):
    assert (SUMMARY / 'best_side_by_side.gif').exists()
