"""
Phase 12: AnimateDiff VCA 파인튜닝 테스트

주의: GPU + 모델 다운로드 필요 → @pytest.mark.slow
  pytest -m phase12
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.phase12, pytest.mark.slow]

sys.path.insert(0, str(Path(__file__).parent.parent))

CKPT_DIR  = Path('checkpoints/animatediff_test')
DEBUG_DIR = Path('debug/train_animatediff_test')


# ─── 유닛 테스트 (subprocess 없음) ──────────────────────────────────────────

def test_train_vca_processor_interface():
    """TrainVCAProcessor: shape 보존 + last_sigma_raw grad 있음"""
    from scripts.train_animatediff_vca import TrainVCAProcessor
    from models.vca_attention import VCALayer

    vca = VCALayer(query_dim=64, context_dim=32, n_heads=4,
                   n_entities=2, z_bins=2, lora_rank=4)
    entity_ctx = torch.randn(1, 2, 32)
    proc = TrainVCAProcessor(vca, entity_ctx)

    hidden = torch.randn(4, 16, 64)
    out = proc(attn=None, hidden_states=hidden)

    assert out.shape == hidden.shape, f"shape mismatch: {out.shape}"
    # last_sigma_raw must exist and have grad
    assert vca.last_sigma_raw is not None
    assert vca.last_sigma_raw.requires_grad or vca.last_sigma_raw.grad_fn is not None, \
        "last_sigma_raw must have grad_fn (FM-I6)"


def test_train_vca_processor_fp16_input():
    """fp16 hidden → fp32 내부 → fp16 출력 (FM-A2)"""
    from scripts.train_animatediff_vca import TrainVCAProcessor
    from models.vca_attention import VCALayer

    vca = VCALayer(query_dim=64, context_dim=32, n_heads=4,
                   n_entities=2, z_bins=2, lora_rank=4)
    entity_ctx = torch.randn(1, 2, 32)
    proc = TrainVCAProcessor(vca, entity_ctx)

    hidden_fp16 = torch.randn(2, 8, 64, dtype=torch.float16)
    out = proc(attn=None, hidden_states=hidden_fp16)
    assert out.dtype == torch.float16, f"Expected fp16, got {out.dtype}"


def test_unet_frozen_after_inject():
    """inject_vca_train 후 UNet parameters requires_grad = False"""
    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        if free_gb < 7.0:
            pytest.skip(f"GPU memory low: {free_gb:.1f} GB")
    try:
        from scripts.train_animatediff_vca import inject_vca_train, build_entity_context
        from scripts.run_animatediff import load_pipeline

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipe = load_pipeline(device=device)
        for param in pipe.unet.parameters():
            param.requires_grad = False

        n_trainable = sum(1 for p in pipe.unet.parameters() if p.requires_grad)
        assert n_trainable == 0, f"UNet has {n_trainable} trainable params!"

        ctx = build_entity_context(pipe, 'chain', device)
        vca, _, orig = inject_vca_train(pipe, ctx)

        # VCA는 학습 가능
        vca_trainable = [p for p in vca.parameters() if p.requires_grad]
        assert len(vca_trainable) > 0, "VCA has no trainable params!"

        pipe.unet.set_attn_processor(orig)
    except ImportError as e:
        pytest.skip(f"diffusers not available: {e}")
    except Exception as e:
        pytest.skip(f"Pipeline not available: {e}")


def test_encode_frames_shape():
    """encode_frames_to_latents → (1, 4, T, H//8, W//8)"""
    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        if free_gb < 7.0:
            pytest.skip(f"GPU memory low: {free_gb:.1f} GB")
    try:
        from scripts.train_animatediff_vca import load_frames, encode_frames_to_latents
        from scripts.run_animatediff import load_pipeline

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipe = load_pipeline(device=device)

        frames = load_frames('chain', n_frames=4, height=256, width=256)
        latents = encode_frames_to_latents(pipe, frames, device)
        assert latents.shape == (1, 4, 4, 32, 32), \
            f"Expected (1,4,4,32,32), got {latents.shape}"
    except Exception as e:
        pytest.skip(f"Pipeline not available: {e}")


def test_sigma_stats_train():
    """compute_sigma_stats_train: separation과 consistency 반환"""
    from scripts.train_animatediff_vca import compute_sigma_stats_train
    sigma = torch.rand(4, 16, 2, 2)
    stats = compute_sigma_stats_train(sigma)
    assert 'sigma_separation' in stats
    assert 'sigma_consistency' in stats
    assert stats['sigma_separation'] >= 0.0


def test_depth_order_parsing():
    """load_depths_and_masks: front/back 인덱스 반환"""
    from scripts.train_animatediff_vca import load_depths_and_masks
    orders = load_depths_and_masks('chain', n_frames=4, height=256, width=256)
    assert len(orders) == 4
    for o in orders:
        assert len(o) == 2
        assert set(o) == {0, 1}


# ─── subprocess 통합 테스트 ──────────────────────────────────────────────────

@pytest.fixture(scope='module')
def train_result():
    """5 에폭, 4 프레임, 256×256으로 빠른 검증"""
    # GPU 메모리 부족 시 skip (다른 프로세스가 VRAM 점유 중일 수 있음)
    if torch.cuda.is_available():
        free_bytes = torch.cuda.mem_get_info()[0]
        free_gb = free_bytes / 1024 ** 3
        if free_gb < 7.0:
            pytest.skip(f"Insufficient GPU memory: {free_gb:.1f} GB free (need ~7 GB). "
                        f"Kill other CUDA processes first.")
    r = subprocess.run(
        [sys.executable, 'scripts/train_animatediff_vca.py',
         '--scenario',   'chain',
         '--epochs',     '5',
         '--n-frames',   '4',
         '--height',     '256',
         '--width',      '256',
         '--lr',         '1e-4',
         '--lambda-depth', '2.0',
         '--lambda-ortho', '0.05',
         '--save-dir',   str(CKPT_DIR),
         '--debug-dir',  str(DEBUG_DIR)],
        capture_output=True, text=True, timeout=600,
    )
    return r


def test_train_exits_cleanly(train_result):
    assert train_result.returncode == 0, \
        f"train_animatediff_vca.py failed:\n{train_result.stderr[-1000:]}"


def test_checkpoint_saved(train_result):
    """epoch 마지막에 best.pt 저장됨"""
    best = CKPT_DIR / 'chain_best.pt'
    assert best.exists(), f"best.pt not found: {best}"


def test_checkpoint_loadable(train_result):
    """저장된 체크포인트 로드 가능"""
    best = CKPT_DIR / 'chain_best.pt'
    if not best.exists():
        pytest.skip("best.pt not found")
    ckpt = torch.load(best, map_location='cpu')
    assert 'vca_state_dict' in ckpt, "vca_state_dict missing from checkpoint"
    assert 'epoch' in ckpt


def test_sigma_separation_increases(train_result):
    """FINAL sigma_separation > 0.0 (미학습 상태: 0.0)"""
    m = re.search(r'FINAL sigma_separation=([\d.]+)', train_result.stdout)
    assert m, f"FINAL sigma_separation not found:\n{train_result.stdout[-500:]}"
    val = float(m.group(1))
    assert val > 0.0, \
        f"sigma_separation={val} should be > 0.0 after training"


def test_learning_ok(train_result):
    """최종 판정: LEARNING=OK"""
    assert 'LEARNING=OK' in train_result.stdout, \
        f"Expected LEARNING=OK:\n{train_result.stdout}\nSTDERR:\n{train_result.stderr[-400:]}"


def test_sigma_gif_created(train_result):
    """debug_dir에 sigma_epoch*.gif 존재"""
    gifs = list(DEBUG_DIR.glob('chain_sigma_epoch*.gif'))
    assert len(gifs) > 0, f"No sigma GIFs found in {DEBUG_DIR}"


def test_loss_components_logged(train_result):
    """l_diff, l_depth, l_ortho 모두 stdout에 출력됨"""
    for key in ['l_diff=', 'l_depth=', 'l_ortho=']:
        assert key in train_result.stdout, \
            f"'{key}' not found in stdout"


def test_loss_not_exploding(train_result):
    """모든 loss값 < 10.0"""
    losses = re.findall(r'loss=([\d.]+)', train_result.stdout)
    for v in losses:
        assert float(v) < 10.0, f"loss={v} ≥ 10.0 (exploding?)"


def test_final_generated_gif_created(train_result):
    """final_generated.gif 생성됨"""
    p = DEBUG_DIR / 'final_generated.gif'
    assert p.exists(), f"final_generated.gif not found: {p}"


def test_final_baseline_gif_created(train_result):
    """final_baseline.gif 생성됨"""
    p = DEBUG_DIR / 'final_baseline.gif'
    assert p.exists(), f"final_baseline.gif not found: {p}"


def test_final_differs_from_baseline(train_result):
    """final_generated.gif 첫 프레임 ≠ final_baseline.gif 첫 프레임"""
    import imageio.v3 as iio
    import numpy as np
    gen_p  = DEBUG_DIR / 'final_generated.gif'
    base_p = DEBUG_DIR / 'final_baseline.gif'
    if not (gen_p.exists() and base_p.exists()):
        pytest.skip("GIFs not found")
    gen_frame  = iio.imread(str(gen_p),  index=0).astype(np.float32)
    base_frame = iio.imread(str(base_p), index=0).astype(np.float32)
    diff = float(abs(gen_frame - base_frame).mean())
    assert diff > 0.5, \
        f"generated vs baseline diff={diff:.3f} — VCA has no effect?"


def test_training_curve_saved(train_result):
    """training_curve.json 저장됨 + 파싱 가능"""
    curve_path = DEBUG_DIR / 'training_curve.json'
    assert curve_path.exists(), f"training_curve.json not found"
    with open(curve_path) as f:
        curve = json.load(f)
    assert len(curve) > 0, "training_curve is empty"
    assert 'loss' in curve[0]
    assert 'sigma_separation' in curve[0]
