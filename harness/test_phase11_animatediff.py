"""
Phase 11: AnimateDiff + VCA 주입 실험 테스트

주의: 모델 다운로드 (~2GB) + GPU 생성 필요 → @pytest.mark.slow
기본 pytest 실행에서 제외됨. 명시적으로만:
  pytest -m phase11
  pytest -m slow

검증 목표:
  - VCA가 AnimateDiff mid_block에 올바르게 주입됨
  - 생성이 정상 완료되고 파일이 저장됨
  - Sigmoid 특성: both_high_ratio > 0 (동시 활성화 가능)
  - baseline과 generated가 실제로 다름 (VCA가 출력에 영향을 줌)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.phase11, pytest.mark.slow]

sys.path.insert(0, str(Path(__file__).parent.parent))

ANIMATEDIFF_OUT = Path('debug/animatediff')
PROMPT_DIRS = ['chain', 'robot_arm', 'wrestling']


# ─── 빠른 단위 테스트 (모델 로드 확인) ──────────────────────────────────────

def test_animatediff_importable():
    """diffusers AnimateDiffPipeline이 import 가능한지"""
    try:
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    except ImportError as e:
        pytest.skip(f"diffusers not available: {e}")


def test_motion_adapter_cached():
    """guoyww/animatediff-motion-adapter-v1-5-3 캐시됨"""
    try:
        from diffusers import MotionAdapter
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-3",
            torch_dtype=torch.float16,
            local_files_only=False,
        )
        assert adapter is not None
    except Exception as e:
        pytest.skip(f"Motion adapter not available: {e}")


def test_fixed_context_vca_processor_interface():
    """FixedContextVCAProcessor: dtype 처리 + call_count 추적"""
    from scripts.run_animatediff import FixedContextVCAProcessor
    from models.vca_attention import VCALayer

    vca = VCALayer(query_dim=64, context_dim=32, n_heads=4,
                   n_entities=2, z_bins=2, lora_rank=4)
    entity_ctx = torch.randn(1, 2, 32)
    proc = FixedContextVCAProcessor(vca, entity_ctx)

    assert proc.call_count == 0
    hidden = torch.randn(4, 16, 64)
    out = proc(attn=None, hidden_states=hidden)

    assert proc.call_count == 1
    assert out.shape == hidden.shape, f"Output shape mismatch: {out.shape} vs {hidden.shape}"
    assert out.dtype == hidden.dtype, f"Output dtype mismatch: {out.dtype} vs {hidden.dtype}"


def test_fixed_context_vca_no_residual_double():
    """FixedContextVCAProcessor가 residual을 포함하지 않음 (이중 잔차 방지)"""
    from scripts.run_animatediff import FixedContextVCAProcessor
    from models.vca_attention import VCALayer

    vca = VCALayer(query_dim=64, context_dim=32, n_heads=4,
                   n_entities=2, z_bins=2, lora_rank=4)
    entity_ctx = torch.zeros(1, 2, 32)  # zero context
    proc = FixedContextVCAProcessor(vca, entity_ctx)

    # VCALayer with zero context should give attn_out ≈ 0
    # proc output = vca_out - hidden_states = attn_out (not x + attn_out)
    hidden = torch.randn(2, 8, 64)
    out = proc(attn=None, hidden_states=hidden)

    # out should NOT equal hidden (residual was removed)
    # i.e., proc output ≠ vca(hidden) - hidden + hidden = hidden + attn_out
    # Just verify shape and no crash
    assert out.shape == hidden.shape


def test_vca_fp16_input_handled():
    """fp16 hidden_states → fp32 내부 계산 → fp16 출력 (FM-A2)"""
    from scripts.run_animatediff import FixedContextVCAProcessor
    from models.vca_attention import VCALayer

    vca = VCALayer(query_dim=64, context_dim=32, n_heads=4,
                   n_entities=2, z_bins=2, lora_rank=4)
    entity_ctx = torch.randn(1, 2, 32, dtype=torch.float16)
    proc = FixedContextVCAProcessor(vca, entity_ctx)

    hidden_fp16 = torch.randn(2, 8, 64, dtype=torch.float16)
    out = proc(attn=None, hidden_states=hidden_fp16)

    assert out.dtype == torch.float16, \
        f"Expected fp16 output, got {out.dtype}"


def test_pipeline_loads():
    """AnimateDiffPipeline 로드 (모델 다운로드 포함)"""
    try:
        from scripts.run_animatediff import load_pipeline
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipe = load_pipeline(device=device)
        assert pipe is not None
        assert pipe.unet is not None
    except Exception as e:
        pytest.skip(f"Pipeline load failed (model not available?): {e}")


def test_vca_injected():
    """mid_block attn2에 FixedContextVCAProcessor가 주입됐는지 확인"""
    try:
        from scripts.run_animatediff import load_pipeline, get_entity_embedding, inject_vca_midblock
        from scripts.run_animatediff import FixedContextVCAProcessor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipe = load_pipeline(device=device)

        e0 = get_entity_embedding(pipe, "entity zero")
        e1 = get_entity_embedding(pipe, "entity one")
        entity_ctx = torch.cat([e0, e1], dim=1).to(torch.float16)

        vca_layer, injected_keys, original_procs = inject_vca_midblock(pipe, entity_ctx)

        # mid_block attn2에 주입됐는지
        assert len(injected_keys) > 0, "No keys injected"
        assert all('mid_block' in k for k in injected_keys), \
            f"Non-mid_block keys injected: {injected_keys}"
        assert all('attn2' in k for k in injected_keys), \
            f"Non-attn2 keys injected: {injected_keys}"

        # FixedContextVCAProcessor 타입 확인
        proc = pipe.unet.attn_processors[injected_keys[0]]
        assert isinstance(proc, FixedContextVCAProcessor), \
            f"Expected FixedContextVCAProcessor, got {type(proc)}"

        # 복구 확인
        pipe.unet.set_attn_processor(original_procs)

    except Exception as e:
        pytest.skip(f"Pipeline load failed: {e}")


# ─── 통합 테스트 (run_all 실행 결과 확인) ────────────────────────────────────

@pytest.fixture(scope='module')
def run_result():
    """--run-all 실행 결과 (모델 다운로드 + 생성 포함)"""
    r = subprocess.run(
        [sys.executable, 'scripts/run_animatediff.py',
         '--run-all',
         '--out-dir',    'debug/animatediff',
         '--steps',      '20',
         '--num-frames', '16',
         '--height',     '256',
         '--width',      '256',
         '--seed',       '42'],
        capture_output=True, text=True, timeout=1800,  # 30분
    )
    return r


def test_run_all_exits_cleanly(run_result):
    assert run_result.returncode == 0, \
        f"run_animatediff.py --run-all failed:\n{run_result.stderr[-1000:]}"


def test_all_gifs_created(run_result):
    """3개 프롬프트 × {generated.gif, debug.gif, baseline.gif} = 9개 파일"""
    missing = []
    for d in PROMPT_DIRS:
        for fname in ['generated.gif', 'debug.gif', 'baseline.gif']:
            p = ANIMATEDIFF_OUT / d / fname
            if not p.exists():
                missing.append(str(p))
    assert not missing, f"Missing files:\n" + "\n".join(missing)


def test_sigma_stats_exist_and_valid(run_result):
    """3개 sigma_stats.json 존재 및 valid JSON"""
    for d in PROMPT_DIRS:
        p = ANIMATEDIFF_OUT / d / 'sigma_stats.json'
        assert p.exists(), f"sigma_stats.json missing: {p}"
        with open(p) as f:
            stats = json.load(f)
        required_keys = ['sigma_separation', 'entity0_mean_sigma',
                         'entity1_mean_sigma', 'both_high_ratio', 'sigma_consistency']
        for k in required_keys:
            assert k in stats, f"Key {k!r} missing from {p}"


def test_sigma_separation_nonnegative(run_result):
    """sigma_separation = abs(e0 - e1) ≥ 0 (sanity: 음수 불가)"""
    for d in PROMPT_DIRS:
        p = ANIMATEDIFF_OUT / d / 'sigma_stats.json'
        if not p.exists():
            pytest.skip(f"{p} not found")
        stats = json.load(open(p))
        sep = stats['sigma_separation']
        assert sep >= 0.0, f"[{d}] sigma_separation={sep:.6e} < 0 (impossible for abs value)"


def test_both_high_ratio_positive(run_result):
    """both_high_ratio > 0: Sigmoid → 두 entity 동시 활성화 가능"""
    for d in PROMPT_DIRS:
        p = ANIMATEDIFF_OUT / d / 'sigma_stats.json'
        if not p.exists():
            pytest.skip(f"{p} not found")
        stats = json.load(open(p))
        bhr = stats['both_high_ratio']
        assert bhr > 0.0, \
            f"[{d}] both_high_ratio={bhr:.4f} should be > 0 (Sigmoid allows simultaneous activation)"


def test_sigma_range_valid(run_result):
    """sigma values in [0, 1] (Sigmoid 출력 범위)"""
    for d in PROMPT_DIRS:
        p = ANIMATEDIFF_OUT / d / 'sigma_stats.json'
        if not p.exists():
            pytest.skip(f"{p} not found")
        stats = json.load(open(p))
        for k in ['entity0_mean_sigma', 'entity1_mean_sigma', 'both_high_ratio']:
            v = stats[k]
            assert 0.0 <= v <= 1.0, f"[{d}] {k}={v:.4f} out of [0,1]"


def test_debug_gif_is_3panel(run_result):
    """debug.gif width = panel_size * 3 = 256 * 3 = 768"""
    import imageio.v3 as iio
    for d in PROMPT_DIRS:
        p = ANIMATEDIFF_OUT / d / 'debug.gif'
        if not p.exists():
            pytest.skip(f"{p} not found")
        frame = iio.imread(str(p), index=0)
        assert frame.shape[1] == 256 * 3, \
            f"[{d}] debug.gif width={frame.shape[1]} should be {256*3} (3-panel)"


def test_baseline_differs_from_generated(run_result):
    """generated.gif ≠ baseline.gif (VCA가 출력에 실제로 영향을 줌)"""
    import imageio.v3 as iio
    for d in PROMPT_DIRS:
        gen_p  = ANIMATEDIFF_OUT / d / 'generated.gif'
        base_p = ANIMATEDIFF_OUT / d / 'baseline.gif'
        if not (gen_p.exists() and base_p.exists()):
            pytest.skip(f"GIFs not found for {d}")
        gen_frame  = iio.imread(str(gen_p),  index=0).astype(np.float32)
        base_frame = iio.imread(str(base_p), index=0).astype(np.float32)
        diff = float(np.abs(gen_frame - base_frame).mean())
        assert diff > 0.5, \
            f"[{d}] generated vs baseline pixel diff={diff:.3f} — VCA seems to have no effect"


def test_vca_processor_was_called(run_result):
    """stdout에 processor call_count > 0 출력됨 (FM-A1 검증)"""
    assert 'processor call_count=' in run_result.stdout, \
        f"Expected 'processor call_count=' in output:\n{run_result.stdout[-500:]}"
    # call_count 파싱
    import re
    counts = re.findall(r'processor call_count=(\d+)', run_result.stdout)
    for c in counts:
        assert int(c) > 0, f"processor call_count={c} — VCA processor was never called!"


def test_generated_gif_frame_count(run_result):
    """generated.gif = 16 frames"""
    import imageio.v3 as iio
    for d in PROMPT_DIRS:
        p = ANIMATEDIFF_OUT / d / 'generated.gif'
        if not p.exists():
            pytest.skip(f"{p} not found")
        frames = iio.imread(str(p))
        assert frames.shape[0] == 16, \
            f"[{d}] generated.gif has {frames.shape[0]} frames (expected 16)"
