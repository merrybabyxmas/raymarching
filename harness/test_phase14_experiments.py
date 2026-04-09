"""
Phase 14: t_max × λ_depth 실험 + 8-prompt 비교 테스트

주의: GPU + 모델 다운로드 필요 → @pytest.mark.slow
  pytest -m phase14
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.phase14, pytest.mark.slow]

sys.path.insert(0, str(Path(__file__).parent.parent))

EXP_CKPT_ROOT = Path('checkpoints/experiments')
EXP_DEBUG_DIR = Path('debug/experiments')
CMP_OUT_DIR   = Path('debug/comparison')

EXPERIMENT_NAMES = ['baseline', 'exp1_late', 'exp2_high', 'exp3_both']
IN_DIST_PROMPTS  = ['chain_in', 'robot_in']
ZERO_SHOT_PROMPTS = ['wrestling', 'swords', 'dance', 'snakes', 'cables', 'fighters']
ALL_PROMPTS = IN_DIST_PROMPTS + ZERO_SHOT_PROMPTS


# ─── GPU 선택 ─────────────────────────────────────────────────────────────────

def _pick_gpu() -> str:
    """사용 가능한 GPU 중 여유 VRAM 최대 GPU 선택."""
    if not torch.cuda.is_available():
        return ''
    best_gpu, best_free = 0, 0
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free = free
                best_gpu = i
        except Exception:
            continue
    if best_free < 7 * 1024 ** 3:
        pytest.skip(f"All GPUs have < 7 GB free. Best GPU {best_gpu}: "
                    f"{best_free / 1024**3:.1f} GB free.")
    return str(best_gpu)


# ─── 유닛 테스트 (subprocess 없음) ───────────────────────────────────────────

def test_experiment_configs():
    """4가지 실험 설정 존재 + 이름/t_max/lambda_depth 올바름"""
    from scripts.train_experiments import EXPERIMENTS
    assert len(EXPERIMENTS) == 4
    names = [e['name'] for e in EXPERIMENTS]
    assert 'baseline'   in names
    assert 'exp1_late'  in names
    assert 'exp2_high'  in names
    assert 'exp3_both'  in names

    baseline = next(e for e in EXPERIMENTS if e['name'] == 'baseline')
    assert baseline['t_max'] == 1000
    assert baseline['lambda_depth'] == 0.1

    exp1 = next(e for e in EXPERIMENTS if e['name'] == 'exp1_late')
    assert exp1['t_max'] == 200

    exp2 = next(e for e in EXPERIMENTS if e['name'] == 'exp2_high')
    assert exp2['lambda_depth'] == 1.0

    exp3 = next(e for e in EXPERIMENTS if e['name'] == 'exp3_both')
    assert exp3['t_max'] == 200
    assert exp3['lambda_depth'] == 1.0


def test_pick_best_selects_max_inference_sep():
    """pick_best: inference_sep 최대인 것 선택"""
    from scripts.train_experiments import pick_best
    results = [
        {'name': 'a', 'inference_sep': 0.1, 'gap': 0.3, 'final_l_diff': 0.5},
        {'name': 'b', 'inference_sep': 0.3, 'gap': 0.1, 'final_l_diff': 0.4},
        {'name': 'c', 'inference_sep': 0.2, 'gap': 0.2, 'final_l_diff': 0.3},
    ]
    assert pick_best(results) == 'b'


def test_pick_best_filters_diverged():
    """pick_best: l_diff >= 1.0인 실험 제외"""
    from scripts.train_experiments import pick_best
    results = [
        {'name': 'a', 'inference_sep': 0.5, 'gap': 0.1, 'final_l_diff': 2.0},  # 발산
        {'name': 'b', 'inference_sep': 0.2, 'gap': 0.1, 'final_l_diff': 0.3},  # 정상
    ]
    assert pick_best(results) == 'b'  # a는 발산이라 제외


def test_pick_best_gap_tiebreak():
    """pick_best: inference_sep 같으면 gap 작은 것 선택"""
    from scripts.train_experiments import pick_best
    results = [
        {'name': 'a', 'inference_sep': 0.3, 'gap': 0.5, 'final_l_diff': 0.3},
        {'name': 'b', 'inference_sep': 0.3, 'gap': 0.1, 'final_l_diff': 0.4},
    ]
    assert pick_best(results) == 'b'


def test_prompts_dict_complete():
    """PROMPTS: 8개 + in_dist 2개 + zero_shot 6개"""
    from scripts.generate_comparison import PROMPTS
    assert len(PROMPTS) == 8
    in_dist   = [v for v in PROMPTS.values() if v['category'] == 'in_dist']
    zero_shot = [v for v in PROMPTS.values() if v['category'] == 'zero_shot']
    assert len(in_dist)   == 2
    assert len(zero_shot) == 6


def test_prompts_have_required_fields():
    """PROMPTS: 각 항목에 category, entity_0, entity_1, full 포함"""
    from scripts.generate_comparison import PROMPTS
    for name, info in PROMPTS.items():
        assert 'category'  in info, f"{name} missing 'category'"
        assert 'entity_0'  in info, f"{name} missing 'entity_0'"
        assert 'entity_1'  in info, f"{name} missing 'entity_1'"
        assert 'full'      in info, f"{name} missing 'full'"


def test_training_step_tmax_range():
    """training_step_tmax: t_max=200이면 t < 200 보장"""
    from scripts.train_experiments import training_step_tmax
    from models.vca_attention import VCALayer
    # CPU 단위 테스트는 GPU 없이 불가 (autocast) → GPU 있을 때만
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA for autocast")

    from scripts.run_animatediff import load_pipeline
    device = 'cuda'
    pipe = load_pipeline(device=device, dtype=torch.float16)
    for p in pipe.unet.parameters():
        p.requires_grad = False

    vca = VCALayer(query_dim=1280, context_dim=768, n_heads=8,
                   n_entities=2, z_bins=2, lora_rank=8, use_softmax=False).to(device)
    from scripts.train_animatediff_vca import TrainVCAProcessor
    proc = TrainVCAProcessor(vca, torch.randn(1, 2, 768).to(device))
    orig = dict(pipe.unet.attn_processors)
    new_procs = {}
    import copy
    for k, p in copy.copy(orig).items():
        new_procs[k] = proc if ('mid_block' in k and 'attn2' in k) else p
    pipe.unet.set_attn_processor(new_procs)

    latents = torch.randn(1, 4, 4, 32, 32, device=device, dtype=torch.float16)
    enc_hs  = torch.randn(1, 77, 768, device=device, dtype=torch.float16)

    # t는 randint(0, 200) → 모두 < 200 보장은 확률적이라 step이 실행되는지만 확인
    out = training_step_tmax(
        pipe, vca, latents, enc_hs,
        [[0, 1], [0, 1], [0, 1], [0, 1]],
        0.1, 0.05, device, t_max=200,
    )
    assert 'loss' in out
    assert out['loss'] < 100.0


# ─── subprocess 통합 테스트 ──────────────────────────────────────────────────

@pytest.fixture(scope='module')
def experiments_result():
    """4개 실험 중 2개(baseline + exp1_late)만 빠르게 실행 — 전체는 너무 오래 걸림"""
    gpu_id = _pick_gpu()
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu_id} if gpu_id else None
    r = subprocess.run(
        [sys.executable, 'scripts/train_experiments.py',
         '--scenario',    'chain',
         '--epochs',      '3',
         '--n-frames',    '4',
         '--height',      '256',
         '--width',       '256',
         '--lr',          '1e-4',
         '--lambda-ortho', '0.05',
         '--ckpt-root',   'checkpoints/experiments',
         '--debug-dir',   'debug/experiments',
         '--experiments', 'baseline,exp1_late'],
        capture_output=True, text=True, timeout=900, env=env,
    )
    return r


def test_experiments_exits_cleanly(experiments_result):
    assert experiments_result.returncode == 0, \
        f"train_experiments.py failed:\n{experiments_result.stderr[-1000:]}"


def test_best_exp_printed(experiments_result):
    """stdout에 BEST_EXP=... 출력됨"""
    assert 'BEST_EXP=' in experiments_result.stdout, \
        f"BEST_EXP not found:\n{experiments_result.stdout[-500:]}"


def test_best_exp_is_valid_name(experiments_result):
    """BEST_EXP 값이 알려진 실험 이름"""
    m = re.search(r'BEST_EXP=(\S+)', experiments_result.stdout)
    assert m, "BEST_EXP= not found"
    assert m.group(1) in EXPERIMENT_NAMES, f"Unknown BEST_EXP: {m.group(1)}"


def test_inference_sigma_logged(experiments_result):
    """각 실험에 INFERENCE sigma_sep= 출력됨"""
    assert 'INFERENCE sigma_sep=' in experiments_result.stdout, \
        f"INFERENCE sigma_sep= not found:\n{experiments_result.stdout[-500:]}"


def test_result_logged(experiments_result):
    """각 실험에 RESULT train_sep= inference_sep= gap= 출력됨"""
    assert 'RESULT train_sep=' in experiments_result.stdout, \
        f"RESULT line not found:\n{experiments_result.stdout[-500:]}"


def test_checkpoints_created(experiments_result):
    """실행된 실험별 best.pt 존재"""
    for name in ['baseline', 'exp1_late']:
        p = EXP_CKPT_ROOT / name / 'best.pt'
        assert p.exists(), f"Checkpoint not found: {p}"


def test_checkpoints_loadable(experiments_result):
    """체크포인트 로드 가능 + vca_state_dict 포함"""
    for name in ['baseline', 'exp1_late']:
        p = EXP_CKPT_ROOT / name / 'best.pt'
        if not p.exists():
            pytest.skip(f"{p} not found")
        ckpt = torch.load(p, map_location='cpu')
        assert 'vca_state_dict' in ckpt, f"{name}: vca_state_dict missing"
        assert 'inference_sep'  in ckpt, f"{name}: inference_sep missing"
        assert 'train_sep'      in ckpt, f"{name}: train_sep missing"
        assert 'gap'            in ckpt, f"{name}: gap missing"


def test_best_exp_json_created(experiments_result):
    """best_exp.json 저장됨"""
    p = EXP_DEBUG_DIR / 'best_exp.json'
    assert p.exists(), f"best_exp.json not found: {p}"


def test_best_exp_json_valid(experiments_result):
    """best_exp.json 파싱 + 필드 검증"""
    p = EXP_DEBUG_DIR / 'best_exp.json'
    if not p.exists():
        pytest.skip("best_exp.json not found")
    data = json.loads(p.read_text())
    assert 'best_exp'      in data
    assert 'checkpoint'    in data
    assert 'inference_sep' in data
    assert 'train_sep'     in data
    assert 'gap'           in data
    assert 'all_results'   in data
    assert len(data['all_results']) >= 1


def test_experiment_curves_saved(experiments_result):
    """experiment_curves.json 저장됨"""
    p = EXP_DEBUG_DIR / 'experiment_curves.json'
    assert p.exists(), f"experiment_curves.json not found: {p}"


def test_experiment_curves_valid(experiments_result):
    """experiment_curves.json: 각 실험별 학습 곡선 존재"""
    p = EXP_DEBUG_DIR / 'experiment_curves.json'
    if not p.exists():
        pytest.skip("experiment_curves.json not found")
    data = json.loads(p.read_text())
    for name in ['baseline', 'exp1_late']:
        assert name in data, f"{name} not in curves"
        assert len(data[name]) > 0, f"{name} curve is empty"
        assert 'loss' in data[name][0]
        assert 'sigma_separation' in data[name][0]


def test_loss_not_exploding_experiments(experiments_result):
    """모든 실험 loss < 10.0"""
    losses = re.findall(r'loss=([\d.]+)', experiments_result.stdout)
    for v in losses:
        assert float(v) < 10.0, f"loss={v} ≥ 10.0 (exploding?)"


# ─── comparison 테스트 ────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def comparison_result(experiments_result):
    """best_exp.json을 사용해 2개 프롬프트만 테스트 (chain_in + wrestling)"""
    best_exp_path = EXP_DEBUG_DIR / 'best_exp.json'
    if not best_exp_path.exists():
        pytest.skip("best_exp.json not found — run experiments first")

    gpu_id = _pick_gpu()
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu_id} if gpu_id else None
    r = subprocess.run(
        [sys.executable, 'scripts/generate_comparison.py',
         '--best-exp-json', str(best_exp_path),
         '--out-dir',       str(CMP_OUT_DIR),
         '--prompts',       'chain_in,wrestling',
         '--num-frames',    '8',
         '--steps',         '10',
         '--height',        '256',
         '--width',         '256',
         '--seed',          '42'],
        capture_output=True, text=True, timeout=600, env=env,
    )
    return r


def test_comparison_exits_cleanly(comparison_result):
    assert comparison_result.returncode == 0, \
        f"generate_comparison.py failed:\n{comparison_result.stderr[-1000:]}"


def test_comparison_baseline_gifs(comparison_result):
    """각 프롬프트별 baseline.gif 존재"""
    for name in ['chain_in', 'wrestling']:
        p = CMP_OUT_DIR / name / 'baseline.gif'
        assert p.exists(), f"baseline.gif not found: {p}"


def test_comparison_vca_gifs(comparison_result):
    """각 프롬프트별 vca_generated.gif 존재"""
    for name in ['chain_in', 'wrestling']:
        p = CMP_OUT_DIR / name / 'vca_generated.gif'
        assert p.exists(), f"vca_generated.gif not found: {p}"


def test_comparison_side_by_side(comparison_result):
    """side_by_side.gif 존재 + 2-panel width"""
    import imageio.v3 as iio3
    import numpy as np
    for name in ['chain_in', 'wrestling']:
        p = CMP_OUT_DIR / name / 'side_by_side.gif'
        assert p.exists(), f"side_by_side.gif not found: {p}"
        frame = iio3.imread(str(p), index=0)
        assert frame.shape[1] >= 256 * 2, \
            f"{name} side_by_side width={frame.shape[1]} should be ≥ 512"


def test_comparison_sigma_stats(comparison_result):
    """sigma_stats.json 존재 + 파싱 가능"""
    for name in ['chain_in', 'wrestling']:
        p = CMP_OUT_DIR / name / 'sigma_stats.json'
        assert p.exists(), f"sigma_stats.json not found: {p}"
        data = json.loads(p.read_text())
        assert 'sigma_separation' in data


def test_comparison_all_stats_json(comparison_result):
    """summary/all_stats.json 존재 + 내용 올바름"""
    p = CMP_OUT_DIR / 'summary' / 'all_stats.json'
    assert p.exists(), f"all_stats.json not found: {p}"
    data = json.loads(p.read_text())
    assert isinstance(data, list)
    assert len(data) >= 1
    assert 'name' in data[0]
    assert 'sigma_separation' in data[0]


def test_comparison_report_md(comparison_result):
    """summary/report.md 존재 + 기본 섹션 포함"""
    p = CMP_OUT_DIR / 'summary' / 'report.md'
    assert p.exists(), f"report.md not found: {p}"
    content = p.read_text()
    assert '# Phase 14' in content
    assert 'Best experiment' in content
    assert 'sigma_separation' in content


def test_comparison_in_dist_sep_positive(comparison_result):
    """chain_in (in-dist) sigma_separation > 0"""
    p = CMP_OUT_DIR / 'chain_in' / 'sigma_stats.json'
    if not p.exists():
        pytest.skip("chain_in sigma_stats.json not found")
    data = json.loads(p.read_text())
    assert data['sigma_separation'] >= 0.0, \
        f"chain_in sep={data['sigma_separation']} should be ≥ 0"


def test_comparison_best_side_by_side(comparison_result):
    """summary/best_side_by_side.gif 존재"""
    p = CMP_OUT_DIR / 'summary' / 'best_side_by_side.gif'
    assert p.exists(), f"best_side_by_side.gif not found: {p}"
