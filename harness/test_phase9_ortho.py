"""Phase 9: L_ortho Ablation — z-collapse 방지 효과 검증"""
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.phase9

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── z0_dominance 단위 테스트 ─────────────────────────────────────────────────

def test_z0_dominance_uniform():
    """sigma가 모든 z에 균등하면 z0_dominance = 1/Z (z=0은 전체 mass의 절반)"""
    from scripts.run_ortho_ablation import z0_dominance
    # Z=2, 균등 분포: z0_sum = BF*S*N*1, total_sum = BF*S*N*Z → ratio = 1/Z = 0.5
    sigma = torch.ones(2, 16, 2, 2)  # (BF, S, N, Z), all equal
    dom = z0_dominance(sigma)
    assert abs(dom - 0.5) < 1e-4, f"Uniform sigma should give z0_dominance≈0.5, got {dom:.4f}"


def test_z0_dominance_collapsed():
    """sigma가 z=0에만 집중되면 z0_dominance ≈ 1.0"""
    from scripts.run_ortho_ablation import z0_dominance
    sigma = torch.zeros(2, 16, 2, 2)
    sigma[:, :, :, 0] = 1.0   # z=0만 활성화
    dom = z0_dominance(sigma)
    assert dom > 0.99, f"z=0 only should give z0_dominance≈1.0, got {dom:.4f}"


def test_l_ortho_reduces_gram_off_diagonal():
    """L_ortho 적용 후 depth_pe gram matrix off-diagonal이 줄어드는지 확인"""
    from models.losses import l_ortho
    # 비-단위 벡터로 시작: gradient가 0이 되지 않도록 약간 다른 방향
    # (단위벡터+정확히 같은 방향이면 normalize Jacobian이 0 → gradient 소실)
    pe = torch.nn.Parameter(torch.zeros(2, 64))
    pe.data[0, 0] = 2.0
    pe.data[1, 0] = 2.0
    pe.data[1, 1] = 0.5   # 약간 다른 방향 → Jacobian이 0 아님
    opt = torch.optim.Adam([pe], lr=0.05)

    initial_loss = l_ortho(pe).item()
    for _ in range(30):
        opt.zero_grad()
        loss = l_ortho(pe)
        loss.backward()
        opt.step()
    final_loss = l_ortho(pe).item()

    assert final_loss < initial_loss, \
        f"L_ortho should reduce Gram off-diag: initial={initial_loss:.4f}, final={final_loss:.4f}"


def test_depth_pe_becomes_orthogonal_with_l_ortho():
    """VCALayer depth_pe가 L_ortho 학습 후 직교에 가까워지는지"""
    from models.vca_attention import VCALayer
    import torch.nn.functional as F
    vca = VCALayer(query_dim=64, context_dim=128, n_heads=4,
                   n_entities=2, z_bins=2, lora_rank=4, use_softmax=False)
    from models.losses import l_ortho
    opt = torch.optim.Adam([vca.depth_pe], lr=0.05)

    for _ in range(50):
        opt.zero_grad()
        l_ortho(vca.depth_pe).backward()
        opt.step()

    pe_norm = F.normalize(vca.depth_pe.detach(), dim=-1)
    gram = pe_norm @ pe_norm.T  # (Z, Z)
    # off-diagonal should be small
    off_diag = gram[0, 1].abs().item()
    assert off_diag < 0.3, \
        f"After L_ortho training, off-diagonal should be <0.3, got {off_diag:.4f}"


# ─── subprocess 통합 테스트 ──────────────────────────────────────────────────

@pytest.fixture(scope='module')
def ortho_result():
    r = subprocess.run(
        [sys.executable, 'scripts/run_ortho_ablation.py',
         '--epochs',   '5',
         '--scenario', 'chain',
         '--out-dir',  'debug/ortho_ablation'],
        capture_output=True, text=True, timeout=300,
    )
    return r


def test_ortho_ablation_exits_cleanly(ortho_result):
    assert ortho_result.returncode == 0, \
        f"run_ortho_ablation.py failed:\n{ortho_result.stderr[-800:]}"


def test_ortho_prints_final_no_ortho(ortho_result):
    assert 'FINAL no_ortho' in ortho_result.stdout, \
        f"Expected 'FINAL no_ortho' in output:\n{ortho_result.stdout[-400:]}"


def test_ortho_prints_final_with_ortho(ortho_result):
    assert 'FINAL with_ortho' in ortho_result.stdout, \
        f"Expected 'FINAL with_ortho' in output:\n{ortho_result.stdout[-400:]}"


def test_ortho_reduces_z0_dominance(ortho_result):
    """핵심 가설: L_ortho 있으면 z0_dominance 감소 (z-collapse 방지)"""
    lines = ortho_result.stdout
    no_match  = re.search(r'FINAL no_ortho\s+z0_dominance=([\d.]+)', lines)
    yes_match = re.search(r'FINAL with_ortho\s+z0_dominance=([\d.]+)', lines)
    if not no_match or not yes_match:
        pytest.skip(f"Could not parse z0_dominance from output:\n{lines[-600:]}")
    dom_no  = float(no_match.group(1))
    dom_yes = float(yes_match.group(1))
    assert dom_no >= dom_yes, \
        f"L_ortho should reduce z0_dominance: no_ortho={dom_no:.4f}, with_ortho={dom_yes:.4f}"


def test_ortho_checkpoints_saved(ortho_result):
    assert Path('debug/ortho_ablation/no_ortho_final.pt').exists(), \
        "no_ortho_final.pt not saved"
    assert Path('debug/ortho_ablation/with_ortho_final.pt').exists(), \
        "with_ortho_final.pt not saved"


def test_ortho_result_line_exists(ortho_result):
    assert 'RESULT:' in ortho_result.stdout, \
        f"Expected 'RESULT:' summary line in output:\n{ortho_result.stdout[-400:]}"
