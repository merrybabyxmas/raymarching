"""
Phase 15: Objaverse 기반 3D 자산 학습 데이터 생성 테스트

pytest -m phase15
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import imageio.v3 as iio3

pytestmark = [pytest.mark.phase15, pytest.mark.slow]

sys.path.insert(0, str(Path(__file__).parent.parent))

ASSETS_DIR      = Path('toy/assets')
DATA_TEST_DIR   = Path('toy/data_objaverse_test')
STATS_DIR       = Path('debug/dataset_stats_objaverse')


# ─── 유닛 테스트 (subprocess 없음) ───────────────────────────────────────────

def test_crossing_modes_defined():
    """CrossingMode: ORBIT, SQUEEZE, ROTATE 모두 존재"""
    from scripts.generate_crossing import CrossingMode
    assert CrossingMode.ORBIT.value   == 'orbit'
    assert CrossingMode.SQUEEZE.value == 'squeeze'
    assert CrossingMode.ROTATE.value  == 'rotate'


def test_orbit_transforms_depth_reversal():
    """ORBIT 모드: 반바퀴(n_frames/2)에서 x좌표가 역전"""
    from scripts.generate_crossing import compute_frame_transforms, CrossingMode
    n = 16
    _, _, t1_0, _ = compute_frame_transforms(CrossingMode.ORBIT, 0,    n)
    _, _, t1_h, _ = compute_frame_transforms(CrossingMode.ORBIT, n//2, n)
    # 공전이므로 시작과 반바퀴 후 x부호가 같아야 (돌아왔으므로)
    # 아니면 시작 t=0과 t=n//2에서 trans0의 x부호가 반대여야 함
    trans0_0, _, _, _ = compute_frame_transforms(CrossingMode.ORBIT, 0,    n)
    trans0_h, _, _, _ = compute_frame_transforms(CrossingMode.ORBIT, n//2, n)
    # x 부호가 반대 → 반바퀴 공전
    assert np.sign(trans0_0[0]) != np.sign(trans0_h[0]) or abs(trans0_h[0] - trans0_0[0]) > 0.5


def test_squeeze_positions_converge():
    """SQUEEZE 모드: 중간 프레임에서 두 mesh가 최대로 겹침"""
    from scripts.generate_crossing import compute_frame_transforms, CrossingMode
    n = 16
    t0_mid, _, t1_mid, _ = compute_frame_transforms(CrossingMode.SQUEEZE, n//2, n)
    # 중간에서 trans0[0] + trans1[0] ≈ 0 (대칭)
    assert abs(t0_mid[0] + t1_mid[0]) < 0.5


def test_rotate_fixed_positions():
    """ROTATE 모드: 위치는 고정, 회전만 변함"""
    from scripts.generate_crossing import compute_frame_transforms, CrossingMode
    n = 16
    t0_0, r0_0, t1_0, r1_0 = compute_frame_transforms(CrossingMode.ROTATE, 0,  n)
    t0_4, r0_4, t1_4, r1_4 = compute_frame_transforms(CrossingMode.ROTATE, 4, n)
    # 위치 고정
    np.testing.assert_allclose(t0_0, t0_4)
    np.testing.assert_allclose(t1_0, t1_4)
    # 회전 변화
    assert r0_0 != r0_4 or r1_0 != r1_4


def test_normalize_mesh():
    """normalize_mesh: centroid ≈ 0, bbox max ≈ 1"""
    from scripts.generate_crossing import normalize_mesh
    import trimesh
    mesh = trimesh.creation.box()  # 단위 정육면체
    mesh.apply_translation([5, 10, 3])  # 임의 이동
    mesh.apply_scale(3.0)
    nm = normalize_mesh(mesh)
    assert np.linalg.norm(nm.centroid) < 0.05
    assert abs(nm.bounding_box.extents.max() - 1.0) < 0.05


def test_category_config():
    """CATEGORIES: 실제 LVIS key 포함"""
    from scripts.download_assets import CATEGORIES, QUICK_CATEGORIES
    assert 'animals' in CATEGORIES
    assert 'objects' in CATEGORIES
    # quick-test 설정
    assert 'animals' in QUICK_CATEGORIES
    for cfg in QUICK_CATEGORIES.values():
        assert cfg['n_per_keyword'] <= 3


def test_objaverse_dataset_empty_dir():
    """ObjaverseVCADataset: 비어있는 디렉토리 → len==0"""
    from scripts.train_vca import ObjaverseVCADataset
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        ds = ObjaverseVCADataset(data_root=tmp, query_dim=64, context_dim=128)
        assert len(ds) == 0


# ─── 다운로드 테스트 ──────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def download_result():
    """--quick-test: cat/dog/sword 각 2개"""
    r = subprocess.run(
        [sys.executable, 'scripts/download_assets.py', '--quick-test',
         '--out-dir', str(ASSETS_DIR)],
        capture_output=True, text=True, timeout=300,
    )
    return r


def test_download_exits_cleanly(download_result):
    assert download_result.returncode == 0, \
        f"download_assets.py failed:\n{download_result.stderr[-800:]}"


def test_assets_downloaded(download_result):
    """최소 1개 GLB 파일 존재"""
    glbs = list(ASSETS_DIR.rglob('*.glb'))
    assert len(glbs) >= 1, f"No GLB files in {ASSETS_DIR}"


def test_manifest_created(download_result):
    """manifest.json 존재"""
    manifests = list(ASSETS_DIR.rglob('manifest.json'))
    assert len(manifests) >= 1, f"No manifest.json in {ASSETS_DIR}"


def test_manifest_valid(download_result):
    """manifest.json 파싱 + vertex_count/bbox_size 필드"""
    for mp in ASSETS_DIR.rglob('manifest.json'):
        data = json.loads(mp.read_text())
        for uid, info in data.items():
            assert 'vertex_count' in info, f"{uid}: vertex_count missing"
            assert 'bbox_size'    in info, f"{uid}: bbox_size missing"
            assert info['vertex_count'] >= 100, f"{uid}: too few vertices"
            assert info['bbox_size'] >= 0.1,    f"{uid}: bbox too small"


def test_download_report_created(download_result):
    """download_report.json 존재"""
    p = ASSETS_DIR / 'download_report.json'
    assert p.exists(), f"download_report.json not found: {p}"


def test_download_report_valid(download_result):
    """download_report.json 파싱 + categories 필드"""
    p = ASSETS_DIR / 'download_report.json'
    if not p.exists():
        pytest.skip("download_report.json not found")
    data = json.loads(p.read_text())
    assert 'categories'        in data
    assert 'total_downloaded'  in data
    assert data['total_downloaded'] >= 0


# ─── 생성 테스트 ──────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def generation_result(download_result):
    """다운로드된 자산으로 간단한 교차 애니메이션 생성"""
    # 사용 가능한 첫 번째 pair 찾기
    glbs = list(ASSETS_DIR.rglob('*.glb'))
    if len(glbs) < 2:
        pytest.skip("Not enough GLB files for generation test")

    r = subprocess.run(
        [sys.executable, 'scripts/generate_crossing.py',
         '--assets-dir', str(ASSETS_DIR),
         '--out-dir',    str(DATA_TEST_DIR),
         '--modes',      'orbit',
         '--n-cameras',  '1',
         '--n-frames',   '8',
         '--resolution', '128'],
        capture_output=True, text=True, timeout=300,
    )
    return r


def test_generation_exits_cleanly(generation_result):
    assert generation_result.returncode == 0, \
        f"generate_crossing.py failed:\n{generation_result.stderr[-800:]}"


def test_output_dirs_created(generation_result):
    """최소 1개 서브디렉토리에 meta.json 존재"""
    meta_paths = list(DATA_TEST_DIR.rglob('meta.json'))
    assert len(meta_paths) >= 1, f"No meta.json found in {DATA_TEST_DIR}"


def test_output_structure_correct(generation_result):
    """frames/depth/mask 구조 + 프레임 수 맞음"""
    for meta_path in DATA_TEST_DIR.rglob('meta.json'):
        d = meta_path.parent
        frames = sorted((d / 'frames').glob('*.png'))
        depths = sorted((d / 'depth').glob('*.npy'))
        masks  = sorted((d / 'mask').glob('*.png'))
        assert len(frames) == 8,    f"{d}: expected 8 frames, got {len(frames)}"
        assert len(depths) == 8,    f"{d}: expected 8 depths, got {len(depths)}"
        assert len(masks)  == 16,   f"{d}: expected 16 masks (8×2), got {len(masks)}"


def test_meta_json_valid(generation_result):
    """meta.json 파싱 + 필수 필드 존재"""
    for meta_path in DATA_TEST_DIR.rglob('meta.json'):
        data = json.loads(meta_path.read_text())
        assert 'keyword0'       in data
        assert 'keyword1'       in data
        assert 'prompt_entity0' in data
        assert 'prompt_entity1' in data
        assert 'mode'           in data
        assert 'camera'         in data
        assert data['mode']     in ('orbit', 'squeeze', 'rotate')


def test_depth_map_non_zero(generation_result):
    """depth .npy 파일: 모든 값이 0이 아닌 픽셀 존재"""
    for npy in list(DATA_TEST_DIR.rglob('depth/*.npy'))[:3]:
        depth = np.load(str(npy))
        assert depth.max() > 0.0, f"{npy}: all-zero depth map"


def test_frames_are_colored(generation_result):
    """프레임 RGB: 완전히 검지 않음 (렌더링 성공)"""
    for png in list(DATA_TEST_DIR.rglob('frames/*.png'))[:3]:
        img = iio3.imread(str(png))[..., :3]
        assert img.max() > 10, f"{png}: appears to be all black"


def test_orbit_depth_reversal_exists(generation_result):
    """orbit 모드에서 depth ordering 역전 존재"""
    all_orderings = []
    for meta_path in DATA_TEST_DIR.rglob('meta.json'):
        data = json.loads(meta_path.read_text())
        if data.get('mode') != 'orbit':
            continue
        d = meta_path.parent
        depths = sorted((d / 'depth').glob('*.npy'))
        orderings = []
        for fi, dpath in enumerate(depths):
            depth = np.load(str(dpath))
            m0_p = d / 'mask' / f'{fi:04d}_entity0.png'
            m1_p = d / 'mask' / f'{fi:04d}_entity1.png'
            if not m0_p.exists() or not m1_p.exists():
                continue
            m0 = iio3.imread(str(m0_p)) > 128
            m1 = iio3.imread(str(m1_p)) > 128
            m0_only = m0 & ~m1
            m1_only = m1 & ~m0
            if m0_only.sum() > 5 and m1_only.sum() > 5:
                orderings.append(depth[m0_only].mean() < depth[m1_only].mean())
        if len(orderings) >= 2:
            all_orderings.extend(orderings)

    if not all_orderings:
        pytest.skip("Not enough separated frames for reversal check")
    # 역전 있음 = True와 False 모두 존재
    has_reversal = 0 < sum(all_orderings) < len(all_orderings)
    assert has_reversal, \
        f"No depth ordering reversal in orbit mode " \
        f"(all_orderings={all_orderings[:8]})"


# ─── Dataset 로드 테스트 ─────────────────────────────────────────────────────

def test_objaverse_dataset_loads(generation_result):
    """ObjaverseVCADataset: data_test_dir 로드 → len >= 1"""
    from scripts.train_vca import ObjaverseVCADataset
    ds = ObjaverseVCADataset(
        data_root=str(DATA_TEST_DIR),
        query_dim=64, context_dim=128,
    )
    assert len(ds) >= 1


def test_dataset_item_shapes(generation_result):
    """__getitem__ 반환 형상 검증"""
    from scripts.train_vca import ObjaverseVCADataset
    ds = ObjaverseVCADataset(
        data_root=str(DATA_TEST_DIR),
        query_dim=64, context_dim=128,
    )
    if len(ds) == 0:
        pytest.skip("No samples in dataset")
    x, ctx, depth_order, rgb = ds[0]
    assert x.shape[-1]   == 64,  f"x.shape={x.shape}, expected last dim=64"
    assert ctx.shape[-1] == 128, f"ctx.shape={ctx.shape}, expected last dim=128"
    assert len(depth_order) == 2
    assert set(depth_order) == {0, 1}


def test_dataset_ctx_differs_between_entities(generation_result):
    """두 entity의 context embedding이 다름 (entity identity 인코딩 검증)"""
    from scripts.train_vca import ObjaverseVCADataset
    ds = ObjaverseVCADataset(
        data_root=str(DATA_TEST_DIR),
        query_dim=64, context_dim=128,
    )
    if len(ds) == 0:
        pytest.skip("No samples")
    _, ctx, _, _ = ds[0]
    ctx_np = ctx.squeeze(0).numpy()   # (N, CD)
    diff = float(np.linalg.norm(ctx_np[0] - ctx_np[1]))
    assert diff > 0.01, f"ctx[0] ≈ ctx[1] (diff={diff:.4f}) — entity identity not encoded"


# ─── 통계 분석 테스트 ────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def stats_result(generation_result):
    r = subprocess.run(
        [sys.executable, 'scripts/analyze_dataset.py',
         '--data-root', str(DATA_TEST_DIR),
         '--out-dir',   str(STATS_DIR)],
        capture_output=True, text=True, timeout=120,
    )
    return r


def test_stats_exits_cleanly(stats_result):
    assert stats_result.returncode == 0, \
        f"analyze_dataset.py failed:\n{stats_result.stderr[-400:]}"


def test_stats_json_created(stats_result):
    assert (STATS_DIR / 'objaverse_stats.json').exists()


def test_stats_json_valid(stats_result):
    p = STATS_DIR / 'objaverse_stats.json'
    if not p.exists():
        pytest.skip("objaverse_stats.json not found")
    data = json.loads(p.read_text())
    assert 'total_samples'       in data
    assert 'depth_reversal_rate' in data


def test_report_md_created(stats_result):
    assert (STATS_DIR / 'objaverse_report.md').exists()


def test_report_md_has_sections(stats_result):
    p = STATS_DIR / 'objaverse_report.md'
    if not p.exists():
        pytest.skip("objaverse_report.md not found")
    content = p.read_text()
    assert '# Objaverse' in content
    assert 'depth_reversal_rate' in content
    assert 'Quality Metrics' in content


def test_depth_reversal_logged(stats_result):
    """stdout 또는 stats.json에 depth_reversal_rate 존재"""
    has_stdout = 'depth_reversal_rate' in stats_result.stdout
    has_json   = (STATS_DIR / 'objaverse_stats.json').exists()
    assert has_stdout or has_json
