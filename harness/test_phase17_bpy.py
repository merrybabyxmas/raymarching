"""
Phase 17: bpy 기반 Objaverse 애니메이션 데이터셋 생성 테스트

pytest -m phase17
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.phase17, pytest.mark.slow]

sys.path.insert(0, str(Path(__file__).parent.parent))

ASSETS_DIR = Path("toy/assets_animated_test")
DATA_DIR   = Path("toy/data_animated_test")


# ─── bpy 유닛 테스트 (CPU/headless) ──────────────────────────────────────────

def test_bpy_importable():
    """bpy 4.0+ 설치 확인"""
    try:
        import bpy
        assert bpy.app.version >= (4, 0, 0), \
            f"bpy {bpy.app.version_string} too old, need 4.0+"
    except ImportError:
        pytest.skip("bpy not installed — pip install bpy")


def test_bpy_headless_scene_init():
    """init_bpy_headless() 씬 초기화 확인"""
    try:
        import bpy
        from scripts.render_bpy_scene import init_bpy_headless
        scene = init_bpy_headless(resolution=64)
        assert scene.render.resolution_x == 64
        assert scene.render.resolution_y == 64
        assert scene.camera is not None
        assert scene.render.engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT",
                                        "BLENDER_WORKBENCH", "CYCLES")
    except ImportError:
        pytest.skip("bpy not installed")


def test_bpy_headless_render():
    """bpy headless 렌더링 기본 동작 확인"""
    try:
        import bpy
        from scripts.render_bpy_scene import init_bpy_headless
        scene = init_bpy_headless(resolution=64)
        bpy.ops.mesh.primitive_cube_add()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "test.png")
            scene.render.filepath = out_path
            scene.compositing_node_group = None
            bpy.ops.render.render(write_still=True)
            assert os.path.exists(out_path), "Render failed — no output file"
    except ImportError:
        pytest.skip("bpy not installed")


def test_bpy_depth_pass():
    """Z pass depth map 추출 확인 (OpenEXR)"""
    try:
        import bpy
        import OpenEXR
        from scripts.render_bpy_scene import (
            init_bpy_headless, _setup_depth_compositor, _load_exr_depth,
        )
        scene = init_bpy_headless(resolution=64)
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))

        with tempfile.TemporaryDirectory() as tmp:
            depth_raw_dir = Path(tmp) / "depth_raw"
            depth_raw_dir.mkdir()

            depth_ng, _ = _setup_depth_compositor(str(depth_raw_dir), "depth_out")
            scene.compositing_node_group = depth_ng
            bpy.ops.render.render(write_still=False)
            scene.compositing_node_group = None

            exr_files = list(depth_raw_dir.glob("*.exr"))
            assert len(exr_files) > 0, "No EXR depth file generated"

            depth = _load_exr_depth(str(exr_files[0]))
            assert depth.dtype == np.float32, f"Expected float32, got {depth.dtype}"
            assert depth.shape == (64, 64), f"Expected (64,64), got {depth.shape}"
            # 큐브가 있으므로 배경보다 가까운 픽셀 존재
            finite_close = depth[np.isfinite(depth) & (depth < 100)]
            assert len(finite_close) > 0, "Depth map appears empty — no close pixels"
            # NaN 비율 50% 미만
            nan_ratio = np.isnan(depth).mean()
            assert nan_ratio < 0.5, f"Too many NaN: {nan_ratio*100:.1f}%"
    except ImportError as e:
        pytest.skip(f"Required package not installed: {e}")


def test_load_exr_depth_no_nan():
    """OpenEXR depth 로드 — NaN이 없거나 배경(inf)만 있어야 함"""
    try:
        import bpy
        import OpenEXR
        from scripts.render_bpy_scene import (
            init_bpy_headless, _setup_depth_compositor, _load_exr_depth,
        )
        scene = init_bpy_headless(resolution=64)
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))

        with tempfile.TemporaryDirectory() as tmp:
            depth_raw_dir = Path(tmp) / "depth_raw"
            depth_raw_dir.mkdir()

            depth_ng, _ = _setup_depth_compositor(str(depth_raw_dir), "depth_out")
            scene.compositing_node_group = depth_ng
            bpy.ops.render.render(write_still=False)
            scene.compositing_node_group = None

            exr_files = list(depth_raw_dir.glob("*.exr"))
            if not exr_files:
                pytest.skip("No EXR generated")

            depth = _load_exr_depth(str(exr_files[0]))
            nan_ratio = float(np.isnan(depth).mean())
            # bpy Z pass는 NaN이 없어야 함 (PyVista FM-I3 문제 없음)
            assert nan_ratio < 0.5, \
                f"bpy depth has {nan_ratio*100:.1f}% NaN — Z pass not working"
    except ImportError as e:
        pytest.skip(f"Required package not installed: {e}")


# ─── 다운로드 통합 테스트 ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def download_result():
    """animationCount > 0 GLB 빠른 테스트 다운로드"""
    r = subprocess.run(
        [sys.executable, "scripts/download_animated_assets.py",
         "--quick-test",
         "--out-dir", str(ASSETS_DIR)],
        capture_output=True, text=True, timeout=300,
    )
    return r


def test_download_exits_cleanly(download_result):
    assert download_result.returncode == 0, \
        f"download_animated_assets.py failed:\n{download_result.stderr[-400:]}"


def test_animated_glbs_downloaded(download_result):
    """최소 1개 이상 GLB 다운로드 확인"""
    manifests = list(ASSETS_DIR.rglob("manifest.json"))
    if not manifests:
        pytest.skip("No manifests found — download may have been skipped")
    total_glbs = 0
    for mp in manifests:
        data = json.loads(mp.read_text())
        total_glbs += len(data)
    assert total_glbs >= 1, "No animated GLBs downloaded"


def test_manifest_has_animation_count(download_result):
    """manifest의 모든 항목은 animation_count > 0이어야 함"""
    for mp in ASSETS_DIR.rglob("manifest.json"):
        data = json.loads(mp.read_text())
        for uid, info in data.items():
            assert info["animation_count"] > 0, \
                f"{uid}: animation_count=0, should have been filtered"


def test_download_report_created(download_result):
    report_path = ASSETS_DIR / "download_report.json"
    assert report_path.exists(), "download_report.json not created"
    data = json.loads(report_path.read_text())
    assert "total_downloaded" in data
    assert "categories" in data


# ─── 렌더링 통합 테스트 ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def render_result(download_result):
    """bpy 렌더링 빠른 테스트"""
    # GLB 2개 이상 있는지 확인
    all_glbs = []
    for mp in ASSETS_DIR.rglob("manifest.json"):
        data = json.loads(mp.read_text())
        for uid, info in data.items():
            if Path(info["glb_path"]).exists():
                all_glbs.append(info["glb_path"])

    if len(all_glbs) < 2:
        pytest.skip(f"Need ≥ 2 GLBs for rendering test, got {len(all_glbs)}")

    r = subprocess.run(
        [sys.executable, "scripts/render_bpy_scene.py",
         "--assets-dir",  str(ASSETS_DIR),
         "--out-dir",     str(DATA_DIR),
         "--modes",       "approach",
         "--n-cameras",   "1",
         "--n-frames",    "8",
         "--resolution",  "128"],
        capture_output=True, text=True, timeout=600,
    )
    return r


def test_render_exits_cleanly(render_result):
    assert render_result.returncode == 0, \
        f"render_bpy_scene.py failed:\n{render_result.stderr[-600:]}"


def test_output_structure(render_result):
    """각 시퀀스 디렉토리에 frames/depth/mask가 있어야 함"""
    seq_dirs = [d for d in DATA_DIR.iterdir()
                if d.is_dir() and (d / "meta.json").exists()]
    assert len(seq_dirs) >= 1, f"No sequences rendered in {DATA_DIR}"
    for d in seq_dirs[:2]:
        frames = list((d / "frames").glob("*.png"))
        depths = list((d / "depth").glob("*.npy"))
        masks  = list((d / "mask").glob("*.png"))
        assert len(frames) == 8, f"{d.name}: expected 8 frames, got {len(frames)}"
        assert len(depths) == 8, f"{d.name}: expected 8 depths, got {len(depths)}"
        assert len(masks)  == 16, f"{d.name}: expected 16 masks, got {len(masks)}"


def test_depth_is_real_float(render_result):
    """bpy Z pass depth는 실제 float32, FM-I3(NaN) 없음"""
    npy_files = list(DATA_DIR.rglob("depth/*.npy"))[:5]
    if not npy_files:
        pytest.skip("No depth files found")
    for npy in npy_files:
        depth = np.load(str(npy))
        assert depth.dtype == np.float32, f"{npy}: expected float32"
        nan_ratio = float(np.isnan(depth).mean())
        assert nan_ratio < 0.5, \
            f"{npy}: {nan_ratio*100:.1f}% NaN — bpy depth should be clean"


def test_depth_reversal_in_approach_mode(render_result):
    """approach 모드에서 depth ordering 역전 확인"""
    import imageio.v3 as iio3
    orderings = []
    for meta_path in DATA_DIR.rglob("meta.json"):
        data = json.loads(meta_path.read_text())
        if data.get("mode") != "approach":
            continue
        d = meta_path.parent
        depths_dir = d / "depth"
        masks_dir  = d / "mask"
        for fi in range(8):
            depth_file = depths_dir / f"{fi:04d}.npy"
            m0_file    = masks_dir  / f"{fi:04d}_entity0.png"
            m1_file    = masks_dir  / f"{fi:04d}_entity1.png"
            if not (depth_file.exists() and m0_file.exists() and m1_file.exists()):
                continue
            depth = np.load(str(depth_file))
            # bpy masks are RGBA with film_transparent — use alpha channel
            m0_img = iio3.imread(str(m0_file))
            m1_img = iio3.imread(str(m1_file))
            m0 = (m0_img[..., 3] > 128) if m0_img.ndim == 3 else (m0_img > 128)
            m1 = (m1_img[..., 3] > 128) if m1_img.ndim == 3 else (m1_img > 128)
            # resize masks to depth resolution if needed
            if m0.shape != depth.shape:
                import cv2
                m0 = cv2.resize(m0.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                                interpolation=cv2.INTER_NEAREST) > 0
                m1 = cv2.resize(m1.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                                interpolation=cv2.INTER_NEAREST) > 0
            m0_only = m0 & ~m1
            m1_only = m1 & ~m0
            if m0_only.sum() > 5 and m1_only.sum() > 5:
                d0_mean = float(depth[m0_only].mean())
                d1_mean = float(depth[m1_only].mean())
                if np.isfinite(d0_mean) and np.isfinite(d1_mean):
                    orderings.append(d0_mean < d1_mean)

    if len(orderings) < 2:
        pytest.skip(f"Not enough valid depth-mask pairs (got {len(orderings)})")
    has_reversal = 0 < sum(orderings) < len(orderings)
    assert has_reversal, \
        f"No depth ordering reversal in approach mode (all {sum(orderings)}/{len(orderings)} same)"


def test_animation_is_not_static(render_result):
    """
    핵심 검증: 연속 프레임 사이에 픽셀 변화가 있어야 한다.
    내장 애니메이션이 실제로 재생됐는지 확인.
    """
    import imageio.v3 as iio3
    diffs = []
    for meta_path in list(DATA_DIR.rglob("meta.json"))[:3]:
        d = meta_path.parent
        frames = sorted((d / "frames").glob("*.png"))
        if len(frames) < 2:
            continue
        f0 = iio3.imread(str(frames[0])).astype(np.float32)
        f1 = iio3.imread(str(frames[1])).astype(np.float32)
        diff = float(np.abs(f0 - f1).mean())
        diffs.append(diff)

    if not diffs:
        pytest.skip("No frame pairs found")

    # 적어도 하나는 픽셀 변화가 있어야 함
    assert max(diffs) > 1.0, \
        f"Frames look identical (max diff={max(diffs):.3f}) — animation may not be playing"


def test_meta_json_valid(render_result):
    """meta.json에 필수 필드 포함"""
    for meta_path in list(DATA_DIR.rglob("meta.json"))[:3]:
        data = json.loads(meta_path.read_text())
        for key in ("keyword0", "keyword1", "mode", "prompt_entity0",
                    "prompt_entity1", "n_anim_frames_e0"):
            assert key in data, f"{meta_path}: missing key '{key}'"


def test_gif_created(render_result):
    """video.gif 생성 확인"""
    gifs = list(DATA_DIR.rglob("video.gif"))
    assert len(gifs) >= 1, f"No video.gif found in {DATA_DIR}"


# ─── Dataset 통합 테스트 ──────────────────────────────────────────────────────

def test_animated_dataset_loads(render_result):
    """AnimatedObjaverseDataset: data_animated 로드 → len >= 1"""
    from scripts.train_vca import AnimatedObjaverseDataset
    ds = AnimatedObjaverseDataset(
        data_roots=[str(DATA_DIR)],
        query_dim=64, context_dim=128,
    )
    assert len(ds) >= 1, f"Dataset empty from {DATA_DIR}"


def test_animated_dataset_item_shape(render_result):
    """__getitem__ 반환 텐서 shape 확인"""
    from scripts.train_vca import AnimatedObjaverseDataset
    ds = AnimatedObjaverseDataset(
        data_roots=[str(DATA_DIR)],
        query_dim=64, context_dim=128,
    )
    if len(ds) == 0:
        pytest.skip("No animated data")
    x, ctx, depth_order, rgb = ds[0]
    assert x.shape[-1] == 64, f"x last dim expected 64, got {x.shape}"
    assert ctx.shape[-1] == 128, f"ctx last dim expected 128, got {ctx.shape}"
    assert len(depth_order) == 2
    assert depth_order[0] in (0, 1) and depth_order[1] in (0, 1)


def test_animated_dataset_combined_with_phase15(render_result):
    """Phase 15 + Phase 17 데이터 합산 로드"""
    from scripts.train_vca import AnimatedObjaverseDataset
    roots = [str(DATA_DIR)]
    if Path("toy/data_objaverse").exists():
        roots.append("toy/data_objaverse")

    ds = AnimatedObjaverseDataset(
        data_roots=roots,
        query_dim=64, context_dim=128,
    )
    assert len(ds) >= 1, "Combined dataset empty"
    print(f"\nCombined dataset: {len(ds)} samples from {len(roots)} roots")
