import subprocess, sys, json
from pathlib import Path
import numpy as np
import pytest

pytestmark = pytest.mark.phase2
SCENARIOS = ["chain", "robot_arm"]
CAM = "front_right"

@pytest.fixture(scope="module")
def generated():
    r = subprocess.run(
        [sys.executable, "toy/generate_toy_data.py",
         "--n-frames", "16", "--n-cameras", "1", "--resolution", "256"],
        capture_output=True, text=True, timeout=300
    )
    return r

def test_exits_cleanly(generated):
    assert generated.returncode == 0, generated.stderr[-600:]

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_video_created(generated, scenario):
    p = Path(f"toy/data/{scenario}/{CAM}/video.mp4")
    assert p.exists() and p.stat().st_size > 5_000

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_frame_count(generated, scenario):
    frames = list(Path(f"toy/data/{scenario}/{CAM}/frames").glob("*.png"))
    assert len(frames) == 16

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_depth_dtype(generated, scenario):
    d = np.load(f"toy/data/{scenario}/{CAM}/depth/0007.npy")
    assert d.dtype == np.float32 and d.shape == (256, 256)

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_masks_created(generated, scenario):
    masks = list(Path(f"toy/data/{scenario}/{CAM}/mask").glob("*.png"))
    assert len(masks) == 32  # 16 frames × 2 entities

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_entity_masks_differ(generated, scenario):
    """두 entity mask가 서로 다른 영역을 커버해야 한다 (IoU < 0.8)"""
    import imageio.v3 as iio
    m0 = iio.imread(f"toy/data/{scenario}/{CAM}/mask/0007_entity0.png") > 128
    m1 = iio.imread(f"toy/data/{scenario}/{CAM}/mask/0007_entity1.png") > 128
    iou = (m0 & m1).sum() / max((m0 | m1).sum(), 1)
    assert iou < 0.8, f"Masks overlap too much IoU={iou:.2f}"

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_depth_ordering_in_overlap(generated, scenario):
    """
    핵심 ground truth 검증:
    두 entity가 겹치는 픽셀에서 depth 값이 달라야 한다.
    이 값이 나중에 VCA sigma 검증의 ground truth가 된다.
    """
    import imageio.v3 as iio
    depth = np.load(f"toy/data/{scenario}/{CAM}/depth/0007.npy")
    m0 = iio.imread(f"toy/data/{scenario}/{CAM}/mask/0007_entity0.png") > 128
    m1 = iio.imread(f"toy/data/{scenario}/{CAM}/mask/0007_entity1.png") > 128
    overlap = m0 & m1
    if overlap.sum() < 10:
        pytest.skip("Overlap too small at frame 7 — occlusion not visible from this angle")
    m0_only = m0 & ~m1
    m1_only = m1 & ~m0
    if m0_only.sum() < 5 or m1_only.sum() < 5:
        pytest.skip("Non-overlapping regions too small — adjust geometry")
    d0 = depth[m0_only].mean()
    d1 = depth[m1_only].mean()
    assert abs(d0 - d1) > 0.01, \
        f"Entity depth planes too similar (d0={d0:.3f}, d1={d1:.3f}) — entities at same Z"

def test_prompts_json(generated):
    data = json.loads(Path("toy/data/prompts.json").read_text())
    for s in SCENARIOS:
        assert "entity_0" in data[s] and "entity_1" in data[s]
