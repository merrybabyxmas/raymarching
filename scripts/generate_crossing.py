"""
Phase 15 Part B: 범용 교차 애니메이션 생성기

두 GLB mesh를 교차 애니메이션으로 렌더링.
3가지 모드 (ORBIT / SQUEEZE / ROTATE) → depth ordering 역전 보장.
SceneRenderer 재사용 (FM-I1/I2/I3 모두 적용됨).

주의사항:
  - trimesh.load()가 Scene 반환 가능 → scene.dump()로 mesh 추출 필수
  - 버텍스 > 50,000 → decimation
  - FM-I2: GIF는 imageio.v2.mimsave()
"""
import argparse
import json
import sys
from enum import Enum
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
import imageio.v3 as iio3
import imageio.v2 as iio2

sys.path.insert(0, str(Path(__file__).parent.parent))

from toy.generate_toy_data import SceneRenderer, CAMERAS


# ─── 색상 ─────────────────────────────────────────────────────────────────────

COLORS = {
    'entity0': (0.85, 0.15, 0.10),   # 빨강 계열
    'entity1': (0.10, 0.25, 0.85),   # 파랑 계열
}


# ─── CrossingMode ─────────────────────────────────────────────────────────────

class CrossingMode(Enum):
    ORBIT   = "orbit"
    SQUEEZE = "squeeze"
    ROTATE  = "rotate"
    LAYERED = "layered"   # depth-separated: entity0 front, entity1 back (occ benchmark)


# ─── mesh 로드 ────────────────────────────────────────────────────────────────

def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.apply_translation(-mesh.centroid)
    scale = 1.0 / (mesh.bounding_box.extents.max() + 1e-6)
    mesh.apply_scale(scale)
    return mesh


def load_mesh_as_pyvista(glb_path: Path) -> pv.PolyData:
    """GLB → PyVista PolyData (normalize 포함)."""
    result = trimesh.load(str(glb_path), force='mesh')
    if isinstance(result, trimesh.Scene):
        meshes = [m for m in result.dump() if hasattr(m, 'faces')]
        if not meshes:
            raise ValueError(f"Empty scene: {glb_path}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(result, trimesh.Trimesh):
        mesh = result
    else:
        raise ValueError(f"Unexpected type: {type(result)}")

    if len(mesh.vertices) > 50_000:
        target_faces = min(10_000, len(mesh.faces) - 1)
        if target_faces > 0 and target_faces < len(mesh.faces):
            try:
                mesh = mesh.simplify_quadric_decimation(target_faces)
            except Exception:
                pass  # decimation 실패 시 원본 사용

    mesh = normalize_mesh(mesh)

    # trimesh → pv.PolyData
    faces_pv = np.hstack([
        np.full((len(mesh.faces), 1), 3, dtype=np.int_),
        mesh.faces,
    ]).flatten()
    return pv.PolyData(mesh.vertices.astype(np.float32), faces_pv)


# ─── 프레임 변환 ──────────────────────────────────────────────────────────────

def compute_frame_transforms(
    mode: CrossingMode,
    frame_idx: int,
    n_frames: int,
) -> tuple:
    """(trans0, rot0_deg, trans1, rot1_deg) 반환."""
    t = frame_idx / max(n_frames - 1, 1)
    pi2 = 2 * np.pi

    if mode == CrossingMode.ORBIT:
        # 두 mesh가 서로를 공전하며 교차
        angle = t * pi2
        r = 0.6
        trans0 = np.array([r * np.cos(angle),       r * np.sin(angle),       0.0])
        trans1 = np.array([r * np.cos(angle + np.pi), r * np.sin(angle + np.pi), 0.0])
        rot0 = float(np.degrees(angle))
        rot1 = float(np.degrees(angle + np.pi))

    elif mode == CrossingMode.SQUEEZE:
        # 정면 충돌 후 통과
        pos = -1.5 + 1.8 * t      # -1.5 → +0.3
        trans0 = np.array([ pos, 0.0, 0.0])
        trans1 = np.array([-pos, 0.0, 0.0])
        rot0 = 0.0
        rot1 = 180.0

    elif mode == CrossingMode.ROTATE:
        # 제자리 회전
        trans0 = np.array([-0.3, 0.0, 0.0])
        trans1 = np.array([ 0.3, 0.0, 0.0])
        rot0 =  frame_idx * (360.0 / n_frames)
        rot1 = -frame_idx * (360.0 / n_frames)

    elif mode == CrossingMode.LAYERED:
        # Depth-separated (occ benchmark): entity0 front (y=+0.45), entity1 back (y=-0.45)
        # Camera looks from +Y direction at origin, so +Y = closer to camera.
        # entity0 slowly sweeps left→right (x: -0.4 → +0.4) while staying in front.
        # entity1 slowly rotates in-place at back.
        # Overlap region mid-animation creates genuine front-back occlusion.
        x_sweep = -0.4 + 0.8 * t          # -0.4 → +0.4
        trans0 = np.array([x_sweep, 0.45, 0.0])   # front entity
        trans1 = np.array([  0.0,  -0.45, 0.0])   # back entity (occluded)
        rot0 = frame_idx * (180.0 / n_frames)      # slow rotation
        rot1 = frame_idx * (360.0 / n_frames)      # full rotation

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return trans0, rot0, trans1, rot1


# ─── 페어 생성 ────────────────────────────────────────────────────────────────

def generate_pairs(assets_dir: Path, seed: int = 42) -> list:
    """assets_dir에서 GLB를 찾아 학습용 페어 생성.
    EXCLUDED_ASSET_IDS에 있는 bad asset은 제외 (육안 검수 결과).
    """
    rng = np.random.RandomState(seed)

    # 육안 검수로 확인된 bad asset IDs (ring, collar 등 동물이 아닌 메쉬)
    EXCLUDED_ASSET_IDS = {
        'faea9af25c584e8aa893b65103c16601',  # dog 폴더에 있으나 실제로는 링/collar 형태
    }

    # 키워드별 GLB 파일 수집
    kw_to_glbs = {}
    for glb in assets_dir.rglob('*.glb'):
        if glb.stem in EXCLUDED_ASSET_IDS:
            continue  # bad asset 제외
        kw = glb.parent.name
        kw_to_glbs.setdefault(kw, []).append(glb)

    available_kws = sorted(kw_to_glbs.keys())
    if not available_kws:
        return []

    print(f"[pairs] available keywords: {available_kws}", flush=True)

    pairs = []

    # 같은 카테고리 다른 종류 페어
    same_pairs = [
        ("cat", "dog"),     ("wolf", "dog"),    ("lion", "bear"),
        ("tiger", "wolf"),  ("snake", "alligator"),
        ("sword", "sword"), ("cat", "cat"),
        ("person", "person"),
    ]

    # 다른 카테고리 페어
    cross_pairs = [
        ("cat", "sword"),   ("dog", "sword"),
        ("person", "snake"),("snake", "sword"),
        ("cat", "snake"),   ("lion", "sword"),
    ]

    for kw0, kw1 in same_pairs + cross_pairs:
        glbs0 = kw_to_glbs.get(kw0, [])
        glbs1 = kw_to_glbs.get(kw1, [])
        if not glbs0 or not glbs1:
            continue
        m0 = glbs0[rng.randint(len(glbs0))]
        m1 = glbs1[rng.randint(len(glbs1))]

        # 같은 종류면 색상 구분 프롬프트
        if kw0 == kw1:
            prompt_full   = f"a red {kw0} and a blue {kw1} tangled together"
            prompt_entity0 = f"a red {kw0}"
            prompt_entity1 = f"a blue {kw1}"
        else:
            prompt_full   = f"a {kw0} and a {kw1} tangled together"
            prompt_entity0 = f"a {kw0}"
            prompt_entity1 = f"a {kw1}"

        pairs.append({
            'keyword0': kw0,
            'keyword1': kw1,
            'mesh0_path': str(m0),
            'mesh1_path': str(m1),
            'prompt_full':    prompt_full,
            'prompt_entity0': prompt_entity0,
            'prompt_entity1': prompt_entity1,
            'color0': list(COLORS['entity0']),
            'color1': list(COLORS['entity1']),
        })

    return pairs


# ─── 단일 페어 렌더링 ─────────────────────────────────────────────────────────

def render_pair(
    pair: dict,
    out_dir: Path,
    renderer: SceneRenderer,
    mode: CrossingMode,
    cam: dict,
    n_frames: int = 16,
):
    """한 페어 × 1 모드 × 1 카메라 렌더링 → 파일 저장."""
    try:
        mesh0_pv = load_mesh_as_pyvista(Path(pair['mesh0_path']))
        mesh1_pv = load_mesh_as_pyvista(Path(pair['mesh1_path']))
    except Exception as e:
        print(f"  [skip] load failed: {e}", flush=True)
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'frames').mkdir(exist_ok=True)
    (out_dir / 'depth').mkdir(exist_ok=True)
    (out_dir / 'mask').mkdir(exist_ok=True)

    frames_rgb = []
    for f_idx in range(n_frames):
        trans0, rot0, trans1, rot1 = compute_frame_transforms(mode, f_idx, n_frames)

        m0 = mesh0_pv.copy()
        m0.rotate_y(rot0, inplace=True)
        m0.translate(trans0, inplace=True)

        m1 = mesh1_pv.copy()
        m1.rotate_y(rot1 + 180.0, inplace=True)
        m1.translate(trans1, inplace=True)

        color0 = tuple(pair.get('color0', COLORS['entity0']))
        color1 = tuple(pair.get('color1', COLORS['entity1']))

        try:
            rgb, depth, masks = renderer.render_frame(
                meshes=[(m0, color0), (m1, color1)],
                camera_pos=cam['position'],
            )
        except Exception as e:
            print(f"  [skip] render failed at frame {f_idx}: {e}", flush=True)
            return False

        iio3.imwrite(str(out_dir / 'frames' / f'{f_idx:04d}.png'), rgb)
        np.save(str(out_dir / 'depth' / f'{f_idx:04d}.npy'), depth)
        for ei, mask in enumerate(masks):
            iio3.imwrite(str(out_dir / 'mask' / f'{f_idx:04d}_entity{ei}.png'), mask)
        frames_rgb.append(rgb)

    # video GIF (FM-I2: imageio.v2)
    iio2.mimsave(str(out_dir / 'video.gif'), frames_rgb, duration=125)

    # meta.json
    # scene_type: "occ" for depth-separated layered mode, "col" for same-depth modes
    scene_type = "occ" if mode == CrossingMode.LAYERED else "col"
    meta = {
        **pair,
        'mode': mode.value,
        'camera': cam['name'],
        'n_frames': n_frames,
        'scene_type': scene_type,
    }
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return True


# ─── 메인 생성 루프 ──────────────────────────────────────────────────────────

def generate_dataset(
    assets_dir: Path,
    out_dir: Path,
    n_cameras: int = 8,
    n_frames: int = 16,
    resolution: int = 256,
    modes: list = None,
    pair_filter: list = None,  # keyword pairs to run, e.g. ["cat_dog"]
    seed: int = 42,
):
    if modes is None:
        modes = list(CrossingMode)

    pairs = generate_pairs(assets_dir, seed=seed)

    if pair_filter:
        filtered = []
        for p in pairs:
            key = f"{p['keyword0']}_{p['keyword1']}"
            rkey = f"{p['keyword1']}_{p['keyword0']}"
            if key in pair_filter or rkey in pair_filter:
                filtered.append(p)
        pairs = filtered

    print(f"[gen] {len(pairs)} pairs × {len(modes)} modes × {n_cameras} cameras "
          f"= {len(pairs)*len(modes)*n_cameras} sequences", flush=True)

    renderer = SceneRenderer(width=resolution, height=resolution)
    cameras  = CAMERAS[:n_cameras]

    n_ok = 0
    n_fail = 0

    for pair in pairs:
        kw = f"{pair['keyword0']}_{pair['keyword1']}"
        for mode in modes:
            for cam in cameras:
                pair_id = f"{kw}_{mode.value}_{cam['name']}"
                out_sub = out_dir / pair_id
                ok = render_pair(pair, out_sub, renderer, mode, cam, n_frames)
                if ok:
                    n_ok += 1
                    print(f"[ok] {pair_id}", flush=True)
                else:
                    n_fail += 1

    print(f"\n[done] ok={n_ok}  fail={n_fail}", flush=True)
    return n_ok, n_fail


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--assets-dir', default='toy/assets',    dest='assets_dir')
    p.add_argument('--out-dir',    default='toy/data_objaverse', dest='out_dir')
    p.add_argument('--n-cameras',  type=int, default=8,     dest='n_cameras')
    p.add_argument('--n-frames',   type=int, default=16,    dest='n_frames')
    p.add_argument('--resolution', type=int, default=256)
    p.add_argument('--modes',      default='',
                   help='comma-separated: orbit,squeeze,rotate,layered (default: all)')
    p.add_argument('--pairs',      default='',
                   help='comma-separated keyword pairs, e.g. cat_dog,knight_ninja')
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--all',        action='store_true',
                   help='Run all pairs/modes/cameras')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    modes = None
    if args.modes:
        mode_map = {m.value: m for m in CrossingMode}
        modes = [mode_map[m.strip()] for m in args.modes.split(',') if m.strip() in mode_map]

    pair_filter = [p.strip() for p in args.pairs.split(',') if p.strip()] or None

    generate_dataset(
        assets_dir  = Path(args.assets_dir),
        out_dir     = Path(args.out_dir),
        n_cameras   = args.n_cameras,
        n_frames    = args.n_frames,
        resolution  = args.resolution,
        modes       = modes,
        pair_filter = pair_filter,
        seed        = args.seed,
    )
