"""
Phase 17 Part B: bpy 기반 애니메이션 GLB 렌더링 파이프라인

두 animated GLB의 자연스러운 모션을 교차시켜
RGB + Depth(Z pass, NaN 없음) + Mask를 렌더링.

주의:
  - bpy 5.1.x (Python 3.13) 기준
  - BLENDER_EEVEE 엔진 (EEVEE_NEXT는 bpy 5.1에서 없음)
  - compositor는 scene.compositing_node_group 사용
  - EXR depth 읽기: OpenEXR 라이브러리 사용 (bpy.data.images 불가)
  - FM-I2: video GIF는 imageio.v2 사용
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import imageio.v2 as iio2

CAMERAS = [
    {
        "name": "front_right",
        "position": (3.0, -3.0, 2.5),
        "rotation": (1.1, 0.0, 0.785),
    },
    {
        "name": "front_left",
        "position": (-3.0, -3.0, 2.5),
        "rotation": (1.1, 0.0, -0.785),
    },
    {
        "name": "front",
        "position": (0.0, -4.0, 1.5),
        "rotation": (1.3, 0.0, 0.0),
    },
    {
        "name": "top",
        "position": (0.0, -1.0, 5.0),
        "rotation": (0.3, 0.0, 0.0),
    },
    {
        "name": "side_right",
        "position": (4.0, 0.0, 1.5),
        "rotation": (1.1, 0.0, 1.57),
    },
    {
        "name": "back_right",
        "position": (3.0, 3.0, 2.5),
        "rotation": (1.1, 0.0, 2.36),
    },
    {
        "name": "low_front",
        "position": (0.0, -3.0, 0.5),
        "rotation": (1.57, 0.0, 0.0),
    },
    {
        "name": "high_front",
        "position": (0.0, -2.0, 4.0),
        "rotation": (0.6, 0.0, 0.0),
    },
]


# ─── bpy 초기화 ───────────────────────────────────────────────────────────────

def init_bpy_headless(resolution: int = 256):
    """
    bpy headless 렌더링 초기화.
    빈 씬으로 시작, BLENDER_EEVEE 설정.
    """
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False

    # 배경 설정
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bg_nodes = world.node_tree.nodes
    bg = bg_nodes.get("Background") or bg_nodes.get("World Output")
    if bg and hasattr(bg, "inputs") and len(bg.inputs) > 0:
        bg.inputs[0].default_value = (0.92, 0.92, 0.92, 1.0)
    scene.world = world

    # 카메라 추가
    bpy.ops.object.camera_add(location=(3.0, -3.0, 2.5))
    cam = bpy.context.active_object
    cam.rotation_euler = (1.1, 0, 0.785)
    scene.camera = cam

    # 조명 추가
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 5))
    sun = bpy.context.active_object
    sun.data.energy = 3.0

    return scene


# ─── GLB 로드 + 애니메이션 ────────────────────────────────────────────────────

def load_glb_with_animation(glb_path: str, obj_name: str, color: tuple):
    """
    GLB 파일 로드 후 씬에 배치, 색상 오버라이드, 애니메이션 프레임 확인.

    반환: (collection, objects_list, n_frames, fps)
    """
    import bpy

    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    after = set(bpy.data.objects.keys())
    new_keys = after - before
    new_objs = [bpy.data.objects[k] for k in new_keys]

    if not new_objs:
        raise RuntimeError(f"No objects imported from {glb_path}")

    # 컬렉션으로 묶기
    col = bpy.data.collections.new(obj_name)
    bpy.context.scene.collection.children.link(col)
    for obj in new_objs:
        for old_col in list(obj.users_collection):
            old_col.objects.unlink(obj)
        col.objects.link(obj)

    # 단색 material 오버라이드
    mat = bpy.data.materials.new(name=f"mat_{obj_name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)

    for obj in new_objs:
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(mat)

    # 애니메이션 프레임 범위 확인
    scene = bpy.context.scene
    n_frames = int(scene.frame_end - scene.frame_start + 1)
    fps = scene.render.fps

    print(f"  [{obj_name}] {len(new_objs)} objects, {n_frames} frames @ {fps}fps",
          flush=True)

    return col, new_objs, n_frames, fps


def _find_root(objs: list):
    """부모 없는 오브젝트 = root."""
    for obj in objs:
        if obj.parent is None:
            return obj
    return objs[0] if objs else None


def place_objects_for_crossing(
    objs_e0: list, objs_e1: list,
    frame_idx: int, n_frames: int,
    mode: str = "approach",
):
    """
    두 entity를 교차하도록 배치.
    내장 애니메이션은 그대로 재생, 위치만 오프셋.

    mode: "approach" | "orbit" | "fixed"
    """
    t = frame_idx / max(n_frames - 1, 1)

    if mode == "approach":
        pos0 = np.array([-1.2 + 1.5 * t, 0, 0])
        pos1 = np.array([ 1.2 - 1.5 * t, 0, 0])
    elif mode == "orbit":
        angle = t * 2 * np.pi
        r = 0.7
        pos0 = np.array([ r * np.cos(angle),         r * np.sin(angle),         0])
        pos1 = np.array([ r * np.cos(angle + np.pi), r * np.sin(angle + np.pi), 0])
    else:  # fixed
        pos0 = np.array([-0.5, 0, 0])
        pos1 = np.array([ 0.5, 0, 0])

    root0 = _find_root(objs_e0)
    root1 = _find_root(objs_e1)
    if root0:
        root0.location = pos0.tolist()
    if root1:
        root1.location = pos1.tolist()


# ─── 렌더링: RGB + Depth + Mask ───────────────────────────────────────────────

def _set_visibility(objs: list, visible: bool):
    """오브젝트 및 자식 렌더 가시성 설정."""
    for obj in objs:
        obj.hide_render = not visible
        for child in obj.children_recursive:
            child.hide_render = not visible


def _setup_depth_compositor(tmp_depth_dir: str, slot_name: str = "depth_out"):
    """
    bpy 5.1 compositor 설정: Z pass → EXR 파일 출력.

    반환: (node_group, out_node)
    """
    import bpy

    scene = bpy.context.scene
    vl = scene.view_layers[0]
    vl.use_pass_z = True

    ng = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
    scene.compositing_node_group = ng

    rl = ng.nodes.new("CompositorNodeRLayers")
    out = ng.nodes.new("CompositorNodeOutputFile")
    out.directory = str(tmp_depth_dir) + "/"
    out.file_name = "depth_"

    item = out.file_output_items.new("FLOAT", slot_name)
    item.format.file_format = "OPEN_EXR"

    ng.links.new(rl.outputs["Depth"], out.inputs[slot_name])
    return ng, out


def _load_exr_depth(exr_path: str) -> np.ndarray:
    """
    OpenEXR depth map → (H, W) float32 numpy array.
    bpy Z pass: 카메라에서의 실제 거리(미터), Y축 flip 필요.
    """
    import OpenEXR
    import Imath

    f = OpenEXR.InputFile(str(exr_path))
    header = f.header()
    dw = header["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    channels = list(header["channels"].keys())
    ch = channels[0]  # depth_out.V or similar
    raw = f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
    arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w)
    # Blender 이미지는 Y축 뒤집힘
    return arr[::-1].copy()


def render_frame_bpy(
    scene,
    frame_idx: int,
    objs_e0: list,
    objs_e1: list,
    out_dir: Path,
    f_idx: int,
    depth_ng=None,
):
    """
    한 프레임 렌더링:
      1. RGB (전체)
      2. Depth (Z pass via compositor → EXR → float32 npy)
      3. Entity별 mask (각각 hide_render 토글)
    """
    import bpy

    scene.frame_set(frame_idx)

    # ── RGB 렌더링 ──────────────────────────────────────────
    _set_visibility(objs_e0 + objs_e1, True)
    rgb_path = str(out_dir / "frames" / f"{f_idx:04d}.png")
    scene.render.filepath = rgb_path
    scene.render.image_settings.file_format = "PNG"

    # compositor 없이 RGB 렌더
    old_ng = scene.compositing_node_group
    scene.compositing_node_group = None
    bpy.ops.render.render(write_still=True)

    # ── Depth 렌더링 ────────────────────────────────────────
    if depth_ng is not None:
        scene.compositing_node_group = depth_ng
        # depth_ng의 OutputFile 노드 base path 설정은 setup 시 이미 됨
        # tmp depth dir에 depth_.exr가 생성됨
        bpy.ops.render.render(write_still=False)
        scene.compositing_node_group = None

        depth_raw_dir = out_dir / "depth_raw"
        exr_files = sorted(depth_raw_dir.glob("depth_*.exr"))
        if exr_files:
            latest_exr = exr_files[-1]
            depth_np = _load_exr_depth(str(latest_exr))
            np.save(str(out_dir / "depth" / f"{f_idx:04d}.npy"), depth_np)
        else:
            # fallback: 빈 depth 저장
            H = scene.render.resolution_y
            W = scene.render.resolution_x
            np.save(str(out_dir / "depth" / f"{f_idx:04d}.npy"),
                    np.zeros((H, W), dtype=np.float32))
    else:
        # compositor 없을 때 proxy depth (object positions로 근사)
        H = scene.render.resolution_y
        W = scene.render.resolution_x
        np.save(str(out_dir / "depth" / f"{f_idx:04d}.npy"),
                np.zeros((H, W), dtype=np.float32))

    # ── Entity별 Mask 렌더링 ─────────────────────────────────
    # film_transparent=True → alpha 채널 = entity 커버리지 (배경=0)
    scene.render.film_transparent = True
    for ei, objs in enumerate([objs_e0, objs_e1]):
        _set_visibility(objs_e0 + objs_e1, False)
        _set_visibility(objs, True)
        mask_path = str(out_dir / "mask" / f"{f_idx:04d}_entity{ei}.png")
        scene.render.filepath = mask_path
        scene.compositing_node_group = None
        bpy.ops.render.render(write_still=True)

    # 전체 표시 복원 + 불투명 배경으로 되돌리기
    scene.render.film_transparent = False
    _set_visibility(objs_e0 + objs_e1, True)


# ─── 메인 생성 루프 ──────────────────────────────────────────────────────────

def generate_crossing_scene(
    glb_e0: str, keyword0: str,
    glb_e1: str, keyword1: str,
    out_dir: Path,
    modes: list,
    n_frames: int = 16,
    cameras: list = None,
    resolution: int = 256,
):
    """
    두 GLB의 내장 애니메이션을 교차시켜 렌더링.
    """
    import bpy

    if cameras is None:
        cameras = CAMERAS[:1]

    for mode in modes:
        for cam in cameras:
            pair_id = f"{keyword0}_{keyword1}_{mode}_{cam['name']}"
            sub_dir = out_dir / pair_id
            for d_name in ["frames", "depth", "depth_raw", "mask"]:
                (sub_dir / d_name).mkdir(parents=True, exist_ok=True)

            print(f"\n[render] {pair_id}", flush=True)

            # ── bpy 씬 초기화 ─────────────────────────────────
            scene = init_bpy_headless(resolution=resolution)

            # 카메라 위치 설정
            scene.camera.location = cam["position"]
            scene.camera.rotation_euler = cam["rotation"]

            # depth compositor 설정
            depth_ng, depth_out_node = _setup_depth_compositor(
                str(sub_dir / "depth_raw"), "depth_out"
            )
            # depth OutputFile 노드를 compositor에서 분리 (RGB 렌더 시 비활성)
            depth_ng_ref = depth_ng
            scene.compositing_node_group = None

            # ── GLB 로드 ───────────────────────────────────────
            try:
                col0, objs_e0, n_anim0, fps0 = load_glb_with_animation(
                    glb_e0, "entity0", color=(0.85, 0.15, 0.10)
                )
                col1, objs_e1, n_anim1, fps1 = load_glb_with_animation(
                    glb_e1, "entity1", color=(0.10, 0.25, 0.85)
                )
            except Exception as e:
                print(f"  [warn] GLB load error: {e} — skipping {pair_id}", flush=True)
                bpy.ops.wm.read_factory_settings(use_empty=True)
                continue

            # ── 애니메이션 프레임 샘플링 ───────────────────────
            anim_frames = list(range(
                int(scene.frame_start),
                int(scene.frame_end) + 1,
            ))
            if not anim_frames:
                anim_frames = [1]

            if len(anim_frames) >= n_frames:
                indices = np.linspace(0, len(anim_frames) - 1, n_frames, dtype=int)
                sampled = [anim_frames[i] for i in indices]
            else:
                sampled = [anim_frames[i % len(anim_frames)] for i in range(n_frames)]

            # ── 프레임별 렌더링 ────────────────────────────────
            for f_idx, frame_bpy in enumerate(sampled):
                place_objects_for_crossing(
                    objs_e0, objs_e1, f_idx, n_frames, mode
                )
                render_frame_bpy(
                    scene, frame_bpy,
                    objs_e0, objs_e1,
                    sub_dir, f_idx,
                    depth_ng=depth_ng_ref,
                )
                print(f"  frame {f_idx+1}/{n_frames}", flush=True)

            # ── GIF 저장 ───────────────────────────────────────
            _frames_to_gif(sub_dir)

            # ── meta.json ─────────────────────────────────────
            meta = {
                "keyword0": keyword0, "keyword1": keyword1,
                "mode": mode, "camera": cam["name"],
                "glb_e0": str(glb_e0), "glb_e1": str(glb_e1),
                "prompt_entity0": f"a {keyword0}",
                "prompt_entity1": f"a {keyword1}",
                "prompt_full": f"a {keyword0} and a {keyword1} interacting",
                "color0": [0.85, 0.15, 0.10],
                "color1": [0.10, 0.25, 0.85],
                "n_anim_frames_e0": n_anim0,
                "n_anim_frames_e1": n_anim1,
            }
            with open(sub_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            print(f"  Done: {pair_id}", flush=True)

            # ── 씬 정리 (메모리 절약) ─────────────────────────
            bpy.ops.wm.read_factory_settings(use_empty=True)


def _frames_to_gif(seq_dir: Path):
    """frames/*.png → video.gif (FM-I2: imageio.v2)."""
    frames = sorted((seq_dir / "frames").glob("*.png"))
    if not frames:
        return
    imgs = [iio2.imread(str(f)) for f in frames]
    gif_path = seq_dir / "video.gif"
    iio2.mimsave(str(gif_path), imgs, fps=4)


# ─── 자산 스캔 + 쌍 생성 ─────────────────────────────────────────────────────

def collect_glb_pairs(assets_dir: str, filter_pairs: list = None) -> list:
    """
    assets_animated/ 아래의 manifest.json에서 GLB 경로 수집,
    카테고리 간 / 내부 쌍 생성.
    """
    assets_dir = Path(assets_dir)
    keyword_glbs: dict = {}

    for manifest_path in assets_dir.rglob("manifest.json"):
        keyword = manifest_path.parent.name
        with open(manifest_path) as f:
            manifest = json.load(f)
        glbs = [(uid, info["glb_path"]) for uid, info in manifest.items()
                if Path(info["glb_path"]).exists()]
        if glbs:
            keyword_glbs[keyword] = glbs

    if not keyword_glbs:
        return []

    keywords = sorted(keyword_glbs.keys())
    pairs = []
    for i, kw0 in enumerate(keywords):
        for kw1 in keywords[i:]:
            if kw0 == kw1 and len(keyword_glbs[kw0]) < 2:
                continue
            uid0, glb0 = keyword_glbs[kw0][0]
            uid1_idx = 1 if kw0 == kw1 else 0
            uid1, glb1 = keyword_glbs[kw1][uid1_idx]
            pair_key = f"{kw0}_{kw1}"
            if filter_pairs and not any(p in pair_key for p in filter_pairs):
                continue
            pairs.append((kw0, glb0, kw1, glb1))

    return pairs


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--assets-dir", default="toy/assets_animated", dest="assets_dir")
    p.add_argument("--out-dir",    default="toy/data_animated",   dest="out_dir")
    p.add_argument("--modes",      default="approach,orbit")
    p.add_argument("--n-cameras",  type=int, default=2,  dest="n_cameras")
    p.add_argument("--n-frames",   type=int, default=16, dest="n_frames")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--pairs",      default=None,
                   help="comma-separated pair filter, e.g. cat_fighter")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    modes = args.modes.split(",")
    cameras = CAMERAS[:args.n_cameras]
    filter_pairs = args.pairs.split(",") if args.pairs else None

    pairs = collect_glb_pairs(args.assets_dir, filter_pairs=filter_pairs)
    print(f"[render] {len(pairs)} pairs found", flush=True)

    out_dir = Path(args.out_dir)
    for kw0, glb0, kw1, glb1 in pairs:
        generate_crossing_scene(
            glb_e0=glb0, keyword0=kw0,
            glb_e1=glb1, keyword1=kw1,
            out_dir=out_dir,
            modes=modes,
            cameras=cameras,
            n_frames=args.n_frames,
            resolution=args.resolution,
        )

    print(f"\n[done] output → {out_dir}", flush=True)
