"""
VCA-Diffusion Toy Data Generator
두 시나리오: Chain Links (A) / Robot Arms (B)
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pyvista as pv
import imageio.v3 as iio
import imageio.v2 as iio2

warnings.filterwarnings("ignore")

# ─── 카메라 앵글 (index 순서 고정) ──────────────────────────────────────────
CAMERAS = [
    {"position": ( 3.0,  3.0, 2.5), "name": "front_right"},
    {"position": (-3.0,  3.0, 2.5), "name": "front_left"},
    {"position": ( 0.0,  4.0, 1.0), "name": "front"},
    {"position": ( 0.0,  0.5, 4.0), "name": "top"},
    {"position": ( 3.0, -3.0, 2.5), "name": "back_right"},
    {"position": (-3.0, -3.0, 2.5), "name": "back_left"},
    {"position": ( 4.0,  0.0, 0.5), "name": "side_right"},
    {"position": (-4.0,  0.0, 0.5), "name": "side_left"},
]


# ─── SceneRenderer ──────────────────────────────────────────────────────────
class SceneRenderer:
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height

    def render_frame(
        self,
        meshes: list,           # [(mesh, rgb_color), ...]
        camera_pos: tuple,
        focal_point=(0, 0, 0),
        view_up=(0, 0, 1),
    ):
        """반환: (rgb HxWx3 uint8, depth HxW float32, [mask_e0, mask_e1, ...])"""
        # ── full scene render ────────────────────────────────────────────────
        pl = pv.Plotter(off_screen=True, window_size=[self.width, self.height])
        for mesh, color in meshes:
            pl.add_mesh(mesh, color=color, smooth_shading=True)
        pl.set_background("black")
        pl.camera_position = [camera_pos, focal_point, view_up]
        pl.render()
        rgb = pl.screenshot(return_img=True)[..., :3].astype(np.uint8)

        pl.close()

        # ── depth: per-vertex distance from camera → render as scalar ────────
        depth = self._render_depth(meshes, camera_pos, focal_point, view_up)

        # ── per-entity mask ──────────────────────────────────────────────────
        masks = []
        for i, (mesh, color) in enumerate(meshes):
            pl2 = pv.Plotter(off_screen=True, window_size=[self.width, self.height])
            pl2.add_mesh(mesh, color=color, smooth_shading=True)
            pl2.set_background("black")
            pl2.camera_position = [camera_pos, focal_point, view_up]
            pl2.render()
            entity_rgb = pl2.screenshot(return_img=True)[..., :3]
            pl2.close()
            mask = (entity_rgb.sum(axis=2) > 10).astype(np.uint8) * 255
            masks.append(mask)

        return rgb, depth, masks

    def _render_depth(self, meshes, camera_pos, focal_point, view_up):
        """카메라에서 각 메시 점까지 유클리드 거리를 depth로 렌더링.
        배경(black) = 0.0, 메시 있는 픽셀 = camera distance (float32)
        """
        cam = np.array(camera_pos)
        # 전체 씬의 depth 범위 계산
        all_dists = []
        for mesh, _ in meshes:
            dists = np.linalg.norm(mesh.points - cam, axis=1)
            all_dists.append(dists)
        all_dists = np.concatenate(all_dists)
        d_min, d_max = all_dists.min(), all_dists.max()
        d_range = d_max - d_min + 1e-6

        # depth를 [0,1] → uint8 로 인코딩해서 렌더링
        pl = pv.Plotter(off_screen=True, window_size=[self.width, self.height])
        pl.set_background("black")
        for mesh, _ in meshes:
            dists = np.linalg.norm(mesh.points - cam, axis=1).astype(np.float32)
            m2 = mesh.copy()
            m2['depth_scalar'] = dists
            pl.add_mesh(m2, scalars='depth_scalar', cmap='gray',
                        clim=[d_min, d_max], show_scalar_bar=False)
        pl.camera_position = [camera_pos, focal_point, view_up]
        pl.render()
        encoded = pl.screenshot(return_img=True)[..., 0].astype(np.float32)  # R channel
        pl.close()

        # 픽셀 값 [0,255] → camera distance
        # 배경(black=0) → depth=0, 나머지 → d_min + value/255 * d_range
        depth = np.where(
            encoded > 0,
            d_min + (encoded / 255.0) * d_range,
            0.0,
        ).astype(np.float32)
        return depth


# ─── Chain Links ────────────────────────────────────────────────────────────
def make_chain_frame(frame_idx, n_frames):
    """두 개의 ㄷ자 링크 (XZ / YZ 평면) - 천천히 회전"""
    t = frame_idx / max(n_frames - 1, 1)
    angle0 = t * 180.0   # Link 0: X축 회전
    angle1 = t * 180.0   # Link 1: Y축 회전

    meshes = []

    # Link 0: XZ 평면, 빨강
    link0 = _make_link(axis='x', angle_deg=angle0, color=(1, 0, 0))
    meshes.append((link0, "red"))

    # Link 1: YZ 평면, 파랑
    link1 = _make_link(axis='y', angle_deg=angle1, color=(0, 0, 1))
    meshes.append((link1, "blue"))

    return meshes


def _make_link(axis, angle_deg, color, radius=0.08, tube_radius=0.05):
    """ㄷ자 링크: 반원 arc + 양쪽 직선으로 구성"""
    # 반원 arc (12 점)
    angles = np.linspace(0, np.pi, 20)
    if axis == 'x':
        # XZ 평면
        arc_pts = np.column_stack([
            radius * np.cos(angles),
            np.zeros_like(angles),
            radius * np.sin(angles),
        ])
        # 양쪽 직선
        left_pts  = np.array([[-radius, 0, 0], [-radius, 0, -0.4]])
        right_pts = np.array([[ radius, 0, 0], [ radius, 0, -0.4]])
    else:
        # YZ 평면
        arc_pts = np.column_stack([
            np.zeros_like(angles),
            radius * np.cos(angles),
            radius * np.sin(angles),
        ])
        left_pts  = np.array([[0, -radius, 0], [0, -radius, -0.4]])
        right_pts = np.array([[0,  radius, 0], [0,  radius, -0.4]])

    all_pts = np.vstack([left_pts[::-1], arc_pts, right_pts])
    spline = pv.Spline(all_pts, 80)
    tube = spline.tube(radius=tube_radius)

    # 회전 적용
    if axis == 'x':
        tube = tube.rotate_x(angle_deg)
    else:
        tube = tube.rotate_y(angle_deg)

    return tube


# ─── Robot Arms ─────────────────────────────────────────────────────────────
def rot_y(deg):
    r = np.radians(deg)
    return np.array([
        [ np.cos(r), 0, np.sin(r)],
        [ 0,         1, 0        ],
        [-np.sin(r), 0, np.cos(r)],
    ])


def _cylinder_between(p0, p1, radius=0.05):
    """두 점 사이 cylinder mesh"""
    center = (p0 + p1) / 2.0
    direction = p1 - p0
    height = np.linalg.norm(direction)
    if height < 1e-6:
        return pv.Sphere(radius=radius, center=center)
    cyl = pv.Cylinder(center=center, direction=direction, radius=radius, height=height)
    return cyl


def _arm_meshes(base_pos, shoulder_deg, elbow_deg, color):
    """3-joint arm: base → shoulder → elbow → wrist"""
    base_pos = np.array(base_pos)
    shoulder_world = base_pos + np.array([0, 0, 0.1])
    elbow_world    = shoulder_world + rot_y(shoulder_deg) @ np.array([0, 0, 0.6])
    wrist_world    = elbow_world    + rot_y(shoulder_deg + elbow_deg) @ np.array([0, 0, 0.5])

    meshes = []
    # base sphere
    meshes.append(pv.Sphere(radius=0.07, center=base_pos))
    # upper arm
    meshes.append(_cylinder_between(shoulder_world, elbow_world, radius=0.06))
    # elbow joint sphere
    meshes.append(pv.Sphere(radius=0.07, center=elbow_world))
    # forearm
    meshes.append(_cylinder_between(elbow_world, wrist_world, radius=0.05))
    # wrist sphere
    meshes.append(pv.Sphere(radius=0.06, center=wrist_world))

    # merge all
    merged = meshes[0].merge(meshes[1:])
    return merged


def make_robot_arm_frame(frame_idx, n_frames):
    """두 로봇팔이 중앙에서 교차"""
    t = frame_idx / max(n_frames - 1, 1)

    # 프레임 7~9에서 교차 최대 (t ≈ 0.47~0.6)
    # shoulder: 0 → 60도, elbow: 0 → -40도
    shoulder = t * 60.0
    elbow    = -(t * 40.0)

    arm0 = _arm_meshes([-0.4, 0, 0],  shoulder,  elbow, "red")
    arm1 = _arm_meshes([ 0.4, 0, 0], -shoulder, -elbow, "blue")

    return [(arm0, "red"), (arm1, "blue")]


# ─── 출력 저장 ────────────────────────────────────────────────────────────────
def save_frame(out_dir: Path, frame_idx: int, rgb, depth, masks):
    fname = f"{frame_idx:04d}"

    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    (out_dir / "mask").mkdir(parents=True, exist_ok=True)

    iio.imwrite(out_dir / "frames" / f"{fname}.png", rgb)
    np.save(out_dir / "depth" / f"{fname}.npy", depth)
    for ei, mask in enumerate(masks):
        iio.imwrite(out_dir / "mask" / f"{fname}_entity{ei}.png", mask)


def make_video(frames_dir: Path, video_path: Path, fps=8):
    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        return
    writer = iio2.get_writer(str(video_path), fps=fps, codec="libx264", quality=7)
    for p in pngs:
        writer.append_data(iio.imread(str(p)))
    writer.close()


# ─── 메인 ────────────────────────────────────────────────────────────────────
def generate(scenario, n_frames, n_cameras, resolution):
    try:
        pv.start_xvfb()
    except Exception:
        pass

    renderer = SceneRenderer(width=resolution, height=resolution)
    cameras = CAMERAS[:n_cameras]

    frame_fn = make_chain_frame if scenario == "chain" else make_robot_arm_frame

    for cam in cameras:
        cam_name = cam["name"]
        cam_pos  = cam["position"]
        out_dir  = Path(f"toy/data/{scenario}/{cam_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        for fi in range(n_frames):
            meshes = frame_fn(fi, n_frames)
            rgb, depth, masks = renderer.render_frame(
                meshes,
                camera_pos=cam_pos,
                focal_point=(0, 0, 0),
                view_up=(0, 0, 1),
            )
            # resize depth to match resolution
            if depth.shape != (resolution, resolution):
                depth = depth[:resolution, :resolution]
            save_frame(out_dir, fi, rgb, depth, masks)

        make_video(out_dir / "frames", out_dir / "video.mp4")
        print(f"  [{scenario}/{cam_name}] {n_frames} frames done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["chain", "robot_arm", "both"], default="both")
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--n-cameras", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    scenarios = ["chain", "robot_arm"] if args.scenario == "both" else [args.scenario]

    Path("toy/data").mkdir(parents=True, exist_ok=True)

    for sc in scenarios:
        print(f"Generating scenario: {sc}")
        generate(sc, args.n_frames, args.n_cameras, args.resolution)

    # prompts.json
    prompts = {
        "chain": {
            "entity_0": "a red chain link rotating in the XZ plane",
            "entity_1": "a blue chain link rotating in the YZ plane",
        },
        "robot_arm": {
            "entity_0": "a red robotic arm reaching toward center from the left",
            "entity_1": "a blue robotic arm reaching toward center from the right",
        },
    }
    Path("toy/data/prompts.json").write_text(json.dumps(prompts, indent=2))
    print("Done. prompts.json written.")


if __name__ == "__main__":
    main()
