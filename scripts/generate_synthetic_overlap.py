"""Generate synthetic high-overlap Phase 65 dataset using basic shapes.

Creates spheres, boxes, and cylinders that collide and overlap.
No external assets needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import imageio.v3 as iio3
import imageio.v2 as iio2

try:
    pv.start_xvfb()
except Exception:
    pass


CAMERAS = [
    {"position": (3.0, 3.0, 2.5), "name": "front_right"},
    {"position": (-3.0, 3.0, 2.5), "name": "front_left"},
    {"position": (0.0, 4.0, 1.0), "name": "front"},
    {"position": (0.0, 0.5, 4.0), "name": "top"},
]

ENTITY_PAIRS = [
    ("sphere", "sphere", "red_sphere", "blue_sphere"),
    ("sphere", "box", "red_sphere", "blue_cube"),
    ("box", "box", "red_cube", "blue_cube"),
    ("cylinder", "sphere", "red_cylinder", "blue_sphere"),
    ("cylinder", "cylinder", "red_cylinder", "blue_cylinder"),
]

COLORS = {
    "entity0": (0.85, 0.15, 0.10),
    "entity1": (0.10, 0.25, 0.85),
}


def create_mesh(shape: str, scale: float = 0.4) -> pv.PolyData:
    if shape == "sphere":
        return pv.Sphere(radius=scale)
    elif shape == "box":
        return pv.Box(bounds=(-scale, scale, -scale, scale, -scale, scale))
    elif shape == "cylinder":
        return pv.Cylinder(radius=scale * 0.7, height=scale * 2)
    else:
        return pv.Sphere(radius=scale)


class SceneRenderer:
    def __init__(self, width=256, height=256):
        self.width = width
        self.height = height

    def render_frame(self, meshes, camera_pos, focal_point=(0, 0, 0), view_up=(0, 0, 1)):
        # Full scene render
        pl = pv.Plotter(off_screen=True, window_size=[self.width, self.height])
        for mesh, color in meshes:
            pl.add_mesh(mesh, color=color, smooth_shading=True)
        pl.set_background("black")
        pl.camera_position = [camera_pos, focal_point, view_up]
        pl.render()
        rgb = pl.screenshot(return_img=True)[..., :3].astype(np.uint8)
        pl.close()

        # Depth
        depth = self._render_depth(meshes, camera_pos, focal_point, view_up)

        # Per-entity masks
        masks = []
        for mesh, color in meshes:
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
        cam = np.array(camera_pos)
        all_dists = []
        for mesh, _ in meshes:
            dists = np.linalg.norm(mesh.points - cam, axis=1)
            all_dists.append(dists)
        all_dists = np.concatenate(all_dists)
        d_min, d_max = all_dists.min(), all_dists.max()
        d_range = d_max - d_min + 1e-6

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
        encoded = pl.screenshot(return_img=True)[..., 0].astype(np.float32)
        pl.close()

        depth = np.where(
            encoded > 0,
            d_min + (encoded / 255.0) * d_range,
            0.0,
        ).astype(np.float32)
        return depth


def compute_collision_transforms(frame_idx: int, n_frames: int, collision_type: str):
    """Compute transforms for high-overlap scenarios."""
    t = frame_idx / max(n_frames - 1, 1)

    if collision_type == "head_on":
        # Head-on collision: entities pass through each other
        pos = -0.6 + 1.2 * t  # -0.6 -> 0.6
        trans0 = np.array([pos, 0.0, 0.0])
        trans1 = np.array([-pos, 0.0, 0.0])
        rot0 = t * 180.0
        rot1 = 180.0

    elif collision_type == "orbit_tight":
        # Very tight orbit - always overlapping
        angle = t * 2 * np.pi
        r = 0.15  # Very small radius
        trans0 = np.array([r * np.cos(angle), r * np.sin(angle), 0.0])
        trans1 = np.array([r * np.cos(angle + np.pi), r * np.sin(angle + np.pi), 0.0])
        rot0 = np.degrees(angle)
        rot1 = np.degrees(angle + np.pi)

    elif collision_type == "vertical_pass":
        # Vertical passing - one goes up, one goes down through each other
        z_pos = -0.5 + 1.0 * t
        trans0 = np.array([0.0, 0.0, z_pos])
        trans1 = np.array([0.0, 0.0, -z_pos])
        rot0 = t * 90.0
        rot1 = -t * 90.0

    elif collision_type == "diagonal":
        # Diagonal collision
        d = -0.6 + 1.2 * t
        trans0 = np.array([d, d * 0.5, 0.0])
        trans1 = np.array([-d, -d * 0.5, 0.0])
        rot0 = t * 120.0
        rot1 = 180.0 - t * 120.0

    else:
        trans0 = np.array([0.3, 0.0, 0.0])
        trans1 = np.array([-0.3, 0.0, 0.0])
        rot0 = 0.0
        rot1 = 0.0

    return trans0, rot0, trans1, rot1


COLLISION_TYPES = ["head_on", "orbit_tight", "vertical_pass", "diagonal"]


def render_sample(
    shape0: str,
    shape1: str,
    name0: str,
    name1: str,
    collision_type: str,
    cam: dict,
    out_dir: Path,
    renderer: SceneRenderer,
    n_frames: int = 16,
):
    """Render one sample."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'frames').mkdir(exist_ok=True)
    (out_dir / 'depth').mkdir(exist_ok=True)
    (out_dir / 'mask').mkdir(exist_ok=True)

    mesh0_base = create_mesh(shape0)
    mesh1_base = create_mesh(shape1)

    frames_rgb = []
    for f_idx in range(n_frames):
        trans0, rot0, trans1, rot1 = compute_collision_transforms(f_idx, n_frames, collision_type)

        m0 = mesh0_base.copy()
        m0.rotate_y(rot0, inplace=True)
        m0.translate(trans0, inplace=True)

        m1 = mesh1_base.copy()
        m1.rotate_y(rot1, inplace=True)
        m1.translate(trans1, inplace=True)

        rgb, depth, masks = renderer.render_frame(
            meshes=[(m0, COLORS["entity0"]), (m1, COLORS["entity1"])],
            camera_pos=cam['position'],
        )

        iio3.imwrite(str(out_dir / 'frames' / f'{f_idx:04d}.png'), rgb)
        np.save(str(out_dir / 'depth' / f'{f_idx:04d}.npy'), depth)
        for ei, mask in enumerate(masks):
            iio3.imwrite(str(out_dir / 'mask' / f'{f_idx:04d}_entity{ei}.png'), mask)
        frames_rgb.append(rgb)

    iio2.mimsave(str(out_dir / 'video.gif'), frames_rgb, duration=125)

    meta = {
        "keyword0": name0,
        "keyword1": name1,
        "prompt_full": f"a {name0} and a {name1} colliding",
        "mode": collision_type,
        "camera": cam['name'],
        "n_frames": n_frames,
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='toy/data_synthetic_overlap')
    parser.add_argument('--n-cameras', type=int, default=4)
    parser.add_argument('--n-frames', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = CAMERAS[:args.n_cameras]
    renderer = SceneRenderer(width=args.resolution, height=args.resolution)

    total = len(ENTITY_PAIRS) * len(COLLISION_TYPES) * len(cameras)
    print(f"Generating {total} samples...")

    n_ok = 0
    for shape0, shape1, name0, name1 in ENTITY_PAIRS:
        for coll_type in COLLISION_TYPES:
            for cam in cameras:
                sample_id = f"{name0}_{name1}_{coll_type}_{cam['name']}"
                sample_dir = out_dir / sample_id
                render_sample(shape0, shape1, name0, name1, coll_type, cam, sample_dir, renderer, args.n_frames)
                n_ok += 1
                if n_ok % 10 == 0:
                    print(f"Generated {n_ok}/{total}...")

    print(f"Done. Generated {n_ok} samples to {out_dir}")


if __name__ == "__main__":
    main()
