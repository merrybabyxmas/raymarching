"""Generate harder Phase 65 synthetic overlap data.

Goals:
- make shape holdout genuinely difficult
- randomize colors to reduce red/blue shortcuts
- store camera pose explicitly for view-aware scene conditioning
- introduce richer primitive families and mild deformations

This script is intended for Phase65 generalization benchmarks rather than the
original easy toy overlap sanity checks.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import imageio.v2 as iio2
import imageio.v3 as iio3
import numpy as np
import pyvista as pv

try:
    pv.start_xvfb()
except Exception:
    pass


BASE_CAMERAS = [
    {'position': (3.0, 3.0, 2.5), 'name': 'front_right'},
    {'position': (-3.0, 3.0, 2.5), 'name': 'front_left'},
    {'position': (0.0, 4.0, 1.0), 'name': 'front'},
    {'position': (0.0, 0.5, 4.0), 'name': 'top'},
]

SHAPE_FAMILIES = [
    'sphere', 'box', 'cylinder', 'ellipsoid', 'cone', 'capsule',
]

COLLISION_TYPES = ['head_on', 'orbit_tight', 'vertical_pass', 'diagonal']


def random_color_pair(rng: random.Random) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    # sample near but non-identical colors to weaken color shortcuts
    base_h = rng.random()
    delta = rng.uniform(0.12, 0.28)
    h1 = base_h
    h2 = (base_h + delta) % 1.0

    def hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (float(r), float(g), float(b))

    return hsv_to_rgb(h1, rng.uniform(0.65, 0.95), rng.uniform(0.75, 0.95)), hsv_to_rgb(h2, rng.uniform(0.65, 0.95), rng.uniform(0.75, 0.95))


def create_mesh(shape: str, scale: float, aspect: tuple[float, float, float]) -> pv.PolyData:
    sx, sy, sz = aspect
    if shape == 'sphere':
        mesh = pv.Sphere(radius=scale, theta_resolution=48, phi_resolution=48)
        mesh.scale([sx, sy, sz], inplace=True)
        return mesh
    if shape == 'ellipsoid':
        mesh = pv.Sphere(radius=scale, theta_resolution=48, phi_resolution=48)
        mesh.scale([sx * 1.5, sy * 0.9, sz * 0.7], inplace=True)
        return mesh
    if shape == 'box':
        return pv.Box(bounds=(-scale * sx, scale * sx, -scale * sy, scale * sy, -scale * sz, scale * sz))
    if shape == 'cylinder':
        return pv.Cylinder(radius=scale * 0.7 * max(sx, sy), height=scale * 2.0 * sz, resolution=64)
    if shape == 'cone':
        return pv.Cone(direction=(0, 0, 1), height=scale * 2.0 * sz, radius=scale * 0.9 * max(sx, sy), resolution=64)
    if shape == 'capsule':
        body = pv.Cylinder(radius=scale * 0.55 * max(sx, sy), height=scale * 1.4 * sz, resolution=64)
        top = pv.Sphere(radius=scale * 0.55 * max(sx, sy), theta_resolution=32, phi_resolution=32)
        top.translate((0, 0, scale * 0.7 * sz), inplace=True)
        bot = pv.Sphere(radius=scale * 0.55 * max(sx, sy), theta_resolution=32, phi_resolution=32)
        bot.translate((0, 0, -scale * 0.7 * sz), inplace=True)
        return body.merge(top).merge(bot)
    mesh = pv.Sphere(radius=scale)
    mesh.scale([sx, sy, sz], inplace=True)
    return mesh


def apply_mild_deformation(mesh: pv.PolyData, rng: random.Random, strength: float = 0.08) -> pv.PolyData:
    pts = mesh.points.copy()
    if len(pts) == 0:
        return mesh
    center = pts.mean(axis=0, keepdims=True)
    rel = pts - center
    norm = np.linalg.norm(rel, axis=1, keepdims=True) + 1e-6
    unit = rel / norm
    wave = np.sin(rel[:, 0:1] * rng.uniform(2.0, 5.0) + rng.uniform(-math.pi, math.pi))
    wave += np.cos(rel[:, 1:2] * rng.uniform(2.0, 5.0) + rng.uniform(-math.pi, math.pi))
    jitter = rng.uniform(0.5, 1.2) * strength * wave * unit
    pts = pts + jitter.astype(np.float32)
    out = mesh.copy()
    out.points = pts
    return out.triangulate()


def sample_camera(rng: random.Random, name: str) -> dict:
    base = next((c for c in BASE_CAMERAS if c['name'] == name), BASE_CAMERAS[0])
    pos = np.array(base['position'], dtype=np.float32)
    pos += np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), rng.uniform(-0.2, 0.2)], dtype=np.float32)
    return {
        'name': name,
        'position': tuple(float(x) for x in pos),
        'pose_vec': [float(pos[0]), float(pos[1]), float(pos[2]), 0.0, 0.0, 0.0, 1.0 if name == 'top' else 0.0, 0.0],
    }


class SceneRenderer:
    def __init__(self, width=256, height=256):
        self.width = width
        self.height = height

    def render_frame(self, meshes, camera_pos, focal_point=(0, 0, 0), view_up=(0, 0, 1)):
        pl = pv.Plotter(off_screen=True, window_size=[self.width, self.height])
        for mesh, color in meshes:
            pl.add_mesh(mesh, color=color, smooth_shading=True)
        pl.set_background('black')
        pl.camera_position = [camera_pos, focal_point, view_up]
        pl.render()
        rgb = pl.screenshot(return_img=True)[..., :3].astype(np.uint8)
        pl.close()

        depth = self._render_depth(meshes, camera_pos, focal_point, view_up)
        masks = []
        for mesh, color in meshes:
            pl2 = pv.Plotter(off_screen=True, window_size=[self.width, self.height])
            pl2.add_mesh(mesh, color=color, smooth_shading=True)
            pl2.set_background('black')
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
        pl.set_background('black')
        for mesh, _ in meshes:
            dists = np.linalg.norm(mesh.points - cam, axis=1).astype(np.float32)
            m2 = mesh.copy()
            m2['depth_scalar'] = dists
            pl.add_mesh(m2, scalars='depth_scalar', cmap='gray', clim=[d_min, d_max], show_scalar_bar=False)
        pl.camera_position = [camera_pos, focal_point, view_up]
        pl.render()
        encoded = pl.screenshot(return_img=True)[..., 0].astype(np.float32)
        pl.close()
        depth = np.where(encoded > 0, d_min + (encoded / 255.0) * d_range, 0.0).astype(np.float32)
        return depth


def compute_collision_transforms(frame_idx: int, n_frames: int, collision_type: str):
    t = frame_idx / max(n_frames - 1, 1)
    if collision_type == 'head_on':
        pos = -0.6 + 1.2 * t
        trans0 = np.array([pos, 0.0, 0.0])
        trans1 = np.array([-pos, 0.0, 0.0])
        rot0 = (t * 180.0, t * 70.0, t * 25.0)
        rot1 = (180.0, -t * 65.0, t * 20.0)
    elif collision_type == 'orbit_tight':
        angle = t * 2 * np.pi
        r = 0.16
        trans0 = np.array([r * np.cos(angle), r * np.sin(angle), 0.0])
        trans1 = np.array([r * np.cos(angle + np.pi), r * np.sin(angle + np.pi), 0.0])
        rot0 = (np.degrees(angle), np.degrees(angle) * 0.5, 0.0)
        rot1 = (np.degrees(angle + np.pi), np.degrees(angle + np.pi) * 0.5, 0.0)
    elif collision_type == 'vertical_pass':
        z_pos = -0.5 + 1.0 * t
        trans0 = np.array([0.0, 0.0, z_pos])
        trans1 = np.array([0.0, 0.0, -z_pos])
        rot0 = (t * 90.0, t * 25.0, t * 15.0)
        rot1 = (-t * 90.0, t * 35.0, -t * 10.0)
    elif collision_type == 'diagonal':
        d = -0.6 + 1.2 * t
        trans0 = np.array([d, d * 0.5, 0.0])
        trans1 = np.array([-d, -d * 0.5, 0.0])
        rot0 = (t * 120.0, t * 40.0, 0.0)
        rot1 = (180.0 - t * 120.0, t * 30.0, 0.0)
    else:
        trans0 = np.array([0.3, 0.0, 0.0])
        trans1 = np.array([-0.3, 0.0, 0.0])
        rot0 = (0.0, 0.0, 0.0)
        rot1 = (0.0, 0.0, 0.0)
    return trans0, rot0, trans1, rot1


def render_sample(out_dir: Path, renderer: SceneRenderer, n_frames: int, shape0: str, shape1: str, collision_type: str, camera_name: str, rng: random.Random):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'frames').mkdir(exist_ok=True)
    (out_dir / 'depth').mkdir(exist_ok=True)
    (out_dir / 'mask').mkdir(exist_ok=True)

    scale0 = rng.uniform(0.32, 0.48)
    scale1 = rng.uniform(0.32, 0.48)
    aspect0 = (rng.uniform(0.7, 1.5), rng.uniform(0.7, 1.5), rng.uniform(0.7, 1.5))
    aspect1 = (rng.uniform(0.7, 1.5), rng.uniform(0.7, 1.5), rng.uniform(0.7, 1.5))
    mesh0_base = apply_mild_deformation(create_mesh(shape0, scale0, aspect0), rng)
    mesh1_base = apply_mild_deformation(create_mesh(shape1, scale1, aspect1), rng)
    color0, color1 = random_color_pair(rng)
    cam = sample_camera(rng, camera_name)

    frames_rgb = []
    for f_idx in range(n_frames):
        trans0, rot0, trans1, rot1 = compute_collision_transforms(f_idx, n_frames, collision_type)
        m0 = mesh0_base.copy()
        m0.rotate_x(rot0[0], inplace=True)
        m0.rotate_y(rot0[1], inplace=True)
        m0.rotate_z(rot0[2], inplace=True)
        m0.translate(trans0, inplace=True)
        m1 = mesh1_base.copy()
        m1.rotate_x(rot1[0], inplace=True)
        m1.rotate_y(rot1[1], inplace=True)
        m1.rotate_z(rot1[2], inplace=True)
        m1.translate(trans1, inplace=True)
        rgb, depth, masks = renderer.render_frame([(m0, color0), (m1, color1)], cam['position'])
        iio3.imwrite(str(out_dir / 'frames' / f'{f_idx:04d}.png'), rgb)
        np.save(str(out_dir / 'depth' / f'{f_idx:04d}.npy'), depth)
        for ei, mask in enumerate(masks):
            iio3.imwrite(str(out_dir / 'mask' / f'{f_idx:04d}_entity{ei}.png'), mask)
        frames_rgb.append(rgb)

    iio2.mimsave(str(out_dir / 'video.gif'), frames_rgb, duration=125)
    meta = {
        'keyword0': f'red_{shape0}',
        'keyword1': f'blue_{shape1}',
        'entity_names': [f'red_{shape0}', f'blue_{shape1}'],
        'entity_specs': [
            {'name': f'red_{shape0}', 'shape': shape0, 'scale': scale0, 'aspect': aspect0, 'color': color0},
            {'name': f'blue_{shape1}', 'shape': shape1, 'scale': scale1, 'aspect': aspect1, 'color': color1},
        ],
        'prompt_full': f'a {shape0} and a {shape1} colliding',
        'mode': collision_type,
        'clip_type': collision_type,
        'camera': cam['name'],
        'camera_pose': cam['pose_vec'],
        'n_frames': n_frames,
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='toy/data_synthetic_overlap_hard')
    parser.add_argument('--n-samples-per-combo', type=int, default=2)
    parser.add_argument('--n-frames', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = SceneRenderer(width=args.resolution, height=args.resolution)

    cameras = [c['name'] for c in BASE_CAMERAS]
    total = len(SHAPE_FAMILIES) * len(SHAPE_FAMILIES) * len(COLLISION_TYPES) * len(cameras) * args.n_samples_per_combo
    print(f'Generating {total} hard samples...')

    count = 0
    for shape0 in SHAPE_FAMILIES:
        for shape1 in SHAPE_FAMILIES:
            if shape0 == shape1 and shape0 in {'cone', 'capsule'}:
                # avoid oversampling unstable rare pairs
                continue
            for collision_type in COLLISION_TYPES:
                for camera_name in cameras:
                    for rep in range(args.n_samples_per_combo):
                        sample_id = f'{shape0}_{shape1}_{collision_type}_{camera_name}_{rep:02d}'
                        render_sample(out_dir / sample_id, renderer, args.n_frames, shape0, shape1, collision_type, camera_name, rng)
                        count += 1
                        if count % 25 == 0:
                            print(f'Generated {count}/{total}...')
    print(f'Done. Generated {count} samples to {out_dir}')


if __name__ == '__main__':
    main()
