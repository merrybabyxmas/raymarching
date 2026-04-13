"""
Phase 62 — Volume Ground-Truth Builder
========================================

Constructs V_gt (K, H_out, W_out) int64 class indices from rendered data:
  - depth_map: per-pixel scene depth
  - entity_masks: amodal masks for each entity
  - visible_masks: visible masks for each entity (preferred)
  - depth_order: front/back ordering fallback

Class indices: 0=background, 1=entity0, 2=entity1.

For each pixel, the entity's actual depth is quantized to a depth bin k,
and the entity's class index is placed at V_gt[k, h, w].

In overlap regions, the front entity occupies the closer bin and the
back entity occupies the farther bin, preserving correct occlusion
geometry in the volume.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


class VolumeGTBuilder:
    """
    Builds a geometry-aware voxel target from depth + amodal/visible masks.

    Usage:
        builder = VolumeGTBuilder(depth_bins=8, spatial_h=16, spatial_w=16)
        V_gt = builder.build(depth_map, entity_masks, depth_order)
        V_gt_batch = builder.build_batch(depth_maps, entity_masks_batch, depth_orders)
    """

    def __init__(
        self,
        depth_bins: int = 8,
        spatial_h: int = 16,
        spatial_w: int = 16,
        render_resolution: int = 128,
        cache_tag: str = "rendered",
    ):
        self.K = depth_bins
        self.H_out = spatial_h
        self.W_out = spatial_w
        self.render_resolution = int(render_resolution)
        self.cache_tag = str(cache_tag)

        try:
            import pyvista as pv
            pv.start_xvfb()
        except Exception:
            pass

    def build(
        self,
        depth_map: np.ndarray,       # (H, W) per-pixel scene depth, float
        entity_masks: np.ndarray,    # (2, S) binary entity masks, S = H_mask * W_mask
        depth_order: tuple,          # (front_idx, back_idx)
        visible_masks: np.ndarray | None = None,  # (2, S) or None
        meta: dict | None = None,
        sample_dir: str | Path | None = None,
    ) -> np.ndarray:
        """
        Build voxel GT from depth + amodal/visible geometry cues.

        For each pixel (h, w):
          1. Get actual depth from depth_map
          2. Quantize to depth bin k using the scene's depth range
          3. Visible entity is placed at the observed depth bin
          4. Occluded entity (amodal - visible) is placed behind the visible one
          5. depth_order is used only when visible masks are ambiguous
          6. Empty voxels = background (0)

        Returns: (K, H_out, W_out) int64 class indices
        """
        if meta is not None and sample_dir is not None:
            try:
                rendered = self._build_from_rendered_entities(
                    frame_idx=0,
                    meta=meta,
                    sample_dir=sample_dir,
                )
                if rendered is not None:
                    return rendered
            except Exception:
                pass

        K = self.K
        H_out = self.H_out
        W_out = self.W_out

        # --- Resize amodal masks to H_out x W_out ---
        S = entity_masks.shape[1]
        H_mask = int(round(S ** 0.5))
        W_mask = H_mask

        mask0_2d = entity_masks[0].reshape(H_mask, W_mask)  # (H_mask, W_mask)
        mask1_2d = entity_masks[1].reshape(H_mask, W_mask)  # (H_mask, W_mask)

        # Use nearest for masks to preserve occupancy geometry.
        mask0_resized = np.array(
            Image.fromarray((mask0_2d * 255).astype(np.uint8)).resize(
                (W_out, H_out), Image.NEAREST),
            dtype=np.float32) / 255.0
        mask1_resized = np.array(
            Image.fromarray((mask1_2d * 255).astype(np.uint8)).resize(
                (W_out, H_out), Image.NEAREST),
            dtype=np.float32) / 255.0

        m0 = (mask0_resized > 0.5).astype(np.bool_)  # (H_out, W_out)
        m1 = (mask1_resized > 0.5).astype(np.bool_)  # (H_out, W_out)

        if visible_masks is not None:
            vis0_2d = visible_masks[0].reshape(H_mask, W_mask)
            vis1_2d = visible_masks[1].reshape(H_mask, W_mask)
            vis0_resized = np.array(
                Image.fromarray((vis0_2d * 255).astype(np.uint8)).resize(
                    (W_out, H_out), Image.NEAREST),
                dtype=np.float32) / 255.0
            vis1_resized = np.array(
                Image.fromarray((vis1_2d * 255).astype(np.uint8)).resize(
                    (W_out, H_out), Image.NEAREST),
                dtype=np.float32) / 255.0
            v0 = (vis0_resized > 0.5).astype(np.bool_)
            v1 = (vis1_resized > 0.5).astype(np.bool_)
        else:
            front_idx = int(depth_order[0])
            v0 = m0.copy()
            v1 = m1.copy()
            overlap_fb = m0 & m1
            if front_idx == 0:
                v1[overlap_fb] = False
            else:
                v0[overlap_fb] = False

        # --- Resize depth to H_out x W_out ---
        H_depth, W_depth = depth_map.shape[:2]
        if H_depth != H_out or W_depth != W_out:
            depth_resized = np.array(
                Image.fromarray(depth_map.astype(np.float32)).resize(
                    (W_out, H_out), Image.BILINEAR),
                dtype=np.float32)  # (H_out, W_out)
        else:
            depth_resized = depth_map.astype(np.float32)

        # --- Compute depth range for binning ---
        d_min = float(depth_resized.min())
        d_max = float(depth_resized.max())
        d_range = d_max - d_min
        if d_range < 1e-8:
            d_range = 1.0  # degenerate: single-depth scene

        # --- Build volume ---
        V_gt = np.zeros((K, H_out, W_out), dtype=np.int64)  # all background

        front_idx = int(depth_order[0])
        back_idx = int(depth_order[1])

        # Class assignments: entity0 -> class 1, entity1 -> class 2
        entity_class = {0: 1, 1: 2}

        # Compute per-pixel depth bin
        depth_norm = (depth_resized - d_min) / d_range  # [0, 1]
        depth_bin = np.clip(
            (depth_norm * K).astype(np.int64), 0, K - 1)  # (H_out, W_out)

        overlap_amodal = m0 & m1
        occ0 = m0 & (~v0)
        occ1 = m1 & (~v1)

        for h in range(H_out):
            for w in range(W_out):
                k_front = int(depth_bin[h, w])
                k_back = min(k_front + 1, K - 1)

                if v0[h, w] and not v1[h, w]:
                    V_gt[k_front, h, w] = entity_class[0]
                    if occ1[h, w] or (overlap_amodal[h, w] and back_idx == 1):
                        V_gt[k_back, h, w] = entity_class[1]
                elif v1[h, w] and not v0[h, w]:
                    V_gt[k_front, h, w] = entity_class[1]
                    if occ0[h, w] or (overlap_amodal[h, w] and back_idx == 0):
                        V_gt[k_back, h, w] = entity_class[0]
                elif v0[h, w] and v1[h, w]:
                    V_gt[k_front, h, w] = entity_class[front_idx]
                    V_gt[k_back, h, w] = entity_class[back_idx]
                elif m0[h, w] and not m1[h, w]:
                    V_gt[k_front, h, w] = entity_class[0]
                elif m1[h, w] and not m0[h, w]:
                    V_gt[k_front, h, w] = entity_class[1]
                elif overlap_amodal[h, w]:
                    V_gt[k_front, h, w] = entity_class[front_idx]
                    V_gt[k_back, h, w] = entity_class[back_idx]

        return V_gt

    def _cache_path(self, sample_dir: str | Path) -> Path:
        sample_dir = Path(sample_dir)
        return sample_dir / (
            f"volume_gt_{self.cache_tag}_k{self.K}_{self.H_out}x{self.W_out}.npy"
        )

    def _camera_position(self, camera_name: str):
        from toy.generate_toy_data import CAMERAS

        for cam in CAMERAS:
            if cam["name"] == camera_name:
                return tuple(cam["position"])
        raise KeyError(f"Unknown camera '{camera_name}'")

    def _resize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        return np.array(
            Image.fromarray(depth_map.astype(np.float32)).resize(
                (self.W_out, self.H_out), Image.BILINEAR
            ),
            dtype=np.float32,
        )

    def _depth_to_bin(self, depth: float, d_min: float, d_max: float) -> int:
        if d_max - d_min < 1e-8:
            return 0
        norm = (float(depth) - d_min) / (d_max - d_min)
        return int(np.clip(np.floor(norm * self.K), 0, self.K - 1))

    def _load_mesh_pair(self, meta: dict):
        from scripts.generate_crossing import load_mesh_as_pyvista

        mesh0 = load_mesh_as_pyvista(Path(meta["mesh0_path"]))
        mesh1 = load_mesh_as_pyvista(Path(meta["mesh1_path"]))
        return mesh0, mesh1

    def _render_entity_depths(
        self,
        mesh0_base,
        mesh1_base,
        meta: dict,
        frame_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        from scripts.generate_crossing import CrossingMode, compute_frame_transforms
        from toy.generate_toy_data import SceneRenderer

        n_frames = int(meta.get("n_frames", 16))
        mode = CrossingMode(str(meta["mode"]))
        camera_pos = self._camera_position(str(meta["camera"]))
        trans0, rot0, trans1, rot1 = compute_frame_transforms(mode, frame_idx, n_frames)

        m0 = mesh0_base.copy()
        m0.rotate_y(float(rot0), inplace=True)
        m0.translate(np.asarray(trans0, dtype=np.float32), inplace=True)

        m1 = mesh1_base.copy()
        m1.rotate_y(float(rot1) + 180.0, inplace=True)
        m1.translate(np.asarray(trans1, dtype=np.float32), inplace=True)

        renderer = SceneRenderer(
            width=self.render_resolution,
            height=self.render_resolution,
        )
        depth0 = renderer._render_depth([(m0, (1.0, 0.0, 0.0))], camera_pos, (0, 0, 0), (0, 0, 1))
        depth1 = renderer._render_depth([(m1, (0.0, 0.0, 1.0))], camera_pos, (0, 0, 0), (0, 0, 1))
        return depth0.astype(np.float32), depth1.astype(np.float32)

    def _frame_from_entity_depths(self, depth0: np.ndarray, depth1: np.ndarray) -> np.ndarray:
        d0 = self._resize_depth(depth0)
        d1 = self._resize_depth(depth1)
        m0 = d0 > 0.0
        m1 = d1 > 0.0

        V_gt = np.zeros((self.K, self.H_out, self.W_out), dtype=np.int64)
        nonzero = np.concatenate([d0[m0], d1[m1]]) if (m0.any() or m1.any()) else np.zeros((0,), dtype=np.float32)
        if nonzero.size == 0:
            return V_gt

        d_min = float(nonzero.min())
        d_max = float(nonzero.max())

        for h in range(self.H_out):
            for w in range(self.W_out):
                has0 = bool(m0[h, w])
                has1 = bool(m1[h, w])
                if not has0 and not has1:
                    continue
                if has0 and not has1:
                    k0 = self._depth_to_bin(d0[h, w], d_min, d_max)
                    V_gt[k0, h, w] = 1
                    continue
                if has1 and not has0:
                    k1 = self._depth_to_bin(d1[h, w], d_min, d_max)
                    V_gt[k1, h, w] = 2
                    continue

                depth_e0 = float(d0[h, w])
                depth_e1 = float(d1[h, w])
                if depth_e0 <= depth_e1:
                    near_cls, near_d = 1, depth_e0
                    far_cls, far_d = 2, depth_e1
                else:
                    near_cls, near_d = 2, depth_e1
                    far_cls, far_d = 1, depth_e0

                k_near = self._depth_to_bin(near_d, d_min, d_max)
                k_far = self._depth_to_bin(far_d, d_min, d_max)
                if k_far <= k_near:
                    if k_near >= self.K - 1:
                        k_near = max(self.K - 2, 0)
                    k_far = min(k_near + 1, self.K - 1)

                V_gt[k_near, h, w] = near_cls
                V_gt[k_far, h, w] = far_cls

        return V_gt

    def _build_from_rendered_entities(
        self,
        frame_idx: int,
        meta: dict,
        sample_dir: str | Path,
    ) -> np.ndarray | None:
        batch = self._build_rendered_batch(meta=meta, sample_dir=sample_dir)
        if batch is None:
            return None
        if frame_idx >= batch.shape[0]:
            return None
        return batch[frame_idx]

    def _build_rendered_batch(
        self,
        meta: dict,
        sample_dir: str | Path,
    ) -> np.ndarray | None:
        sample_dir = Path(sample_dir)
        cache_path = self._cache_path(sample_dir)
        if cache_path.exists():
            arr = np.load(cache_path)
            if arr.ndim == 4 and arr.shape[1:] == (self.K, self.H_out, self.W_out):
                return arr.astype(np.int64, copy=False)

        if "mesh0_path" not in meta or "mesh1_path" not in meta or "mode" not in meta or "camera" not in meta:
            return None

        mesh0_base, mesh1_base = self._load_mesh_pair(meta)
        n_frames = int(meta.get("n_frames", 16))
        V = np.zeros((n_frames, self.K, self.H_out, self.W_out), dtype=np.int64)
        for frame_idx in range(n_frames):
            depth0, depth1 = self._render_entity_depths(mesh0_base, mesh1_base, meta, frame_idx)
            V[frame_idx] = self._frame_from_entity_depths(depth0, depth1)

        np.save(cache_path, V)
        return V

    def build_batch(
        self,
        depth_maps: np.ndarray,       # (T, H, W) per-frame depth
        entity_masks: np.ndarray,     # (T, 2, S)
        depth_orders: list,           # list of (front_idx, back_idx) per frame
        visible_masks: np.ndarray | None = None,  # (T, 2, S) or None
        meta: dict | None = None,
        sample_dir: str | Path | None = None,
    ) -> np.ndarray:
        """
        Build V_gt for a batch of T frames.
        Returns: (T, K, H_out, W_out) int64
        """
        if meta is not None and sample_dir is not None:
            rendered = self._build_rendered_batch(meta=meta, sample_dir=sample_dir)
            if rendered is not None:
                T = min(
                    rendered.shape[0],
                    depth_maps.shape[0] if depth_maps is not None else rendered.shape[0],
                    entity_masks.shape[0],
                    len(depth_orders),
                )
                return rendered[:T]

        T = min(depth_maps.shape[0], entity_masks.shape[0], len(depth_orders))
        V_gt_batch = np.zeros((T, self.K, self.H_out, self.W_out), dtype=np.int64)

        for t in range(T):
            V_gt_batch[t] = self.build(
                depth_map=depth_maps[t],
                entity_masks=entity_masks[t],
                depth_order=depth_orders[t],
                visible_masks=(visible_masks[t] if visible_masks is not None else None),
                meta=meta,
                sample_dir=sample_dir,
            )

        return V_gt_batch
