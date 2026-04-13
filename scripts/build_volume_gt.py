"""
Phase 62 — Build Volume Ground-Truth from Rendered 3D Data
============================================================

Constructs V_gt (K, H_out, W_out) int64 class indices from:
  - depth_map: per-pixel scene depth
  - entity_masks: binary masks for each entity
  - depth_order: front/back ordering

Class indices: 0=background, 1=entity0, 2=entity1.

For each pixel, the entity's actual depth is quantized to a depth bin k,
and the entity's class index is placed at V_gt[k, h, w].

In overlap regions, the front entity occupies the closer bin and the
back entity occupies the farther bin, preserving correct occlusion
geometry in the volume.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def build_volume_gt(
    depth_map: np.ndarray,       # (H, W) per-pixel scene depth, float
    entity_masks: np.ndarray,    # (2, S) binary entity masks, S = H_mask * W_mask
    depth_order: tuple,          # (front_idx, back_idx)
    K: int = 8,                  # number of depth bins
    H_out: int = 32,             # output spatial resolution (latent)
    W_out: int = 32,
) -> np.ndarray:
    """
    Build voxel GT from actual rendered 3D data.

    For each pixel (h, w):
      1. Get actual depth from depth_map
      2. Quantize to depth bin k using the scene's depth range
      3. If entity0 mask is 1: V_gt[k, h, w] = 1
      4. If entity1 mask is 1: V_gt[k, h, w] = 2
      5. In overlap: front entity → closer bin, back entity → farther bin
      6. Empty voxels = background (0)

    Returns: (K, H_out, W_out) int64 class indices
    """
    # --- Resize masks to H_out x W_out ---
    S = entity_masks.shape[1]
    H_mask = int(round(S ** 0.5))
    W_mask = H_mask

    mask0_2d = entity_masks[0].reshape(H_mask, W_mask)  # (H_mask, W_mask)
    mask1_2d = entity_masks[1].reshape(H_mask, W_mask)  # (H_mask, W_mask)

    # Resize via PIL bilinear, then threshold at 0.5 for binary
    mask0_resized = np.array(
        Image.fromarray((mask0_2d * 255).astype(np.uint8)).resize(
            (W_out, H_out), Image.BILINEAR),
        dtype=np.float32) / 255.0
    mask1_resized = np.array(
        Image.fromarray((mask1_2d * 255).astype(np.uint8)).resize(
            (W_out, H_out), Image.BILINEAR),
        dtype=np.float32) / 255.0

    m0 = (mask0_resized > 0.5).astype(np.bool_)  # (H_out, W_out)
    m1 = (mask1_resized > 0.5).astype(np.bool_)  # (H_out, W_out)

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

    # Class assignments: entity0 → class 1, entity1 → class 2
    entity_class = {0: 1, 1: 2}

    # Compute per-pixel depth bin
    depth_norm = (depth_resized - d_min) / d_range  # [0, 1]
    depth_bin = np.clip(
        (depth_norm * K).astype(np.int64), 0, K - 1)  # (H_out, W_out)

    # Overlap region
    overlap = m0 & m1  # (H_out, W_out)
    only_e0 = m0 & (~m1)
    only_e1 = m1 & (~m0)

    # --- Non-overlap: place entity at its depth bin ---
    for h in range(H_out):
        for w in range(W_out):
            if only_e0[h, w]:
                k = depth_bin[h, w]
                V_gt[k, h, w] = entity_class[0]
            elif only_e1[h, w]:
                k = depth_bin[h, w]
                V_gt[k, h, w] = entity_class[1]
            elif overlap[h, w]:
                # Front entity → depth bin from depth map
                # Back entity → one bin farther (or clamp to K-1)
                k_front = depth_bin[h, w]
                k_back = min(k_front + 1, K - 1)
                # If front and back land on the same bin, shift back
                if k_front == k_back and k_front > 0:
                    k_front = k_front - 1
                V_gt[k_front, h, w] = entity_class[front_idx]
                V_gt[k_back, h, w] = entity_class[back_idx]

    return V_gt


def build_volume_gt_batch(
    depth_maps: np.ndarray,       # (T, H, W) per-frame depth
    entity_masks: np.ndarray,     # (T, 2, S)
    depth_orders: list,           # list of (front_idx, back_idx) per frame
    K: int = 8,
    H_out: int = 32,
    W_out: int = 32,
) -> np.ndarray:
    """
    Build V_gt for a batch of T frames.
    Returns: (T, K, H_out, W_out) int64
    """
    T = min(depth_maps.shape[0], entity_masks.shape[0], len(depth_orders))
    V_gt_batch = np.zeros((T, K, H_out, W_out), dtype=np.int64)

    for t in range(T):
        V_gt_batch[t] = build_volume_gt(
            depth_map=depth_maps[t],
            entity_masks=entity_masks[t],
            depth_order=depth_orders[t],
            K=K, H_out=H_out, W_out=W_out,
        )

    return V_gt_batch
