"""
Phase 40 — Solo Render Data Generator & Dataset Extension
==========================================================

Phase 40의 핵심 loss (L_solo_feat_visible, L_id_contrast)는 solo reference features가
필요합니다. 이 스크립트는 두 가지를 제공합니다.

1. Pseudo-solo frame 생성 (mask-based approximation):
   - 기존 composite frame × entity0_mask → entity0 pseudo-solo
   - 기존 composite frame × entity1_mask → entity1 pseudo-solo
   - 진짜 solo render보다 정확도는 낮지만 즉시 사용 가능
   - Phase 40 stage1/2는 이 방법으로도 충분히 학습 가능

2. 실제 solo render (Blender 사용, 선택적):
   - generate_solo_renders_blender() 함수 제공
   - meta.json의 mesh0_path, mesh1_path 사용
   - Blender 설치 필요: blender --background --python ...

3. Visible mask 생성:
   - compute_visible_masks(): 기존 mask + depth ordering → visible mask
   - 추가 렌더링 불필요

4. ObjaverseDatasetPhase40 클래스:
   - Phase 39 ObjaverseDatasetWithMasks 확장
   - solo frames + visible masks 추가 반환

Usage
-----
# 전체 dataset에 pseudo-solo 생성 (5분 이내)
python scripts/generate_solo_renders.py --data-root toy/data_objaverse --method pseudo

# Blender 사용 (진짜 solo render)
python scripts/generate_solo_renders.py --data-root toy/data_objaverse --method blender --blender-path /path/to/blender

# 생성 후 dataset 테스트
python scripts/generate_solo_renders.py --data-root toy/data_objaverse --verify
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_phase31 import ObjaverseDatasetWithMasks


# =============================================================================
# Visible mask computation (renderer-free)
# =============================================================================

def compute_visible_masks_np(
    entity_masks_TNS: np.ndarray,   # (T, 2, S) float/bool
    depth_orders_T:   list,
) -> np.ndarray:
    """
    GT visible masks from full masks + depth ordering.

    front entity: fully visible
    back entity: visible only where front entity mask = 0

    Args
    ----
    entity_masks_TNS: (T, 2, S) — full entity masks, S = H*W (downsampled)
    depth_orders_T  : list of (front_idx, back_idx) per frame

    Returns
    -------
    visible (T, 2, S) float32
    """
    T = entity_masks_TNS.shape[0]
    visible = np.zeros_like(entity_masks_TNS, dtype=np.float32)

    for t in range(min(T, len(depth_orders_T))):
        front = int(depth_orders_T[t][0])
        back  = int(depth_orders_T[t][1])
        m_front = entity_masks_TNS[t, front].astype(float)
        m_back  = entity_masks_TNS[t, back].astype(float)
        visible[t, front] = m_front
        visible[t, back]  = m_back * (1.0 - m_front)

    return visible


# =============================================================================
# Pseudo-solo frame generation (mask-based, no renderer)
# =============================================================================

def make_pseudo_solo_frames(
    composite_frames: np.ndarray,        # (T, H, W, 3) uint8
    mask_dir:         Path,
    n_frames:         int,
    bg_color:         Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    mask-based pseudo-solo frame 생성.

    entity0 pseudo-solo[t] = composite[t] * mask0[t] + bg_color * (1 - mask0[t])
    entity1 pseudo-solo[t] = composite[t] * mask1[t] + bg_color * (1 - mask1[t])

    완벽한 solo render는 아니지만:
    - entity0 visible region의 appearance는 정확히 보존
    - occluded region에서만 bg_color로 채워짐
    - Phase 40 L_solo_feat_visible 학습에 충분히 유효

    Args
    ----
    composite_frames : (T, H, W, 3) uint8
    mask_dir         : {sample_dir}/mask/
    n_frames         : 프레임 수

    Returns
    -------
    solo_e0, solo_e1 : (T, H, W, 3) uint8
    """
    T, H, W, _ = composite_frames.shape
    solo_e0 = np.full_like(composite_frames, bg_color[0])
    solo_e1 = np.full_like(composite_frames, bg_color[0])

    for t in range(min(T, n_frames)):
        mask0_path = mask_dir / f"{t:04d}_entity0.png"
        mask1_path = mask_dir / f"{t:04d}_entity1.png"

        if not mask0_path.exists() or not mask1_path.exists():
            continue

        m0 = np.array(Image.open(mask0_path).convert("L").resize((W, H), Image.NEAREST))
        m1 = np.array(Image.open(mask1_path).convert("L").resize((W, H), Image.NEAREST))

        # threshold to binary
        m0_bin = (m0 > 127).astype(np.float32)[..., None]   # (H, W, 1)
        m1_bin = (m1 > 127).astype(np.float32)[..., None]

        frame = composite_frames[t].astype(np.float32)   # (H, W, 3)
        bg    = np.array(bg_color, dtype=np.float32)

        solo_e0[t] = (frame * m0_bin + bg * (1.0 - m0_bin)).clip(0, 255).astype(np.uint8)
        solo_e1[t] = (frame * m1_bin + bg * (1.0 - m1_bin)).clip(0, 255).astype(np.uint8)

    return solo_e0, solo_e1


def generate_pseudo_solo_for_sample(sample_dir: Path, overwrite: bool = False) -> bool:
    """
    한 sample에 대해 pseudo-solo frames 생성 → solo_entity0/, solo_entity1/ 저장.

    Returns True if generated successfully.
    """
    frames_dir = sample_dir / "frames"
    mask_dir   = sample_dir / "mask"
    solo_e0_dir = sample_dir / "solo_entity0"
    solo_e1_dir = sample_dir / "solo_entity1"

    if not frames_dir.exists() or not mask_dir.exists():
        return False

    if solo_e0_dir.exists() and not overwrite:
        return True   # already generated

    # load meta for n_frames
    meta_path = sample_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    n_frames = int(meta.get("n_frames", 16))

    # load composite frames
    frames = []
    for t in range(n_frames):
        fp = frames_dir / f"{t:04d}.png"
        if fp.exists():
            frames.append(np.array(Image.open(fp).convert("RGB")))
        else:
            break
    if not frames:
        return False

    composite = np.stack(frames, axis=0)   # (T, H, W, 3)
    solo_e0, solo_e1 = make_pseudo_solo_frames(composite, mask_dir, n_frames)

    solo_e0_dir.mkdir(parents=True, exist_ok=True)
    solo_e1_dir.mkdir(parents=True, exist_ok=True)

    for t in range(len(frames)):
        Image.fromarray(solo_e0[t]).save(solo_e0_dir / f"{t:04d}.png")
        Image.fromarray(solo_e1[t]).save(solo_e1_dir / f"{t:04d}.png")

    return True


def generate_solo_data_for_dataset(
    data_root: str,
    method:    str = "pseudo",
    overwrite: bool = False,
    verbose:   bool = True,
):
    """
    전체 dataset에 solo render 데이터 생성.

    method:
      "pseudo"  — mask-based pseudo-solo (즉시 사용 가능, Blender 불필요)
      "blender" — 실제 solo render (Blender 필요, generate_solo_renders_blender 사용)
    """
    data_root = Path(data_root)
    sample_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])

    if verbose:
        print(f"[generate_solo] {len(sample_dirs)} samples, method={method}")

    n_ok = 0
    for i, sample_dir in enumerate(sample_dirs):
        if method == "pseudo":
            ok = generate_pseudo_solo_for_sample(sample_dir, overwrite=overwrite)
        elif method == "blender":
            raise NotImplementedError(
                "Blender solo render 생성은 generate_solo_renders_blender() 사용. "
                "blender --background --python tools/render_solo_blender.py "
                "-- --mesh-path <mesh> --output-dir <dir> 를 직접 호출해야 합니다."
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        if ok:
            n_ok += 1
        if verbose and (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(sample_dirs)}] done={n_ok}")

    if verbose:
        print(f"[generate_solo] 완료: {n_ok}/{len(sample_dirs)} samples")


# =============================================================================
# Extended dataset for Phase 40
# =============================================================================

class ObjaverseDatasetPhase40(ObjaverseDatasetWithMasks):
    """
    Phase 40용 확장 dataset.

    Phase 39 return: frames_np, depths_np, depth_orders, meta, entity_masks

    Phase 40 return (추가):
      visible_masks : (T, 2, S) float32 — computed from masks + depth_orders
      solo_e0_frames: (T, H, W, 3) uint8 — entity0 solo frames (pseudo or real)
      solo_e1_frames: (T, H, W, 3) uint8 — entity1 solo frames

    solo renders가 없으면:
      - solo_e0_frames = solo_e1_frames = None (train_phase40.py가 fallback 사용)
    """

    def __init__(self, data_root: str, n_frames: int = 8, require_solo: bool = False):
        super().__init__(data_root, n_frames=n_frames)
        self.require_solo = require_solo

    def __getitem__(self, idx):
        base = super().__getitem__(idx)
        frames_np, depths_np, depth_orders, meta, entity_masks = base

        # ── visible masks ───────────────────────────────────────────────
        visible_masks = compute_visible_masks_np(
            entity_masks.astype(np.float32), depth_orders)

        # ── solo frames ─────────────────────────────────────────────────
        sample_dir  = self._get_sample_dir(idx)
        solo_e0_dir = sample_dir / "solo_entity0"
        solo_e1_dir = sample_dir / "solo_entity1"

        solo_e0 = None
        solo_e1 = None

        if solo_e0_dir.exists() and solo_e1_dir.exists():
            try:
                solo_e0 = self._load_frames_from_dir(solo_e0_dir)
                solo_e1 = self._load_frames_from_dir(solo_e1_dir)
            except Exception:
                solo_e0 = solo_e1 = None

        if self.require_solo and (solo_e0 is None or solo_e1 is None):
            raise RuntimeError(
                f"Solo renders required but missing for {sample_dir}. "
                "Run: python scripts/generate_solo_renders.py --method pseudo"
            )

        return frames_np, depths_np, depth_orders, meta, entity_masks, visible_masks, solo_e0, solo_e1

    def _get_sample_dir(self, idx: int) -> Path:
        """sample index → directory path (uses base class sample list)."""
        return self.samples[idx]["dir"]

    def _load_frames_from_dir(self, frame_dir: Path) -> Optional[np.ndarray]:
        """PNG frames → (T, H, W, 3) uint8."""
        pngs = sorted(frame_dir.glob("*.png"))
        if not pngs:
            return None
        frames = [np.array(Image.open(p).convert("RGB")) for p in pngs[:self.n_frames]]
        return np.stack(frames, axis=0)

    def has_solo_renders(self, idx: int) -> bool:
        """solo renders가 생성되어 있는지 확인."""
        sample_dir = self._get_sample_dir(idx)
        return ((sample_dir / "solo_entity0").exists()
                and (sample_dir / "solo_entity1").exists())

    def count_solo_renders(self) -> int:
        return sum(1 for i in range(len(self)) if self.has_solo_renders(i))


def make_pseudo_solo_from_composite(
    frames_np:     np.ndarray,   # (T, H, W, 3) uint8
    sample_dir:    Path,
    n_frames:      int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fallback: composite frames + masks → pseudo-solo (runtime, no file I/O).
    solo_entity0/, solo_entity1/가 없을 때 train_phase40.py가 호출.
    """
    mask_dir = sample_dir / "mask"
    if not mask_dir.exists():
        return None, None
    return make_pseudo_solo_frames(frames_np, mask_dir, n_frames)


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Phase 40 solo render data generator")
    p.add_argument("--data-root", type=str, default="toy/data_objaverse")
    p.add_argument("--method",    type=str, default="pseudo",
                   choices=["pseudo", "blender"],
                   help="pseudo: mask-based (fast), blender: actual 3D render")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify",    action="store_true",
                   help="Check how many samples have solo renders")
    args = p.parse_args()

    if args.verify:
        ds  = ObjaverseDatasetPhase40(args.data_root, require_solo=False)
        cnt = ds.count_solo_renders()
        print(f"Solo renders: {cnt}/{len(ds)} samples")
        return

    generate_solo_data_for_dataset(
        data_root=args.data_root,
        method=args.method,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
