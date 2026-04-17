"""Convert existing toy/data_objaverse to Phase 65 format.

Phase 65 expects:
  data/phase65/
    sample_000001/
      meta.json
      frames/0000.png ...
      visible_masks/0000_e0.png, 0000_e1.png ...
      amodal_masks/0000_e0.png, 0000_e1.png ...
      depth/0000.npy ...

Usage:
    python scripts/convert_to_phase65.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def convert_sample(src_dir: Path, dst_dir: Path, sample_idx: int) -> bool:
    """Convert a single sample from old format to Phase 65 format."""
    meta_path = src_dir / "meta.json"
    if not meta_path.exists():
        return False

    meta = json.loads(meta_path.read_text())

    # Create destination directory
    sample_name = f"sample_{sample_idx:06d}"
    sample_dir = dst_dir / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Convert meta.json
    entity_names = [meta.get("keyword0", "entity0"), meta.get("keyword1", "entity1")]
    clip_type = meta.get("mode", "unknown")
    text_prompt = meta.get("prompt_full", f"{entity_names[0]} and {entity_names[1]}")
    num_frames = meta.get("n_frames", 16)

    new_meta = {
        "entity_names": entity_names,
        "clip_type": clip_type,
        "text_prompt": text_prompt,
        "num_frames": num_frames,
        "camera": meta.get("camera", "front"),
    }
    (sample_dir / "meta.json").write_text(json.dumps(new_meta, indent=2))

    # Copy frames
    frames_src = src_dir / "frames"
    frames_dst = sample_dir / "frames"
    frames_dst.mkdir(exist_ok=True)
    if frames_src.exists():
        for f in sorted(frames_src.glob("*.png")):
            shutil.copy(f, frames_dst / f.name)

    # Copy depth
    depth_src = src_dir / "depth"
    depth_dst = sample_dir / "depth"
    depth_dst.mkdir(exist_ok=True)
    if depth_src.exists():
        for f in sorted(depth_src.glob("*.npy")):
            shutil.copy(f, depth_dst / f.name)

    # Convert masks (entity0 -> e0, entity1 -> e1)
    # For simple data, visible_masks = amodal_masks (no occlusion in source)
    mask_src = src_dir / "mask"
    visible_dst = sample_dir / "visible_masks"
    amodal_dst = sample_dir / "amodal_masks"
    visible_dst.mkdir(exist_ok=True)
    amodal_dst.mkdir(exist_ok=True)

    if mask_src.exists():
        for f in sorted(mask_src.glob("*_entity0.png")):
            stem = f.stem.replace("_entity0", "_e0")
            shutil.copy(f, visible_dst / f"{stem}.png")
            shutil.copy(f, amodal_dst / f"{stem}.png")

        for f in sorted(mask_src.glob("*_entity1.png")):
            stem = f.stem.replace("_entity1", "_e1")
            shutil.copy(f, visible_dst / f"{stem}.png")
            shutil.copy(f, amodal_dst / f"{stem}.png")

    return True


def main():
    dst_root = Path("data/phase65")
    dst_root.mkdir(parents=True, exist_ok=True)

    # Source directories to convert
    src_dirs = [
        Path("toy/data_objaverse"),
        Path("toy/data_objaverse_occ"),
        Path("toy/data_synthetic_overlap"),
    ]

    all_samples = []
    for src_root in src_dirs:
        if src_root.exists():
            samples = sorted([p for p in src_root.iterdir() if p.is_dir()])
            all_samples.extend(samples)
            print(f"Found {len(samples)} samples in {src_root}")

    if not all_samples:
        print("No source samples found")
        return

    converted = 0
    for idx, sample_dir in enumerate(all_samples, start=1):
        if convert_sample(sample_dir, dst_root, idx):
            converted += 1
            if converted % 20 == 0:
                print(f"Converted {converted} samples...")

    print(f"Done. Converted {converted} samples to {dst_root}")


if __name__ == "__main__":
    main()
