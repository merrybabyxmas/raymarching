from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _read_meta(sample_dir: Path) -> dict:
    return json.loads((sample_dir / 'meta.json').read_text())


def _infer_tags(meta: dict) -> dict:
    names = meta.get('entity_names') or [meta.get('keyword0', 'entity0'), meta.get('keyword1', 'entity1')]
    clip_type = meta.get('clip_type') or meta.get('mode', 'unknown')
    camera = meta.get('camera', 'unknown')
    joined = ' '.join(str(x) for x in names).lower()
    tags = {
        'entity_names': names,
        'camera': camera,
        'clip_type': clip_type,
        'has_cylinder': 'cylinder' in joined,
        'has_box': 'box' in joined or 'cube' in joined,
        'has_sphere': 'sphere' in joined,
    }
    return tags


def build_splits(sample_dirs: List[Path], mode: str, seed: int = 42, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    records = []
    for p in sample_dirs:
        meta = _read_meta(p)
        rec = {'sample_id': p.name, **_infer_tags(meta)}
        records.append(rec)

    train, val, test = [], [], []

    if mode == 'random':
        ids = [r['sample_id'] for r in records]
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test = ids[:n_test]
        val = ids[n_test:n_test + n_val]
        train = ids[n_test + n_val:]
    elif mode == 'shape_holdout':
        test = [r['sample_id'] for r in records if r['has_cylinder']]
        remain = [r['sample_id'] for r in records if not r['has_cylinder']]
        rng.shuffle(remain)
        n_val = max(1, int(len(remain) * val_ratio))
        val = remain[:n_val]
        train = remain[n_val:]
    elif mode == 'camera_holdout':
        holdout = {'top', 'front_right'}
        test = [r['sample_id'] for r in records if r['camera'] in holdout]
        remain = [r['sample_id'] for r in records if r['camera'] not in holdout]
        rng.shuffle(remain)
        n_val = max(1, int(len(remain) * val_ratio))
        val = remain[:n_val]
        train = remain[n_val:]
    elif mode == 'collision_holdout':
        holdout = {'diagonal', 'vertical_pass'}
        test = [r['sample_id'] for r in records if r['clip_type'] in holdout]
        remain = [r['sample_id'] for r in records if r['clip_type'] not in holdout]
        rng.shuffle(remain)
        n_val = max(1, int(len(remain) * val_ratio))
        val = remain[:n_val]
        train = remain[n_val:]
    else:
        raise ValueError(f'Unknown split mode: {mode}')

    return {
        'mode': mode,
        'seed': seed,
        'train': train,
        'val': val,
        'test': test,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/phase65')
    parser.add_argument('--mode', type=str, default='random', choices=['random', 'shape_holdout', 'camera_holdout', 'collision_holdout'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    root = Path(args.root)
    sample_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    splits = build_splits(sample_dirs, mode=args.mode, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(splits, indent=2))
    print(f'Wrote split file to {out_path}')
    print({k: len(v) if isinstance(v, list) else v for k, v in splits.items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
