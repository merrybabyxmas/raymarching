"""Build splits for ablation study: with/without hard synthetic data."""
from __future__ import annotations

import json
import random
from pathlib import Path


def build_splits(sample_ids: list[str], seed: int = 42) -> dict:
    """70/15/15 split."""
    random.seed(seed)
    ids = list(sample_ids)
    random.shuffle(ids)
    n = len(ids)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    return {
        'train': ids[:n_train],
        'val': ids[n_train:n_train + n_val],
        'test': ids[n_train + n_val:],
    }


def main():
    output_dir = Path('outputs/phase65/splits')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Original data only (samples 1-316)
    original_ids = [f'sample_{i:06d}' for i in range(1, 317)]
    splits_original = build_splits(original_ids, seed=42)
    print(f"Original only: train={len(splits_original['train'])}, val={len(splits_original['val'])}, test={len(splits_original['test'])}")
    (output_dir / 'ablation_original_seed42.json').write_text(json.dumps(splits_original, indent=2))

    # All data (samples 1-1404)
    all_ids = [f'sample_{i:06d}' for i in range(1, 1405)]
    splits_all = build_splits(all_ids, seed=42)
    print(f"All data: train={len(splits_all['train'])}, val={len(splits_all['val'])}, test={len(splits_all['test'])}")
    (output_dir / 'ablation_all_seed42.json').write_text(json.dumps(splits_all, indent=2))

    print(f"Splits saved to {output_dir}")


if __name__ == '__main__':
    main()
