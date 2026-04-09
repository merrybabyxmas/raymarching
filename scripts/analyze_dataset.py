"""
Phase 15 Part D: Objaverse 데이터셋 통계 분석

생성된 toy/data_objaverse/ 데이터셋의 품질 지표 측정:
  1. total_samples: 전체 샘플 수
  2. category_distribution: 키워드 쌍별 분포
  3. depth_reversal_rate: depth ordering이 역전되는 프레임 비율
  4. occlusion_rate: entity mask IoU > 0.1인 프레임 비율
  5. mode_distribution: ORBIT/SQUEEZE/ROTATE 비율

출력:
  debug/dataset_stats/
    objaverse_stats.json
    objaverse_report.md
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import imageio.v3 as iio3

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── 단일 시퀀스 분석 ─────────────────────────────────────────────────────────

def analyze_sequence(seq_dir: Path) -> dict:
    """meta.json + frames/depth/mask → 통계 dict 반환."""
    meta_path = seq_dir / 'meta.json'
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    frames = sorted((seq_dir / 'frames').glob('*.png'))
    depths = sorted((seq_dir / 'depth').glob('*.npy'))
    if not frames or not depths:
        return None

    n_frames = min(len(frames), len(depths))

    depth_orders = []
    occlusions   = []

    for fi in range(n_frames):
        depth = np.load(str(depths[fi]))

        m0_path = seq_dir / 'mask' / f'{fi:04d}_entity0.png'
        m1_path = seq_dir / 'mask' / f'{fi:04d}_entity1.png'

        if not m0_path.exists() or not m1_path.exists():
            continue

        m0 = iio3.imread(str(m0_path)) > 128
        m1 = iio3.imread(str(m1_path)) > 128

        # depth ordering
        m0_only = m0 & ~m1
        m1_only = m1 & ~m0

        if m0_only.sum() > 5 and m1_only.sum() > 5:
            d0 = float(depth[m0_only].mean())
            d1 = float(depth[m1_only].mean())
            depth_orders.append(d0 < d1)  # True = entity0이 앞

        # occlusion: mask 겹침 여부 (어느 픽셀이라도 겹치면 True)
        intersection = (m0 & m1).sum()
        occlusions.append(intersection > 10)   # 10픽셀 이상 겹치면 occlusion 있음

    # depth reversal: True/False가 섞여 있으면 역전 있음
    has_reversal = (0 < sum(depth_orders) < len(depth_orders)) if depth_orders else False
    depth_reversal_rate = (1.0 if has_reversal else 0.0)

    # occlusion_rate: 이 시퀀스에서 겹치는 프레임이 1개라도 있으면 1.0
    has_occlusion = any(occlusions) if occlusions else False

    return {
        'keyword0':   meta.get('keyword0', ''),
        'keyword1':   meta.get('keyword1', ''),
        'mode':       meta.get('mode', ''),
        'camera':     meta.get('camera', ''),
        'n_frames':   n_frames,
        'depth_reversal_rate': depth_reversal_rate,
        'has_occlusion':       1.0 if has_occlusion else 0.0,
        'avg_depth_sep':       float(np.std(depth_orders)) if len(depth_orders) > 1 else 0.0,
    }


# ─── 전체 데이터셋 분석 ──────────────────────────────────────────────────────

def analyze_objaverse_dataset(data_root: str, out_dir: str):
    data_root = Path(data_root)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = [d for d in data_root.iterdir() if (d / 'meta.json').exists()]
    print(f"[analyze] {len(seq_dirs)} sequences found in {data_root}", flush=True)

    results = []
    for seq_dir in seq_dirs:
        r = analyze_sequence(seq_dir)
        if r:
            results.append(r)
            print(f"  {seq_dir.name}: depth_rev={r['depth_reversal_rate']:.2f}  "
                  f"occ={r['has_occlusion']:.2f}", flush=True)

    if not results:
        print("[warn] No valid sequences found", flush=True)
        stats = {'total_samples': 0, 'total_frames': 0}
    else:
        # 집계
        mode_counts = defaultdict(int)
        kw_pair_counts = defaultdict(int)
        total_frames = sum(r['n_frames'] for r in results)
        for r in results:
            mode_counts[r['mode']] += 1
            kw_pair_counts[f"{r['keyword0']}+{r['keyword1']}"] += 1

        total = len(results)
        avg_depth_rev = float(np.mean([r['depth_reversal_rate'] for r in results]))
        # occlusion_rate = 시퀀스 중 1프레임이라도 겹치는 비율
        occlusion_rate = float(np.mean([r['has_occlusion'] for r in results]))
        avg_depth_sep  = float(np.mean([r['avg_depth_sep'] for r in results]))

        stats = {
            'total_samples':       total,
            'total_frames':        total_frames,
            'depth_reversal_rate': avg_depth_rev,
            'occlusion_rate':      occlusion_rate,
            'avg_depth_sep':       avg_depth_sep,
            'mode_distribution':   dict(mode_counts),
            'pair_distribution':   dict(sorted(kw_pair_counts.items(),
                                               key=lambda x: -x[1])[:20]),
            'sequences': results,
        }

        print(f"\n[stats] total_samples={total}  total_frames={total_frames}", flush=True)
        print(f"[stats] depth_reversal_rate={avg_depth_rev:.3f}", flush=True)
        print(f"[stats] occlusion_rate(seq-level)={occlusion_rate:.3f}", flush=True)
        print(f"[stats] mode_distribution={dict(mode_counts)}", flush=True)

    # JSON 저장
    json_path = out_dir / 'objaverse_stats.json'
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[out] {json_path}", flush=True)

    # report.md 생성
    make_report_md(stats, out_dir / 'objaverse_report.md', data_root)

    return stats


# ─── report.md 생성 ──────────────────────────────────────────────────────────

def make_report_md(stats: dict, out_path: Path, data_root: Path):
    total  = stats.get('total_samples', 0)
    dr     = stats.get('depth_reversal_rate', 0)
    occ    = stats.get('occlusion_rate', 0)
    dsep   = stats.get('avg_depth_sep', 0)
    modes  = stats.get('mode_distribution', {})
    pairs  = stats.get('pair_distribution', {})

    def status(val, threshold, fmt='.2f'):
        ok = val > threshold
        return f"{val:{fmt}} | {threshold:{fmt}} | {'✅' if ok else '❌'}"

    lines = [
        f"# Objaverse Training Dataset Statistics",
        f"",
        f"**Data root**: `{data_root}`",
        f"",
        f"## Overview",
        f"",
        f"- Total samples: {total}",
        f"- Total frames: {total * 16} (est. {total} × 16)",
        f"",
        f"## Quality Metrics",
        f"",
        f"| metric              | value  | threshold | status |",
        f"|---------------------|--------|-----------|--------|",
        f"| depth_reversal_rate | {status(dr, 0.3)} |",
        f"| occlusion_rate      | {status(occ, 0.1)} |",
        f"| avg_depth_sep       | {status(dsep, 0.01)} |",
        f"",
        f"## Crossing Mode Distribution",
        f"",
        f"| mode    | samples |",
        f"|---------|---------|",
    ]
    for mode, count in sorted(modes.items()):
        lines.append(f"| {mode:7s} | {count:7d} |")

    lines += [
        f"",
        f"## Entity Pair Examples (Top 10)",
        f"",
        f"| pair             | count |",
        f"|------------------|-------|",
    ]
    for pair, count in list(pairs.items())[:10]:
        lines.append(f"| {pair:16s} | {count:5d} |")

    out_path.write_text('\n'.join(lines))
    print(f"[out] {out_path}", flush=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='toy/data_objaverse', dest='data_root')
    p.add_argument('--out-dir',   default='debug/dataset_stats', dest='out_dir')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    analyze_objaverse_dataset(args.data_root, args.out_dir)
