"""
Phase 17 Part A: animationCount > 0 Objaverse GLB 다운로드

Objaverse LVIS annotation에서 키워드 매칭 후
animationCount > 0인 GLB만 선택해서 다운로드.

출력:
  toy/assets_animated/
    {category}/{keyword}/
      manifest.json    ← {uid: {animation_count, name, glb_path}}
    download_report.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ANIMATED_CATEGORIES = {
    "animals": [
        "cat", "dog", "wolf", "horse",
        "bear", "tiger", "dragon", "dinosaur",
    ],
    "humans": [
        "person", "human", "fighter",
        "warrior", "ninja", "knight", "zombie",
    ],
    "creatures": [
        "monster", "creature", "alien",
    ],
}

QUICK_CATEGORIES = {
    "animals": ["cat", "wolf"],
    "humans":  ["person"],
}


def download_animated_assets(
    categories: dict,
    n_per_keyword: int = 10,
    out_dir: str = "toy/assets_animated",
    seed: int = 42,
) -> dict:
    """
    Objaverse LVIS에서 키워드로 검색 후
    animationCount > 0인 GLB만 선택해서 다운로드.
    """
    import objaverse

    lvis = objaverse.load_lvis_annotations()
    results = {}
    report = {"categories": {}, "total_animated": 0, "total_downloaded": 0}

    for category, keywords in categories.items():
        report["categories"][category] = {}
        for keyword in keywords:
            # LVIS에서 keyword 매칭 카테고리 찾기
            matched_uids = []
            for lvis_key, uids in lvis.items():
                if keyword.lower() in lvis_key.lower():
                    matched_uids.extend(uids)

            if not matched_uids:
                print(f"  [{keyword}] No LVIS match — skipping", flush=True)
                continue

            # annotation 로드해서 animationCount > 0 필터
            sample_uids = matched_uids[:50]  # 최대 50개 확인
            print(f"  [{keyword}] Checking {len(sample_uids)} UIDs for animations...",
                  flush=True)
            annotations = objaverse.load_annotations(sample_uids)

            animated_uids = [
                uid for uid, ann in annotations.items()
                if ann.get("animationCount", 0) > 0
            ]
            print(f"  [{keyword}] {len(animated_uids)} animated GLBs found", flush=True)

            if not animated_uids:
                print(f"  [{keyword}] WARNING: no animated GLBs — skipping", flush=True)
                report["categories"][category][keyword] = {"animated": 0, "downloaded": 0}
                continue

            # n_per_keyword개 선택
            rng = random.Random(seed)
            selected = rng.sample(animated_uids, min(n_per_keyword, len(animated_uids)))

            # 다운로드
            objects = objaverse.load_objects(selected)

            # 저장 + manifest
            kw_dir = Path(out_dir) / category / keyword
            kw_dir.mkdir(parents=True, exist_ok=True)
            manifest = {}

            for uid, glb_path in objects.items():
                ann = annotations.get(uid, {})
                manifest[uid] = {
                    "animation_count": ann.get("animationCount", 0),
                    "name":            ann.get("name", ""),
                    "glb_path":        str(glb_path),
                }
                print(f"    [{keyword}] {uid}: {ann.get('animationCount')} animations",
                      flush=True)

            with open(kw_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            results[keyword] = manifest
            report["categories"][category][keyword] = {
                "animated": len(animated_uids),
                "downloaded": len(manifest),
            }
            report["total_animated"]   += len(animated_uids)
            report["total_downloaded"] += len(manifest)

    # 전체 리포트 저장
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = out_path / "download_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] {report_path}", flush=True)
    print(f"[done] total_animated={report['total_animated']}  "
          f"total_downloaded={report['total_downloaded']}", flush=True)

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--categories", default="animals,humans", dest="categories")
    p.add_argument("--n-per-keyword", type=int, default=10, dest="n_per_keyword")
    p.add_argument("--out-dir", default="toy/assets_animated", dest="out_dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick-test", action="store_true", dest="quick_test")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick_test:
        cats = QUICK_CATEGORIES
        n = 3
    else:
        cat_keys = args.categories.split(",")
        cats = {k: v for k, v in ANIMATED_CATEGORIES.items() if k in cat_keys}
        n = args.n_per_keyword
    download_animated_assets(cats, n_per_keyword=n, out_dir=args.out_dir, seed=args.seed)
