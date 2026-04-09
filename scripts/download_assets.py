"""
Phase 15 Part A: Objaverse 3D 자산 다운로드 + 필터링

카테고리별로 GLB 파일을 다운로드하고 품질 필터링 후
toy/assets/ 아래 저장.

주의사항:
  - trimesh.load()가 Scene 반환 가능 → scene.dump()로 단일 mesh 추출
  - 다운로드 실패/파손 시 skip, download_report.json에 기록
  - 버텍스 50,000 초과 시 decimation으로 간소화
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── 카테고리 설정 (실제 LVIS annotation key 사용) ────────────────────────────

CATEGORIES = {
    "animals": {
        "keywords": ["cat", "dog", "wolf", "horse", "bear", "tiger", "lion", "snake"],
        "n_per_keyword": 5,
    },
    "humans": {
        "keywords": ["person", "fighter_jet"],   # LVIS에 warrior/ninja 없음
        "n_per_keyword": 5,
    },
    "objects": {
        "keywords": ["sword", "alligator"],       # chain/rope/robot 없음, 대체
        "n_per_keyword": 5,
    },
}

# quick-test용 축소 버전
QUICK_CATEGORIES = {
    "animals": {
        "keywords": ["cat", "dog"],
        "n_per_keyword": 2,
    },
    "objects": {
        "keywords": ["sword"],
        "n_per_keyword": 2,
    },
}


# ─── mesh 정규화 ──────────────────────────────────────────────────────────────

def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """centroid 원점 이동 + bounding box 최대 축 1.0으로 스케일."""
    mesh.apply_translation(-mesh.centroid)
    ext = mesh.bounding_box.extents
    scale = 1.0 / (ext.max() + 1e-6)
    mesh.apply_scale(scale)
    return mesh


def load_and_normalize(glb_path: Path) -> trimesh.Trimesh:
    """GLB 로드 → 단일 Trimesh 반환 (Scene인 경우 병합)."""
    result = trimesh.load(str(glb_path), force='mesh')
    if isinstance(result, trimesh.Scene):
        meshes = [m for m in result.dump() if hasattr(m, 'faces')]
        if not meshes:
            raise ValueError(f"Empty scene: {glb_path}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(result, trimesh.Trimesh):
        mesh = result
    else:
        raise ValueError(f"Unexpected type: {type(result)}")

    # 버텍스 과도 시 decimation (face_count 기준, 실패 시 원본 사용)
    if len(mesh.vertices) > 50_000:
        target_faces = min(10_000, len(mesh.faces) - 1)
        if target_faces > 0 and target_faces < len(mesh.faces):
            try:
                mesh = mesh.simplify_quadric_decimation(target_faces)
            except Exception:
                pass  # decimation 실패 시 원본 사용

    return normalize_mesh(mesh)


# ─── 다운로드 ─────────────────────────────────────────────────────────────────

def download_category(keyword: str, n: int, out_dir: Path,
                      rng: np.random.RandomState) -> dict:
    """
    LVIS annotation에서 keyword로 UID 검색 → n개 다운로드 + 필터링.
    manifest.json 저장 후 결과 dict 반환.
    """
    import objaverse

    anns = objaverse.load_lvis_annotations()

    # keyword와 일치하는 LVIS 카테고리 탐색 (완전 매칭 우선, 없으면 부분 매칭)
    exact = [k for k in anns.keys() if k.lower() == keyword.lower()]
    partial = [k for k in anns.keys() if keyword.lower() in k.lower() and k not in exact]
    matched_keys = exact + partial

    if not matched_keys:
        print(f"  [skip] '{keyword}' not found in LVIS", flush=True)
        return {'keyword': keyword, 'found': 0, 'downloaded': 0, 'failed': 0}

    # UID 수집
    all_uids = []
    for k in matched_keys:
        all_uids.extend(anns[k])
    all_uids = list(set(all_uids))

    # n개 랜덤 선택
    n_select = min(n, len(all_uids))
    selected = rng.choice(all_uids, n_select, replace=False).tolist()
    print(f"  [{keyword}] LVIS matches={matched_keys[:3]}  total_uids={len(all_uids)}  "
          f"selecting={n_select}", flush=True)

    # 다운로드
    out_dir.mkdir(parents=True, exist_ok=True)
    objects = objaverse.load_objects(uids=selected)  # {uid: local_path}

    manifest = {}
    n_ok = 0
    n_fail = 0

    for uid, glb_path_str in objects.items():
        if glb_path_str is None:
            n_fail += 1
            continue
        glb_src = Path(glb_path_str)
        if not glb_src.exists():
            n_fail += 1
            continue

        try:
            mesh = load_and_normalize(glb_src)
        except Exception as e:
            print(f"    [fail] {uid}: {e}", flush=True)
            n_fail += 1
            continue

        # 품질 필터
        if len(mesh.vertices) < 100:
            print(f"    [skip] {uid}: too few vertices ({len(mesh.vertices)})", flush=True)
            n_fail += 1
            continue
        ext = mesh.bounding_box.extents
        if ext.max() < 0.1:
            print(f"    [skip] {uid}: bbox too small ({ext.max():.4f})", flush=True)
            n_fail += 1
            continue

        # 저장
        dst = out_dir / f"{uid}.glb"
        mesh.export(str(dst))

        manifest[uid] = {
            'name': keyword,
            'vertex_count': len(mesh.vertices),
            'face_count': len(mesh.faces),
            'bbox_size': float(ext.max()),
        }
        n_ok += 1
        print(f"    [ok] {uid}  verts={len(mesh.vertices)}  bbox={ext.max():.3f}", flush=True)

    # manifest 저장
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"  [{keyword}] done: ok={n_ok}  fail={n_fail}", flush=True)
    return {'keyword': keyword, 'found': len(selected), 'downloaded': n_ok, 'failed': n_fail}


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main(args):
    rng = np.random.RandomState(args.seed)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.quick_test:
        cats = QUICK_CATEGORIES
        print("[quick-test] Using reduced category set", flush=True)
    else:
        cats = CATEGORIES

    # 카테고리 필터
    if args.categories:
        cat_names = [c.strip() for c in args.categories.split(',')]
        cats = {k: v for k, v in cats.items() if k in cat_names}

    report = {'categories': {}, 'total_downloaded': 0, 'total_failed': 0}

    for cat_name, cfg in cats.items():
        print(f"\n[category] {cat_name}", flush=True)
        cat_results = []
        for keyword in cfg['keywords']:
            kw_dir = out_root / cat_name / keyword
            res = download_category(
                keyword, cfg['n_per_keyword'], kw_dir, rng,
            )
            cat_results.append(res)
            report['total_downloaded'] += res['downloaded']
            report['total_failed']     += res['failed']
        report['categories'][cat_name] = cat_results

    # 저장
    report_path = out_root / 'download_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n[done] total_downloaded={report['total_downloaded']}  "
          f"total_failed={report['total_failed']}", flush=True)
    print(f"[done] report → {report_path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--categories', default='',
                   help='comma-separated: animals,humans,objects (default: all)')
    p.add_argument('--n-per-keyword', type=int, default=5, dest='n_per_keyword')
    p.add_argument('--out-dir',  default='toy/assets', dest='out_dir')
    p.add_argument('--seed',     type=int, default=42)
    p.add_argument('--quick-test', action='store_true', dest='quick_test',
                   help='Download 2 models per keyword for cats/dogs/swords only')
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
