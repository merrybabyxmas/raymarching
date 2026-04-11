"""
Dataset Inspector — Streamlit 앱
toy/data_objaverse 내 각 샘플을 시각적으로 검사하고
텍스트 라벨과 실제 렌더링이 맞지 않는 bad asset을 식별.

실행:
  streamlit run tools/dataset_inspector.py
"""
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ─── 설정 ────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path(__file__).parent.parent / "toy" / "data_objaverse"
EXCLUDE_CFG = Path(__file__).parent.parent / "toy" / "excluded_assets.json"

st.set_page_config(
    page_title="Dataset Inspector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 상태 관리 ────────────────────────────────────────────────────────────────
if "excluded" not in st.session_state:
    # 이전 실행에서 저장된 exclusion 로드
    if EXCLUDE_CFG.exists():
        st.session_state.excluded = set(json.load(open(EXCLUDE_CFG)))
    else:
        st.session_state.excluded = set()

# ─── 데이터 로드 ──────────────────────────────────────────────────────────────
@st.cache_data
def load_samples(data_root: Path):
    samples = []
    for sample_dir in sorted(data_root.iterdir()):
        meta_path = sample_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        frames_dir = sample_dir / "frames"
        frame_files = sorted(frames_dir.glob("*.png")) if frames_dir.exists() else []
        samples.append({
            "name":     sample_dir.name,
            "path":     sample_dir,
            "meta":     meta,
            "frames":   frame_files,
            "mesh0":    Path(meta.get("mesh0_path", "")).stem,
            "mesh1":    Path(meta.get("mesh1_path", "")).stem,
        })
    return samples


def color_swatch(rgb_float, size=20):
    """(R,G,B) float → PIL Image swatch."""
    r, g, b = [int(c * 255) for c in rgb_float]
    img = Image.new("RGB", (size, size), (r, g, b))
    return img


samples = load_samples(DATA_ROOT)

# ─── 사이드바: 필터 + 통계 ────────────────────────────────────────────────────
st.sidebar.title("Dataset Inspector")
st.sidebar.metric("총 샘플", len(samples))
st.sidebar.metric("제외 표시됨", len(st.session_state.excluded))

# 키워드 필터
all_kw = sorted({s["meta"].get("keyword0", "") for s in samples}
                | {s["meta"].get("keyword1", "") for s in samples})
filter_kw = st.sidebar.multiselect("키워드 필터", all_kw, default=[])

# 모드 필터
all_modes = sorted({s["meta"].get("mode", "") for s in samples})
filter_mode = st.sidebar.multiselect("모드 필터", all_modes, default=[])

# bad-only 필터
show_excluded_only = st.sidebar.checkbox("제외 표시된 것만 보기", value=False)
show_good_only     = st.sidebar.checkbox("정상 샘플만 보기",    value=False)

st.sidebar.divider()

# 저장
if st.sidebar.button("💾 제외 목록 저장", type="primary"):
    EXCLUDE_CFG.parent.mkdir(parents=True, exist_ok=True)
    excluded_list = sorted(st.session_state.excluded)
    json.dump(excluded_list, open(EXCLUDE_CFG, "w"), indent=2)
    st.sidebar.success(f"저장 완료: {len(excluded_list)}개 asset 제외")
    st.sidebar.code("\n".join(excluded_list[:20]))

st.sidebar.divider()
if st.sidebar.button("모두 초기화"):
    st.session_state.excluded.clear()
    st.rerun()

# ─── 메인: 샘플 그리드 ───────────────────────────────────────────────────────
st.title("Toy Dataset Inspector")
st.caption("텍스트 라벨과 실제 렌더링이 맞는지 확인. 불일치 샘플을 체크하면 excluded_assets.json에 저장됩니다.")

# 필터 적용
filtered = samples
if filter_kw:
    filtered = [s for s in filtered
                if s["meta"].get("keyword0") in filter_kw
                or s["meta"].get("keyword1") in filter_kw]
if filter_mode:
    filtered = [s for s in filtered if s["meta"].get("mode") in filter_mode]
if show_excluded_only:
    filtered = [s for s in filtered
                if s["mesh0"] in st.session_state.excluded
                or s["mesh1"] in st.session_state.excluded]
if show_good_only:
    filtered = [s for s in filtered
                if s["mesh0"] not in st.session_state.excluded
                and s["mesh1"] not in st.session_state.excluded]

st.write(f"**{len(filtered)}개** 샘플 표시 중")

# 3열 그리드
COLS_PER_ROW = 3
rows = [filtered[i:i+COLS_PER_ROW] for i in range(0, len(filtered), COLS_PER_ROW)]

for row_samples in rows:
    cols = st.columns(COLS_PER_ROW)
    for col, sample in zip(cols, row_samples):
        meta    = sample["meta"]
        mesh0   = sample["mesh0"]
        mesh1   = sample["mesh1"]
        frames  = sample["frames"]

        kw0 = meta.get("keyword0", "?")
        kw1 = meta.get("keyword1", "?")
        c0  = meta.get("color0", [1, 0, 0])
        c1  = meta.get("color1", [0, 0, 1])

        is_excluded0 = mesh0 in st.session_state.excluded
        is_excluded1 = mesh1 in st.session_state.excluded
        is_any_excluded = is_excluded0 or is_excluded1

        with col:
            # 타이틀 (bad면 빨간 배경 강조)
            title_color = "🔴" if is_any_excluded else "🟢"
            st.markdown(f"**{title_color} {sample['name']}**")

            # 엔티티 색상 + 이름 표시
            kw0_hex = "#{:02x}{:02x}{:02x}".format(*[int(x*255) for x in c0])
            kw1_hex = "#{:02x}{:02x}{:02x}".format(*[int(x*255) for x in c1])
            st.markdown(
                f"<span style='background:{kw0_hex};color:white;padding:2px 6px;"
                f"border-radius:3px;font-weight:bold'>{kw0}</span> "
                f"<span style='background:{kw1_hex};color:white;padding:2px 6px;"
                f"border-radius:3px;font-weight:bold'>{kw1}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"mode: {meta.get('mode')} | cam: {meta.get('camera')}")

            # 첫 프레임 (프레임 슬라이더)
            if frames:
                n_f = len(frames)
                fi  = st.slider("frame", 0, n_f - 1, 0,
                                key=f"sl_{sample['name']}", label_visibility="collapsed")
                img = Image.open(frames[fi])
                # 테두리 색: bad면 빨강, good면 초록
                border_color = (200, 50, 50) if is_any_excluded else (50, 200, 50)
                bordered = Image.new("RGB",
                                     (img.width + 4, img.height + 4),
                                     border_color)
                bordered.paste(img, (2, 2))
                st.image(bordered, use_container_width=True)
            else:
                st.warning("프레임 없음")

            # Mesh ID 표시
            st.caption(f"mesh0 (RED): `{mesh0[:16]}...`")
            st.caption(f"mesh1 (BLU): `{mesh1[:16]}...`")

            # 체크박스로 bad 마킹
            bad0 = st.checkbox(
                f"❌ '{kw0}' mesh가 잘못됨",
                value=is_excluded0,
                key=f"bad0_{sample['name']}",
            )
            bad1 = st.checkbox(
                f"❌ '{kw1}' mesh가 잘못됨",
                value=is_excluded1,
                key=f"bad1_{sample['name']}",
            )

            # 상태 업데이트
            if bad0:
                st.session_state.excluded.add(mesh0)
            else:
                st.session_state.excluded.discard(mesh0)

            if bad1:
                st.session_state.excluded.add(mesh1)
            else:
                st.session_state.excluded.discard(mesh1)

            st.divider()

# ─── 하단: 제외 목록 요약 ────────────────────────────────────────────────────
if st.session_state.excluded:
    st.subheader("제외 표시된 Asset IDs")
    st.info(
        "아래 ID들을 `generate_crossing.py`의 `EXCLUDED_ASSET_IDS`에 추가하면\n"
        "데이터 재생성 시 제외됩니다."
    )
    # asset ID별로 어떤 키워드인지 역매핑
    id_to_kw = {}
    for s in samples:
        id_to_kw[s["mesh0"]] = s["meta"].get("keyword0", "?")
        id_to_kw[s["mesh1"]] = s["meta"].get("keyword1", "?")

    rows_out = []
    for asset_id in sorted(st.session_state.excluded):
        kw = id_to_kw.get(asset_id, "?")
        # 이 asset을 사용하는 sample 수
        n_affected = sum(
            1 for s in samples
            if s["mesh0"] == asset_id or s["mesh1"] == asset_id
        )
        rows_out.append({"asset_id": asset_id, "keyword": kw, "affected_samples": n_affected})

    import pandas as pd
    st.dataframe(pd.DataFrame(rows_out), use_container_width=True)

    st.code(
        "EXCLUDED_ASSET_IDS = {\n"
        + "".join(f"    '{aid}',  # {id_to_kw.get(aid, '?')}\n"
                  for aid in sorted(st.session_state.excluded))
        + "}",
        language="python",
    )
