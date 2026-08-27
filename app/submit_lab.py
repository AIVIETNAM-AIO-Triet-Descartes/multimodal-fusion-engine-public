"""Bàn nộp — tải cả gói đề lên, chạy một lượt, xếp tay, xuất CSV/ZIP.

    streamlit run app/submit_lab.py

Khác `app/rerank_lab.py` ở **vòng lặp**, không ở thuật toán. `rerank_lab` trả
lời "tầng này dời cái gì đi đâu" cho **một** truy vấn. Cái này là vòng lặp lúc
thi: nhận `query-*.txt` của BTC, chạy hết, rồi người vận hành duyệt từng câu,
kéo thả sửa thứ hạng, và xuất ra đúng những file BTC chấm.

Ba thứ ở đây không được phép sai, và cả ba đều hỏng IM LẶNG
------------------------------------------------------------
1. **Dạng đề đọc từ hậu tố tên file** (`-kis` / `-qa` / `-trake`), và tên file
   nộp là tên đề cộng `.csv`. Cả hai luật nằm ở `mfe.eval.query_package`, không
   viết lại ở đây — nhân bản là cách để hai đường phân kỳ rồi một hôm file QA ra
   thiếu cột đáp án mà vẫn nộp được.
2. **`frame_id` là frame của video gốc**, đọc từ `RelationalMap`, KHÔNG suy ra
   từ `keyframe_id`. `L21_V001#031124` là frame 31.124 và tình cờ là keyframe
   thứ 250 — hai số cách nhau 17 phút video.
3. **Thứ tự dòng có tính điểm.** Final Score = trung bình R@k tại
   k ∈ {1, 5, 20, 50, 100}, nên đáp án đúng ở dòng 1 ăn hơn hẳn cùng đáp án ở
   dòng 30. Đó là toàn bộ lý do có phần kéo thả.

Vì sao đi qua `build_flat_cascade` chứ không tự ghép arm
---------------------------------------------------------
`rerank_lab` tự gọi BM25 và tích vô hướng tay. Tiện cho một bàn thử, nhưng ở đây
sai một chút là nộp sai: giao diện này phải chạy **đúng** hệ mà
`scripts/measure_rerank.py` và `mfe.eval.official` đo. `build_flat_cascade` là
cùng một đường đó, kể cả trọng số, chuẩn hoá RRF và phép gom nhóm.

TRAKE đi đường riêng: nó không xếp hạng keyframe mà ghép **N khung tăng dần
trong cùng một video** qua `dante.align`, nên nó không có phần kéo thả — kéo một
dòng TRAKE lên đầu không có nghĩa gì nếu chuỗi bên trong nó sai.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfe.contracts.errors import ContractError  # noqa: E402
from mfe.contracts.tools import Candidate  # noqa: E402
from mfe.core.config import load_config  # noqa: E402
from mfe.eval.metrics.final_score import MAX_ANSWERS_PER_QUERY  # noqa: E402
from mfe.eval.query_package import PackagedQuery, parse_query  # noqa: E402
from mfe.eval.submission import to_csv_text  # noqa: E402
from mfe.eval.submission_bundle import (  # noqa: E402
    apply_manual_order,
    build_zip,
    run_package,
    submissions_for,
)
from mfe.mapping.relational import RelationalMap  # noqa: E402
from mfe.offline.transport import load_dotenv  # noqa: E402
from mfe.online import dante  # noqa: E402
from mfe.online.flat_arms import build_flat_cascade, load_flat_bundle  # noqa: E402
from mfe.online.keyframe_images import KeyframeImages  # noqa: E402

CONFIG = ROOT / "config" / "btc_b1.yaml"
FLAT_ROOT = ROOT / "stores" / "flat"

st.set_page_config(page_title="Bàn nộp — AIC 2026", page_icon="📦", layout="wide")

# Streamlit khong nap `.env`; thieu dong nay thi tang dich va cac tang goi API
# deu bao "chua co API_KEY" trong khi key nam san trong file.
for _key, _value in load_dotenv(ROOT / ".env").items():
    os.environ.setdefault(_key, _value)

try:
    from streamlit_sortables import sort_items

    HAS_SORTABLES = True
except ImportError:  # pragma: no cover - phu thuoc tuy chon
    HAS_SORTABLES = False


# ---------------------------------------------------------------------------
# Nạp một lần, giữ suốt phiên
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Nạp kho phẳng + tháp text SigLIP …")
def load_engine():
    """`(config, cascade, relational)`. Nạp ~10s, nên đúng một lần mỗi phiên."""
    config = load_config(str(CONFIG))
    bundle = load_flat_bundle(str(FLAT_ROOT))
    cascade = build_flat_cascade(config, str(FLAT_ROOT), flat_bundle=bundle)
    relational = RelationalMap(config.paths.relational_db)
    return config, cascade, relational


@st.cache_resource
def load_images() -> KeyframeImages:
    return KeyframeImages()


# ---------------------------------------------------------------------------
# Chạy cả gói
# ---------------------------------------------------------------------------


def run_all(queries: list[PackagedQuery], pool_k: int, penalty: float) -> dict[str, list]:
    """`run_package` của thư viện, nối với thanh tiến trình của Streamlit."""
    _, cascade, relational = load_engine()
    bar = st.progress(0.0, text="Bắt đầu …")

    def tick(index: int, query: PackagedQuery) -> None:
        bar.progress(
            (index - 1) / len(queries),
            text=f"{query.query_id} ({index}/{len(queries)})",
        )

    out = run_package(
        cascade, relational, queries,
        pool_k=pool_k, penalty=penalty, on_progress=tick,
    )
    bar.progress(1.0, text=f"Xong {len(queries)} câu")
    return out


# ---------------------------------------------------------------------------
# Thứ tự do người xếp
# ---------------------------------------------------------------------------


def manual_order(query_id: str) -> list[str] | None:
    """Thứ tự người vận hành đã kéo thả, lấy khỏi `session_state`.

    Lưu bằng `keyframe_id` chứ không bằng chỉ số — lý do nằm ở
    `submission_bundle.apply_manual_order`, cùng chỗ áp dụng nó.
    """
    return st.session_state.get(f"order_{query_id}")


def ordered(query_id: str, candidates: list[Candidate]) -> list[Candidate]:
    return apply_manual_order(candidates, manual_order(query_id))


def rows_for(query: PackagedQuery, result, relational, top_k: int):
    return submissions_for(
        query, result, relational, limit=top_k,
        manual_order=manual_order(query.query_id),
        answer=st.session_state.get(f"answer_{query.query_id}", ""),
    )


# ---------------------------------------------------------------------------
# Thanh bên
# ---------------------------------------------------------------------------

st.sidebar.header("📥 Gói đề")
uploads = st.sidebar.file_uploader(
    "Thả các file `query-*.txt` vào đây",
    type=["txt"], accept_multiple_files=True,
    help="Dạng đề đọc từ hậu tố tên file: `-kis`, `-qa`, `-trake`. "
         "Tên file nộp = tên đề + `.csv`, nên ĐỪNG đổi tên file.",
)

st.sidebar.header("⚙ Tuỳ chọn")
top_k = st.sidebar.slider(
    "Số dòng nộp mỗi câu", 10, MAX_ANSWERS_PER_QUERY, MAX_ANSWERS_PER_QUERY, 10,
    help="R@100 vào Final Score và nó chỉ tăng khi nộp thêm dòng — "
         "hạ xuống chỉ khi có lý do.",
)
pool_k = st.sidebar.slider(
    "Độ sâu pool để xem", 50, 500, 200, 50,
    help="Số ứng viên truy hồi để duyệt. Nộp vẫn chỉ lấy `top_k` dòng đầu.",
)
grid_cols = st.sidebar.slider("Số cột lưới ảnh", 2, 8, 5)
show_images = st.sidebar.checkbox("Hiện ảnh (tải từ Modal)", value=True)

penalty = st.sidebar.select_slider(
    "λ — phạt khoảng cách (TRAKE)",
    options=[0.0, 0.001, 0.003, 0.01, 0.03], value=dante.PENALTY,
)

if not HAS_SORTABLES:
    st.sidebar.warning(
        "Chưa có `streamlit-sortables` nên phần kéo thả thay bằng ô nhập thứ tự.\n\n"
        "`pip install --no-deps streamlit-sortables`"
    )

# ---------------------------------------------------------------------------
# Thân
# ---------------------------------------------------------------------------

st.title("📦 Bàn nộp — cả gói đề, một lượt")

if not uploads:
    st.info(
        "Thả các file `query-*.txt` vào thanh bên. Đợt đánh giá 1 phải chép file "
        "vào codebase rồi mới chạy được; ở đây tải thẳng lên."
    )
    st.stop()

parsed: list[PackagedQuery] = []
errors: list[str] = []
for upload in uploads:
    try:
        parsed.append(parse_query(upload.name, upload.getvalue().decode("utf-8")))
    except Exception as err:  # noqa: BLE001 - hien loi cho nguoi dung, khong nuot
        errors.append(f"`{upload.name}` — {type(err).__name__}: {err}")

for message in errors:
    st.error(message)
if not parsed:
    st.stop()

# Sắp theo SỐ trong tên, không theo chuỗi: sắp chuỗi đặt `query-p1-10` trước
# `query-p1-2`, và thứ tự đó là thứ tự người vận hành đối chiếu với đề trên màn
# hình. Không đổi điểm, đổi khả năng phát hiện mình đang nhìn nhầm câu.
parsed.sort(key=lambda q: tuple(
    int(part) if part.isdigit() else part
    for part in re.split(r"(\d+)", q.query_id)
))

counts = {t: sum(1 for q in parsed if q.type == t) for t in ("KIS", "QA", "TRAKE")}
st.success(
    f"Đọc được **{len(parsed)}** đề — "
    + " · ".join(f"{t} {n}" for t, n in counts.items() if n)
)

if st.button("▶️ Chạy cả gói", type="primary"):
    st.session_state["results"] = run_all(parsed, pool_k, penalty)
    st.session_state["ran_ids"] = [q.query_id for q in parsed]

results = st.session_state.get("results")
if not results:
    st.stop()

_, _, relational = load_engine()

# --- chọn câu để xem -------------------------------------------------------
labels = {
    q.query_id: f"{q.query_id}  [{q.type}]  — {len(results.get(q.query_id, []))} ứng viên"
    for q in parsed if q.query_id in results
}
picked_id = st.selectbox(
    "Xem câu nào", list(labels), format_func=lambda k: labels[k],
)
query = next(q for q in parsed if q.query_id == picked_id)
result = results[picked_id]

st.markdown(f"**Đề `{query.query_id}` — {query.type}**")
st.caption(query.text)
if query.type == "TRAKE":
    for index, event in enumerate(query.events, start=1):
        st.caption(f"  E{index}: {event}")

# --- tải trước ảnh của câu đang xem ----------------------------------------
images = load_images()
if show_images and query.type != "TRAKE":
    with st.spinner(f"Tải {min(len(result), pool_k)} ảnh …"):
        images.prefetch([c.keyframe_id for c in result[:pool_k]])

# --- QA cần một đáp án ------------------------------------------------------
if query.type == "QA":
    st.text_input(
        "Đáp án (cột thứ ba của mọi dòng)", key=f"answer_{picked_id}",
        help="Một đáp án cho cả câu. Dấu phẩy trong đáp án được `csv` trích dẫn "
             "đúng — không tự nối chuỗi bằng dấu phẩy.",
    )
    if not st.session_state.get(f"answer_{picked_id}", "").strip():
        st.warning("Chưa nhập đáp án — file QA xuất ra sẽ có cột thứ ba rỗng.")

# --- xếp tay ----------------------------------------------------------------
if query.type == "TRAKE":
    st.info(
        "TRAKE không có phần xếp tay: mỗi dòng là **một video kèm N khung tăng "
        "dần**, do `dante.align` ghép. Kéo một dòng lên đầu không sửa được chuỗi "
        "bên trong nó."
    )
    for position, item in enumerate(result[:20], start=1):
        st.code(
            f"{position}. {item.video_id}," + ",".join(str(f) for f in item.frame_ids),
            language=None,
        )
else:
    ranked = ordered(picked_id, result)
    depth = st.number_input(
        "Xếp tay bao nhiêu dòng đầu", 5, min(100, len(ranked)),
        min(20, len(ranked)), 5,
        help="Chỉ phần đầu đáng xếp: R@k thưởng nặng nhất ở k nhỏ.",
    )
    head = ranked[: int(depth)]

    st.subheader(f"✋ Xếp tay — {int(depth)} dòng đầu")
    if HAS_SORTABLES:
        new_order = sort_items(
            [c.keyframe_id for c in head],
            direction="vertical", key=f"sort_{picked_id}_{depth}",
        )
        if new_order and new_order != [c.keyframe_id for c in head]:
            st.session_state[f"order_{picked_id}"] = new_order + [
                c.keyframe_id for c in ranked[int(depth):]
            ]
            st.rerun()
    else:
        typed = st.text_area(
            "Thứ tự mong muốn — mỗi dòng một `keyframe_id`",
            value="\n".join(c.keyframe_id for c in head), height=240,
            key=f"typed_{picked_id}",
        )
        if st.button("Áp thứ tự này", key=f"apply_{picked_id}"):
            st.session_state[f"order_{picked_id}"] = [
                line.strip() for line in typed.splitlines() if line.strip()
            ] + [c.keyframe_id for c in ranked[int(depth):]]
            st.rerun()

    if st.session_state.get(f"order_{picked_id}") and st.button(
        "↩️ Trả về thứ tự máy", key=f"reset_{picked_id}"
    ):
        st.session_state.pop(f"order_{picked_id}")
        st.rerun()

    # --- lưới ảnh -----------------------------------------------------------
    st.subheader(f"🖼 Keyframe hệ thống truy được ({len(ranked)})")
    machine_rank = {c.keyframe_id: i for i, c in enumerate(result, start=1)}
    columns = st.columns(grid_cols)
    for position, candidate in enumerate(ranked[: int(pool_k)], start=1):
        with columns[(position - 1) % grid_cols]:
            if show_images:
                path = images.get(candidate.keyframe_id)
                if path:
                    st.image(str(path), use_container_width=True)
            was = machine_rank.get(candidate.keyframe_id)
            moved = f"  ✋ (máy xếp {was})" if was and was != position else ""
            try:
                frame_id = relational.get_keyframe(candidate.keyframe_id).frame_idx
                frame_text = f"frame `{frame_id}`"
            except Exception:  # noqa: BLE001
                frame_text = ":red[không tra được frame_id]"
            st.markdown(
                f"**{position}.** `{candidate.keyframe_id}`{moved}  \n{frame_text}"
            )

# --- xuất -------------------------------------------------------------------
st.markdown("---")
st.subheader("⬇️ Xuất file nộp")

ran = [q for q in parsed if q.query_id in results]
bundle = build_zip(
    ran, results, relational, limit=top_k,
    manual_orders={
        q.query_id: order for q in ran if (order := manual_order(q.query_id))
    },
    answers={
        q.query_id: st.session_state.get(f"answer_{q.query_id}", "") for q in ran
    },
)

left, right = st.columns(2)
with left:
    try:
        subs = rows_for(query, result, relational, top_k)
    except ContractError as err:
        st.error(str(err))
    else:
        st.download_button(
            f"Câu này — `{query.csv_name}`",
            data=to_csv_text(subs, limit=top_k).encode("utf-8"),
            file_name=query.csv_name, mime="text/csv", type="primary",
        )
        st.caption(f"{len(subs)} dòng")
with right:
    st.download_button(
        f"Cả gói — {len(ran) - len(bundle.skipped)}/{len(ran)} file trong một ZIP",
        data=bundle.payload, file_name="submissions.zip", mime="application/zip",
    )

if bundle.skipped:
    # Một ZIP thiếu file trông y hệt một ZIP đủ file cho tới lúc đếm, và lúc
    # đếm thì đã nộp rồi.
    st.warning(f"**{len(bundle.skipped)} câu KHÔNG có file trong ZIP:**")
    for query_id, reason in bundle.skipped.items():
        st.caption(f"· `{query_id}` — {reason}")

st.caption(
    "Không dòng tiêu đề · tối đa 100 dòng · **thứ tự dòng có tính điểm** "
    "(Final Score = trung bình R@k tại k = 1, 5, 20, 50, 100). "
    "`frame_id` đọc từ Relational Map, không suy ra từ `keyframe_id`."
)
