"""Video browser — ba dạng truy vấn của AIC 2026, một giao diện.

    streamlit run src/mfe/browser/app.py

    docs/context/Thong-tin-vong-So-tuyen-AIC2026.pdf §2.1 (định dạng nộp)
    docs/context/buoi-2.md (vòng lặp truy hồi → hiển thị → phản hồi)

Vòng sơ tuyển có **đúng ba** dạng truy vấn, và chúng nộp ba hình dạng khác nhau:

    KIS    video_id, frame_id
    QA     video_id, frame_id, answer
    TRAKE  video_id, frame_id × N   (đúng thứ tự sự kiện)

Nên giao diện có ba tab, không phải một ô tìm kiếm dùng chung. Chúng khác nhau ở
chỗ sâu hơn ô nhập: TRAKE chạy `dante.align` chứ không xếp hạng keyframe, và QA
phải bày **bằng chứng văn bản** ra để người vận hành đọc câu trả lời.

Ba thứ chỉ số Final Score thưởng, và cả ba đều nằm trong giao diện này
----------------------------------------------------------------------
Final Score = trung bình R@k tại k ∈ {1, 5, 20, 50, 100} (PDF §2.2). Hệ quả:

1. **Thứ tự dòng có tính điểm.** Đáp án đúng ở dòng 1 ăn hơn hẳn cùng đáp án ở
   dòng 30. Giỏ đáp án giữ nguyên thứ tự xếp hạng và cho đẩy một dòng lên đầu.
2. **Nộp 100 dòng gần như luôn tốt hơn nộp 1.** R@100 chỉ tăng. Nút "lấy cả
   top-k vào giỏ" có mặt vì thế.
3. **Sai video là 0 điểm** ở TRAKE, ngay lập tức — nên tab đó xếp theo *video*
   chứ không theo keyframe.

Nút Submit **không** gửi đi đâu cả: nó ghi một file CSV đúng định dạng BTC. Tách
"xem" khỏi "nộp" là yêu cầu của `docs/context/buoi-2.md` §8.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mfe.contracts.tools import Candidate
from mfe.eval.metrics.kis import KISSubmission
from mfe.eval.metrics.qa import QASubmission
from mfe.eval.metrics.trake import TRAKESubmission
from mfe.eval.submission import to_csv_text
from mfe.online import answering, dante
from mfe.online.cascade import Cascade
from mfe.online.tools.get_neighbors import get_neighbors

logger = logging.getLogger(__name__)

try:
    import streamlit as st

    HAS_STREAMLIT = True
except ImportError:  # pragma: no cover - chỉ xảy ra ngoài môi trường browser
    HAS_STREAMLIT = False

#: PDF §2 — mỗi truy vấn tối đa 100 câu trả lời.
MAX_ANSWERS = 100


# ---------------------------------------------------------------------------
# Ảnh keyframe
# ---------------------------------------------------------------------------


def get_keyframe_image(keyframe_id: str, width: int = 320, height: int = 180) -> Any:
    """Ảnh keyframe từ đĩa, hoặc một ô xem trước sinh tại chỗ.

    Ô xem trước **không phải** ảnh thật và không được trông giống ảnh thật: nó in
    thẳng `keyframe_id` lên nền màu. Một chỗ trống trông như ảnh mờ sẽ khiến
    người vận hành tưởng mình đang nhìn khung hình.
    """
    # Tên file theo SỐ THỨ TỰ, `keyframe_id` theo FRAME GỐC — hai trục khác nhau,
    # và Relational Map là cầu nối duy nhất. Bản trước tách thẳng phần sau dấu `#`
    # ra làm tên file nên **không bao giờ tìm thấy ảnh nào**; vì có sẵn ô giữ chỗ
    # nên nó không báo lỗi, chỉ hiện một ô màu trên mọi thẻ.
    found = answering.keyframe_image_path(keyframe_id, st.session_state.get("relational"))
    if found is not None:
        return str(found)

    try:
        from PIL import Image, ImageDraw

        seed = sum(ord(c) for c in keyframe_id)
        image = Image.new(
            "RGB",
            (width, height),
            color=((seed * 37) % 120 + 30, (seed * 59) % 120 + 30, (seed * 83) % 120 + 30),
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle([4, 4, width - 5, height - 5], outline=(255, 255, 255), width=2)
        draw.text((14, 14), "KHONG CO ANH TREN DIA", fill=(255, 210, 90))
        draw.text((14, height // 2 - 6), keyframe_id, fill=(255, 255, 255))
        return image
    except Exception:  # pragma: no cover - PIL vắng mặt
        return None


# ---------------------------------------------------------------------------
# Giỏ đáp án — chung cho ba tab
# ---------------------------------------------------------------------------


def basket(kind: str) -> list:
    return st.session_state.setdefault(f"basket_{kind}", [])


def add_to_basket(kind: str, item) -> None:
    items = basket(kind)
    if item in items:
        return
    if len(items) >= MAX_ANSWERS:
        st.warning(f"Giỏ đã đủ {MAX_ANSWERS} đáp án — BTC không chấm quá số này.")
        return
    items.append(item)


def render_basket(kind: str, label: str) -> None:
    """Giỏ đáp án + xuất CSV. Thứ tự dòng ở đây **là** thứ tự nộp."""
    items = basket(kind)
    st.subheader(f"📋 Giỏ đáp án — {label} ({len(items)}/{MAX_ANSWERS})")
    if not items:
        st.info(
            "Chưa có đáp án nào. Final Score là trung bình R@k tại k = 1, 5, 20, 50, 100 "
            "— nộp nhiều dòng gần như luôn tốt hơn nộp một."
        )
        return

    for index, item in enumerate(items):
        cols = st.columns([6, 1, 1])
        with cols[0]:
            st.code(",".join(str(x) for x in _row_preview(item)), language=None)
        with cols[1]:
            if index > 0 and st.button("⬆️ lên đầu", key=f"top_{kind}_{index}"):
                items.insert(0, items.pop(index))
                st.rerun()
        with cols[2]:
            if st.button("🗑️", key=f"del_{kind}_{index}"):
                items.pop(index)
                st.rerun()

    csv_text = to_csv_text(items)
    st.download_button(
        "⬇️ Tải file nộp (.csv)",
        data=csv_text.encode("utf-8"),
        file_name=f"{st.session_state.get('query_id', 'query')}.csv",
        mime="text/csv",
        type="primary",
        key=f"dl_{kind}",
    )
    st.caption(
        "Không dòng tiêu đề, tối đa 100 dòng, **thứ tự dòng có tính điểm** — "
        "đáp án đúng ở dòng 1 ăn hơn hẳn cùng đáp án đó ở dòng 30 (PDF §2.2)."
    )
    if st.button("Xoá sạch giỏ", key=f"clear_{kind}"):
        st.session_state[f"basket_{kind}"] = []
        st.rerun()


def _row_preview(item) -> list:
    if isinstance(item, QASubmission):
        return [item.video_id, item.frame_id, item.answer]
    if isinstance(item, TRAKESubmission):
        return [item.video_id, *item.frame_ids]
    return [item.video_id, item.frame_id]


# ---------------------------------------------------------------------------
# Lưới ứng viên
# ---------------------------------------------------------------------------


def frame_id_of(candidate: Candidate) -> int | None:
    """`frame_idx` đọc từ Relational Map — **không suy ra** từ `keyframe_id`.

    Trả `None` khi không tra được, và người dùng phải thấy điều đó: nộp một
    `frame_id` đoán bừa là lỗi trắng (`docs/context/buoi-1.md` §8).
    """
    relational = st.session_state.get("relational")
    if relational is None:
        return None
    try:
        return relational.get_keyframe(candidate.keyframe_id).frame_idx
    except Exception:
        return None


def render_grid(candidates: list[Candidate], kind: str, on_pick) -> None:
    st.subheader(f"🔍 Ứng viên ({len(candidates)})")
    if not candidates:
        st.info("Không có kết quả. Thử nới truy vấn hoặc bật thêm arm ở thanh bên.")
        return

    per_row = 3
    for start in range(0, len(candidates), per_row):
        cols = st.columns(per_row)
        for col, candidate in zip(cols, candidates[start : start + per_row], strict=False):
            with col, st.container(border=True):
                frame_id = frame_id_of(candidate)
                st.markdown(f"**`{candidate.keyframe_id}`**")
                if frame_id is None:
                    st.error("không tra được frame_id — không nộp được")
                else:
                    st.caption(f"frame `{frame_id}` · điểm `{candidate.score:.4f}`")
                image = get_keyframe_image(candidate.keyframe_id)
                if image is not None:
                    st.image(image, use_container_width=True)
                pick_cols = st.columns(2)
                with pick_cols[0]:
                    if st.button("🎞️ lân cận", key=f"nb_{kind}_{candidate.keyframe_id}"):
                        st.session_state[f"timeline_{kind}"] = candidate.keyframe_id
                        st.rerun()
                with pick_cols[1]:
                    if frame_id is not None and st.button(
                        "➕ vào giỏ", key=f"add_{kind}_{candidate.keyframe_id}"
                    ):
                        on_pick(candidate, frame_id)
                        st.rerun()


def render_timeline(kind: str) -> None:
    keyframe_id = st.session_state.get(f"timeline_{kind}")
    if not keyframe_id:
        return
    st.subheader(f"🎞️ Lân cận quanh `{keyframe_id}`")
    known = [c.keyframe_id for c in st.session_state.get(f"results_{kind}", [])]
    result = get_neighbors(keyframe_id=keyframe_id, window=5, known_keyframe_ids=known)
    if not result.ok or not result.items:
        st.warning(f"Không dựng được timeline: {result.error}")
    else:
        cols = st.columns(min(len(result.items), 5))
        for index, row in enumerate(result.items):
            with cols[index % 5], st.container(border=True):
                st.caption(f"`{row.keyframe_id}`")
                image = get_keyframe_image(row.keyframe_id, 160, 90)
                if image is not None:
                    st.image(image, use_container_width=True)
    if st.button("Đóng timeline", key=f"close_tl_{kind}"):
        st.session_state[f"timeline_{kind}"] = None
        st.rerun()
    st.markdown("---")


def run_search(query_text: str, top_k: int, run_id: str, text_en: str | None = None):
    started = time.perf_counter()
    results = st.session_state["cascade"].search(
        query_text=query_text, top_k=top_k, run_id=run_id, query_text_en=text_en or None
    )
    elapsed = (time.perf_counter() - started) * 1000
    st.toast(f"{len(results)} ứng viên trong {elapsed:.0f}ms", icon="⚡")
    return results


# ---------------------------------------------------------------------------
# Tab 1 — Textual KIS
# ---------------------------------------------------------------------------


def tab_kis(top_k: int) -> None:
    st.caption(
        "PDF §2.1.1 — nộp `video_id, frame_id`. Đúng nếu khớp video **và** frame "
        "rơi vào khoảng đáp án."
    )
    query = st.text_area("Mô tả khoảnh khắc cần tìm", key="kis_q", height=80)
    text_en = st.text_input(
        "Bản tiếng Anh (tuỳ chọn)",
        key="kis_en",
        help="Cho arm có tháp văn bản không đọc được tiếng Việt (SigLIP SO400M). "
             "Bỏ trống thì những arm đó ngồi ngoài và ghi lý do vào log.",
    )
    if st.button("🔍 Tìm", type="primary", key="kis_go") and query.strip():
        st.session_state["results_kis"] = run_search(query, top_k, "ui_kis", text_en)
        st.session_state["timeline_kis"] = None

    results = st.session_state.get("results_kis", [])
    if results and st.button(f"➕ Lấy cả {len(results)} ứng viên vào giỏ", key="kis_all"):
        for candidate in results:
            frame_id = frame_id_of(candidate)
            if frame_id is not None:
                add_to_basket(
                    "kis", KISSubmission(video_id=candidate.video_id, frame_id=frame_id)
                )
        st.rerun()

    render_timeline("kis")
    render_grid(
        results,
        "kis",
        lambda c, f: add_to_basket("kis", KISSubmission(video_id=c.video_id, frame_id=f)),
    )
    st.markdown("---")
    render_basket("kis", "KIS")


# ---------------------------------------------------------------------------
# Tab 2 — Q&A
# ---------------------------------------------------------------------------


def tab_qa(top_k: int) -> None:
    st.caption(
        "PDF §2.1.2 — nộp `video_id, frame_id, answer`. Cả ba phải đúng: khớp "
        "video, frame trong khoảng, **và** câu trả lời khớp ngữ nghĩa."
    )
    question = st.text_area("Câu hỏi của BTC", key="qa_q", height=68)
    retrieval_text = st.text_input(
        "Câu truy hồi (tuỳ chọn)",
        key="qa_rt",
        help="Một câu HỎI không phải một câu MÔ TẢ, mà CLIP được huấn luyện trên "
             "caption. Viết lại thành mô tả cảnh thường kéo keyframe đúng lên "
             "hạng cao hơn hẳn — xem `QAQuery.retrieval_text`.",
    )
    if st.button("🔍 Tìm", type="primary", key="qa_go") and question.strip():
        st.session_state["results_qa"] = run_search(
            retrieval_text.strip() or question, top_k, "ui_qa"
        )
        st.session_state["timeline_qa"] = None

    results = st.session_state.get("results_qa", [])
    render_timeline("qa")

    # ---- MỘT câu trả lời cho cả truy vấn -------------------------------
    #
    # `R = I(v = GT ∧ id ∈ [s,e] ∧ a = GT)` (§2.1.2). `a` là thuộc tính của CÂU
    # HỎI, không phải của keyframe: một truy vấn có một đáp án đúng, và 100 dòng
    # là 100 phỏng đoán về CHỖ NÀO. Nên gõ một lần rồi nhân ra cả 100 dòng.
    #
    # Bản đầu của tab này để một ô trả lời trên MỖI ứng viên — tức bắt gõ 100 lần
    # cùng một câu. Sai hình dạng bài toán, không phải thiếu tính năng.
    st.markdown("### ✍️ Câu trả lời")
    if results and st.session_state.get("reader") is not None:
        if st.button("🤖 Gemini đề xuất đáp án", key="qa_ask"):
            _ask_reader(question, results)
            st.rerun()
        proposal = st.session_state.get("qa_proposal")
        if proposal is not None:
            st.success(
                f"**{proposal.text}** — đọc từ `{proposal.keyframe_id}` "
                f"({proposal.reader})"
            )
            st.caption(
                "Đề xuất, không phải đáp án. Kiểm lại bằng bằng chứng bên dưới rồi "
                "chép vào ô sau — bạn là người chịu trách nhiệm cho dòng được nộp."
            )
        elif st.session_state.get("qa_asked"):
            st.warning(
                "Gemini không đọc ra được từ những khung hình này. Đọc bằng chứng "
                "bên dưới và tự trả lời, hoặc nới truy vấn."
            )

    answer = st.text_input(
        "Đáp án cho truy vấn này (áp cho mọi dòng)",
        key="qa_answer",
        placeholder="vd: màu xanh · 5 · Nguyễn Văn A",
    )
    hedge = st.text_input(
        "Đáp án dự phòng (tuỳ chọn)",
        key="qa_hedge",
        help="Không chắc thì cược cả hai. Final Score lấy R-Score CAO NHẤT trong "
             "k dòng đầu, nên một đáp án sai xen giữa không trừ điểm — nó chỉ "
             "chiếm chỗ. Đáp án dự phòng được thêm SAU toàn bộ đáp án chính.",
    )

    if results and answer.strip():
        cols = st.columns(2)
        with cols[0]:
            if st.button(f"➕ Lấy cả {len(results)} ứng viên với đáp án này", key="qa_all"):
                _fill_basket(results, answer.strip())
                if hedge.strip():
                    _fill_basket(results, hedge.strip())
                st.rerun()
        with cols[1]:
            st.caption(
                f"Giỏ sẽ có tới {min(len(results) * (2 if hedge.strip() else 1), MAX_ANSWERS)} "
                "dòng — cùng đáp án, khác frame."
            )
    elif results:
        st.info("Gõ đáp án ở trên rồi mới đẩy ứng viên vào giỏ — Q&A cần đủ ba trường.")

    # ---- Bằng chứng để ĐỌC RA đáp án đó --------------------------------
    if results:
        st.markdown("### 🧾 Bằng chứng văn bản")
        st.caption(
            "Mọi văn bản đang giữ về từng keyframe. Đây là thứ đọc ra câu trả lời — "
            "hệ thống **không tự đoán**, vì một câu trả lời bịa ra trông y hệt một "
            "câu đúng cho tới lúc chấm điểm."
        )
        depth = st.slider("Xem bằng chứng của bao nhiêu ứng viên đầu", 1, 20, 5, key="qa_depth")
        cascade = st.session_state["cascade"]
        try:
            evidence = answering.gather_evidence(
                [c.keyframe_id for c in results[:depth]], cascade.text_store
            )
        except Exception as err:
            st.error(f"Không lấy được bằng chứng: {err}")
            evidence = []

        for item in evidence:
            with st.expander(f"`{item.keyframe_id}` — frame {item.frame_id}", expanded=True):
                cols = st.columns([1, 2])
                with cols[0]:
                    image = get_keyframe_image(item.keyframe_id, 260, 146)
                    if image is not None:
                        st.image(image, use_container_width=True)
                with cols[1]:
                    if not item.texts:
                        st.warning("Không có văn bản nào cho keyframe này.")
                    for source, text in item.ordered():
                        st.markdown(f"**{source}** — {text}")

    render_grid(results, "qa", lambda c, f: _add_qa(c, f, st.session_state.get("qa_answer", "")))
    st.markdown("---")
    render_basket("qa", "Q&A")


def _ask_reader(question: str, results: list[Candidate]) -> None:
    """Hỏi bộ đọc, cất đề xuất vào session. Không tự điền vào ô đáp án.

    Không điền là có chủ ý: một ô đã có sẵn chữ sẽ được bấm qua mà không đọc. Đề
    xuất hiện ở trên, người vận hành chép xuống — một thao tác, nhưng là một
    thao tác **có ý thức**.
    """
    cascade = st.session_state["cascade"]
    relational = st.session_state.get("relational")
    reader = st.session_state["reader"]
    top = results[: getattr(reader, "images", 3)]
    evidence = answering.gather_evidence(
        [c.keyframe_id for c in top], cascade.text_store
    )
    images = [
        path
        for path in (
            answering.keyframe_image_path(c.keyframe_id, relational) for c in top
        )
        if path is not None
    ]
    if not images:
        st.warning(
            "Không tìm thấy ảnh keyframe trên đĩa — Gemini chỉ đọc được phần văn bản."
        )
    st.session_state["qa_asked"] = True
    st.session_state["qa_proposal"] = answering.propose(
        question, evidence, reader=reader, image_paths=images
    )


def _add_qa(candidate: Candidate, frame_id: int, answer: str) -> None:
    if not answer.strip():
        st.warning("Chưa có đáp án — Q&A cần đủ ba trường.")
        return
    add_to_basket(
        "qa",
        QASubmission(
            video_id=candidate.video_id, frame_id=frame_id, answer=answer.strip()
        ),
    )


def _fill_basket(results: list[Candidate], answer: str) -> None:
    """Cùng một đáp án, mọi ứng viên — giữ nguyên thứ tự xếp hạng."""
    for candidate in results:
        frame_id = frame_id_of(candidate)
        if frame_id is not None:
            add_to_basket(
                "qa",
                QASubmission(
                    video_id=candidate.video_id, frame_id=frame_id, answer=answer
                ),
            )


# ---------------------------------------------------------------------------
# Tab 3 — TRAKE (DANTE)
# ---------------------------------------------------------------------------


def tab_trake(top_k: int, penalty: float) -> None:
    st.caption(
        "PDF §2.1.3 — nộp `video_id` rồi N frame **đúng thứ tự sự kiện**. "
        "Sai video là **0 điểm ngay**; đúng video thì điểm là tỉ lệ khoảnh khắc "
        "rơi đúng cửa sổ."
    )
    st.markdown(
        "Ghép bằng **DANTE** (arXiv:2512.13169 §4.3): quy hoạch động gán N sự kiện "
        "vào N vị trí tăng dần trong cùng một video, có phạt khoảng cách thời gian."
    )

    count = st.number_input("Số sự kiện", min_value=2, max_value=10, value=4, key="trake_n")
    events: list[str] = []
    for index in range(int(count)):
        events.append(
            st.text_input(f"Sự kiện {index + 1}", key=f"trake_e{index}", placeholder="…")
        )

    if st.button("🔍 Tìm & ghép", type="primary", key="trake_go"):
        filled = [text.strip() for text in events]
        if not all(filled):
            st.error("Phải điền đủ mọi sự kiện — DANTE gán N sự kiện, không bỏ trống được.")
        else:
            relational = st.session_state.get("relational")
            if relational is None:
                st.error("Chưa mở được Relational Map — không có `keyframe_index` để xếp thứ tự.")
            else:
                per_event = []
                for index, text in enumerate(filled):
                    hits = run_search(text, top_k, f"ui_trake_e{index}")
                    per_event.append(_to_event_hits(hits, relational))
                st.session_state["results_trake"] = dante.align(
                    per_event, penalty=penalty, limit=MAX_ANSWERS
                )

    alignments = st.session_state.get("results_trake", [])
    if alignments:
        st.subheader(f"🎯 {len(alignments)} video ghép được")
        for rank, item in enumerate(alignments[:20], 1):
            with st.container(border=True):
                head = st.columns([3, 1])
                with head[0]:
                    st.markdown(
                        f"**#{rank} · `{item.video_id}`** — điểm `{item.score:.4f}` · "
                        f"bằng chứng thật `{item.support}/{len(item.hits)}` sự kiện"
                    )
                    if item.support < len(item.hits):
                        st.caption(
                            "Sự kiện không có bằng chứng đi nhờ vị trí do sự kiện "
                            "khác tìm ra — kiểm bằng mắt trước khi nộp."
                        )
                with head[1]:
                    if st.button("➕ vào giỏ", key=f"tr_add_{item.video_id}"):
                        add_to_basket(
                            "trake",
                            TRAKESubmission(
                                video_id=item.video_id, frame_ids=item.frame_ids
                            ),
                        )
                        st.rerun()
                cols = st.columns(len(item.hits))
                for index, (col, keyframe) in enumerate(zip(cols, item.hits, strict=True)):
                    with col:
                        st.caption(f"SK{index + 1} · frame `{keyframe.frame_id}`")
                        image = get_keyframe_image(keyframe.keyframe_id, 200, 112)
                        if image is not None:
                            st.image(image, use_container_width=True)

        if st.button(f"➕ Lấy cả {min(len(alignments), MAX_ANSWERS)} video vào giỏ", key="tr_all"):
            for item in alignments[:MAX_ANSWERS]:
                add_to_basket(
                    "trake",
                    TRAKESubmission(video_id=item.video_id, frame_ids=item.frame_ids),
                )
            st.rerun()

    st.markdown("---")
    render_basket("trake", "TRAKE")


def _to_event_hits(candidates: list[Candidate], relational) -> list[dante.EventHit]:
    hits: list[dante.EventHit] = []
    for candidate in candidates:
        try:
            row = relational.get_keyframe(candidate.keyframe_id)
        except Exception:
            continue  # keyframe khong co trong map — bo, khong doan toa do
        hits.append(
            dante.EventHit(
                keyframe_id=candidate.keyframe_id,
                video_id=candidate.video_id,
                ordinal=row.keyframe_index,
                frame_id=row.frame_idx,
                score=float(candidate.score),
            )
        )
    return hits


# ---------------------------------------------------------------------------
# Vào chương trình
# ---------------------------------------------------------------------------


def main() -> None:
    if not HAS_STREAMLIT:
        print("Chưa cài Streamlit. Chạy: pip install -e '.[browser]'")
        return

    st.set_page_config(page_title="MFE — AIC 2026", page_icon="🎬", layout="wide")
    st.title("🎬 Multimodal Fusion Engine — AIC 2026")

    if "cascade" not in st.session_state:
        st.session_state["cascade"] = Cascade()
    if "reader" not in st.session_state:
        st.session_state["reader"] = None
    if "relational" not in st.session_state:
        relational, reason = _open_relational()
        st.session_state["relational"] = relational
        st.session_state["relational_error"] = reason

    cascade = st.session_state["cascade"]

    with st.sidebar:
        st.header("⚙️ Tuỳ chọn")
        st.text_input("Mã truy vấn (tên file nộp)", value="query-01", key="query_id")
        top_k = st.slider("Top-K", 10, MAX_ANSWERS, MAX_ANSWERS, 10)
        st.caption(
            "Mặc định 100 vì R@100 vào Final Score và nó chỉ tăng khi nộp thêm dòng."
        )
        penalty = st.select_slider(
            "λ — phạt khoảng cách (TRAKE)",
            options=[0.0, 0.001, 0.003, 0.01, 0.03],
            value=dante.PENALTY,
            help="Paper hiệu chỉnh 0,001 khi khoảng cách sự kiện là 3–15 keyframe, "
                 "0,01 khi 1–3. λ=0 là bỏ phạt, chỉ còn ràng buộc tăng dần.",
        )

        if st.session_state["relational"] is None:
            reason = st.session_state.get("relational_error", "?")
            st.error(
                "**Không mở được Relational Map.** Tab TRAKE không chạy được và "
                "lưới không hiện `frame_id`, nên không nộp được gì."
            )
            st.caption(f"Lý do: `{reason}`")
            st.caption(
                f"Config đang đọc: `{DEFAULT_CONFIG}` — đặt `MFE_CONFIG` để đổi."
            )

        st.markdown("**Bộ đọc Q&A**")
        use_reader = st.checkbox(
            "Gemini đề xuất đáp án", value=False, key="use_reader",
            help="BTC cho phép gọi API trong vòng sơ tuyển. Cần biến môi trường "
                 "GEMINI_API_KEY.",
        )
        if use_reader and st.session_state.get("reader") is None:
            from mfe.online.readers import GeminiAnswerReader

            st.session_state["reader"] = GeminiAnswerReader()
        elif not use_reader:
            st.session_state["reader"] = None
        reader = st.session_state.get("reader")
        if reader is not None and reader.calls:
            st.caption(f"đã gọi {reader.calls} lần · bỏ cuộc {reader.bails}")

        st.markdown("---")
        st.markdown("**Tầng truy hồi**")
        for arm in cascade.arms:
            if arm.available:
                cascade.arms.set_enabled(
                    arm.name,
                    st.checkbox(
                        arm.name, value=arm.enabled, help=arm.description,
                        key=f"arm_{arm.name}",
                    ),
                )
                cascade.arms.set_weight(
                    arm.name,
                    st.slider(
                        f"↳ trọng số {arm.name}", 0.0, 4.0, float(arm.weight), 0.5,
                        key=f"w_{arm.name}", label_visibility="collapsed",
                    ),
                )
            else:
                st.checkbox(arm.name, value=False, disabled=True, key=f"arm_{arm.name}")
                st.caption(f"↳ không chạy được: {arm.unavailable_reason}")

        st.markdown("---")
        st.caption(
            "Trọng số ở đây là để **thử**, không phải để chốt. Trọng số chốt đặt "
            "bằng leo dốc trên tập dev rồi công bố trên heldout."
        )

    kis, qa, trake = st.tabs(["🔎 Textual KIS", "❓ Q&A", "🎯 TRAKE"])
    with kis:
        tab_kis(top_k)
    with qa:
        tab_qa(top_k)
    with trake:
        tab_trake(top_k, penalty)


#: Config mặc định của giao diện. `MFE_CONFIG` ghi đè.
DEFAULT_CONFIG = "config/pilot.yaml"


def _open_relational() -> tuple[Any, str]:
    """`(RelationalMap, "")` hoặc `(None, lý do)`.

    Trả **lý do** ra ngoài chứ không chỉ ghi log, và đó là một lần sửa: bản đầu
    nuốt lỗi vào `except` rồi hiện một dòng "không mở được" chung chung. Lỗi thật
    là `load_config() missing 1 required positional argument` — một dòng sai
    tham số, mất mười giây để sửa **nếu đọc được nó**. Không đọc được thì nó là
    "tab TRAKE tự nhiên biến mất", và lưới thì im lặng không hiện `frame_id`.
    """
    import os

    path = os.environ.get("MFE_CONFIG", DEFAULT_CONFIG)
    try:
        from mfe.core.config import load_config
        from mfe.mapping.relational import RelationalMap

        config = load_config(path)
        relational = RelationalMap(config.paths.relational_db)
        relational.init_schema()
        return relational, ""
    except Exception as err:
        logger.exception("khong mo duoc Relational Map tu %s", path)
        return None, f"{type(err).__name__}: {err}"


if __name__ == "__main__":
    main()
