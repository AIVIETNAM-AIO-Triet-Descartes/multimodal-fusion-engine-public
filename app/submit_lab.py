"""Bàn nộp — tải cả gói đề lên, chạy, duyệt theo dạng, xếp tay, xuất CSV/ZIP.

    streamlit run app/submit_lab.py

Khác `app/rerank_lab.py` ở **vòng lặp**, không ở thuật toán. `rerank_lab` trả
lời "tầng này dời cái gì đi đâu" cho **một** truy vấn. Cái này là vòng lặp lúc
thi: nhận `query-*.txt` của BTC, chạy hết, rồi người vận hành duyệt từng câu,
kéo thả sửa thứ hạng, và xuất ra đúng những file BTC chấm.

Ba tab, vì ba dạng đề nộp ba hình dạng khác nhau
------------------------------------------------
    KIS    video_id, frame_id
    QA     video_id, frame_id, answer
    TRAKE  video_id, frame_id × N   (đúng thứ tự sự kiện)

Dạng đọc từ **hậu tố tên file** (`-kis` / `-qa` / `-trake`), nên tải một gói lên
là câu tự về đúng tab. Không có ô "chọn dạng" nào cả — đoán dạng là chỗ sai
lặng lẽ nhất: mặc định về KIS sẽ ghi một file QA thiếu hẳn cột đáp án, và file
đó **nộp được**.

Ba thứ ở đây không được phép sai, và cả ba đều hỏng IM LẶNG
------------------------------------------------------------
1. **Dạng đề và tên file nộp** — cả hai luật nằm ở `mfe.eval.query_package`,
   không viết lại ở đây.
2. **`frame_id` là frame của video gốc**, đọc từ `RelationalMap`, KHÔNG suy ra
   từ `keyframe_id`. `L21_V001#031124` là frame 31.124 và tình cờ là keyframe
   thứ 250 — hai số cách nhau 17 phút video.
3. **Thứ tự dòng có tính điểm.** Final Score = trung bình R@k tại
   k ∈ {1, 5, 20, 50, 100}. Đó là toàn bộ lý do có phần kéo thả.

Vì sao đi qua `build_flat_cascade` chứ không tự ghép arm
---------------------------------------------------------
`rerank_lab` tự gọi BM25 và tích vô hướng tay. Tiện cho một bàn thử, nhưng ở đây
sai một chút là nộp sai: giao diện này phải chạy **đúng** hệ mà
`scripts/measure_rerank.py` và `mfe.eval.official` đo.

Bộ tuỳ chọn có PHẠM VI
-----------------------
Bật/tắt arm và tầng rerank áp được cho **một câu**, **một dạng**, hoặc **cả
gói**, theo thứ tự ưu tiên câu > dạng > gói. Có phạm vi vì các dạng cần thứ khác
nhau: QA sống nhờ OCR/phụ đề, KIS nhiều cảnh cần thứ tự cảnh, còn TRAKE thì
`vlm` vô nghĩa vì mỗi dòng là một chuỗi chứ không phải một khung.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfe.browser.sortable_grid import sortable_grid  # noqa: E402
from mfe.contracts.errors import ContractError  # noqa: E402
from mfe.contracts.tools import Candidate  # noqa: E402
from mfe.core.config import load_config  # noqa: E402
from mfe.eval.metrics.final_score import MAX_ANSWERS_PER_QUERY  # noqa: E402
from mfe.eval.query_package import (  # noqa: E402
    ADHOC_PREFIX,
    PackagedQuery,
    is_adhoc,
    load_package,
    parse_query,
)
from mfe.eval.submission import to_csv_text  # noqa: E402
from mfe.eval.submission_bundle import (  # noqa: E402
    SearchText,
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
from mfe.online.query_options import (  # noqa: E402
    QUERY_TYPES,
    Options,
    resolve,
)
from mfe.online.query_text import has_description, strip_questions  # noqa: E402
from mfe.online.rerank_tiers import DEFAULT_DEPTH  # noqa: E402
from mfe.online.scene_order import MIN_CLAUSES_FOR_PRIOR, split_clauses  # noqa: E402
from mfe.online.vqa_answer import DEFAULT_DEPTH as VQA_DEPTH  # noqa: E402
from mfe.online.vqa_answer import available as vqa_available  # noqa: E402
from mfe.online.vqa_answer import read_answer  # noqa: E402

CONFIG = ROOT / "config" / "btc_b1.yaml"
FLAT_ROOT = ROOT / "stores" / "flat"
TYPES = QUERY_TYPES
TAB_LABEL = {"KIS": "🔎 KIS", "QA": "❓ VQA", "TRAKE": "🎯 TRAKE"}

# `initial_sidebar_state="expanded"`: Streamlit tự thu thanh bên tuỳ bề rộng cửa
# sổ, mà ô tải gói đề nằm trong đó — thu lại là giao diện trông như không có chỗ
# nhập gì cả. Đã xảy ra thật.
st.set_page_config(
    page_title="Bàn nộp — AIC 2026", page_icon="📦", layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit khong nap `.env`; thieu dong nay thi tang dich va cac tang goi API
# deu bao "chua co API_KEY" trong khi key nam san trong file.
for _key, _value in load_dotenv(ROOT / ".env").items():
    os.environ.setdefault(_key, _value)


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
# Bộ tuỳ chọn và phạm vi áp
# ---------------------------------------------------------------------------


def options_for(query: PackagedQuery) -> Options:
    """Bộ tuỳ chọn hiệu lực cho một câu — phép xếp nằm ở `query_options.resolve`."""
    return resolve(
        st.session_state.setdefault("options", {}), query.query_id, query.type
    )


def apply_options(cascade, options: Options):
    """Đặt arm rồi trả về context ghi đè config. Dùng trong `with`.

    Arm đặt **ngoài** context vì `ArmRegistry` không có ngăn xếp — nó là trạng
    thái của cascade. Đó là chấp nhận được ở đây vì mọi lượt chạy đều đặt lại
    đủ bộ arm trước khi tìm, nên không có lượt nào thừa hưởng lựa chọn của lượt
    trước.
    """
    for name, enabled in options.arms.items():
        if name in cascade.arms:
            cascade.arms.set_enabled(name, enabled)
    return cascade.overriding(
        rerank_tiers=None if options.tiers is None else list(options.tiers),
        scene_order_weight=options.scene_order,
    )


def render_options(scope_key: str, defaults, cascade) -> None:
    """Bảng bật/tắt cho một phạm vi. `scope_key` là khoá trong `session_state`."""
    store = st.session_state.setdefault("options", {})
    current = store.get(scope_key, Options())

    st.caption("**Arm truy hồi**")
    arms: dict[str, bool] = {}
    for arm in cascade.arms:
        if not arm.available:
            st.checkbox(
                f"{arm.name} — không chạy được", value=False, disabled=True,
                key=f"{scope_key}_arm_{arm.name}",
            )
            st.caption(f"↳ {arm.unavailable_reason}")
            continue
        arms[arm.name] = st.checkbox(
            arm.name, value=current.arms.get(arm.name, arm.enabled),
            help=arm.description, key=f"{scope_key}_arm_{arm.name}",
        )

    st.caption("**Tầng rerank** — đo trên 163 câu dev, nền 0,515")
    base_tiers = current.tiers if current.tiers is not None else list(defaults.rerank_tiers)
    tiers = st.multiselect(
        "Tầng", list(DEFAULT_DEPTH), default=base_tiers,
        format_func=lambda n: {
            "vlm": "vlm  +0,015  (27,6 s/câu)",
            "ta": "ta  −0,010",
            "hf_cross": "hf_cross  −0,029",
            "colpali": "colpali  −0,033  (36,6 s/câu)",
        }.get(n, n),
        key=f"{scope_key}_tiers", label_visibility="collapsed",
        help="Chỉ `vlm` có bằng chứng dương. Ba tầng kia đo ra ÂM — bật chúng "
             "là đi ngược số đo, không phải thêm một lớp an toàn.",
    )

    scene = st.slider(
        "Thứ tự cảnh (KIS ≥3 mệnh đề)", 0.0, 1.0,
        float(current.scene_order if current.scene_order is not None
              else defaults.scene_order_weight),
        0.1, key=f"{scope_key}_scene",
    )
    store[scope_key] = Options(
        arms=arms, tiers=tuple(tiers), scene_order=scene
    )


# ---------------------------------------------------------------------------
# Chạy
# ---------------------------------------------------------------------------


def run_queries(queries: list[PackagedQuery], pool_k: int, penalty: float) -> None:
    """Chạy một nhóm câu, mỗi câu với bộ tuỳ chọn hiệu lực của **chính nó**.

    Chạy từng câu chứ không gọi `run_package` một lượt: bộ tuỳ chọn khác nhau
    theo câu, nên một lượt gọi chung sẽ áp nhầm cấu hình cho phần lớn số câu.
    """
    _, cascade, relational = load_engine()
    results = st.session_state.setdefault("results", {})
    bar = st.progress(0.0, text="Bắt đầu …")
    for index, query in enumerate(queries, start=1):
        bar.progress(
            (index - 1) / len(queries), text=f"{query.query_id} ({index}/{len(queries)})"
        )
        override = manual_search(query.query_id)
        with apply_options(cascade, options_for(query)):
            results.update(run_package(
                cascade, relational, [query], pool_k=pool_k, penalty=penalty,
                clauses={query.query_id: manual_clauses(query.query_id)}
                if manual_clauses(query.query_id) else None,
                searches={query.query_id: override} if override else None,
            ))
    bar.progress(1.0, text=f"Xong {len(queries)} câu")


# ---------------------------------------------------------------------------
# Chuỗi tìm do người ghi đè — cần gạt thứ hai, cạnh ô mệnh đề
# ---------------------------------------------------------------------------


def manual_search(query_id: str) -> SearchText | None:
    return st.session_state.get(f"search_{query_id}")


def render_search_editor(query: PackagedQuery, pool_k: int, penalty: float) -> None:
    """Ô ghi đè chuỗi đi tìm + nút tìm lại, **ghi đè tại chỗ** câu đang mở.

    Khác thanh "tìm riêng" ở đầu tab: thanh đó **đẻ ra** một câu mới
    (`tim-tay-N-*`) không vào ZIP; ô này **thay kết quả của chính câu đề**, nên
    `query_id` và tên file nộp không đổi.

    Vì sao đây là cần gạt chứ không phải tiện ích: xem `SearchText`. Tóm tắt —
    pool 500 phủ 40/47 câu QA dev khi tìm bằng `retrieval_text` viết tay, so với
    **32/47** khi tìm bằng nguyên văn câu đề, và **10 câu rơi hẳn khỏi pool**.
    """
    override = manual_search(query.query_id)
    cut = strip_questions(query.text)
    vi_key = f"search_vi_{query.query_id}"
    en_key = f"search_en_{query.query_id}"

    # Đổ chữ vào ô TRƯỚC khi ô được dựng. Streamlit ném `StreamlitAPIException`
    # khi ghi `session_state[k]` sau lúc widget khoá `k` đã dựng trong cùng lượt
    # chạy, mà mọi nút ở đây đều nằm dưới hai ô. Nên nút chỉ **hẹn** qua khoá
    # `*_fill_*`, còn việc đổ nằm ở đầu lượt kế tiếp — sau `st.rerun()`.
    for key, pending_key in ((vi_key, "vi"), (en_key, "en")):
        pending = st.session_state.pop(f"search_fill_{pending_key}_{query.query_id}", None)
        if pending is not None:
            st.session_state[key] = pending
    st.session_state.setdefault(vi_key, override.vi if override else query.text)
    st.session_state.setdefault(en_key, (override.en if override else None) or "")

    state = ":green[đang ghi đè]" if override else "đang dùng nguyên văn câu đề"
    with st.expander(f"🔍 Chuỗi tìm — {state}"):
        _search_help(query)
        # KHÔNG truyền `value=`: khi đã có `key` thì `session_state` là nguồn duy
        # nhất, và đưa cả hai vào là hai nguồn sự thật cho cùng một ô.
        vi = st.text_area(
            "Chuỗi tìm (tiếng Việt)", height=90, key=vi_key,
            help="Cái hệ đi TÌM bằng. Không đổi câu đề, không đổi tên file nộp.",
        )
        en = st.text_input(
            "Tiếng Anh — đè lên bản dịch máy (để trống thì envit5 tự dịch)",
            key=en_key,
            help="Đi THẲNG vào tháp text SigLIP. envit5 dịch `xe ô tô lội nước` "
                 "thành `the trucks wading` và làm hạng tụt từ 71 xuống 540.",
        )

        buttons = st.columns([2, 2, 2])
        with buttons[0], st.container():
            if st.button(
                "▶️ Tìm lại câu này", key=f"research_{query.query_id}",
                type="primary",
            ):
                st.session_state[f"search_{query.query_id}"] = SearchText(
                    vi=vi.strip() or query.text, en=en.strip() or None
                )
                # Thu tu tay cu la cua mot danh sach ung vien KHAC.
                st.session_state.pop(f"order_{query.query_id}", None)
                run_queries([query], pool_k, penalty)
                st.rerun()
        with buttons[1], st.container():
            # Điền sẵn, KHÔNG tự chạy: người vận hành phải thấy chuỗi trước khi
            # nó thành kết quả. Vô hiệu khi câu đề chỉ có đúng một câu hỏi — bấm
            # một nút rồi không đổi gì trông y hệt một nút hỏng.
            if st.button(
                "✂️ Bỏ câu hỏi", key=f"strip_{query.query_id}",
                disabled=not has_description(query.text),
                help="Điền phần mô tả (bỏ mọi câu kết thúc bằng `?`) vào ô trên. "
                     "Chưa chạy — bấm Tìm lại sau khi xem." if
                     has_description(query.text) else
                     "Câu đề này chỉ có một câu hỏi, không có phần mô tả để giữ.",
            ):
                st.session_state[f"search_fill_vi_{query.query_id}"] = cut
                st.rerun()
        with buttons[2], st.container():
            if override and st.button(
                "↩️ Trả về câu đề gốc", key=f"reset_search_{query.query_id}"
            ):
                st.session_state.pop(f"search_{query.query_id}", None)
                st.session_state.pop(f"order_{query.query_id}", None)
                # Trả hai ô về mặc định qua đường `*_fill_*` như trên, chứ không
                # xoá khoá widget: xoá xong ô vẫn dựng lại từ `setdefault` ở lượt
                # sau, nhưng đường ghi thì vẫn là đường Streamlit cấm.
                st.session_state[f"search_fill_vi_{query.query_id}"] = query.text
                st.session_state[f"search_fill_en_{query.query_id}"] = ""
                run_queries([query], pool_k, penalty)
                st.rerun()

    if override:
        # Ngoài expander, vì kết quả đang hiện KHÔNG phải của câu đề hiện phía
        # trên, và người vận hành phải thấy điều đó mà không cần mở gì ra.
        st.caption(
            f":green[🔍 Kết quả dưới đây tìm bằng chuỗi ghi đè:] `{override.vi}`"
            + (f"  ·  EN: `{override.en}`" if override.en else "")
        )


def _search_help(query: PackagedQuery) -> None:
    with st.popover("❓ Khi nào nên sửa chuỗi tìm?"):
        st.markdown(
            """
**Sửa khi** chạy xong mà lưới ảnh không có khung cần tìm. Hai lỗi hay gặp, và
chúng **cộng dồn** chứ không thay nhau:

**1. Câu hỏi làm loãng.** Tháp thị giác học trên *caption* — mô tả một tấm ảnh —
còn câu đề QA là một *câu hỏi*. Bấm **✂️ Bỏ câu hỏi** để giữ lại phần mô tả.

**2. Bản dịch máy sai một chữ.** envit5 dịch *"xe ô tô lội nước"* thành
*"the trucks wading"* — đổi *ô tô* thành **xe tải** và bỏ mất *nước*. Gõ tay vào
ô tiếng Anh để đè lên.

Đo trên `query-p1-9-qa` (đợt 1), hạng của khung đáp án trong 176.707 khung:

| chuỗi đi tìm | hạng |
|---|---|
| nguyên văn câu đề (envit5) | **540** — ngoài pool 500 |
| sửa `trucks` → `cars` | 71 |
| + `through water` | 34 |
| bỏ câu hỏi + sửa dịch | **12** |

---

**Chỉ đổi thứ hệ đi TÌM.** `query_id` không đổi, tên file nộp không đổi, đáp án
đã gõ không mất. Thứ tự xếp tay cũ **bị xoá**, vì danh sách ứng viên đã khác.
"""
        )


# ---------------------------------------------------------------------------
# Mệnh đề do người tách — cần gạt độ phủ của tầng thứ tự cảnh
# ---------------------------------------------------------------------------


def manual_clauses(query_id: str) -> list[str] | None:
    return st.session_state.get(f"clauses_{query_id}")


def _clause_help() -> None:
    """Hướng dẫn "khi nào nên tách" — phần khó nhất của tính năng này.

    Cơ chế thì đơn giản, còn phán đoán thì không: người đứng bàn phím phải phân
    biệt **ba cảnh nối tiếp** với **một cảnh tả kỹ**, và tách nhầm loại thứ hai
    là chủ động làm tụt hạng. Bộ luật tự động cũng vấp đúng chỗ đó (chỉ 3/38 đề
    KIS thật đạt ngưỡng), nên chú thích này gánh phần bộ luật không gánh nổi.
    """
    with st.popover("❓ Khi nào nên tách mệnh đề?"):
        st.markdown(
            f"""
**Tách khi** câu kể **≥{MIN_CLAUSES_FOR_PRIOR} cảnh nối tiếp nhau** trong cùng
một video. Dấu hiệu là từ nối trình tự: *"Đầu tiên… Sau đó… Cuối cùng"*,
*"Tiếp theo"*, *"Trước đó là"*.

**Đừng tách khi** câu chỉ tả **một khoảnh khắc**, dù dài. Liệt kê chi tiết trong
cùng một khung — số người, màu áo, vật bên trái/phải — **không phải** trình tự;
tách ra chỉ làm mỗi mệnh đề mất ngữ cảnh và tụt hạng. Riêng *"lần lượt"* hầu như
luôn là liệt kê: **3/3** lần xuất hiện trong bộ đề thật đều đếm vật trong một
cảnh.

**Thứ tự dòng = trình tự thời gian**, không phải thứ tự chữ trong đề. *"Người đàn
ông ngã xuống. Trước đó anh ta đang chạy"* → dòng 1 là *đang chạy*, dòng 2 là
*ngã xuống*.

**Tách xong vẫn nộp một khung.** Tầng không đổi thứ mình nộp; nó cộng điểm cho
**video** nào có đủ các cảnh xuất hiện đúng thứ tự, rồi mọi khung trong video đó
lên theo.

**Sửa xong phải bấm ▶️ Chạy lại câu này** — gõ vào ô không tự chạy lại. Thứ tự
xếp tay cũ sẽ bị xoá, vì danh sách ứng viên đã khác.

---

Đo trên 14 câu KIS ba cảnh: hạng TB **10,3 → 4,2**, 4 câu lên / 0 câu xuống, và
DANTE xếp đúng video ở hạng nhất **14/14** trên 873 video. Dưới
{MIN_CLAUSES_FOR_PRIOR} mệnh đề thì tầng tự tắt: với 2 cảnh, hai vị trí bất kỳ
xếp đúng thứ tự với xác suất ~½ nên ràng buộc gần như không lọc gì — đo được là
**luôn** gây hại.
"""
        )


def render_clause_editor(query: PackagedQuery, pool_k: int, penalty: float) -> None:
    """Ô sửa mệnh đề + nút chạy lại RIÊNG câu này.

    **Đây là cần gạt độ phủ, không phải tiện ích.** Tầng thứ tự cảnh đo được
    RỘNG 0,800 → 0,843 (hạng TB 10,3 → 4,2) và DANTE xếp đúng video hạng nhất
    **14/14** câu — nhưng nó chỉ chạy cho câu tách được **≥3** mệnh đề, và trên
    38 đề KIS thật bộ luật chỉ đạt ngưỡng đó ở **3 câu**. Người đọc không có
    đánh đổi mà bộ luật phải chịu, và lúc thi thì có người ngồi ở bàn phím.
    """
    weight = options_for(query).scene_order or 0.0
    auto = [c.text for c in split_clauses(query.text)]
    current = manual_clauses(query.query_id) or auto
    enough = len(current) >= MIN_CLAUSES_FOR_PRIOR

    # Nhãn nói ĐANG CÓ BAO NHIÊU trên NGƯỠNG, không phải một con số trần trụi:
    # "(2)" đứng cạnh dòng "cần ít nhất 3" đọc ra như một cái trần.
    state = "đang chạy" if enough and weight > 0 else f"cần ≥{MIN_CLAUSES_FOR_PRIOR}"

    text_key = f"clause_text_{query.query_id}"
    pending = st.session_state.pop(f"clause_fill_{query.query_id}", None)
    if pending is not None:
        st.session_state[text_key] = pending
    st.session_state.setdefault(text_key, "\n".join(current))

    with st.expander(
        f"✂️ Mệnh đề — {len(current)}/≥{MIN_CLAUSES_FOR_PRIOR} dòng · {state}"
    ):
        _clause_help()
        if weight <= 0:
            st.warning(
                "Thứ tự cảnh đang ở trọng số 0 cho câu này — tách mệnh đề sẽ "
                "**không đổi gì**. Chỉnh ở bảng Tuỳ chọn."
            )
        st.caption(
            "Mỗi dòng một cảnh, **theo trình tự thời gian** (với 'Trước đó là "
            f"cảnh X' thì X lên đầu). Không giới hạn số dòng, nhưng dưới "
            f"{MIN_CLAUSES_FOR_PRIOR} thì tầng tự tắt: câu hai cảnh đo được là "
            "luôn gây hại, vì hai vị trí bất kỳ xếp đúng thứ tự với xác suất ~½."
        )
        # KHÔNG `value=`: đã có `key` thì `session_state` là nguồn duy nhất.
        typed = st.text_area(
            "Mệnh đề", height=150, key=text_key, label_visibility="collapsed",
        )
        lines = [line.strip() for line in typed.splitlines() if line.strip()]
        if len(lines) < MIN_CLAUSES_FOR_PRIOR:
            st.caption(f":orange[{len(lines)} dòng — tầng sẽ ngồi ngoài.]")

        left, right = st.columns(2)
        with left, st.container():
            if st.button(
                "▶️ Chạy lại câu này",
                key=f"rerun_{query.query_id}",
                help=(
                    "Bắt buộc sau khi sửa ô mệnh đề — gõ xong không tự chạy lại. "
                    "Thứ tự xếp tay cũ sẽ bị xoá vì danh sách ứng viên đã khác."
                ),
            ):
                st.session_state[f"clauses_{query.query_id}"] = lines
                # Thu tu tay cu la cua mot danh sach ung vien KHAC.
                st.session_state.pop(f"order_{query.query_id}", None)
                run_queries([query], pool_k, penalty)
                st.rerun()
        with right, st.container():
            if manual_clauses(query.query_id) and st.button(
                "↩️ Trả về bộ luật", key=f"reset_clause_{query.query_id}"
            ):
                st.session_state.pop(f"clauses_{query.query_id}", None)
                # `clause_text_*` là khoá WIDGET — đụng vào nó sau khi ô đã dựng
                # là `StreamlitAPIException`. Hẹn qua `clause_fill_*` rồi đổ ở
                # đầu lượt sau, y như hai ô chuỗi tìm.
                st.session_state[f"clause_fill_{query.query_id}"] = "\n".join(auto)
                st.rerun()


# ---------------------------------------------------------------------------
# Đáp án QA — gõ tay, hoặc để VLM đọc khỏi ảnh
# ---------------------------------------------------------------------------


def answer_for(query_id: str) -> str:
    """Đáp án QA đang giữ, đọc từ khoá KHÔNG-widget.

    Đọc thẳng `answer_{id}` (khoá widget) là sai: Streamlit thu hồi nó khi ô
    không được vẽ ở lượt trước — và mọi nút gọi `st.rerun()` sớm đều tạo ra một
    lượt như thế. Kết quả là ZIP bỏ câu trong khi màn hình vẫn hiện đáp án.
    """
    return st.session_state.get(f"answer_keep_{query_id}", "")


def render_answer_box(query: PackagedQuery, result, images) -> None:
    """Ô đáp án + nút bảo VLM nhìn top khung rồi đọc ra.

    Cột thứ ba **giống nhau ở cả 100 dòng**, nên đây là đúng một chuỗi — và một
    chuỗi sai làm mất **cả 100 dòng**, không phải một. Đó là lý do nút này chỉ
    *điền vào ô*, không tự nộp: người vận hành vẫn phải nhìn.

    `Answer.frame` là chỗ VLM khai đã đọc đáp án ra, và nó có ích hơn con số tin
    cậy: kiểm một tấm ảnh nhanh hơn kiểm tám tấm.
    """
    ok, why = vqa_available()
    answer_key = f"answer_{query.query_id}"

    # Nhân bản ra một khoá KHÔNG-widget, và đó không phải thừa. Streamlit thu hồi
    # trạng thái của widget nào không được vẽ trong lượt chạy trước, mà `st.rerun()`
    # trong nút "Tìm lại câu này" cắt script ngay tại đó — ô đáp án chưa kịp dựng.
    # Đã xảy ra thật: ô vẫn HIỆN `24.84` trên màn hình còn `build_zip` thì báo
    # "chưa có đáp án" và bỏ câu khỏi gói. Đúng loại hỏng im lặng tệ nhất ở đây.
    keep_key = f"answer_keep_{query.query_id}"

    # Đổ đáp án VLM vào ô TRƯỚC khi ô được dựng. Streamlit cấm ghi
    # `session_state[k]` sau khi widget khoá `k` đã dựng trong cùng lượt chạy
    # (`StreamlitAPIException`), mà nút nằm ngay cạnh ô nên nó luôn chạy sau. Nên
    # nút chỉ **hẹn** qua một khoá riêng, còn việc đổ nằm ở đầu lượt kế tiếp.
    pending = st.session_state.pop(f"vlm_fill_{query.query_id}", None)
    st.session_state.setdefault(answer_key, st.session_state.get(keep_key, ""))
    if pending is not None:
        st.session_state[answer_key] = pending

    left, right = st.columns([4, 1])
    with left, st.container():
        typed = st.text_input(
            "Đáp án (cột thứ ba của mọi dòng)", key=answer_key,
            help="Dấu phẩy trong đáp án được `csv` trích dẫn đúng.",
        )
    st.session_state[keep_key] = typed
    with right, st.container():
        st.caption("")  # đẩy nút xuống ngang hàng với ô nhập
        if st.button(
            "🤖 VLM đọc hộ", key=f"vlm_answer_{query.query_id}", disabled=not ok,
            help=why or f"Cho model nhìn {VQA_DEPTH} khung đầu rồi đọc đáp án ra. "
                        "Chỉ điền vào ô — không tự nộp.",
        ):
            head = [c.keyframe_id for c in result[:VQA_DEPTH]]
            with st.spinner(f"Tải {len(head)} ảnh rồi hỏi VLM …"):
                images.prefetch(head)
                answer = read_answer(
                    query.text, result, images.cached, depth=VQA_DEPTH
                )
            st.session_state[f"vlm_answer_out_{query.query_id}"] = answer
            if answer.ok:
                st.session_state[f"vlm_fill_{query.query_id}"] = answer.text
            st.rerun()

    answer = st.session_state.get(f"vlm_answer_out_{query.query_id}")
    if answer is not None:
        if answer.ok:
            source = f" · đọc từ `{answer.frame}`" if answer.frame else (
                " · :orange[không chỉ được khung nào]"
            )
            st.caption(
                f"🤖 VLM: **{answer.text}** — tin cậy *{answer.confidence or '?'}*"
                f"{source} · nhìn {answer.looked_at} khung · {answer.seconds:.1f}s"
            )
            if answer.why:
                st.caption(f"↳ {answer.why}")
        else:
            # Không đọc được PHẢI hiện ra. Một nút bấm xong im lặng trông y hệt
            # một nút chạy rồi mà vô ích, và ở đây "vô ích" nghĩa là ô đáp án vẫn
            # trống trong khi người vận hành tưởng đã xong.
            st.warning(f"🤖 VLM không điền được: {answer.reason}")

    if not typed.strip():
        st.warning("Chưa nhập đáp án — câu này sẽ bị BỎ khỏi ZIP.")


# ---------------------------------------------------------------------------
# Thứ tự do người xếp
# ---------------------------------------------------------------------------


def manual_order(query_id: str) -> list[str] | None:
    return st.session_state.get(f"order_{query_id}")


def ordered(query_id: str, candidates: list[Candidate]) -> list[Candidate]:
    return apply_manual_order(candidates, manual_order(query_id))


def rows_for(query: PackagedQuery, result, relational, top_k: int):
    return submissions_for(
        query, result, relational, limit=top_k,
        manual_order=manual_order(query.query_id),
        answer=answer_for(query.query_id),
    )


# ---------------------------------------------------------------------------
# Thanh bên
# ---------------------------------------------------------------------------

_, _cascade, _relational = load_engine()
_defaults = load_engine()[0].retrieval

st.sidebar.header("📥 Gói đề")
uploads = st.sidebar.file_uploader(
    "Thả các file `query-*.txt` vào đây",
    type=["txt"], accept_multiple_files=True,
    help="Dạng đề đọc từ hậu tố tên file: `-kis`, `-qa`, `-trake`. "
         "Tên file nộp = tên đề + `.csv`, nên ĐỪNG đổi tên file.",
)

folder = st.sidebar.text_input(
    "…hoặc đường dẫn thư mục chứa `query-*.txt`",
    placeholder="docs/context/bo-de-dot-1",
    help="Cùng bộ đọc với ô tải lên (`mfe.eval.query_package`). Tiện khi gói đề "
         "đã nằm trên đĩa — 25 file thì chọn tay khá mệt.",
)

st.sidebar.header("⚙ Tuỳ chọn")
top_k = st.sidebar.slider(
    "Số dòng nộp mỗi câu", 10, MAX_ANSWERS_PER_QUERY, MAX_ANSWERS_PER_QUERY, 10,
    help="R@100 vào Final Score và nó chỉ tăng khi nộp thêm dòng.",
)
pool_k = st.sidebar.slider("Độ sâu pool để xem", 50, 500, 200, 50)
grid_cols = st.sidebar.slider("Số cột lưới ảnh", 2, 8, 5)
show_images = st.sidebar.checkbox("Hiện ảnh (tải từ Modal)", value=True)
penalty = st.sidebar.select_slider(
    "λ — phạt khoảng cách (TRAKE)",
    options=[0.0, 0.001, 0.003, 0.01, 0.03], value=dante.PENALTY,
)

# ---------------------------------------------------------------------------
# Thân
# ---------------------------------------------------------------------------

st.title("📦 Bàn nộp — AIC 2026")

parsed: list[PackagedQuery] = st.session_state.setdefault("parsed", [])
seen = {q.query_id for q in parsed}
for upload in uploads or []:
    try:
        query = parse_query(upload.name, upload.getvalue().decode("utf-8"))
    except Exception as err:  # noqa: BLE001 - hien loi, khong nuot
        st.error(f"`{upload.name}` — {type(err).__name__}: {err}")
        continue
    if query.query_id not in seen:
        parsed.append(query)
        seen.add(query.query_id)

if folder.strip():
    try:
        # `load_package` chứ không phải tự `glob`: nó sắp theo SỐ trong tên và
        # ném khi thư mục rỗng. Hai luật đó không nên có bản thứ hai.
        for query in load_package(folder.strip()):
            if query.query_id not in seen:
                parsed.append(query)
                seen.add(query.query_id)
    except Exception as err:  # noqa: BLE001 - hien loi cho nguoi dung
        st.sidebar.error(f"{type(err).__name__}: {err}")

if not parsed:
    st.info(
        "Thả các file `query-*.txt` vào thanh bên. Đợt đánh giá 1 phải chép file "
        "vào codebase mới chạy được; ở đây tải thẳng lên."
    )
    st.stop()

# Sắp theo SỐ trong tên, không theo chuỗi: sắp chuỗi đặt `query-p1-10` trước
# `query-p1-2`, và thứ tự đó là thứ tự người vận hành đối chiếu với đề.
parsed.sort(key=lambda q: tuple(
    int(part) if part.isdigit() else part for part in re.split(r"(\d+)", q.query_id)
))

results: dict[str, list] = st.session_state.setdefault("results", {})
by_type = {t: [q for q in parsed if q.type == t] for t in TYPES}

head = st.columns([3, 1, 1])
with head[0]:
    st.markdown(
        "**" + str(len(parsed)) + " đề** — "
        + " · ".join(f"{t} {len(by_type[t])}" for t in TYPES if by_type[t])
        + f"  ·  đã chạy {len(results)}/{len(parsed)}"
    )
with head[1]:
    if st.button("▶️ Chạy cả gói", type="primary"):
        run_queries(parsed, pool_k, penalty)
        st.rerun()
with head[2]:
    if st.button("🗑️ Xoá gói"):
        for key in ("parsed", "results"):
            st.session_state.pop(key, None)
        st.rerun()

with st.expander("⚙ Bộ tuỳ chọn — áp cho CẢ GÓI", expanded=False):
    render_options("goi", _defaults, _cascade)

st.markdown("---")

tabs = st.tabs([f"{TAB_LABEL[t]} ({len(by_type[t])})" for t in TYPES])

for tab, qtype in zip(tabs, TYPES, strict=True):
    with tab:
        pool = by_type[qtype]

        # --- tìm riêng, kết quả nối vào chính tab này -----------------------
        with st.form(key=f"adhoc_{qtype}", clear_on_submit=True):
            typed = st.text_input(
                f"Tìm riêng một câu {qtype} — kết quả sẽ được **thêm vào tab này**",
                placeholder="gõ mô tả rồi Enter…",
            )
            st.caption(
                f"Đẻ ra một câu **mới** (`{ADHOC_PREFIX}-N-{qtype.lower()}`) để dò — "
                "**không** vào ZIP nộp. Muốn thay kết quả của một đề có sẵn thì "
                "dùng ô `🔍 Chuỗi tìm` của chính câu đó."
            )
            if st.form_submit_button("🔍 Tìm và thêm vào tab") and typed.strip():
                # Đi qua `parse_query` với một tên file dựng sẵn, nên câu tự tay
                # có đủ `query_id`, `csv_name`, tra được `frame_id` — mọi luật đọc
                # đề chỉ có một bản. Cái duy nhất khác là `build_zip` loại nó ra:
                # tên file nộp phải là tên đề BTC ra.
                index = sum(1 for q in parsed if is_adhoc(q.query_id)) + 1
                name = f"{ADHOC_PREFIX}-{index}-{qtype.lower()}.txt"
                extra = parse_query(name, typed.strip())
                parsed.append(extra)
                run_queries([extra], pool_k, penalty)
                st.rerun()

        if not pool:
            st.info(f"Chưa có đề {qtype} nào trong gói.")
            continue

        with st.expander(f"⚙ Bộ tuỳ chọn — áp cho MỌI câu {qtype}"):
            render_options(f"dang_{qtype}", _defaults, _cascade)
        if st.button(f"▶️ Chạy lại {len(pool)} câu {qtype}", key=f"run_{qtype}"):
            run_queries(pool, pool_k, penalty)
            st.rerun()

        labels = {
            q.query_id: (
                f"{'✓' if q.query_id in results else '·'} {q.query_id}"
                f"  — {len(results.get(q.query_id, []))} ứng viên"
            )
            for q in pool
        }
        # `labels.get` chứ không phải `lambda k: labels[k]`: closure trong vòng
        # lặp bắt biến chứ không bắt giá trị, nên lambda sẽ đọc `labels` của tab
        # CUỐI cho cả ba tab. Streamlit vẽ ba tab trong cùng một lượt chạy nên
        # lỗi đó hiện ra ngay, nhưng ở dạng "nhãn tab KIS hiện tên câu TRAKE" —
        # trông như lẫn dữ liệu chứ không như một lỗi closure.
        picked_id = st.selectbox(
            "Câu nào", list(labels), format_func=labels.get, key=f"pick_{qtype}",
        )
        query = next(q for q in pool if q.query_id == picked_id)
        st.caption(query.text)
        if query.type == "TRAKE":
            for index, event in enumerate(query.events, start=1):
                st.caption(f"  E{index}: {event}")

        with st.expander("⚙ Bộ tuỳ chọn — áp cho RIÊNG câu này"):
            render_options(f"cau_{picked_id}", _defaults, _cascade)

        if picked_id not in results:
            if st.button("▶️ Chạy câu này", key=f"run_one_{picked_id}", type="primary"):
                run_queries([query], pool_k, penalty)
                st.rerun()
            continue

        result = results[picked_id]
        images = load_images()

        if query.type != "TRAKE":
            render_search_editor(query, pool_k, penalty)
            render_clause_editor(query, pool_k, penalty)

        if query.type == "QA":
            render_answer_box(query, result, images)

        # --- xếp tay -------------------------------------------------------
        if query.type == "TRAKE":
            st.info(
                "TRAKE không có phần xếp tay: mỗi dòng là **một video kèm N "
                "khung tăng dần** do `dante.align` ghép. Kéo một dòng lên đầu "
                "không sửa được chuỗi bên trong nó."
            )
            for position, item in enumerate(result[:20], start=1):
                st.code(
                    f"{position}. {item.video_id},"
                    + ",".join(str(f) for f in item.frame_ids),
                    language=None,
                )
        else:
            ranked = ordered(picked_id, result)
            machine_rank = {c.keyframe_id: i for i, c in enumerate(result, start=1)}
            depth = st.slider(
                "Xếp tay bao nhiêu dòng đầu", 10,
                min(MAX_ANSWERS_PER_QUERY, len(ranked)),
                min(MAX_ANSWERS_PER_QUERY, len(ranked)), 10,
                key=f"depth_{picked_id}",
            )
            grid = ranked[: int(depth)]
            st.caption(
                "Kéo **chính tấm ảnh** sang chỗ mới. Mép đỏ báo nó chèn vào "
                "trước hay sau ô đang thả."
            )
            if show_images:
                with st.spinner(f"Tải {len(grid)} ảnh …"):
                    images.prefetch([c.keyframe_id for c in grid])

            dragged = sortable_grid(
                [c.keyframe_id for c in grid],
                # `cached`, KHONG `get`: `prefetch` da tai song song 8 luong.
                # `get` se tai TUAN TU tung anh thieu ngay giua vong dung luoi.
                images.cached if show_images else (lambda _: None),
                machine_rank=machine_rank, cols=grid_cols,
                key=f"grid_{picked_id}_{depth}",
            )
            if dragged and dragged != [c.keyframe_id for c in grid]:
                # Đuôi ngoài `depth` giữ thứ tự máy và VẪN được nộp — cắt nó đi
                # là vứt bốn trong năm số hạng của Final Score.
                st.session_state[f"order_{picked_id}"] = list(dragged) + [
                    c.keyframe_id for c in ranked[int(depth):]
                ]
                st.rerun()

            if manual_order(picked_id) and st.button(
                "↩️ Trả về thứ tự máy", key=f"reset_{picked_id}"
            ):
                st.session_state.pop(f"order_{picked_id}")
                st.rerun()

            with st.expander(f"📄 Toạ độ nộp — {len(ranked)} khung"):
                for position, candidate in enumerate(ranked[: int(pool_k)], start=1):
                    was = machine_rank.get(candidate.keyframe_id)
                    moved = f"  ✋ (máy xếp {was})" if was and was != position else ""
                    try:
                        frame_id = _relational.get_keyframe(
                            candidate.keyframe_id
                        ).frame_idx
                        frame_text = f"frame `{frame_id}`"
                    except Exception:  # noqa: BLE001
                        frame_text = ":red[không tra được frame_id]"
                    st.markdown(
                        f"**{position}.** `{candidate.keyframe_id}` — "
                        f"{frame_text}{moved}"
                    )

        try:
            subs = rows_for(query, result, _relational, top_k)
        except ContractError as err:
            st.error(str(err))
        else:
            st.download_button(
                f"⬇️ `{query.csv_name}` — {len(subs)} dòng",
                data=to_csv_text(subs, limit=top_k).encode("utf-8"),
                file_name=query.csv_name, mime="text/csv",
                key=f"dl_{picked_id}",
            )

# ---------------------------------------------------------------------------
# Xuất cả gói
# ---------------------------------------------------------------------------

st.markdown("---")
ran = [q for q in parsed if q.query_id in results]
if ran:
    bundle = build_zip(
        ran, results, _relational, limit=top_k,
        manual_orders={
            q.query_id: order for q in ran if (order := manual_order(q.query_id))
        },
        answers={
            q.query_id: answer_for(q.query_id) for q in ran
        },
    )
    # Mẫu số là số ĐỀ BTC, không phải số câu đã chạy: câu tự gõ để dò không bao
    # giờ thuộc gói nộp, nên đếm chúng vào mẫu số sẽ làm một gói ĐỦ file trông
    # như thiếu.
    total = sum(1 for q in parsed if not is_adhoc(q.query_id))
    written = len(ran) - len(bundle.skipped)
    st.download_button(
        f"⬇️ Cả gói — {written}/{total} file trong ZIP",
        data=bundle.payload, file_name="submissions.zip",
        mime="application/zip", type="primary",
    )
    missing = {k: v for k, v in bundle.skipped.items() if not is_adhoc(k)}
    adhoc = [k for k in bundle.skipped if is_adhoc(k)]
    if missing:
        # Một ZIP thiếu file trông y hệt một ZIP đủ file cho tới lúc đếm, và lúc
        # đếm thì đã nộp rồi.
        st.warning(f"**{len(missing)} câu KHÔNG có file trong ZIP:**")
        for query_id, reason in missing.items():
            st.caption(f"· `{query_id}` — {reason}")
    if adhoc:
        st.caption(
            f"ℹ️ {len(adhoc)} câu tự gõ để dò không vào ZIP (đúng như thiết kế): "
            + ", ".join(f"`{k}`" for k in adhoc)
        )

st.caption(
    "Không dòng tiêu đề · tối đa 100 dòng · **thứ tự dòng có tính điểm**. "
    "`frame_id` đọc từ Relational Map, không suy ra từ `keyframe_id`."
)
