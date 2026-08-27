"""Bàn thử rerank — bật/tắt từng tầng và nhìn thứ hạng đổi.

    streamlit run app/rerank_lab.py

Giao diện dựng theo `SOTA_AIC/ui_app.py` của Nhật Bảo để hai bên so được với nhau,
nhưng nó trả lời một câu hỏi khác. Bên Bảo là công cụ **tìm**; cái này là công cụ
**đo**: mỗi tầng in ra chạm bao nhiêu ứng viên, mất bao lâu, và **dời cái gì đi
đâu**. Một tầng rerank đổi thứ hạng mà không nói nó đổi gì thì không kiểm được.

Bốn tầng, xếp theo thứ tự chạy
-------------------------------
    ta        SigLIP đa ngữ chấm lại pixel      — của mình, đã có số đo
    colpali   late-interaction MaxSim           — tầng 3 của Bảo
    hf_cross  cross-encoder trên text metadata  — tầng 4 của Bảo
    vlm       VLM xếp hạng bằng ảnh             — tầng 5 của Bảo

Thứ tự là một lựa chọn: tầng sau ghi đè tầng trước ở phần nó chạm tới. Rẻ và sâu
chạy trước, đắt và nông chạy sau, để cái đắt chỉ nói lời cuối trên phần đã được
đẩy lên.

Không tầng nào bật mặc định, vì **chi phí và độ trễ** — không phải vì dữ liệu.
BTC cho phép chia sẻ dữ liệu tự do (Lead xác nhận 23-08-2026), nên `vlm` gửi ảnh
ra ngoài là chuyện bình thường; nó chỉ chậm (32s cho 12 ảnh) và tính tiền.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mfe.contracts.tools import Candidate  # noqa: E402
from mfe.offline.transport import load_dotenv  # noqa: E402
from mfe.online.flat_arms import FlatBundle  # noqa: E402
from mfe.online.fusion import weighted_fusion  # noqa: E402
from mfe.online.keyframe_images import KeyframeImages  # noqa: E402
from mfe.online.rerank_tiers import (  # noqa: E402
    ColPaliTier,
    HfCrossEncoderTier,
    VlmTier,
    apply_tiers,
)
from mfe.stores.inverted.index import _normalize_text, _tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FLAT_ROOT = ROOT / "stores" / "flat"
CONFIG = ROOT / "config" / "btc_b1.yaml"

TEXT_ARMS = ("ocr_vietocr", "ocr", "ocr_paddle", "subtitle", "object")
DENSE_ARMS = ("siglip", "clap")

st.set_page_config(page_title="Bàn thử rerank", layout="wide")

# Streamlit khong nap `.env`. Khong co dong nay thi ba tang cua Bao deu bao
# "chua co API_KEY" trong khi key nam san trong file — mot loi trong tro thanh
# "tang do khong dung duoc" thay vi "tang do chua duoc cam dien".
for _key, _value in load_dotenv(ROOT / ".env").items():
    os.environ.setdefault(_key, _value)


# ---------------------------------------------------------------------------
# Nạp một lần, giữ suốt phiên. Đây là thứ CLI không làm được: nạp tháp text
# SigLIP mất ~10s mỗi tiến trình, nên tra tay bằng CLI trả giá đó mỗi truy vấn.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Nạp kho phẳng …")
def load_bundle(sources: tuple[str, ...], models: tuple[str, ...]):
    bundle = FlatBundle.load(FLAT_ROOT, sources=set(sources), models=set(models))
    for source in sources:
        if source in bundle.texts:
            bundle.build_bm25(source)
    return bundle


@st.cache_resource(show_spinner="Nạp tháp text SigLIP …")
def load_siglip():
    from transformers import AutoModel, AutoTokenizer

    name = "google/siglip-so400m-patch14-384"
    return AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()


@st.cache_resource(show_spinner="Nạp tháp text CLAP …")
def load_clap():
    from transformers import ClapModel, ClapProcessor

    name = "laion/clap-htsat-unfused"
    return ClapProcessor.from_pretrained(name), ClapModel.from_pretrained(name).eval()


@st.cache_resource(show_spinner="Nạp bộ dịch envit5 …")
def load_translator():
    from mfe.online.translate import MachineTranslator

    return MachineTranslator()


@st.cache_resource
def load_images():
    return KeyframeImages()


class AssetShim:
    """Cho `Reranker` của ta thứ nó chờ đợi: `get_asset(keyframe_id) -> bytes`.

    `Reranker` được viết để đọc Asset Store thật. Ở đường phẳng không có store
    nào, chỉ có ảnh cache trên đĩa — nên chỗ nối nằm ở đây, và `Reranker` không
    phải biết gì về chuyện đó.
    """

    def __init__(self, images: KeyframeImages) -> None:
        self.images = images

    def get_asset(self, keyframe_id: str) -> bytes:
        path = self.images.get(keyframe_id)
        if path is None:
            raise FileNotFoundError(keyframe_id)
        return path.read_bytes()


@st.cache_resource
def load_our_reranker(_images: KeyframeImages, budget_ms: int):
    from mfe.online.rerank import Reranker

    return Reranker(asset_store=AssetShim(_images), budget_ms=budget_ms)


def encode_dense(arm: str, text_en: str) -> np.ndarray:
    import torch

    if arm == "siglip":
        tokenizer, model = load_siglip()
        batch = tokenizer(
            [text_en], padding="max_length", max_length=64,
            truncation=True, return_tensors="pt",
        )
    else:
        processor, model = load_clap()
        batch = processor(text=[text_en], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.get_text_features(**batch)
    pooled = getattr(out, "pooler_output", None)
    vector = (pooled if pooled is not None else out).numpy()
    return (vector / np.linalg.norm(vector, axis=1, keepdims=True))[0]


def to_candidates(keys, arm: str) -> list[Candidate]:
    return [
        Candidate(keyframe_id=k, video_id=k.split("#")[0], score=1.0 / rank,
                  evidence_id=f"{arm}:lab:{rank}")
        for rank, k in enumerate(keys, start=1)
    ]


# ---------------------------------------------------------------------------
# Thanh bên
# ---------------------------------------------------------------------------

st.sidebar.header("⚙ Arm truy hồi")
arms_on = {
    arm: st.sidebar.checkbox(arm, value=arm in ("siglip", "ocr_vietocr", "subtitle"))
    for arm in (*DENSE_ARMS, *TEXT_ARMS)
}
arms_on["subtitle_exact"] = st.sidebar.checkbox("subtitle_exact (khớp chuỗi con)", value=True)
pool_size = st.sidebar.slider("Mỗi arm nộp bao nhiêu ứng viên", 50, 500, 200, 50)

st.sidebar.header("🎯 Tầng rerank")
use_ours = st.sidebar.checkbox("**Của ta** — SigLIP đa ngữ trên pixel", value=False)
use_colpali = st.sidebar.checkbox("Tầng 3 — ColPali MaxSim", value=False)
use_hf = st.sidebar.checkbox("Tầng 4 — Cross-encoder HF API", value=False)
use_vlm = st.sidebar.checkbox("Tầng 5 — VLM (Llama-90B / Gemini)", value=False)
rerank_depth = st.sidebar.slider("Rerank sâu bao nhiêu", 10, 100, 30, 10)
our_budget_ms = st.sidebar.slider("Ngân sách tầng của ta (ms)", 3000, 60000, 20000, 1000)

with st.sidebar.expander("Vì sao không cái nào bật sẵn"):
    st.markdown(
        "**Tiền và thời gian, không phải dữ liệu.** BTC cho phép chia sẻ dữ liệu "
        "tự do (Lead xác nhận 23-08-2026), nên `vlm` gửi ảnh ra ngoài không phải "
        "rào cản — đo được 32 giây cho 12 ảnh, và mỗi lượt là một lần gọi API "
        "tính phí.\n\n"
        "`colpali` tải model 6 GB. Bản của Bảo dựng sẵn ma trận token cho cả kho "
        "và phải tắt vì RAM; ở đây nó mã hoá **lúc chạy**, chỉ top ứng viên, nên "
        "kho bằng 0 nhưng mỗi truy vấn chậm hơn."
    )

st.sidebar.header("🧠 Tầng 0 — QUEST")
use_quest = st.sidebar.checkbox(
    "Gọi tên thứ đề không nêu tên", value=False,
    help="Đụng 7/49 đề chính thức: 'phim Spielberg 1975' -> 'cá mập'. "
         "Chỉ THÊM từ vào cuối truy vấn, không thay câu. Cần GEMINI_API_KEY.",
)

st.sidebar.header("🖼 Ảnh")
show_images = st.sidebar.checkbox("Hiện ảnh (tải từ Modal)", value=True)
grid_cols = st.sidebar.slider("Số cột", 2, 8, 5)

# ---------------------------------------------------------------------------
# Truy vấn
# ---------------------------------------------------------------------------

st.title("Bàn thử rerank")
query_vi = st.text_input("Truy vấn (tiếng Việt)", "sạt lở đèo gây ách tắc giao thông")
col_a, col_b = st.columns([3, 1])
query_en = col_a.text_input("Bản dịch tiếng Anh (để trống thì dịch bằng envit5)", "")
top_k = col_b.number_input("Hiện bao nhiêu", 5, 100, 20, 5)
go = st.button("Tìm", type="primary")

if go and query_vi.strip():
    dense_wanted = tuple(a for a in DENSE_ARMS if arms_on.get(a))
    text_wanted = tuple(
        {("subtitle" if a == "subtitle_exact" else a)
         for a in (*TEXT_ARMS, "subtitle_exact") if arms_on.get(a)}
    )
    bundle = load_bundle(text_wanted, dense_wanted)

    if not query_en.strip() and dense_wanted:
        with st.spinner("Dịch bằng envit5 …"):
            query_en = load_translator().translate(query_vi)
        st.info(f"Đã dịch: **{query_en}**")

    if use_quest:
        from mfe.online.quest import expanded_queries, quest_expand

        with st.spinner("QUEST đang gọi tên …"):
            terms = quest_expand(query_vi)
        if terms:
            # In ra thứ nó thêm VÀ vì sao. Một truy vấn âm thầm bị viết lại là
            # một bảng kết quả không giải thích được.
            st.info(
                "QUEST thêm: "
                + " · ".join(f"**{t.vi}** / {t.en} ({t.why})" for t in terms)
            )
            query_vi, query_en = expanded_queries(query_vi, query_en or None, terms)
        else:
            st.caption("QUEST: không có gì để gọi tên — tìm bằng nguyên câu.")

    # --- truy hồi -----------------------------------------------------------
    runs: dict[str, list[Candidate]] = {}
    timings: dict[str, float] = {}
    for arm in (*dense_wanted, *TEXT_ARMS, "subtitle_exact"):
        if not arms_on.get(arm):
            continue
        started = time.perf_counter()
        if arm in DENSE_ARMS:
            if arm not in bundle.vectors:
                st.warning(f"{arm}: {bundle.missing.get(arm, 'không nạp được')}")
                continue
            if not query_en:
                st.warning(f"{arm}: cần bản dịch tiếng Anh — arm ngồi ngoài")
                continue
            scores = bundle.vectors[arm] @ encode_dense(arm, query_en)
            order = np.argsort(-scores)[:pool_size]
            runs[arm] = to_candidates([bundle.vector_keys[arm][i] for i in order], arm)
        else:
            source = "subtitle" if arm == "subtitle_exact" else arm
            if source not in bundle.texts:
                st.warning(f"{arm}: {bundle.missing.get(source, 'không có nguồn')}")
                continue
            keys = bundle.text_keys[source]
            if arm == "subtitle_exact":
                needle = _normalize_text(query_vi)
                hits = [k for k in keys
                        if needle in _normalize_text(bundle.texts[source][k])]
                runs[arm] = to_candidates(hits[:pool_size], arm)
            else:
                model = bundle.bm25.get(source) or bundle.build_bm25(source)
                scores = model.get_scores(_tokenize(query_vi))
                order = [i for i in np.argsort(-scores)[:pool_size] if scores[i] > 0]
                runs[arm] = to_candidates([keys[i] for i in order], arm)
        timings[arm] = time.perf_counter() - started

    if not runs:
        st.error("Không arm nào chạy được.")
        st.stop()

    import yaml

    weights_all = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["retrieval"][
        "fusion_weights"
    ]
    key_of = {"subtitle": "subtitle_partial"}
    weights = {a: weights_all.get(key_of.get(a, a), 1.0) for a in runs}
    fused = weighted_fusion(runs, weights=weights, run_id="lab",
                            normalizer="rrf", rrf_k=60)
    baseline = list(fused)

    st.caption(
        "arm: " + " · ".join(f"{a}({len(c)}, {timings[a]:.2f}s)" for a, c in runs.items())
        + f" → hợp nhất {len(fused)} ứng viên · trọng số {weights}"
    )

    # --- ảnh ----------------------------------------------------------------
    images = load_images()
    needs_pixels = use_ours or use_colpali or use_vlm
    if show_images or needs_pixels:
        depth_needed = max(rerank_depth if needs_pixels else 0, int(top_k))
        with st.spinner(f"Tải {depth_needed} ảnh …"):
            ready = images.prefetch([c.keyframe_id for c in fused[:depth_needed]])
        if ready < depth_needed:
            st.warning(f"Chỉ lấy được {ready}/{depth_needed} ảnh.")

    # --- rerank -------------------------------------------------------------
    outcomes = []
    if use_ours:
        # `OurRerankTier` chứ không phải một bản chuyển thể viết tại đây: bàn thử
        # này và `scripts/measure_rerank.py` phải chạy CÙNG MỘT tầng, nếu không
        # hai bên báo hai con số dưới cùng một cái tên.
        from mfe.online.rerank_tiers import OurRerankTier

        fused, outcome = OurRerankTier(
            load_our_reranker(images, our_budget_ms)
        ).rerank(fused, query_vi=query_vi, depth=rerank_depth)
        outcomes.append(outcome)

    tiers = []
    if use_colpali:
        tiers.append(ColPaliTier())
    if use_hf:
        tiers.append(HfCrossEncoderTier())
    if use_vlm:
        tiers.append(VlmTier())

    if tiers:
        def texts_of(keyframe_id: str) -> str:
            parts = [bundle.texts[s].get(keyframe_id, "") for s in bundle.texts]
            return " ".join(p for p in parts if p)

        with st.spinner("Chạy các tầng rerank …"):
            fused, tier_outcomes = apply_tiers(
                fused, tiers, query_vi=query_vi, query_en=query_en or None,
                images=images, texts=texts_of, depth=rerank_depth,
            )
        outcomes.extend(tier_outcomes)

    # --- báo cáo từng tầng ---------------------------------------------------
    if outcomes:
        st.subheader("Các tầng đã làm gì")
        for outcome in outcomes:
            if outcome.ran:
                st.success(outcome.summary())
                if outcome.moved:
                    ups = sorted(outcome.moved.items(), key=lambda kv: kv[1][1] - kv[1][0])
                    lines = [f"`{k}` {a} → **{b}**" for k, (a, b) in ups[:8]]
                    st.caption("dời nhiều nhất: " + " · ".join(lines))
            else:
                st.warning(outcome.summary())
    else:
        st.caption("Chưa bật tầng rerank nào — đang xem kết quả hợp nhất thô.")

    # --- kết quả -------------------------------------------------------------
    st.subheader(f"Top {int(top_k)}")
    rank_before = {c.keyframe_id: i for i, c in enumerate(baseline, start=1)}
    contributors: dict[str, list[str]] = {}
    for arm, candidate_list in runs.items():
        for candidate in candidate_list:
            contributors.setdefault(candidate.keyframe_id, []).append(arm)

    columns = st.columns(grid_cols)
    for position, candidate in enumerate(fused[: int(top_k)], start=1):
        slot = columns[(position - 1) % grid_cols]
        with slot:
            path = images.get(candidate.keyframe_id) if show_images else None
            if path:
                st.image(str(path), width="stretch")
            was = rank_before.get(candidate.keyframe_id)
            delta = ""
            if was and was != position:
                arrow = "▲" if was > position else "▼"
                delta = f"  {arrow}{abs(was - position)} (từ {was})"
            st.markdown(
                f"**{position}.** `{candidate.keyframe_id}`{delta}  \n"
                f"<span style='color:#888;font-size:12px'>"
                f"{','.join(contributors.get(candidate.keyframe_id, []))}</span>",
                unsafe_allow_html=True,
            )
