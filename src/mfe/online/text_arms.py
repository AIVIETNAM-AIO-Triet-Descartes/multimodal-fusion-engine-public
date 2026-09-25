"""Stage-A arms backed by Công Quang's Elasticsearch text store.

One arm per `source` tag, so `ocr`, `ocr_sviptr`, `subtitle`, `object` and
`caption` are weighted and toggled independently rather than merged into one
"text" signal. The store already discriminates by `source`
(`stores/production/elasticsearch.py`); this is the online half of that.

**Four OCR arms, and that is a measurement rather than indecision.** `ocr`
(easyocr), `ocr_sviptr` (Qwen2-VL — the name is legacy, see notebooks/README.md),
`ocr_paddle` and `ocr_vietocr` read the same frames and fail in different places.
Measured on 40 shared keyframes of `L21_V001`, with `ocr_paddle` and `ocr_vietocr`
sharing one detector so only the recogniser differs: VietOCR keeps 20.6% of its
letters accented and transcribes Vietnamese headlines almost exactly, but turns
the station logo into Vietnamese words; PaddleOCR keeps 10.7%, reads the logo and
clock cleanly, and drops diacritics systematically. Fusion adds them; it does not
choose between them (ke-hoach-merge-sota.md §3.4b, handoff §8.6).

That 40-keyframe table justifies *generating* both. It does **not** set their
weights — those come from hill-climbing on dev and are published on held-out.

`subtitle` ships as **two** arms over the same source, because exact-phrase and
token-overlap are different evidence and lumping them produced the score
discontinuity described in ke-hoach-merge-sota.md §3.4b: Bảo's engine scores an
exact substring hit 100.0 against [0.05, 1.0] for everything else, which is a
hard filter wearing a score's clothes. Elasticsearch gives both for free —
`match_phrase` and `match` — so the split costs nothing and keeps each arm's
score scale internally coherent.
"""

from __future__ import annotations

import logging

from mfe.contracts.ids import build_keyframe_id
from mfe.contracts.tools import Candidate
from mfe.online.arms import Arm
from mfe.online.tools.evidence import create_evidence_id

logger = logging.getLogger(__name__)

#: Elasticsearch caps `size`; the store itself refuses top_k > 500.
_MAX_TOP_K = 500

#: `arm name -> (source, phrase, query_language)`. Order is display order.
#:
#: ⚠ Cột thứ BA thêm 21-09-2026, và nó sửa một lỗi im lặng có trọng số 1,0.
#:
#: Trước đó bảng này chỉ có `(source, phrase)`, nên **không có cách nào khai ngôn
#: ngữ** và mọi arm văn bản nhận mặc định `"vi"` của dataclass `Arm`. Nhưng
#: `caption_qwen` đọc `qwen_captions.json` — nửa `[CAPTION]`, **viết tiếng Anh**
#: theo đúng quyết định 25-08. Nên một arm chỉ mục tiếng Anh đã bị tìm bằng truy
#: vấn tiếng Việt suốt từ đó.
#:
#: Người viết BIẾT điều này: `_DESCRIPTIONS["caption_qwen"]` ghi thẳng "cần bản
#: dịch truy vấn", và CLAUDE.md §4 ghi arm này "dùng truy vấn đã dịch". Cả hai
#: mô tả một hành vi mà cơ chế khai báo không cho phép nói ra.
#:
#: Đây đúng lớp lỗi §5 đã trả giá: "dịch một chuỗi rồi tìm bằng chuỗi khác" —
#: lần trước kéo `siglip` từ 34/42 xuống 31/42.
TEXT_ARM_SPECS: dict[str, tuple[str, bool, str]] = {
    # Phụ đề là tiếng Việt -> truy vấn tiếng Việt. Đúng sẵn.
    "subtitle_exact": ("subtitle", True, "vi"),
    "subtitle_partial": ("subtitle", False, "vi"),
    # OCR chép chữ trên màn hình, phần lớn tiếng Việt, và prompt CẤM dịch nửa này.
    "ocr": ("ocr", False, "vi"),
    "ocr_paddle": ("ocr_paddle", False, "vi"),
    "ocr_vietocr": ("ocr_vietocr", False, "vi"),
    "ocr_qwen25": ("ocr_qwen25", False, "vi"),
    # Nhãn vật thể đã dịch sang tiếng Việt lúc dựng chỉ mục (`labels_vi`).
    "object": ("object", False, "vi"),
    # Hai arm caption: chỉ mục TIẾNG ANH, nên phải ăn bản dịch.
    "caption": ("caption", False, "en"),
    "caption_qwen": ("caption_qwen", False, "en"),
    "caption_gemini": ("caption_gemini", False, "en"),
    "ocr_gemini": ("ocr_gemini", False, "vi"),
    # Window prose is generated in English.  It is expanded back to real
    # keyframe ids by the flat backend before fusion.
    "window_bm25": ("window_bm25", False, "en"),
}

_DESCRIPTIONS = {
    "subtitle_exact": "Khớp nguyên văn cụm trong phụ đề (match_phrase)",
    "subtitle_partial": "Chồng lấn từ khoá trong phụ đề (match)",
    "ocr": "OCR easyocr — phủ mọi frame, lẫn nhiễu từ đồng hồ",
    "ocr_paddle": "OCR PaddleOCR — logo/đồng hồ sạch, RỤNG DẤU tiếng Việt",
    "ocr_vietocr": "OCR VietOCR trên vùng chữ của PaddleOCR — giữ dấu, lẫn ở logo",
    "object": "Nhãn vật thể phát hiện được",
    "caption": "Mô tả ảnh sinh tự động (TIẾNG ANH — truy vấn tiếng Việt sẽ trượt)",
    # Cùng cảnh báo ngôn ngữ như `caption`: prompt của Bảo yêu cầu mô tả bằng
    # TIẾNG ANH, nên arm này chỉ khớp khi truy vấn đã được dịch. Nửa `[TEXT]` thì
    # ngược lại — nó chép chữ trên màn hình, phần lớn là tiếng Việt.
    "caption_qwen": "Mô tả ảnh bởi Qwen2.5-VL (TIẾNG ANH — cần bản dịch truy vấn)",
    "caption_gemini": "Mô tả ảnh Gemini mới nhất (TIẾNG ANH — cần bản dịch truy vấn)",
    "ocr_gemini": "OCR Gemini mới nhất trên batch 2",
    "ocr_qwen25": "Chữ trên màn hình do Qwen2.5-VL chép, cùng lượt sinh với caption",
    "window_bm25": "Bản ghi cửa sổ 15 giây (TIẾNG ANH — cần bản dịch truy vấn)",
}


def _to_candidates(hits, arm_name: str, run_id: str) -> list[Candidate]:
    """`StorageHit` -> `Candidate`.

    Pure formatting, no lookup: since `keyframe_id` became
    `{video_id}#{frame_id:06d}`, the store's logical key and our id are two
    spellings of the same pair. Before that change this step had to consult the
    Relational Map on every hit of every arm.
    """
    candidates: list[Candidate] = []
    for rank, hit in enumerate(hits, start=1):
        candidates.append(
            Candidate(
                keyframe_id=build_keyframe_id(hit.video_id, hit.frame_id),
                video_id=hit.video_id,
                score=float(hit.score),
                evidence_id=create_evidence_id(arm_name, run_id, rank),
            )
        )
    return candidates


def build_text_arms(
    text_store,
    weights: dict[str, float] | None = None,
    enabled: dict[str, bool] | None = None,
) -> list[Arm]:
    """Build one `Arm` per entry of `TEXT_ARM_SPECS`.

    `text_store` may be None — the arms are still registered, marked unavailable
    with a reason, so the browser shows the full switch list and says why a row
    is dark instead of quietly omitting it.
    """
    weights = weights or {}
    enabled = enabled or {}
    arms: list[Arm] = []

    for arm_name, (source, phrase, lang) in TEXT_ARM_SPECS.items():
        if text_store is None:
            arms.append(
                Arm(
                    name=arm_name,
                    weight=float(weights.get(arm_name, 1.0)),
                    unavailable_reason="Elasticsearch text store not configured",
                    description=_DESCRIPTIONS[arm_name],
                    query_language=lang,
                )
            )
            continue

        def search(query, top_k, run_id, _source=source, _phrase=phrase, _name=arm_name):
            hits = text_store.search(
                query, top_k=min(top_k, _MAX_TOP_K), source=_source, phrase=_phrase
            )
            return _to_candidates(hits, _name, run_id)

        arms.append(
            Arm(
                name=arm_name,
                search=search,
                weight=float(weights.get(arm_name, 1.0)),
                enabled=bool(enabled.get(arm_name, True)),
                description=_DESCRIPTIONS[arm_name],
                query_language=lang,
            )
        )

    return arms
