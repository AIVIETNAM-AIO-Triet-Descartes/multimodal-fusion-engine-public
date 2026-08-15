"""Stage-A arms backed by Công Quang's Elasticsearch text store.

One arm per `source` tag, so `ocr`, `ocr_sviptr`, `subtitle`, `object` and
`caption` are weighted and toggled independently rather than merged into one
"text" signal. The store already discriminates by `source`
(`stores/production/elasticsearch.py`); this is the online half of that.

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

#: `arm name -> (source, phrase)`. Order is display order in the browser.
TEXT_ARM_SPECS: dict[str, tuple[str, bool]] = {
    "subtitle_exact": ("subtitle", True),
    "subtitle_partial": ("subtitle", False),
    "ocr_sviptr": ("ocr_sviptr", False),
    "ocr": ("ocr", False),
    "object": ("object", False),
    "caption": ("caption", False),
}

_DESCRIPTIONS = {
    "subtitle_exact": "Khớp nguyên văn cụm trong phụ đề (match_phrase)",
    "subtitle_partial": "Chồng lấn từ khoá trong phụ đề (match)",
    "ocr_sviptr": "OCR SVIPTR — đọc băng rôn, im lặng khi không có chữ",
    "ocr": "OCR easyocr — phủ mọi frame, lẫn nhiễu từ đồng hồ",
    "object": "Nhãn vật thể phát hiện được",
    "caption": "Mô tả ảnh sinh tự động (TIẾNG ANH — truy vấn tiếng Việt sẽ trượt)",
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

    for arm_name, (source, phrase) in TEXT_ARM_SPECS.items():
        if text_store is None:
            arms.append(
                Arm(
                    name=arm_name,
                    weight=float(weights.get(arm_name, 1.0)),
                    unavailable_reason="Elasticsearch text store not configured",
                    description=_DESCRIPTIONS[arm_name],
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
            )
        )

    return arms
