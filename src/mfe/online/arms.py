"""Stage-A arm registry — what the cascade fuses, and what the UI can toggle.

Stage A used to be two hardcoded `if` branches (`vector_search`, `search_text`)
plus a `_WEIGHT_KEY` dict translating tool names into config's signal names. That
shape stopped fitting the moment the merge plan called for eight arms
(docs/triet/ke-hoach-merge-sota.md §6.0 step 9), and the indirection was pure
overhead: an arm's name *is* its weight key.

One registry serves three needs that would otherwise be three mechanisms:

  * **toggling** — the browser's on/off switches set `enabled`
  * **per-arm scoring** — §3.4b B1 runs each arm alone by enabling exactly one
  * **plugging in** — a new arm is a registration, not an edit to `_coarse`

**`enabled` and `available` are different things, deliberately.** `enabled=False`
means a human turned it off. `available=False` means it cannot run — no store, no
model, no data. Collapsing them is exactly the bug in Bảo's UI: the ColPali switch
defaults to on while `search_engine.py` pins `has_colpali = False`, so the button
looks live, does nothing, and says nothing. An arm that cannot run must be able to
explain why.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from mfe.contracts.tools import Candidate

logger = logging.getLogger(__name__)

#: `(query_text, top_k, run_id) -> candidates`
SearchFn = Callable[[str, int, str], list[Candidate]]

#: Arm nhận `ocr_keywords` thay cho nguyên câu truy vấn — thứ `mfe.online.quest`
#: tách ra từ câu hỏi QA ("dòng chữ nào hiện trên màn hình").
#:
#: Danh sách này từng ghi `["bm25", "ocr", "subtitles", "objects", "blip"]` — ba
#: cái tên **không tồn tại**. Tên arm thật là `subtitle_partial`, `object`, và
#: không có arm nào tên `blip`. Nên nhánh này chỉ chạm tới `bm25` và `ocr`, tức
#: đúng hai arm không có dữ liệu trên đường phẳng: quest tách được từ khoá rồi
#: gửi chúng cho hai arm im lặng. Không có gì báo, vì `ocr_keywords` là tuỳ chọn
#: và một arm không nhận nó vẫn chạy bình thường với nguyên câu.
OCR_KEYWORD_ARMS = frozenset(
    {"bm25", "ocr", "ocr_paddle", "ocr_vietocr", "object", "subtitle_partial"}
)


@dataclass(frozen=True)
class Arm:
    """One named retrieval arm in stage A."""

    name: str
    search: SearchFn | None = None
    weight: float = 1.0
    enabled: bool = True
    #: Why this arm cannot run. Set whenever `search` is None; shown in the UI.
    unavailable_reason: str | None = None
    #: One line for the operator: what this arm actually matches on.
    description: str = ""
    #: Which language this arm's query must be in.
    #:
    #:   "vi"  the arm matches Vietnamese text (OCR, subtitle, Vietnamese labels)
    #:   "en"  the arm's model only understands English (SigLIP SO400M's text
    #:         tower, CLAP, PaliGemma/ColPali) — feeding it Vietnamese does not
    #:         error, it just retrieves badly, which is worse
    #:   "any" the model is cross-lingual (BAAI/bge-m3), so either works
    #:
    #: An "en" arm with no English query available goes unavailable with a reason
    #: rather than silently searching with the wrong language.
    query_language: str = "vi"

    @property
    def available(self) -> bool:
        return self.search is not None

    @property
    def active(self) -> bool:
        """Runs on the next query: switched on *and* able to run."""
        return self.enabled and self.available

    def status(self) -> str:
        if not self.available:
            return f"unavailable — {self.unavailable_reason or 'no reason recorded'}"
        return "on" if self.enabled else "off"


class ArmRegistry:
    """Ordered, name-addressed collection of arms.

    Order is preserved for display only; fusion is order-independent.
    """

    #: Số arm chạy SONG SONG trong `run()`. Đặt 1 để quay về tuần tự.
    #:
    #: Vì sao song song: đo 25-09 trên 8 câu dev, một truy vấn mất **7,66 s**
    #: trung vị, và **6,7 s trong đó là 8 arm BM25 chạy lần lượt** —
    #: `caption_qwen` 1.284 ms · `object` 1.208 ms · `ocr_paddle` 994 ms ·
    #: `subtitle_partial` 910 ms · `ocr_vietocr` 854 ms · `ocr` 772 ms ·
    #: `ocr_qwen25` 460 ms · `subtitle_exact` 242 ms. Bốn arm dense chỉ tốn
    #: 49–152 ms, nên tối ưu vector là tối ưu 4% sai chỗ.
    #:
    #: Các arm ĐỘC LẬP: mỗi arm đọc chỉ mục riêng, không arm nào thấy kết quả
    #: của arm khác, và `weighted_fusion` không phụ thuộc thứ tự. Nên chạy song
    #: song KHÔNG đổi kết quả một chút nào — chỉ đổi đồng hồ.
    #:
    #: ⚠ ĐO RỒI: song song chỉ được **10%** (7.663 → 6.904 ms), KHÔNG phải 3 lần.
    #:
    #: Vì `rank_bm25.get_scores` giữ GIL gần như toàn bộ — thân của nó là một
    #: vòng lặp PYTHON THUẦN trên 168.000 dict, cho MỖI từ của truy vấn:
    #:
    #:     q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
    #:
    #: `np.array(...)` chỉ gói kết quả; phần nặng là list comprehension, và nó
    #: không nhả GIL. Phần 10% thu được là của bốn arm dense (torch/numpy có
    #: nhả) cộng chút chồng lấn I/O.
    #:
    #: Vẫn giữ song song vì nó **đúng về mặt ngữ nghĩa** và sẽ trả cổ tức ngay
    #: khi BM25 được thay bằng ma trận thưa — lúc đó phần nặng nằm trong scipy
    #: và GIL không còn chặn. 760 ms cũng không phải không có gì.
    #:
    #: Luồng chứ không phải tiến trình: tiến trình con phải chép cả chỉ mục
    #: 168k tài liệu qua pickle cho mỗi lượt gọi, và trên Windows `spawn` còn
    #: phải nạp lại cả module.
    max_workers: int = 8

    def __init__(self, arms: list[Arm] | None = None, *, max_workers: int | None = None) -> None:
        self._arms: dict[str, Arm] = {}
        if max_workers is not None:
            self.max_workers = max_workers
        for arm in arms or []:
            self.register(arm)

    def register(self, arm: Arm) -> None:
        if arm.name in self._arms:
            raise ValueError(f"arm {arm.name!r} is already registered")
        if arm.search is None and arm.unavailable_reason is None:
            raise ValueError(
                f"arm {arm.name!r} has no search function and no unavailable_reason; "
                "an arm that cannot run must say why (see module docstring)"
            )
        self._arms[arm.name] = arm

    def __len__(self) -> int:
        return len(self._arms)

    def __iter__(self):
        return iter(self._arms.values())

    def __contains__(self, name: object) -> bool:
        return name in self._arms

    def get(self, name: str) -> Arm:
        if name not in self._arms:
            raise KeyError(f"no arm named {name!r}; have {sorted(self._arms)}")
        return self._arms[name]

    def active(self) -> list[Arm]:
        return [arm for arm in self._arms.values() if arm.active]

    def needs_english(self) -> bool:
        """Whether any arm about to run would consume an English query.

        Lets a caller skip translating — and skip loading a translation model
        that weighs more than a gigabyte — when nothing would read the result.
        Asked of the *active* arms, not all of them: an English arm whose
        collection never opened is not a reason to translate.
        """
        return any(arm.query_language == "en" for arm in self.active())

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Flip one switch. Enabling an unavailable arm is allowed and still
        will not run it — the UI shows the reason rather than silently lying."""
        self._arms[name] = replace(self.get(name), enabled=enabled)

    def set_weight(self, name: str, weight: float) -> None:
        self._arms[name] = replace(self.get(name), weight=float(weight))

    def only(self, name: str) -> None:
        """Enable exactly one arm. The per-arm measurement of §3.4b B1."""
        self.get(name)
        for arm_name in self._arms:
            self.set_enabled(arm_name, arm_name == name)

    def weights(self) -> dict[str, float]:
        return {arm.name: arm.weight for arm in self._arms.values()}

    def report(self) -> list[dict[str, object]]:
        """Rows for the UI and for `tiers_enabled` in an eval run (§3.6)."""
        return [
            {
                "name": arm.name,
                "status": arm.status(),
                "weight": arm.weight,
                "description": arm.description,
            }
            for arm in self._arms.values()
        ]

    def run_named(
        self,
        names: list[str],
        query_text: str,
        top_k: int,
        run_id: str,
        query_text_en: str | None = None,
        ocr_keywords: str | None = None,
    ) -> dict[str, list[Candidate]]:
        """`run()` restricted to specific arms, ignoring their `enabled` flags.

        For measuring one arm in isolation without mutating registry state.
        """
        saved = {arm.name: arm.enabled for arm in self}
        try:
            for name in saved:
                self.set_enabled(name, name in names)
            return self.run(
                query_text, top_k, run_id,
                query_text_en=query_text_en, ocr_keywords=ocr_keywords,
            )
        finally:
            for name, was in saved.items():
                self.set_enabled(name, was)

    def run(
        self,
        query_text: str,
        top_k: int,
        run_id: str,
        query_text_en: str | None = None,
        ocr_keywords: str | None = None,
    ) -> dict[str, list[Candidate]]:
        """Run every active arm. A failing arm is dropped with a warning, never
        raised: one broken arm must not take the whole query down.

        `query_text` is the Vietnamese query the user typed. `query_text_en` is
        the optional English form; arms declaring `query_language="en"` need it
        and are skipped (loudly) without it.
        """
        viec: list[tuple[Arm, str]] = []
        for arm in self.active():
            text = query_text
            if arm.query_language == "en":
                if not query_text_en:
                    logger.warning(
                        "arm %s needs an English query and none was supplied; skipping. "
                        "Add `text_en` to the query, or disable the arm.",
                        arm.name,
                    )
                    continue
                text = query_text_en
            elif ocr_keywords and arm.name in OCR_KEYWORD_ARMS:
                text = ocr_keywords
            viec.append((arm, text))

        if not viec:
            return {}

        def chay(cap: tuple[Arm, str]):
            arm, text = cap
            try:
                return arm.name, arm.search(text, top_k, run_id), None  # type: ignore[misc]
            except ValueError as err:
                # e.g. a query with no indexable term — a miss, not a crash.
                return arm.name, None, ("info", err)
            except Exception as err:  # noqa: BLE001 — degrade, do not abort the query
                return arm.name, None, ("warn", err)

        if len(viec) == 1 or self.max_workers <= 1:
            ket = [chay(c) for c in viec]
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(viec))) as pool:
                ket = list(pool.map(chay, viec))

        # Ghi log ở LUỒNG GỌI, không trong worker: thứ tự dòng log là thứ tự
        # arm, không phải thứ tự luồng nào về trước. Một bảng log xáo trộn giữa
        # các lượt chạy là thứ không đối chiếu được.
        results: dict[str, list[Candidate]] = {}
        for ten, hits, loi in ket:
            if loi is not None:
                muc, err = loi
                if muc == "info":
                    logger.info("arm %s returned nothing for %r (%s)", ten, query_text, err)
                else:
                    logger.warning("arm %s failed (%s); continuing without it", ten, err)
                continue
            if hits:
                results[ten] = hits
        return results
