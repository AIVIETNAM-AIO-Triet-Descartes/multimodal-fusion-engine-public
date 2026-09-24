"""Load SOTA_AIC's text exports into our production text store.

Each of Bảo's JSON files is a flat ``{sota_keyframe_id: text}`` map. They become
`StorageRecord`s carrying a `source` tag, which is the field Công Quang's
Elasticsearch adapter keys documents by — one frame may legitimately hold an OCR,
a caption and a subtitle document at once (`stores/production/elasticsearch.py`).

**The two OCR files are kept apart on purpose.** Bảo's engine concatenates them
into one string. Measured on the pilot export, they are not the same quality:

    ocr_text.json     "IA 71? 06:30:11 giây"          <- easyocr, misreading the clock
    sviptr_ocr.json   "CHỐNG NGẬP CAO TỐC PHAN THIẾT" <- reads the banner correctly

Concatenating lets the noisy source dilute the useful one, and destroys any way to
measure which is carrying the weight. They ship as `ocr` and `ocr_sviptr`; the
fusion weights decide what each is worth (ke-hoach-merge-sota.md §3.4b).

**Two key grammars, both accepted** (ke-hoach-offline-ingest.md §1.2, §9). Bảo's
bundles key on the keyframe *ordinal*; the bundles our own notebooks produce key
on `keyframe_id`, because an ordinal is a property of the extraction run and not
of the video. Both are read here, told apart by structure rather than by guess —
`validate_video_id` forbids `#`, so a `#` can only be ours. See
`SotaIdMap.resolve_key`. Nothing about the ordinal path changed: it is a second
branch, not a rewrite, and `tests/unit/offline/test_sota_payloads.py` is what
holds that still.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.ids import parse_keyframe_id
from mfe.offline.adapters.sota_ids import SotaIdMap
from mfe.stores.production.contracts import StorageRecord

logger = logging.getLogger(__name__)

#: `source` tag -> flat filename under a bundle's `data/vectors/`. The tag must
#: satisfy `SOURCE_PATTERN` in stores/production/contracts.py:
#: `[a-z][a-z0-9_-]{0,31}`.
#:
#: This registry pairs with `bundle.ARTIFACTS`, and `_assert_registries_agree`
#: refuses a mismatch: a bundle writing `subtitle.json` while the reader looks
#: for `subtitles.json` produces an index short one arm and no error anywhere.
SOURCE_FILES: dict[str, str] = {
    "subtitle": "subtitles.json",
    "ocr": "ocr_text.json",
    "ocr_sviptr": "sviptr_ocr.json",
    "object": "objects_text.json",
    # `caption` KHONG con artifact nao cua ta sinh ra: `n6_recap` (ReCap qua
    # Gemini/Qwen2-VL) da bi xoa 25-08-2026, nhom doi han sang `caption_qwen`.
    #
    # DONG NAY VAN PHAI O LAI, va no khong phai dau vet sot. `dense_captions.json`
    # la file **Bao xuat ra** (xem `SOTA_SOURCES` ngay duoi va
    # `notebooks/reference/dense_captions_qwen.json`); go no di la mat kha nang
    # doc bundle cua Bao, chu khong phai don mot nhanh chet.
    #
    # Vi the `caption` la source DUY NHAT o day khong co `Artifact` doi ung —
    # `_assert_registries_agree` chi soi chieu THEO HUONG artifact -> source, nen
    # tinh trang do hop le va co chu y.
    "caption": "dense_captions.json",
    "ocr_paddle": "paddle_ocr.json",
    "ocr_vietocr": "viet_ocr.json",
    # Hai nua cua MOT lan sinh Qwen2.5-VL (`n6_caption_qwen` / `n6_ocr_qwen25`).
    "caption_qwen": "qwen_captions.json",
    "ocr_qwen25": "qwen_ocr.json",
    
    # Bundle BTC batch 1 and 2
    "window_bm25": "window_text.json",
    "caption_gemini": "gemini_captions.json",
    "ocr_gemini": "gemini_ocr.json",
}

#: The subset SOTA_AIC's own export ships. **Not the same set as `SOURCE_FILES`,
#: and that gap is the point**: `ocr_paddle` and `ocr_vietocr` are produced by our
#: own lượt 4 (`notebooks/btc/06-btc-paddle-vietocr.ipynb`) and Bảo's export has
#: never contained them.
#:
#: It exists so a test can say "every source Bảo ships must load" without that
#: claim silently becoming "no source may exist unless Bảo ships it" — which is
#: what asserting over `SOURCE_FILES` meant, and what would have blocked adding
#: any producer of our own.
SOTA_SOURCES: frozenset[str] = frozenset(
    {"subtitle", "ocr", "ocr_sviptr", "object", "caption"}
)


@dataclass
class LoadReport:
    """What one file produced, and what it silently would have dropped.

    Every number here exists because "the OCR arm returns nothing" must never be
    a mystery. A file that is present but yields zero records looks identical to
    a file that is absent, unless someone counted.
    """

    source: str
    path: str
    present: bool = True
    keys_read: int = 0
    records: int = 0
    unknown_ids: list[str] = field(default_factory=list)
    empty_texts: int = 0
    #: Keys that parse as neither grammar. Kept apart from `unknown_ids` because
    #: the two mean opposite things about who is at fault. An unknown id is
    #: **expected** — an export may cover videos we did not ingest. A malformed id
    #: is a **broken producer**: some notebook wrote a key no reader can resolve.
    #: With five members writing payloads, conflating them would hide the second
    #: behind the first, which is the one we learned to tolerate.
    malformed_ids: list[str] = field(default_factory=list)
    #: `"keyframe_id"`, `"ordinal"`, `"mixed"` or `""` when nothing was read.
    #: `"mixed"` means one file carries both grammars — which no correct producer
    #: does, so it means a run changed convention half way and the caller should
    #: refuse rather than pick a winner.
    key_format: str = ""

    def summary(self) -> str:
        if not self.present:
            return f"{self.source}: file not found ({self.path})"
        parts = [f"{self.source}: {self.records}/{self.keys_read} records"]
        if self.key_format:
            parts.append(f"khoa={self.key_format}")
        if self.unknown_ids:
            parts.append(f"{len(self.unknown_ids)} unknown id(s)")
        if self.malformed_ids:
            parts.append(f"{len(self.malformed_ids)} MALFORMED id(s)")
        if self.empty_texts:
            parts.append(f"{self.empty_texts} empty")
        return ", ".join(parts)


def load_payload(path: str | Path, source: str, id_map: SotaIdMap) -> tuple[
    list[StorageRecord], LoadReport
]:
    """Read one `{sota_id: text}` file into records tagged with `source`."""
    if source not in SOURCE_FILES:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(SOURCE_FILES)}")

    path = Path(path)
    report = LoadReport(source=source, path=str(path))
    if not path.is_file():
        report.present = False
        return [], report

    with path.open(encoding="utf-8") as handle:
        payload: dict[str, str] = json.load(handle)
    report.keys_read = len(payload)

    records: list[StorageRecord] = []
    grammars: set[str] = set()
    for key, text in payload.items():
        grammars.add("keyframe_id" if "#" in key else "ordinal")
        if not isinstance(text, str) or not text.strip():
            # An empty string is not a document. Elasticsearch would reject it
            # anyway (StorageRecord.validate_text), so drop it here and count it.
            report.empty_texts += 1
            continue
        try:
            keyframe_id = id_map.resolve_key(key)
        except ContractError as err:
            # Two outcomes, deliberately reported apart — see LoadReport.
            if err.code is ErrorCode.KEYFRAME_NOT_FOUND:
                report.unknown_ids.append(key)
            else:
                report.malformed_ids.append(key)
            continue
        video_id, frame_id = parse_keyframe_id(keyframe_id)
        records.append(
            StorageRecord.from_values(
                video_id=video_id, frame_id=frame_id, text=text.strip(), source=source
            )
        )

    report.records = len(records)
    if len(grammars) > 1:
        report.key_format = "mixed"
    elif grammars:
        report.key_format = next(iter(grammars))
    return records, report


def load_all(
    vectors_dir: str | Path,
    id_map: SotaIdMap,
    sources: list[str] | None = None,
) -> tuple[list[StorageRecord], list[LoadReport]]:
    """Load every configured source from SOTA_AIC's `data/vectors/`.

    A missing file is reported, not raised: four of the five files have no
    producer in Bảo's repo (ke-hoach-merge-sota.md §1b), so "absent" is the
    expected state for some of them until those scripts arrive.
    """
    vectors_dir = Path(vectors_dir)
    wanted = sources if sources is not None else list(SOURCE_FILES)

    all_records: list[StorageRecord] = []
    reports: list[LoadReport] = []
    for source in wanted:
        records, report = load_payload(vectors_dir / SOURCE_FILES[source], source, id_map)
        all_records.extend(records)
        reports.append(report)
        logger.info("%s", report.summary())

    return all_records, reports
