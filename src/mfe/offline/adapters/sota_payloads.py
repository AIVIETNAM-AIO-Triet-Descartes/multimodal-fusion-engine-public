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
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from mfe.contracts.ids import parse_keyframe_id
from mfe.offline.adapters.sota_ids import SotaIdMap
from mfe.stores.production.contracts import StorageRecord

logger = logging.getLogger(__name__)

#: `source` tag -> filename in SOTA_AIC's `data/vectors/`. The tag must satisfy
#: `SOURCE_PATTERN` in stores/production/contracts.py: [a-z][a-z0-9_-]{0,31}.
SOURCE_FILES: dict[str, str] = {
    "subtitle": "subtitles.json",
    "ocr": "ocr_text.json",
    "ocr_sviptr": "sviptr_ocr.json",
    "object": "objects_text.json",
    "caption": "dense_captions.json",
}


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

    def summary(self) -> str:
        if not self.present:
            return f"{self.source}: file not found ({self.path})"
        parts = [f"{self.source}: {self.records}/{self.keys_read} records"]
        if self.unknown_ids:
            parts.append(f"{len(self.unknown_ids)} unknown id(s)")
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
    for sota_id, text in payload.items():
        if not isinstance(text, str) or not text.strip():
            # An empty string is not a document. Elasticsearch would reject it
            # anyway (StorageRecord.validate_text), so drop it here and count it.
            report.empty_texts += 1
            continue
        try:
            keyframe_id = id_map.to_keyframe_id(sota_id)
        except Exception:  # noqa: BLE001 — an id we did not ingest is expected
            report.unknown_ids.append(sota_id)
            continue
        video_id, frame_id = parse_keyframe_id(keyframe_id)
        records.append(
            StorageRecord.from_values(
                video_id=video_id, frame_id=frame_id, text=text.strip(), source=source
            )
        )

    report.records = len(records)
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
