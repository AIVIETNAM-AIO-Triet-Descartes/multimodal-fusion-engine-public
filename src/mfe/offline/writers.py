"""Write records into the 4 stores — step [4] of docs/02 §2.

The seam between Quốc Đạt and everyone else. docs/01 §1 lists Quốc Đạt<->Công Quang as a sync
point whose failure mode is "Công Quang đọc sai, phải chạy lại offline" — re-running
offline is the expensive kind of mistake, so validate here rather than trusting.

Mapping of step -> store (docs/02 §4):

    keyframe image      -> Asset Store      (AssetRecord)
    keyframe coordinates-> Relational Map   (KeyframeRow)  [Nhật Bảo owns the store]
    CLIP embedding      -> Vector Index     (VectorRecord)
    object/OCR/metadata -> Inverted Index   (InvertedRecord)
"""

from __future__ import annotations

from collections.abc import Iterable

from mfe.contracts.records import (
    AssetRecord,
    InvertedRecord,
    KeyframeRow,
    VectorRecord,
    VideoRow,
)


def write_videos(rows: Iterable[VideoRow]) -> int:
    """Write the `videos` table of the Relational Map (docs/05 §1.1).

    Owner: Quốc Đạt (writes) / Nhật Bảo (owns the schema). TODO(Quốc Đạt): implement.
    Must land before any KeyframeRow: keyframes carry a foreign key to it.
    """
    raise NotImplementedError("TODO(Quốc Đạt): write VideoRow -> Relational Map")


def write_keyframes(rows: Iterable[KeyframeRow]) -> int:
    """Write the `keyframes` table of the Relational Map (docs/05 §1.2).

    Owner: Quốc Đạt (writes) / Nhật Bảo (owns the schema). TODO(Quốc Đạt): implement.
    This is the source of truth for every submission — a wrong row here is a
    wrong answer even when retrieval was perfect (docs/context/buoi-1.md §8).
    """
    raise NotImplementedError("TODO(Quốc Đạt): write KeyframeRow -> Relational Map")


def write_vectors(records: Iterable[VectorRecord]) -> int:
    """Write to the Vector Index + its sidecar (docs/03 §1).

    Owner: Quốc Đạt (writes) / Công Quang (owns the store). TODO(Quốc Đạt): implement.
    `internal_id` is Faiss's own position and must match the sidecar row order
    exactly; an off-by-one translates every search hit to the wrong keyframe.
    """
    raise NotImplementedError("TODO(Quốc Đạt): write VectorRecord -> Vector Index")


def write_inverted(records: Iterable[InvertedRecord]) -> int:
    """Write to the Inverted Index (docs/03 §2).

    Owner: Quốc Đạt (writes) / Công Quang (owns the store). TODO(Quốc Đạt): implement.
    Emit a row per keyframe even when every text field is empty — an absent row
    and a row with no text mean different things to a filter.
    """
    raise NotImplementedError("TODO(Quốc Đạt): write InvertedRecord -> Inverted Index")


def write_assets(records: Iterable[AssetRecord]) -> int:
    """Write the Asset Store manifest (docs/03 §3).

    Owner: Quốc Đạt (writes) / Công Quang (owns the store). TODO(Quốc Đạt): implement.
    Bytes go through `mfe.core.storage`, not `open()` — see core/README.md.
    """
    raise NotImplementedError("TODO(Quốc Đạt): write AssetRecord -> Asset Store manifest")


def verify_stores_agree() -> None:
    """Cross-check that all 4 stores cover the same keyframe_id set.

    Owner: Quốc Đạt. TODO(Quốc Đạt): implement.

    Cheap here, expensive later. docs/00 §1 requires mapping to be 100% correct
    on the sample, and docs/02 §6 requires ids to match across all four stores.
    A keyframe present in the Vector Index but missing from the Relational Map
    is a candidate the validator will refuse at submit time — better to catch it
    at ingest than during a timed run.
    """
    raise NotImplementedError("TODO(Quốc Đạt): assert keyframe_id sets match across 4 stores")
