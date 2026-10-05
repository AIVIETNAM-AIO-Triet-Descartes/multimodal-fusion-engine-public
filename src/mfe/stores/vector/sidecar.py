"""Sidecar: `internal_id` <-> `keyframe_id` translation. docs/03 §1.

Faiss stores vectors, not identity. This table is the only thing that makes a
search result mean anything.

docs/03 §4 pins it as a handover criterion: "Sidecar dịch internal_id <->
keyframe_id chính xác 100%". Not 99% — an off-by-one here returns a real,
plausible, wrong keyframe for every query, and nothing about the result looks
broken.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_SIDECAR_FILENAME = "sidecar.parquet"
_SIDECAR_SCHEMA = pa.schema(
    [
        ("keyframe_id", pa.string()),
        ("video_id", pa.string()),
    ]
)


class Sidecar:
    """Row `i` describes Faiss `internal_id == i`.

    Owner: Công Quang. Backend: an in-memory list persisted as inspectable
    Parquet.
    """

    def __init__(self, store_path: str) -> None:
        self.store_path = store_path
        self._entries: list[dict[str, str]] = []
        self._keyframe_to_internal_id: dict[str, int] = {}

    def append(self, keyframe_id: str, video_id: str) -> int:
        """
        Append one entry, returning the `internal_id` it was assigned.
        """
        if keyframe_id in self._keyframe_to_internal_id:
            raise ValueError(f"Duplicate keyframe_id: {keyframe_id!r}")

        internal_id = len(self._entries)

        entry = {
            "keyframe_id": keyframe_id,
            "video_id": video_id,
        }

        self._entries.append(entry)
        self._keyframe_to_internal_id[keyframe_id] = internal_id

        return internal_id

    def to_keyframe_id(self, internal_id: int) -> str:
        """Translate a Faiss hit back to the global key.

        Raise on an unknown id — never return a guess or a placeholder. A wrong
        keyframe_id here becomes a wrong submission
        (docs/context/buoi-1.md §8: "tìm đúng hình nhưng submit sai ID/frame là
        lỗi trắng").
        """

        if not 0 <= internal_id < len(self._entries):
            raise IndexError(f"Unknown internal_id: {internal_id}")

        return self._entries[internal_id]["keyframe_id"]

    def to_internal_id(self, keyframe_id: str) -> int:
        try:
            return self._keyframe_to_internal_id[keyframe_id]
        except KeyError:
            raise KeyError(f"Unknown keyframe_id: {keyframe_id}") from None

    def video_id_of(self, internal_id: int) -> str:
        """Used to filter results by video without touching another store."""

        if not 0 <= internal_id < len(self._entries):
            raise IndexError(f"Unknown internal_id: {internal_id}")

        return self._entries[internal_id]["video_id"]

    def verify_alignment(self, faiss_size: int) -> None:
        sidecar_size = len(self)

        if sidecar_size != faiss_size:
            raise AssertionError(
                f"Sidecar size {sidecar_size} does not match "
                f"Faiss size {faiss_size}"
            )

    def save(self) -> None:
        """Persist entries in order so row number remains the internal ID."""
        store_dir = Path(self.store_path)
        store_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(self._entries, schema=_SIDECAR_SCHEMA)
        target = store_dir / _SIDECAR_FILENAME
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        pq.write_table(table, temporary)
        temporary.replace(target)

    @classmethod
    def load(cls, store_path: str) -> Sidecar:
        """Load entries while preserving their Parquet row order."""
        table = pq.read_table(Path(store_path) / _SIDECAR_FILENAME)
        sidecar = cls(store_path=store_path)
        for entry in table.to_pylist():
            sidecar.append(
                keyframe_id=entry["keyframe_id"],
                video_id=entry["video_id"],
            )
        return sidecar

    def __len__(self) -> int:
        return len(self._entries)
