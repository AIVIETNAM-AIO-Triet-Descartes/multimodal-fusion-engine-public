"""Sidecar: `internal_id` <-> `keyframe_id` translation. docs/03 §1.

Faiss stores vectors, not identity. This table is the only thing that makes a
search result mean anything.

docs/03 §4 pins it as a handover criterion: "Sidecar dịch internal_id <->
keyframe_id chính xác 100%". Not 99% — an off-by-one here returns a real,
plausible, wrong keyframe for every query, and nothing about the result looks
broken.
"""

from __future__ import annotations


class Sidecar:
    """Row `i` describes Faiss `internal_id == i`.

    Owner: TV2. TODO(TV2): implement. Backend: a parallel array, or parquet for
    something inspectable at scale.
    """

    def __init__(self, store_path: str) -> None:
        self.store_path = store_path

    def append(self, keyframe_id: str, video_id: str) -> int:
        """Append one entry, returning the `internal_id` it was assigned.

        TODO(TV2): implement. The returned id must be exactly the position Faiss
        will assign to the matching `add` — that coupling is the whole contract.
        """
        raise NotImplementedError("TODO(TV2): append -> internal_id")

    def to_keyframe_id(self, internal_id: int) -> str:
        """Translate a Faiss hit back to the global key.

        TODO(TV2): implement. Raise on an unknown id — never return a guess or
        a placeholder. A wrong keyframe_id here becomes a wrong submission
        (docs/context/buoi-1.md §8: "tìm đúng hình nhưng submit sai ID/frame là
        lỗi trắng").
        """
        raise NotImplementedError("TODO(TV2): internal_id -> keyframe_id")

    def to_internal_id(self, keyframe_id: str) -> int:
        """Reverse lookup — needed by `search_image` when the query is a
        keyframe already in the index (docs/04 §2)."""
        raise NotImplementedError("TODO(TV2): keyframe_id -> internal_id")

    def video_id_of(self, internal_id: int) -> str:
        """Used to filter results by video without touching another store."""
        raise NotImplementedError("TODO(TV2): internal_id -> video_id")

    def verify_alignment(self, faiss_size: int) -> None:
        """Assert the sidecar and the Faiss index are the same length.

        TODO(TV2): implement, and call it on every `load`. This is the cheap
        check that catches a half-finished ingest before it silently corrupts a
        competition run.
        """
        raise NotImplementedError("TODO(TV2): assert len(self) == faiss_size")

    def __len__(self) -> int:
        raise NotImplementedError("TODO(TV2)")
