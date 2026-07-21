"""Relational Map — the source of truth. docs/05 §1.

Why mapping and the validator are one person's job (docs/05 preamble): the
relational store *is* the validator core. Split them across two people and the
schema drifts.

Why it is worth over-investing in during Sprint 1 despite the tiny sample: this
is the part that "generalize 100% sang data mới" — it is ID logic, with no
dependence on any pretrained model. Every hour spent here survives 2026's data;
much of the rest may not.
"""

from __future__ import annotations

from mfe.contracts.records import KeyframeRow, VideoRow


class RelationalMap:
    """Owner: Nhật Bảo. TODO(Nhật Bảo): implement. Backend: `backends/sqlite.py`."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def init_schema(self) -> None:
        """Create tables from `schema.sql`. Idempotent."""
        raise NotImplementedError("TODO(Nhật Bảo): execute schema.sql")

    # --- write side: called by Quốc Đạt's ingest -------------------------------

    def upsert_video(self, video: VideoRow) -> None:
        raise NotImplementedError("TODO(Nhật Bảo): insert/update videos row")

    def upsert_keyframes(self, rows: list[KeyframeRow]) -> int:
        """Bulk insert. TODO(Nhật Bảo): implement.

        Insert inside one transaction. A partial ingest leaves keyframes whose
        video row is missing, and the validator then blocks perfectly good
        answers with VIDEO_NOT_FOUND during a timed run.
        """
        raise NotImplementedError("TODO(Nhật Bảo): bulk insert keyframes")

    # --- read side: validator only, NOT retrieval -------------------------
    #
    # docs/00 §2: "retrieval không query thẳng vào Relational Map". If Phạm Trọng wants
    # to call something here, that is a design smell worth raising with the Vỹ Triết
    # rather than quietly satisfying — except get_neighbors, see below.

    def get_keyframe(self, keyframe_id: str) -> KeyframeRow:
        """Look up one keyframe. TODO(Nhật Bảo): implement.

        Raise ContractError(KEYFRAME_NOT_FOUND) on a miss. docs/05 §5:
        "keyframe_id không tồn tại -> báo lỗi, không đoán".
        """
        raise NotImplementedError("TODO(Nhật Bảo): keyframe_id -> KeyframeRow")

    def get_video(self, video_id: str) -> VideoRow:
        """TODO(Nhật Bảo): implement. Raise ContractError(VIDEO_NOT_FOUND) on a miss."""
        raise NotImplementedError("TODO(Nhật Bảo): video_id -> VideoRow")

    def get_neighbors(self, keyframe_id: str, window: int) -> list[KeyframeRow]:
        """Keyframes temporally adjacent to `keyframe_id`.

        TODO(Nhật Bảo): implement.

        The one read path retrieval legitimately needs (it backs the
        `get_neighbors` tool, docs/04 §2). It is a narrow exception to the
        no-retrieval-reads rule: it returns context, not verification. Temporal
        checking is the third-hardest thing in docs/context/buoi-1.md §8 — "một
        frame giống mô tả chưa chắc là đúng sự kiện".
        """
        raise NotImplementedError("TODO(Nhật Bảo): neighbors within window")

    def count_keyframes(self, video_id: str | None = None) -> int:
        raise NotImplementedError("TODO(Nhật Bảo)")

    def verify_mapping(self) -> list[str]:
        """Self-check the map. Returns a list of problems; empty means clean.

        TODO(Nhật Bảo): implement. docs/00 §1 requires mapping to be 100% correct on
        the sample, and docs/05 §5 makes it a handover criterion — so it needs
        to be checkable by running something, not by reading rows.

        Worth checking:
          - every keyframe_id parses, and agrees with its own video_id/keyframe_index
          - every keyframes.video_id resolves to a videos row
          - 0 <= frame_idx < videos.total_frames
          - 0 <= timestamp_ms <= videos.duration_ms
          - timestamp_ms == round(pts_time * 1000), i.e. the one derived column
            agrees with its source. Do NOT check frame_idx against
            round(pts_time * fps) — that formula is wrong on this dataset
            (docs/00 §3, adr/0002); frame_idx is authoritative from the CSV.
        """
        raise NotImplementedError("TODO(Nhật Bảo): return list of mapping violations")
