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

import sqlite3
from pathlib import Path

from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.contracts.errors import ContractError, ErrorCode


class RelationalMap:
    """Owner: Nhật Bảo. Backend: `backends/sqlite.py` or inline sqlite3."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")

    def init_schema(self) -> None:
        """Create tables from `schema.sql`. Idempotent."""
        schema_path = Path(__file__).parent / "schema.sql"
        schema_sql = schema_path.read_text(encoding="utf-8")
        self._conn.executescript(schema_sql)
        self._conn.commit()

    # --- write side: called by Quốc Đạt's ingest -------------------------------

    def upsert_video(self, video: VideoRow) -> None:
        """Insert or replace a video row."""
        sql = """
            INSERT INTO videos (video_id, source_path, fps, duration_ms, total_frames, width, height)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                source_path=excluded.source_path,
                fps=excluded.fps,
                duration_ms=excluded.duration_ms,
                total_frames=excluded.total_frames,
                width=excluded.width,
                height=excluded.height
        """
        self._conn.execute(
            sql,
            (
                video.video_id,
                video.source_path,
                video.fps,
                video.duration_ms,
                video.total_frames,
                video.width,
                video.height,
            ),
        )
        self._conn.commit()

    def upsert_keyframes(self, rows: list[KeyframeRow]) -> int:
        """Bulk insert. Insert inside one transaction."""
        sql = """
            INSERT INTO keyframes (keyframe_id, video_id, keyframe_index, frame_idx, pts_time, timestamp_ms, shot_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(keyframe_id) DO UPDATE SET
                video_id=excluded.video_id,
                keyframe_index=excluded.keyframe_index,
                frame_idx=excluded.frame_idx,
                pts_time=excluded.pts_time,
                timestamp_ms=excluded.timestamp_ms,
                shot_id=excluded.shot_id
        """
        with self._conn:
            self._conn.executemany(
                sql,
                [
                    (
                        r.keyframe_id,
                        r.video_id,
                        r.keyframe_index,
                        r.frame_idx,
                        r.pts_time,
                        r.timestamp_ms,
                        r.shot_id,
                    )
                    for r in rows
                ],
            )
        return len(rows)

    # --- read side: validator only, NOT retrieval -------------------------

    def get_keyframe(self, keyframe_id: str) -> KeyframeRow:
        """Look up one keyframe. Raise ContractError(KEYFRAME_NOT_FOUND) on a miss."""
        row = self._conn.execute(
            "SELECT * FROM keyframes WHERE keyframe_id = ?", (keyframe_id,)
        ).fetchone()
        if row is None:
            raise ContractError(ErrorCode.KEYFRAME_NOT_FOUND, f"Keyframe {keyframe_id} not found")
        return KeyframeRow(**dict(row))

    def get_video(self, video_id: str) -> VideoRow:
        """Raise ContractError(VIDEO_NOT_FOUND) on a miss."""
        row = self._conn.execute(
            "SELECT * FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        if row is None:
            raise ContractError(ErrorCode.VIDEO_NOT_FOUND, f"Video {video_id} not found")
        return VideoRow(**dict(row))

    def get_neighbors(self, keyframe_id: str, window: int) -> list[KeyframeRow]:
        """Keyframes temporally adjacent to `keyframe_id`."""
        try:
            center = self.get_keyframe(keyframe_id)
        except ContractError:
            return []

        sql = """
            SELECT * FROM keyframes
            WHERE video_id = ? AND keyframe_index >= ? AND keyframe_index <= ?
            ORDER BY keyframe_index ASC
        """
        rows = self._conn.execute(
            sql, (center.video_id, center.keyframe_index - window, center.keyframe_index + window)
        ).fetchall()
        
        return [KeyframeRow(**dict(r)) for r in rows]

    def count_keyframes(self, video_id: str | None = None) -> int:
        if video_id is None:
            row = self._conn.execute("SELECT COUNT(*) as count FROM keyframes").fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) as count FROM keyframes WHERE video_id = ?", (video_id,)).fetchone()
        return row["count"]

    def verify_mapping(self) -> list[str]:
        """Self-check the map. Returns a list of problems; empty means clean."""
        problems = []
        
        # Check if every keyframe_id agrees with its video_id and index
        rows = self._conn.execute("SELECT keyframe_id, video_id, keyframe_index FROM keyframes").fetchall()
        for r in rows:
            expected_id = f"{r['video_id']}#{r['keyframe_index']:03d}"
            if r['keyframe_id'] != expected_id:
                problems.append(f"keyframe_id {r['keyframe_id']} mismatch: expected {expected_id}")
        
        # Check orphaned keyframes (no video)
        orphans = self._conn.execute("""
            SELECT k.keyframe_id 
            FROM keyframes k 
            LEFT JOIN videos v ON k.video_id = v.video_id 
            WHERE v.video_id IS NULL
        """).fetchall()
        for o in orphans:
            problems.append(f"keyframe {o['keyframe_id']} refers to missing video")
            
        # Check bounds: 0 <= frame_idx < videos.total_frames
        out_of_bounds = self._conn.execute("""
            SELECT k.keyframe_id, k.frame_idx, v.total_frames
            FROM keyframes k
            JOIN videos v ON k.video_id = v.video_id
            WHERE k.frame_idx < 0 OR k.frame_idx >= v.total_frames
        """).fetchall()
        for b in out_of_bounds:
            problems.append(f"keyframe {b['keyframe_id']} frame_idx {b['frame_idx']} out of bounds (0, {b['total_frames']})")
            
        # Check timestamp bounds: 0 <= timestamp_ms <= videos.duration_ms
        time_out_of_bounds = self._conn.execute("""
            SELECT k.keyframe_id, k.timestamp_ms, v.duration_ms
            FROM keyframes k
            JOIN videos v ON k.video_id = v.video_id
            WHERE k.timestamp_ms < 0 OR k.timestamp_ms > v.duration_ms
        """).fetchall()
        for t in time_out_of_bounds:
            problems.append(f"keyframe {t['keyframe_id']} timestamp {t['timestamp_ms']} out of bounds (0, {t['duration_ms']})")
            
        # Check timestamp_ms == round(pts_time * 1000)
        time_mismatch = self._conn.execute("""
            SELECT keyframe_id, pts_time, timestamp_ms
            FROM keyframes
            WHERE timestamp_ms != ROUND(pts_time * 1000)
        """).fetchall()
        for tm in time_mismatch:
            problems.append(f"keyframe {tm['keyframe_id']} timestamp_ms {tm['timestamp_ms']} != round({tm['pts_time']} * 1000)")
            
        return problems
