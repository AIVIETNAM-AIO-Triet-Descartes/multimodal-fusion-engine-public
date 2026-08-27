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

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.ids import build_keyframe_id, frame_idx_from_pts, pts_time_to_timestamp_ms
from mfe.contracts.records import KeyframeRow, VideoRow


class RelationalMap:
    """Owner: Nhật Bảo. Backend: sqlite3 inlined here for Sprint 1 (foreign_keys
    ON, executemany-in-transaction, Row factory). Extract into `backends/` when
    adding Postgres — see backends/README.md."""

    def __init__(self, db_path: str) -> None:
        """`check_same_thread=False` là BẮT BUỘC cho giao diện Streamlit.

        Streamlit chạy **mỗi lần vẽ lại trong một luồng khác**, còn
        `@st.cache_resource` thì giữ nguyên đối tượng qua các lần đó. Mặc định
        `sqlite3` cấm dùng một connection ngoài luồng đã tạo ra nó, nên lần vẽ
        đầu chạy được rồi mọi thao tác sau ném `ProgrammingError: SQLite objects
        created in a thread can only be used in that same thread` — đúng lúc
        người dùng bấm nút xuất file. Đã xảy ra thật ở `app/submit_lab.py`.

        An toàn ở đây vì mọi lối dùng đều **tuần tự**: Streamlit chạy một lần vẽ
        tại một thời điểm cho mỗi phiên, còn ingest thì một luồng. Nới cờ này
        KHÔNG cho phép ghi đồng thời; nếu sau này có hai luồng cùng ghi thì phải
        thêm khoá, chứ không phải gỡ cờ này ra.
        """
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")

    def close(self) -> None:
        """Close the SQLite connection. On Windows an open connection keeps a
        lock on the db file, so callers (and tests) must release it."""
        self._conn.close()

    def __enter__(self) -> RelationalMap:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

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
            INSERT INTO videos
                (video_id, source_path, fps, duration_ms, total_frames, width, height)
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
            INSERT INTO keyframes
                (keyframe_id, video_id, keyframe_index, frame_idx, pts_time, timestamp_ms, shot_id)
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

    def find_bracketing_keyframes(
        self, video_id: str, pts_time: float
    ) -> tuple[KeyframeRow | None, KeyframeRow | None]:
        """The official keyframes immediately before and after `pts_time`.

        The entry point for validating a keyframe the team extracted itself
        (DAKE or otherwise). `get_neighbors` cannot do this: it is keyed by
        `keyframe_index`, so it needs the keyframe to already be a row in this
        table — which a self-extracted one, by definition, is not.

        Boundaries are deliberate and asymmetric:

            prev = last official row with pts_time <= t
            next = first official row with pts_time >  t

        So an exact hit lands in `prev`, and a caller detects "this is already
        an official keyframe" by testing `prev.pts_time == pts_time`. Splitting
        it the other way would make an exact match look like it sat strictly
        between two rows, which is a different claim.

        Returns `(None, next)` before the first keyframe and `(prev, None)`
        after the last. Both ends are real situations, not errors — but a caller
        that treats a missing side as "nothing to check, therefore fine" is
        asserting something it did not verify.

        That distinction was audited over the whole batch-1 mapping: the
        interpolation-ratio check built on this function measured **0% false
        alarms and 98.1% of injected fps errors caught**. The 1.9% it misses are
        exactly the one-sided cases above — a keyframe at a video boundary has
        no bracket to sit inside, so there is nothing to interpolate. Callers
        must report that as *unverified*, never as *passed*.

        Uses two indexed single-row lookups (`idx_keyframes_video_pts`), not a
        scan: batch 1 alone is ~177k keyframes.
        """
        previous_row = self._conn.execute(
            """
            SELECT * FROM keyframes
            WHERE video_id = ? AND pts_time <= ?
            ORDER BY pts_time DESC LIMIT 1
            """,
            (video_id, pts_time),
        ).fetchone()
        next_row = self._conn.execute(
            """
            SELECT * FROM keyframes
            WHERE video_id = ? AND pts_time > ?
            ORDER BY pts_time ASC LIMIT 1
            """,
            (video_id, pts_time),
        ).fetchone()

        return (
            KeyframeRow(**dict(previous_row)) if previous_row is not None else None,
            KeyframeRow(**dict(next_row)) if next_row is not None else None,
        )

    def count_keyframes(self, video_id: str | None = None) -> int:
        if video_id is None:
            row = self._conn.execute("SELECT COUNT(*) as count FROM keyframes").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) as count FROM keyframes WHERE video_id = ?", (video_id,)
            ).fetchone()
        return row["count"]

    def sample_keyframe_ids(self, limit: int = 1) -> list[str]:
        """A few real keyframe_ids, for callers that need a known-good one.

        The eval harness uses this to build its "valid submit must NOT be
        blocked" probe (docs/00 §1) — a validator that blocked everything would
        otherwise pass a bad-submits-only test.
        """
        rows = self._conn.execute(
            "SELECT keyframe_id FROM keyframes ORDER BY keyframe_id LIMIT ?", (limit,)
        ).fetchall()
        return [row["keyframe_id"] for row in rows]

    def verify_mapping(self) -> list[str]:
        """Self-check the map. Returns a list of problems; empty means clean."""
        problems = []
        
        # Check if every keyframe_id agrees with its video_id and frame_idx.
        # Built through contracts.ids rather than an f-string here: spelling the
        # format a second time is how the two drift apart.
        rows = self._conn.execute(
            "SELECT keyframe_id, video_id, frame_idx FROM keyframes"
        ).fetchall()
        for r in rows:
            expected_id = build_keyframe_id(r["video_id"], r["frame_idx"])
            if r['keyframe_id'] != expected_id:
                problems.append(f"keyframe_id {r['keyframe_id']} mismatch: expected {expected_id}")
        
        # Check frame_idx against floor(pts_time * fps).
        #
        # The whole map can be internally consistent and still be wrong about
        # the one number that gets submitted. This is the check that catches an
        # ingest run that used the wrong rounding mode, or the exact NTSC
        # rational instead of the fps the dataset states — both of which put a
        # plausible, wrong frame on the scoreboard (contracts/ids.py).
        for row in self._conn.execute(
            """
            SELECT k.keyframe_id, k.frame_idx, k.pts_time, v.fps
            FROM keyframes k
            JOIN videos v ON k.video_id = v.video_id
            """
        ).fetchall():
            expected = frame_idx_from_pts(row["pts_time"], row["fps"])
            if row["frame_idx"] != expected:
                problems.append(
                    f"keyframe {row['keyframe_id']} frame_idx {row['frame_idx']} "
                    f"!= floor(pts_time {row['pts_time']} * fps {row['fps']}) = {expected}"
                )

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
            problems.append(
                f"keyframe {b['keyframe_id']} frame_idx {b['frame_idx']} "
                f"out of bounds (0, {b['total_frames']})"
            )
            
        # Check timestamp bounds: 0 <= timestamp_ms <= videos.duration_ms
        time_out_of_bounds = self._conn.execute("""
            SELECT k.keyframe_id, k.timestamp_ms, v.duration_ms
            FROM keyframes k
            JOIN videos v ON k.video_id = v.video_id
            WHERE k.timestamp_ms < 0 OR k.timestamp_ms > v.duration_ms
        """).fetchall()
        for t in time_out_of_bounds:
            problems.append(
                f"keyframe {t['keyframe_id']} timestamp {t['timestamp_ms']} "
                f"out of bounds (0, {t['duration_ms']})"
            )
            
        # Check timestamp_ms == pts_time_to_timestamp_ms(pts_time). Computed in
        # Python, NOT with SQL ROUND(): SQLite rounds half-away (0.5->1) while
        # Python round() is banker's (0.5->0), so a SQL check false-flags a
        # keyframe whose timestamp was computed correctly at ingest.
        for tm in self._conn.execute(
            "SELECT keyframe_id, pts_time, timestamp_ms FROM keyframes"
        ).fetchall():
            expected = pts_time_to_timestamp_ms(tm["pts_time"])
            if tm["timestamp_ms"] != expected:
                problems.append(
                    f"keyframe {tm['keyframe_id']} timestamp_ms {tm['timestamp_ms']} "
                    f"!= {expected} (= round(pts_time {tm['pts_time']} * 1000))"
                )
            
        return problems
