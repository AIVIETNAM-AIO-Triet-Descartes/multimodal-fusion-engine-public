-- Relational Map — validator core. docs/05 §1.
--
-- The source of truth for ID <-> frame <-> timestamp. Retrieval never queries
-- this (docs/00 §2); only the validator does, at submit time.
--
-- Owner: Nhật Bảo. Backend: SQLite for Sprint 1, Postgres when it needs to scale
-- (docs/05 §1). Keep this file backend-neutral enough that the move is mostly
-- mechanical.

CREATE TABLE IF NOT EXISTS videos (
    video_id     TEXT PRIMARY KEY,
    source_path  TEXT    NOT NULL,          -- reopen the video to verify an answer
    fps          REAL    NOT NULL CHECK (fps > 0),   -- per video; never assumed
    duration_ms  INTEGER NOT NULL CHECK (duration_ms >= 0),
    total_frames INTEGER NOT NULL CHECK (total_frames >= 0),
    width        INTEGER NOT NULL CHECK (width > 0),
    height       INTEGER NOT NULL CHECK (height > 0)
);

CREATE TABLE IF NOT EXISTS keyframes (
    keyframe_id    TEXT PRIMARY KEY,          -- {video_id}#{frame_idx:06d}
    video_id       TEXT    NOT NULL REFERENCES videos(video_id),
    keyframe_index INTEGER NOT NULL CHECK (keyframe_index >= 0),  -- ordinal n, names the image file
    -- frame_idx and pts_time come straight from map-keyframes/{video_id}.csv:
    -- every catalogue row has them, so copying beats computing. The banned
    -- function is round(pts_time*fps), wrong on 83-93% of 29.97fps rows;
    -- floor(pts_time*fps) is 100% exact and is how a frame OUTSIDE the
    -- catalogue gets its id (CLAUDE.md §2). Both columns stored because BTC has
    -- not fixed which the submission uses (TO_CONFIRM).
    frame_idx      INTEGER NOT NULL CHECK (frame_idx >= 0),       -- real source-video frame
    pts_time       REAL    NOT NULL CHECK (pts_time >= 0),        -- seconds
    timestamp_ms   INTEGER NOT NULL CHECK (timestamp_ms >= 0),    -- pts_time * 1000
    shot_id        TEXT,                       -- optional in Sprint 1
    UNIQUE(video_id, keyframe_index),
    -- keyframe_id is built from (video_id, frame_idx), so this is the same
    -- constraint spelled a second way -- deliberately. 614 CSV row pairs in
    -- batch 1 share a frame_idx (two images less than one frame apart); ingest
    -- collapses them. If a future extractor stops collapsing, this fails the
    -- insert instead of letting one keyframe silently overwrite another.
    UNIQUE(video_id, frame_idx)
);

-- The validator's "Trong biên" rule joins keyframes to videos on every submit
-- (docs/05 §3); without this the check degrades to a scan.
CREATE INDEX IF NOT EXISTS idx_keyframes_video_id ON keyframes(video_id);

-- Backs get_neighbors (docs/04 §2) and temporal grouping (docs/04 §3.2).
CREATE INDEX IF NOT EXISTS idx_keyframes_video_index ON keyframes(video_id, keyframe_index);

-- Backs find_bracketing_keyframes(): given an arbitrary pts_time from a keyframe
-- the team extracted itself, find the official keyframes either side of it.
-- The index above cannot serve that query — it is keyed by ordinal, and a
-- self-extracted keyframe has no ordinal in this table. Without this index the
-- lookup degrades to a scan of every keyframe in the video, on a mapping that
-- is 177k rows for batch 1 alone.
CREATE INDEX IF NOT EXISTS idx_keyframes_video_pts ON keyframes(video_id, pts_time);

CREATE INDEX IF NOT EXISTS idx_keyframes_shot_id ON keyframes(shot_id);
