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
    keyframe_id  TEXT PRIMARY KEY,          -- {video_id}#{frame_number:06d}
    video_id     TEXT    NOT NULL REFERENCES videos(video_id),
    frame_number INTEGER NOT NULL CHECK (frame_number >= 0),
    -- Both columns are stored even though either derives from the other via
    -- fps, because BTC has not fixed which the submission uses
    -- (TO_CONFIRM, docs/00 §3). Never re-derive one at submit time.
    timestamp_ms INTEGER NOT NULL CHECK (timestamp_ms >= 0),
    shot_id      TEXT                       -- optional in Sprint 1
);

-- The validator's "Trong biên" rule joins keyframes to videos on every submit
-- (docs/05 §3); without this the check degrades to a scan.
CREATE INDEX IF NOT EXISTS idx_keyframes_video_id ON keyframes(video_id);

-- Backs get_neighbors (docs/04 §2) and temporal grouping (docs/04 §3.2).
CREATE INDEX IF NOT EXISTS idx_keyframes_video_frame ON keyframes(video_id, frame_number);

CREATE INDEX IF NOT EXISTS idx_keyframes_shot_id ON keyframes(shot_id);

-- TODO(Nhật Bảo): a video_id + frame_number pair should be unique, and it is implied
-- by keyframe_id being the PK — but only if every writer builds the id through
-- mfe.contracts.ids. A UNIQUE(video_id, frame_number) constraint makes that
-- independent of writer discipline. Add it unless there is a reason not to.
