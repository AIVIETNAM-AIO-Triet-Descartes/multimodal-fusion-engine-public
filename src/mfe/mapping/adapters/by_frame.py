"""Submit by frame: {video_id, frame_id}. docs/05 §2 — the format BTC scores."""

from __future__ import annotations

from typing import Any

from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.mapping.adapters.base import SubmitAdapter


class ByFrameAdapter(SubmitAdapter):
    """Owner: Nhật Bảo. The Sprint 1 default — `submit.schema_format: by_frame`.

    Emits BTC's field name, `frame_id`, not our internal `frame_idx`. The two
    hold the same number; only the name differs, and the rename happens here so
    it happens once. Everything upstream (Relational Map, ingest, the CSV) keeps
    the dataset's vocabulary; everything downstream of this line uses BTC's.

    Which integer is it? The **real source-video frame**, read straight off
    `keyframe.frame_idx`. This was open for a while — the alternative was
    `keyframe_index`, the ordinal `n` that names the image file — and the two
    are wildly different numbers for the same keyframe (`L23_V001#008` is
    ordinal 8 but frame 375). PDF §2.1.1 settles it: its worked example scores
    `[500, 510]` against a submitted `505`, a range of source frames, not of
    keyframe ordinals. `mfe.eval.metrics.kis` implements exactly that.

    Never recompute it from `pts_time` and `fps`. The number is copied from
    `map-keyframes/*.csv` at ingest; our own docs once carried three mutually
    inconsistent worked examples of that formula (adr/0002), and deriving at
    submit time is how that class of bug reaches a scoreboard.
    """

    format_name = "by_frame"

    def build(self, keyframe: KeyframeRow, video: VideoRow) -> dict[str, Any]:
        return {"video_id": video.video_id, "frame_id": keyframe.frame_idx}
