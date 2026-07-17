"""Submit by frame range: {video_id, frame_start, frame_end}. docs/05 §2."""

from __future__ import annotations

from typing import Any

from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.mapping.adapters.base import SubmitAdapter


class ByFrameRangeAdapter(SubmitAdapter):
    """Expand a window around the keyframe. Owner: TV4. TODO(TV4): implement.

    Why this format is plausible at all: docs/context/buoi-1.md §7 records the
    lecturer explaining BTC may treat a *contiguous range* of frames as correct
    — a frame inside the range scores, one outside does not.

    - Window width comes from `submit.frame_range_window` in config, not a
      literal here.
    - Clamp to `[0, video.total_frames - 1]`. An unclamped window around frame 5
      submits a negative start, and the validator's "Trong biên" rule then
      blocks an answer that was actually right.
    """

    format_name = "by_frame_range"

    def build(self, keyframe: KeyframeRow, video: VideoRow) -> dict[str, Any]:
        raise NotImplementedError("TODO(TV4): {video_id, frame_start, frame_end}, clamped")
