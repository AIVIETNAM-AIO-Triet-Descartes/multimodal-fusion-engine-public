"""Submit by frame number: {video_id, frame_number}. docs/05 §2."""

from __future__ import annotations

from typing import Any

from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.mapping.adapters.base import SubmitAdapter


class ByFrameAdapter(SubmitAdapter):
    """Owner: TV4. TODO(TV4): implement.

    Read `keyframe.frame_number` directly.

    TODO(TV4, Lead): if BTC scores by frame, confirm their extraction convention
    and FPS. docs/context/buoi-1.md §7: "nếu dùng frame thì cần FPS/config trích
    xuất". Our frame numbers index the source video (docs/00 §3); if BTC numbers
    frames within *their* keyframe set instead, every submission is off by the
    sampling stride and the mapping work has to be redone against their index.
    """

    format_name = "by_frame"

    def build(self, keyframe: KeyframeRow, video: VideoRow) -> dict[str, Any]:
        raise NotImplementedError("TODO(TV4): {video_id, frame_number}")
