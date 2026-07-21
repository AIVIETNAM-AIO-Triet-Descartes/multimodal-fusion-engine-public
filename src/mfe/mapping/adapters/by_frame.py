"""Submit by frame: {video_id, frame_idx}. docs/05 §2."""

from __future__ import annotations

from typing import Any

from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.mapping.adapters.base import SubmitAdapter


class ByFrameAdapter(SubmitAdapter):
    """Owner: Nhật Bảo. TODO(Nhật Bảo): implement.

    Read `keyframe.frame_idx` directly — the real source-video frame from the
    map-keyframes CSV. Never recompute it from `pts_time` and `fps`
    (docs/00 §3, adr/0002).

    TODO(Nhật Bảo, Vỹ Triết): confirm with BTC which integer they score on —
    `frame_idx` (real source frame) or `keyframe_index` (the ordinal n / image
    filename). Both live on the row; picking the wrong one is a silent miss.
    docs/context/buoi-1.md §7: "nếu dùng frame thì cần FPS/config trích xuất".
    """

    format_name = "by_frame"

    def build(self, keyframe: KeyframeRow, video: VideoRow) -> dict[str, Any]:
        raise NotImplementedError("TODO(Nhật Bảo): {video_id, frame_idx}")
