"""Submit by timestamp: {video_id, timestamp_ms}. docs/05 §2."""

from __future__ import annotations

from typing import Any

from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.mapping.adapters.base import SubmitAdapter


class ByTimestampAdapter(SubmitAdapter):
    """Owner: Nhật Bảo. TODO(Nhật Bảo): implement.

    Read `keyframe.timestamp_ms` directly. Do not compute it from
    `frame_number` and `video.fps` — that is the whole reason docs/00 §3 stores
    both columns.

    TODO(Nhật Bảo, Vỹ Triết): ask BTC whether the unit is milliseconds or seconds.
    docs/context/buoi-1.md §7 has the lecturer saying "timestamp đến mili-giây",
    which is guidance, not a rule. A 1000x unit error submits a valid-looking
    record pointing at the wrong moment.
    """

    format_name = "by_timestamp"

    def build(self, keyframe: KeyframeRow, video: VideoRow) -> dict[str, Any]:
        raise NotImplementedError("TODO(Nhật Bảo): {video_id, timestamp_ms}")
