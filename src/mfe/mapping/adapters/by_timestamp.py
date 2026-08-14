"""Submit by timestamp: {video_id, timestamp_ms}. docs/05 §2."""

from __future__ import annotations

from typing import Any

from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.mapping.adapters.base import SubmitAdapter


class ByTimestampAdapter(SubmitAdapter):
    """Owner: Nhật Bảo. **Not used** — BTC scores frames, not timestamps.

    PDF §2.1.1 answers the question this adapter was waiting on: the submission
    is `<video_id>, <frame_id>` and the ground truth is a frame interval. The
    millisecond-versus-second worry that used to sit here is therefore moot, and
    asking BTC about it would waste a question. Kept because a format we have
    tested is cheaper to revive than one we have to write under time pressure.

    Read `keyframe.timestamp_ms` directly (it is `pts_time * 1000`, both from the
    CSV). Do not recompute anything from `fps` — that is the whole reason
    docs/00 §3 stores the columns instead of deriving them.
    """

    format_name = "by_timestamp"

    def build(self, keyframe: KeyframeRow, video: VideoRow) -> dict[str, Any]:
        return {"video_id": video.video_id, "timestamp_ms": keyframe.timestamp_ms}
