"""`get_video_segment` — docs/04 §2.

    purpose:     Trả về video gốc + khung thời gian để phát/xem một đoạn segment.
    input:       keyframe_ids (list), lead_pad_ms?, trail_pad_ms?
    output:      {video_path, start_ms, end_ms}
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from mfe.contracts.errors import ErrorCode
from mfe.contracts.tools import ToolError, ToolResult
from mfe.online.tools.build_segment import build_segment


def get_video_segment(
    keyframe_ids: list[str],
    lead_pad_ms: int = 500,
    trail_pad_ms: int = 500,
    video_path_map: Mapping[str, str] | None = None,
    pts_map: Mapping[str, int] | None = None,
    duration_map: Mapping[str, int] | None = None,
) -> ToolResult[dict[str, Any]]:
    """Retrieve video file path and time boundary for keyframes.

    Args:
        keyframe_ids: Non-empty list of keyframe IDs.
        lead_pad_ms: Lead padding in milliseconds (default 500).
        trail_pad_ms: Trail padding in milliseconds (default 500).
        video_path_map: Optional mapping of video_id -> absolute file path.
        pts_map: Optional mapping of keyframe_id -> pts_time_ms.
        duration_map: Optional mapping of video_id -> duration_ms.

    Returns:
        ToolResult[dict[str, Any]] containing {'video_path': ..., 'start_ms': ..., 'end_ms': ...}
    """
    start_t = time.perf_counter()
    tool_name = "get_video_segment"

    seg_res = build_segment(
        keyframe_ids=keyframe_ids,
        lead_pad_ms=lead_pad_ms,
        trail_pad_ms=trail_pad_ms,
        pts_map=pts_map,
        duration_map=duration_map,
    )

    if not seg_res.ok or not seg_res.items:
        latency = (time.perf_counter() - start_t) * 1000.0
        return ToolResult(
            tool=tool_name,
            items=[],
            error=seg_res.error or ToolError(code=ErrorCode.NOT_FOUND, message="Failed to build segment", tool=tool_name),
            latency_ms=latency,
        )

    seg_info = seg_res.items[0]
    video_id = seg_info["video_id"]

    if video_path_map and video_id in video_path_map:
        video_path = video_path_map[video_id]
    else:
        # Fallback mock path
        video_path = f"data/videos/{video_id}.mp4"

    result_payload = {
        "video_path": video_path,
        "start_ms": seg_info["start_ms"],
        "end_ms": seg_info["end_ms"],
    }

    latency = (time.perf_counter() - start_t) * 1000.0
    return ToolResult(
        tool=tool_name,
        items=[result_payload],
        error=None,
        latency_ms=latency,
    )
