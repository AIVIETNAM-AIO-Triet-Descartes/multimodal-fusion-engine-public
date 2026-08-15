"""`build_segment` — docs/04 §2.

    purpose:     Ghép 1 hoặc nhiều keyframe_id thành một khung thời gian start_ms/end_ms.
    input:       keyframe_ids (list, >=1), lead_pad_ms (default 500), trail_pad_ms (default 500)
    output:      {video_id, start_ms, end_ms}
    constraints: start_ms = max(0, min(pts_time) - lead_pad_ms)
                 end_ms = min(duration_ms, max(pts_time) + trail_pad_ms)
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from mfe.contracts.errors import ErrorCode
from mfe.contracts.ids import parse_keyframe_id
from mfe.contracts.tools import ToolError, ToolResult


def build_segment(
    keyframe_ids: list[str],
    lead_pad_ms: int = 500,
    trail_pad_ms: int = 500,
    pts_map: Mapping[str, int] | None = None,
    duration_map: Mapping[str, int] | None = None,
) -> ToolResult[dict[str, Any]]:
    """Construct a continuous video segment from one or more keyframe IDs.

    Args:
        keyframe_ids: Non-empty list of keyframe IDs.
        lead_pad_ms: Lead padding in milliseconds (default 500).
        trail_pad_ms: Trail padding in milliseconds (default 500).
        pts_map: Optional mapping of keyframe_id -> pts_time_ms.
        duration_map: Optional mapping of video_id -> duration_ms.

    Returns:
        ToolResult[dict[str, Any]] containing {'video_id': ..., 'start_ms': ..., 'end_ms': ...}
    """
    start_t = time.perf_counter()
    tool_name = "build_segment"

    if not keyframe_ids:
        latency = (time.perf_counter() - start_t) * 1000.0
        return ToolResult(
            tool=tool_name,
            items=[],
            error=ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message="keyframe_ids cannot be empty",
                tool=tool_name,
            ),
            latency_ms=latency,
        )

    # Validate video_id consistency across keyframes
    video_ids = set()
    for kf_id in keyframe_ids:
        try:
            vid, _ = parse_keyframe_id(kf_id)
            video_ids.add(vid)
        except ValueError as err:
            latency = (time.perf_counter() - start_t) * 1000.0
            return ToolResult(
                tool=tool_name,
                items=[],
                error=ToolError(
                    code=ErrorCode.INVALID_ARGUMENT,
                    message=f"Invalid keyframe_id '{kf_id}': {err}",
                    tool=tool_name,
                ),
                latency_ms=latency,
            )

    if len(video_ids) > 1:
        latency = (time.perf_counter() - start_t) * 1000.0
        return ToolResult(
            tool=tool_name,
            items=[],
            error=ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"Keyframes belong to multiple videos: {sorted(video_ids)}",
                tool=tool_name,
            ),
            latency_ms=latency,
        )

    video_id = next(iter(video_ids))
    timestamps_ms: list[int] = []

    for kf_id in keyframe_ids:
        if pts_map and kf_id in pts_map:
            timestamps_ms.append(pts_map[kf_id])
        else:
            # No fallback. This used to guess `ordinal * 1000ms`, which was
            # already a mock and became nonsense once the id started encoding a
            # frame number (frame 90 is 3s in, not 90s). A segment with invented
            # bounds plays the wrong part of the video and looks entirely normal
            # doing it, so the tool refuses instead.
            latency = (time.perf_counter() - start_t) * 1000.0
            return ToolResult(
                tool=tool_name,
                items=[],
                error=ToolError(
                    code=ErrorCode.INVALID_ARGUMENT,
                    message=(
                        f"no timestamp for {kf_id}: pass pts_map with an entry for "
                        "every keyframe. Timestamps cannot be derived from the id."
                    ),
                    tool=tool_name,
                ),
                latency_ms=latency,
            )

    min_pts = min(timestamps_ms)
    max_pts = max(timestamps_ms)

    start_ms = max(0, min_pts - lead_pad_ms)

    # Upper bound clamping against duration_ms if duration_map is provided
    duration_ms = duration_map.get(video_id, 3_600_000) if duration_map else 3_600_000
    end_ms = min(duration_ms, max_pts + trail_pad_ms)

    result_payload = {
        "video_id": video_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
    }

    latency = (time.perf_counter() - start_t) * 1000.0
    return ToolResult(
        tool=tool_name,
        items=[result_payload],
        error=None,
        latency_ms=latency,
    )
