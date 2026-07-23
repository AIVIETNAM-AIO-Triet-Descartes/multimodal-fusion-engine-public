"""`get_neighbors` — docs/04 §2.

    purpose: Lấy keyframe lân cận theo thời gian để xem ngữ cảnh trước/sau.
    input:   keyframe_id, window
    output:  [{keyframe_id, timestamp_ms}]
"""

from __future__ import annotations

import time
from typing import Mapping

from mfe.contracts.errors import ErrorCode
from mfe.contracts.ids import build_keyframe_id, parse_keyframe_id
from mfe.contracts.tools import Neighbor, ToolError, ToolResult


def get_neighbors(
    keyframe_id: str,
    window: int = 5,
    pts_map: Mapping[str, int] | None = None,
) -> ToolResult[Neighbor]:
    """Retrieve temporally adjacent keyframes around keyframe_id.

    Args:
        keyframe_id: Target keyframe identifier.
        window: Number of neighboring keyframes on each side (default 5).
        pts_map: Optional mapping of keyframe_id -> pts_time_ms.

    Returns:
        ToolResult[Neighbor] containing list of Neighbor instances.
    """
    start_t = time.perf_counter()
    tool_name = "get_neighbors"

    try:
        video_id, kf_idx = parse_keyframe_id(keyframe_id)
    except ValueError as err:
        latency = (time.perf_counter() - start_t) * 1000.0
        return ToolResult(
            tool=tool_name,
            items=[],
            error=ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"Invalid keyframe_id '{keyframe_id}': {err}",
                tool=tool_name,
            ),
            latency_ms=latency,
        )

    if window <= 0:
        latency = (time.perf_counter() - start_t) * 1000.0
        return ToolResult(
            tool=tool_name,
            items=[],
            error=ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"window must be > 0, got {window}",
                tool=tool_name,
            ),
            latency_ms=latency,
        )

    # Collect neighbors within [max(0, kf_idx - window), kf_idx + window]
    start_idx = max(0, kf_idx - window)
    end_idx = kf_idx + window

    neighbors: list[Neighbor] = []
    for idx in range(start_idx, end_idx + 1):
        if idx == kf_idx:
            continue
        n_id = build_keyframe_id(video_id, idx)
        ts_ms = pts_map[n_id] if (pts_map and n_id in pts_map) else idx * 1000
        neighbors.append(Neighbor(keyframe_id=n_id, timestamp_ms=ts_ms))

    latency = (time.perf_counter() - start_t) * 1000.0
    return ToolResult(
        tool=tool_name,
        items=neighbors,
        error=None,
        latency_ms=latency,
    )
