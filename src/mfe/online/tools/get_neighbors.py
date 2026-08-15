"""`get_neighbors` — docs/04 §2.

    purpose: Lấy keyframe lân cận theo thời gian để xem ngữ cảnh trước/sau.
    input:   keyframe_id, window
    output:  [{keyframe_id, timestamp_ms}]
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from mfe.contracts.errors import ErrorCode
from mfe.contracts.ids import parse_keyframe_id
from mfe.contracts.tools import Neighbor, ToolError, ToolResult


def get_neighbors(
    keyframe_id: str,
    window: int = 5,
    pts_map: Mapping[str, int] | None = None,
    known_keyframe_ids: Sequence[str] | None = None,
) -> ToolResult[Neighbor]:
    """Retrieve temporally adjacent keyframes around keyframe_id.

    Args:
        keyframe_id: Target keyframe identifier.
        window: Number of neighbouring keyframes on each side (default 5).
        pts_map: Optional mapping of keyframe_id -> pts_time_ms. Doubles as the
            set of keyframes known to exist when `known_keyframe_ids` is absent.
        known_keyframe_ids: The keyframes that exist. Required (here or via
            `pts_map`) — see the body for why they cannot be derived.

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

    # Which keyframes actually exist has to come from outside. This used to walk
    # the id arithmetically — `kf_idx - 1`, `kf_idx + 1`, ... — which worked only
    # while the id encoded a dense ordinal. Now that it encodes a frame number,
    # frame-1 is almost never a keyframe, so minting neighbour ids would return a
    # list of keyframes that do not exist. Refusing is the honest answer.
    universe = known_keyframe_ids if known_keyframe_ids is not None else (
        list(pts_map) if pts_map else None
    )
    if universe is None:
        latency = (time.perf_counter() - start_t) * 1000.0
        return ToolResult(
            tool=tool_name,
            items=[],
            error=ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message=(
                    "get_neighbors needs the set of keyframes that exist for this "
                    "video (known_keyframe_ids or pts_map). Neighbours cannot be "
                    "derived from the id alone: it encodes a frame number, not a "
                    "dense ordinal."
                ),
                tool=tool_name,
            ),
            latency_ms=latency,
        )

    # Rank order within the video, by frame. The window is in *keyframes* here
    # (unlike dedup's, which is in frames) — "the 5 either side of this one".
    same_video: list[tuple[int, str]] = []
    for candidate_id in universe:
        try:
            candidate_video, candidate_frame = parse_keyframe_id(candidate_id)
        except ValueError:
            continue
        if candidate_video == video_id:
            same_video.append((candidate_frame, candidate_id))
    same_video.sort()

    try:
        centre = next(i for i, (frame, _) in enumerate(same_video) if frame == kf_idx)
    except StopIteration:
        latency = (time.perf_counter() - start_t) * 1000.0
        return ToolResult(
            tool=tool_name,
            items=[],
            error=ToolError(
                code=ErrorCode.KEYFRAME_NOT_FOUND,
                message=f"{keyframe_id} is not among the keyframes given for {video_id}",
                tool=tool_name,
            ),
            latency_ms=latency,
        )

    neighbors: list[Neighbor] = []
    for i in range(max(0, centre - window), min(len(same_video), centre + window + 1)):
        if i == centre:
            continue
        frame, n_id = same_video[i]
        ts_ms = pts_map[n_id] if (pts_map and n_id in pts_map) else frame * 1000
        neighbors.append(Neighbor(keyframe_id=n_id, timestamp_ms=ts_ms))

    latency = (time.perf_counter() - start_t) * 1000.0
    return ToolResult(
        tool=tool_name,
        items=neighbors,
        error=None,
        latency_ms=latency,
    )
