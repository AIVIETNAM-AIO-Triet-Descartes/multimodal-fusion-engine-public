"""`search_text` — docs/04 §2.

    purpose:     Tìm keyframe theo mô tả văn bản (semantic + text).
    input:       query_text, top_k
    output:      [{keyframe_id, video_id, score, evidence_id}]
    constraints: top_k <= 500
    failure:     structured error code, never a fabricated keyframe
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from mfe.contracts.errors import ErrorCode
from mfe.contracts.tools import Candidate, ToolError, ToolResult


def search_text(
    query_text: str,
    top_k: int = 100,
    run_id: str = "default_run",
    cascade: Any = None,
    mock_candidates: Sequence[Candidate] | None = None,
) -> ToolResult[Candidate]:
    """Execute text search (semantic CLIP + text BM25).

    This is the tool-contract wrapper (docs/00 §5) that the browser and the
    Buổi 3 agent call; the retrieval itself lives in `mfe.online.cascade`.

    Args:
        query_text: The user query text description.
        top_k: Number of candidates to retrieve (must be <= 500).
        run_id: Run identifier for evidence tracing.
        cascade: The `Cascade` to search through. Required unless mocking.
        mock_candidates: Optional pre-constructed candidate list for testing.

    Returns:
        ToolResult[Candidate] envelope containing items or ToolError.
    """
    start_t = time.perf_counter()
    tool_name = "search_text"

    if top_k > 500:
        latency = (time.perf_counter() - start_t) * 1000.0
        return ToolResult(
            tool=tool_name,
            items=[],
            error=ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"top_k ({top_k}) exceeds maximum constraint of 500",
                tool=tool_name,
            ),
            latency_ms=latency,
        )

    if not query_text or not query_text.strip():
        latency = (time.perf_counter() - start_t) * 1000.0
        return ToolResult(
            tool=tool_name,
            items=[],
            error=ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message="query_text cannot be empty",
                tool=tool_name,
            ),
            latency_ms=latency,
        )

    if mock_candidates is not None:
        items = list(mock_candidates[:top_k])
    elif cascade is not None:
        items = cascade.search(query_text=query_text, top_k=top_k, run_id=run_id)
    else:
        # No cascade and no mock: say so. Returning invented keyframes here is
        # exactly what docs/04 §2 forbids ("không bịa keyframe") — they would
        # look like real hits all the way to the scoreboard.
        latency = (time.perf_counter() - start_t) * 1000.0
        return ToolResult(
            tool=tool_name,
            items=[],
            error=ToolError(
                code=ErrorCode.STORE_UNAVAILABLE,
                message="search_text needs a Cascade to search; pass cascade=...",
                tool=tool_name,
            ),
            latency_ms=latency,
        )

    latency = (time.perf_counter() - start_t) * 1000.0
    return ToolResult(
        tool=tool_name,
        items=items,
        error=None,
        latency_ms=latency,
    )
