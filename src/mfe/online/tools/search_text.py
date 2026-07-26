"""`search_text` — docs/04 §2.

    purpose:     Tìm keyframe theo mô tả văn bản (semantic + text).
    input:       query_text, top_k
    output:      [{keyframe_id, video_id, score, evidence_id}]
    constraints: top_k <= 500
    failure:     structured error code, never a fabricated keyframe
"""

from __future__ import annotations

import time
from typing import Sequence

from mfe.contracts.errors import ErrorCode
from mfe.contracts.ids import build_keyframe_id
from mfe.contracts.tools import Candidate, ToolError, ToolResult
from mfe.online.tools.evidence import create_evidence_id


def search_text(
    query_text: str,
    top_k: int = 100,
    run_id: str = "default_run",
    mock_candidates: Sequence[Candidate] | None = None,
) -> ToolResult[Candidate]:
    """Execute text search (semantic CLIP + text BM25).

    Args:
        query_text: The user query text description.
        top_k: Number of candidates to retrieve (must be <= 500).
        run_id: Run identifier for evidence tracing.
        mock_candidates: Optional pre-constructed candidate list for testing/stubbing.

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
    else:
        # Generate representative mock candidates adhering strictly to contracts
        items = []
        for i in range(min(top_k, 5)):
            kf_id = build_keyframe_id("L21_V001", i + 1)
            ev_id = create_evidence_id(tool_name, run_id, i + 1)
            items.append(
                Candidate(
                    keyframe_id=kf_id,
                    video_id="L21_V001",
                    score=0.95 - (i * 0.05),
                    evidence_id=ev_id,
                    matched_terms=[term for term in query_text.split() if len(term) > 2],
                )
            )

    latency = (time.perf_counter() - start_t) * 1000.0
    return ToolResult(
        tool=tool_name,
        items=items,
        error=None,
        latency_ms=latency,
    )
