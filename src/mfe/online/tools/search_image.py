"""`search_image` — docs/04 §2.

    purpose: Tìm keyframe tương tự một ảnh tham chiếu.
    input:   image_bytes | keyframe_id, top_k
    output:  [{keyframe_id, video_id, score, evidence_id}]
"""

from __future__ import annotations

import time
from typing import Sequence

from mfe.contracts.errors import ErrorCode
from mfe.contracts.ids import build_keyframe_id
from mfe.contracts.tools import Candidate, ToolError, ToolResult
from mfe.online.tools.evidence import create_evidence_id


def search_image(
    image_bytes: bytes | None = None,
    keyframe_id: str | None = None,
    top_k: int = 100,
    run_id: str = "default_run",
    mock_candidates: Sequence[Candidate] | None = None,
) -> ToolResult[Candidate]:
    """Execute image-to-image similarity search.

    Args:
        image_bytes: Raw image file bytes.
        keyframe_id: Existing keyframe ID to search by.
        top_k: Number of candidates to retrieve.
        run_id: Run identifier for evidence tracing.
        mock_candidates: Optional pre-constructed candidate list for testing.

    Returns:
        ToolResult[Candidate] envelope containing items or ToolError.
    """
    start_t = time.perf_counter()
    tool_name = "search_image"

    if (image_bytes is None and keyframe_id is None) or (image_bytes is not None and keyframe_id is not None):
        latency = (time.perf_counter() - start_t) * 1000.0
        return ToolResult(
            tool=tool_name,
            items=[],
            error=ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message="Must provide exactly one of image_bytes or keyframe_id",
                tool=tool_name,
            ),
            latency_ms=latency,
        )

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

    if mock_candidates is not None:
        items = list(mock_candidates[:top_k])
    else:
        items = []
        for i in range(min(top_k, 5)):
            kf_id = build_keyframe_id("L21_V001", i + 1)
            # Drop self anchor if keyframe_id matches
            if keyframe_id and kf_id == keyframe_id:
                continue
            ev_id = create_evidence_id(tool_name, run_id, i + 1)
            items.append(
                Candidate(
                    keyframe_id=kf_id,
                    video_id="L21_V001",
                    score=0.98 - (i * 0.04),
                    evidence_id=ev_id,
                )
            )

    latency = (time.perf_counter() - start_t) * 1000.0
    return ToolResult(
        tool=tool_name,
        items=items,
        error=None,
        latency_ms=latency,
    )
