"""`rerank` tool — docs/04 §2.

    purpose: Xếp lại tập nhỏ bằng cross-encoder trên ảnh gốc.
    input:   query, candidate_ids
    output:  [{keyframe_id, score, evidence_id}]
"""

from __future__ import annotations

import time

from mfe.contracts.tools import Candidate, ToolResult
from mfe.online.rerank import Reranker
from mfe.online.tools.evidence import create_evidence_id


def rerank(
    query: str,
    candidate_ids: list[str],
    budget_ms: int = 2000,
    run_id: str = "default_run",
    reranker: Reranker | None = None,
) -> ToolResult[Candidate]:
    """Execute cross-encoder reranking over a candidate ID list.

    Args:
        query: User text query.
        candidate_ids: List of candidate keyframe IDs to rerank.
        budget_ms: Latency budget in milliseconds.
        run_id: Run identifier for evidence tracing.
        reranker: Optional Reranker instance.

    Returns:
        ToolResult[Candidate] envelope containing items or ToolError.
    """
    start_t = time.perf_counter()
    tool_name = "rerank"

    if not candidate_ids:
        latency = (time.perf_counter() - start_t) * 1000.0
        return ToolResult(
            tool=tool_name,
            items=[],
            error=None,
            latency_ms=latency,
        )

    # Convert candidate IDs to candidate stubs
    cands = [
        Candidate(
            keyframe_id=cid,
            video_id=cid.split("#")[0] if "#" in cid else cid,
            score=1.0 - (idx * 0.01),
            evidence_id=create_evidence_id("coarse", run_id, idx + 1),
        )
        for idx, cid in enumerate(candidate_ids)
    ]

    r = reranker or Reranker(model_name="mock-cross-encoder", budget_ms=budget_ms)

    # No try/except around this call: `Reranker.rerank` truncates on budget
    # rather than raising (CHỐT, see mfe.online.rerank), so a handler here would
    # be unreachable code that looks like a safety net.
    res_cands = r.rerank(query=query, candidates=cands, run_id=run_id)

    latency = (time.perf_counter() - start_t) * 1000.0
    return ToolResult(
        tool=tool_name,
        items=res_cands,
        error=None,
        latency_ms=latency,
    )
