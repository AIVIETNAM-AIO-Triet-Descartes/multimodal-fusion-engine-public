"""`rerank` tool — docs/04 §2.

    purpose: Xếp lại tập nhỏ bằng cross-encoder trên ảnh gốc.
    input:   query, candidate_ids
    output:  [{keyframe_id, score, evidence_id}]

Thin wrapper over `mfe.online.rerank.Reranker` so the capability is reachable
from the tool registry — i.e. by the Buổi 3 agent — and not only from inside the
cascade.
"""

from __future__ import annotations

from mfe.contracts.tools import Candidate, ToolResult


def rerank(query: str, candidate_ids: list[str]) -> ToolResult[Candidate]:
    """Owner: TV3. TODO(TV3): implement.

    - Enforce `retrieval.rerank_topk` from config. This tool is the expensive
      one; an agent handed 500 ids will happily wait, blowing the query budget
      (docs/context/buoi-3.md §7 guardrails).
    - Over budget -> ToolResult with TOOL_TIMEOUT, and the caller keeps the
      coarse ranking (docs/04 §1). The tool must fail cleanly, not hang.
    - The tool description must say when *not* to use it
      (docs/context/buoi-3.md §7): reranking 5 candidates buys nothing and costs
      seconds.
    """
    raise NotImplementedError("TODO(TV3): wrap Reranker with topk + budget enforcement")
