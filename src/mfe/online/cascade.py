"""The retrieval cascade — docs/04 §1.

    query text (+ reference image, later)
      -> [A] late-fusion coarse: vector_search + text_search over everything -> top 200-500
      -> [B] dedup + temporal grouping                                        -> top 50-100
      -> [C] early-fusion rerank: cross-encoder on original images            -> top 10-20
      -> [D] browser: grid + timeline for a human to verify
      -> candidate (+ evidence_id) -> Validator (Nhật Bảo)

The idea (docs/context/buoi-2.md §7): cheap model over the whole store, expensive
model over a small candidate set. Never the expensive one over everything.

The candidate sizes above are a starting point, not a law — docs/04 §1 says
benchmark them against the actual GPU and time budget.
"""

from __future__ import annotations

from mfe.contracts.tools import Candidate


class Cascade:
    """Owner: Phạm Trọng. TODO(Phạm Trọng): implement.

    This is the sprint's longest critical path (docs/04 preamble) — build it in
    the order docs/01 §4 gives: `search_text` end-to-end first, then B, then C.
    A working thin slice beats a half-built wide one.
    """

    def __init__(self, config: object) -> None:
        self.config = config

    def search(self, query_text: str, top_k: int | None = None) -> list[Candidate]:
        """Run A -> B -> C and return the final ranking.

        Owner: Phạm Trọng. TODO(Phạm Trọng): implement.

        **The fallback is not optional** (docs/04 §1): if C fails or blows its
        latency budget, return B's (or A's) ranking rather than raising. A
        timeout at the moment the answer is needed costs the round. Log that the
        fallback fired — a silent degradation looks like a quality regression.
        """
        raise NotImplementedError("TODO(Phạm Trọng): A -> B -> C with fallback")

    def _coarse(self, query_text: str) -> list[Candidate]:
        """Stage A — late fusion over the whole store. See `fusion.py`."""
        raise NotImplementedError("TODO(Phạm Trọng): vector + text search -> fused candidates")

    def _group(self, candidates: list[Candidate]) -> list[Candidate]:
        """Stage B — dedup + temporal grouping. See `dedup.py`."""
        raise NotImplementedError("TODO(Phạm Trọng): dedup near-duplicates, group by shot")

    def _rerank(self, query_text: str, candidates: list[Candidate]) -> list[Candidate]:
        """Stage C — early fusion on original images. See `rerank.py`."""
        raise NotImplementedError("TODO(Phạm Trọng): cross-encoder rerank within budget")
