"""Early-fusion rerank — stage C of docs/04 §1, §3.3.

A cross-encoder model scoring query against keyframe images. Includes latency budget
checking to ensure queries do not timeout.
"""

from __future__ import annotations

import time

from mfe.contracts.tools import Candidate
from mfe.online.tools.evidence import create_evidence_id


class Reranker:
    """Cross-encoder reranker for Stage C early fusion."""

    def __init__(self, model_name: str = "cross-encoder-v1", budget_ms: int = 2000) -> None:
        self.model_name = model_name
        self.budget_ms = budget_ms

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        run_id: str = "rerank_run",
    ) -> list[Candidate]:
        """Re-score candidates within the allocated latency budget.

        Args:
            query: Query text.
            candidates: Candidate list to rerank.
            run_id: Tracing run ID.

        Returns:
            Reranked candidate list.
        """
        if not candidates:
            return []

        start_time = time.perf_counter()
        scored_candidates: list[Candidate] = []
        unscored_candidates: list[Candidate] = list(candidates)

        for idx, candidate in enumerate(candidates, start=1):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if elapsed_ms >= self.budget_ms:
                # Budget exceeded: stop scoring, append remaining candidates
                break

            score = self._score_pair(query, candidate.keyframe_id)
            ev_id = create_evidence_id("rerank", run_id, idx)
            scored_candidates.append(
                Candidate(
                    keyframe_id=candidate.keyframe_id,
                    video_id=candidate.video_id,
                    score=score,
                    evidence_id=ev_id,
                    matched_terms=candidate.matched_terms,
                )
            )

        # Sort scored candidates descending by new cross-encoder score
        scored_candidates.sort(key=lambda c: c.score, reverse=True)

        # Append remaining candidates that were not scored due to budget limit
        scored_ids = {c.keyframe_id for c in scored_candidates}
        remaining = [c for c in unscored_candidates if c.keyframe_id not in scored_ids]

        return scored_candidates + remaining

    def _score_pair(self, query: str, keyframe_id: str) -> float:
        """Simulate cross-encoder forward pass scoring."""
        # Simple deterministic scoring stub for testing
        hash_val = sum(ord(ch) for ch in query + keyframe_id)
        return round((hash_val % 100) / 100.0, 4)
