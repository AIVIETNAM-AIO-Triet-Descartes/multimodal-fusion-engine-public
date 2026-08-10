"""Early-fusion rerank — stage C of docs/04 §1, §3.3.

A cross-encoder model scoring query against keyframe images, under a latency
budget.

**Budget policy: truncate, never raise** (CHỐT — Vỹ Triết). Over budget, the
reranker stops scoring and returns what it has, with the unscored tail appended
in its incoming (coarse) order. It does not raise, and callers therefore do not
wrap it in a try/except pretending to catch a timeout — an except branch that
nothing can reach reads as insurance while providing none.

This is a safe policy precisely because the tail keeps coarse order: the result
is never worse than stage B's ranking, which is what "fallback bắt buộc" asks
for. Degradation is graded rather than all-or-nothing.
"""

from __future__ import annotations

import time

from mfe.contracts.tools import Candidate
from mfe.online.tools.evidence import create_evidence_id


class Reranker:
    """Cross-encoder reranker for Stage C early fusion.

    **Still a stub.** `_score_pair` hashes the strings; it does not look at the
    image. `Cascade` therefore does not build one by default — measured against
    the real stores, this stub reorders a good ranking into a bad one. Pass an
    instance explicitly only in tests, or once `_score_pair` runs a real model.
    """

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
            Reranked candidate list: scored candidates first (best first), then
            whatever the budget did not reach, in coarse order.

        Never raises on timeout — see the module docstring.
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
        """STUB — deterministic hash, not a model. TODO(Phạm Trọng).

        A real implementation loads a cross-encoder and scores the query
        against `AssetStore.get_asset(keyframe_id)` — the *image*, which is what
        makes stage C "early fusion" rather than a second text pass. Until then
        this returns a number that correlates with nothing, which is why the
        cascade leaves stage C switched off.
        """
        hash_val = sum(ord(ch) for ch in query + keyframe_id)
        return round((hash_val % 100) / 100.0, 4)
