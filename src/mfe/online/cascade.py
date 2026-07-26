"""The retrieval cascade — docs/04 §1.

    query text
      -> [A] late-fusion coarse: vector_search + text_search over everything -> top 200-500
      -> [B] dedup + temporal grouping                                        -> top 50-100
      -> [C] early-fusion rerank: cross-encoder on original images            -> top 10-20
      -> [D] browser: grid + timeline for human verification
      -> candidate (+ evidence_id) -> Validator (Nhật Bảo)
"""

from __future__ import annotations

import logging
from typing import Any

from mfe.contracts.tools import Candidate
from mfe.online.dedup import dedup
from mfe.online.fusion import weighted_fusion
from mfe.online.rerank import Reranker
from mfe.online.tools.search_text import search_text

logger = logging.getLogger(__name__)


class Cascade:
    """Online retrieval cascade pipeline owner: Phạm Trọng."""

    def __init__(
        self,
        config: Any = None,
        fusion_weights: dict[str, float] | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config
        self.fusion_weights = fusion_weights or {"search_text": 1.0, "vector_search": 1.0}
        self.reranker = reranker or Reranker(model_name="default_reranker", budget_ms=2000)

    def search(
        self,
        query_text: str,
        top_k: int | None = None,
        run_id: str = "cascade_run",
    ) -> list[Candidate]:
        """Run Stage A -> Stage B -> Stage C cascade with mandatory fallback.

        Args:
            query_text: Input query description.
            top_k: Target number of top candidates to return.
            run_id: Tracing run identifier.

        Returns:
            List of Candidate objects.
        """
        # Stage A: Late-fusion coarse retrieval
        coarse_candidates = self._coarse(query_text, run_id=run_id)
        if not coarse_candidates:
            return []

        # Stage B: Dedup + temporal grouping
        grouped_candidates = self._group(coarse_candidates)

        # Stage C: Early-fusion reranking with mandatory fallback
        try:
            reranked_candidates = self._rerank(query_text, grouped_candidates, run_id=run_id)
            final_ranking = reranked_candidates
        except Exception as err:
            logger.warning(
                "Stage C reranking failed or timed out (%s). Falling back to Stage B candidates.",
                err,
            )
            final_ranking = grouped_candidates

        if top_k is not None:
            return final_ranking[:top_k]
        return final_ranking

    def _coarse(self, query_text: str, run_id: str = "cascade_run") -> list[Candidate]:
        """Stage A — late fusion over the whole store."""
        text_res = search_text(query_text=query_text, top_k=200, run_id=run_id)
        if not text_res.ok or not text_res.items:
            return []

        # Simulated vector search list for late fusion demo/stubbing
        vector_res_items = text_res.items[::-1]

        result_lists = {
            "search_text": text_res.items,
            "vector_search": vector_res_items,
        }
        return weighted_fusion(result_lists, weights=self.fusion_weights, run_id=run_id)

    def _group(self, candidates: list[Candidate]) -> list[Candidate]:
        """Stage B — dedup + temporal grouping."""
        return dedup(candidates, temporal_window=5)

    def _rerank(
        self,
        query_text: str,
        candidates: list[Candidate],
        run_id: str = "cascade_run",
    ) -> list[Candidate]:
        """Stage C — early fusion on original images."""
        return self.reranker.rerank(query=query_text, candidates=candidates, run_id=run_id)
