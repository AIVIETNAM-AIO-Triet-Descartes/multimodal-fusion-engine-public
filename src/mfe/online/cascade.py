"""The retrieval cascade — docs/04 §1.

    query text
      -> [A] late-fusion coarse: vector_search + text_search over everything -> top 200-500
      -> [B] dedup + temporal grouping                                        -> top 50-100
      -> [C] early-fusion rerank: cross-encoder on original images            -> top 10-20
      -> [D] browser: grid + timeline for human verification
      -> candidate (+ evidence_id) -> Validator (Nhật Bảo)

Wired to the real stores (CHỐT — Vỹ Triết). Stage A now runs both arms:

    vector : encode the query with clip-ViT-B-32, cosine over the Faiss index
    text   : BM25 over object labels / OCR / metadata

The vector arm was impossible until the checkpoint was pinned — BTC ships
keyframe features but not the model, so a text query had no way into the space.
Verified at cosine 0.9992-1.0000 against the shipped vectors
(`scripts/verify_clip_checkpoint.py`).

**Scoring shapes this file.** Official PDF §2.2: a team may submit up to 100
answers and the score is the mean of R@{1,5,20,50,100}. Returning fewer than 100
candidates throws away four of the five terms for free. So the cascade is built
to hand the browser a deep list, not a short one, and stage B keeps the tail
instead of discarding it.
"""

from __future__ import annotations

import logging
from typing import Any

from mfe.contracts.tools import Candidate
from mfe.online.dedup import dedup
from mfe.online.fusion import weighted_fusion
from mfe.online.rerank import Reranker

logger = logging.getLogger(__name__)

#: Config `fusion_weights` names the *signals* (clip, bm25); fusion keys the
#: *tools* that produced them. One place to map between the two.
_WEIGHT_KEY = {"vector_search": "clip", "search_text": "bm25"}


class Cascade:
    """Online retrieval cascade. Owner: Phạm Trọng.

    Stores are injected rather than opened here, so the eval harness, the
    browser and a test can each hand it whatever they already have open.
    Use `from_config` for the ordinary case.
    """

    def __init__(
        self,
        config: Any = None,
        vector_index: Any = None,
        inverted_index: Any = None,
        reranker: Reranker | None = None,
        encode_query: Any = None,
    ) -> None:
        self.config = config
        self.vector_index = vector_index
        self.inverted_index = inverted_index
        self.encode_query = encode_query
        self.fusion_weights = self._resolve_weights(config)
        # No default reranker. `Reranker` currently scores with a hash stub, and
        # measured against the real stores it *destroys* the ranking: for
        # "a person riding a motorbike", stage A+B put Motorcycle/Wheel/Tire
        # keyframes on top and the stub replaced them with a Laptop/Glasses
        # frame it happened to hash higher. A fake scorer in the final stage is
        # strictly worse than no stage at all, because it is the last word on
        # order. Stage C switches on the day a real cross-encoder is passed in.
        self.reranker = reranker

    @classmethod
    def from_config(cls, config: Any, **overrides: Any) -> Cascade:
        """Open the stores named in config and build a cascade over them.

        A store that will not load is a warning, not a failure: a repo where
        ingest has not run should still start the browser and say what is
        missing, rather than refusing at import time.
        """
        from mfe.online.encoder import DEFAULT_TEXT_CHECKPOINT, encode_query_text
        from mfe.stores.inverted.index import InvertedIndex
        from mfe.stores.vector.index import VectorIndex

        text_checkpoint = getattr(
            getattr(config, "retrieval", None), "clip_text_checkpoint", DEFAULT_TEXT_CHECKPOINT
        )

        def load(label: str, loader: Any) -> Any:
            try:
                return loader()
            except Exception as err:  # noqa: BLE001 — "not built yet" is expected
                logger.warning("%s unavailable (%s); run mfe-ingest", label, err)
                return None

        return cls(
            config=config,
            vector_index=overrides.get(
                "vector_index",
                load(
                    "vector index",
                    lambda: VectorIndex.load(
                        config.paths.vector_store, config.retrieval.embed_dim
                    ),
                ),
            ),
            inverted_index=overrides.get(
                "inverted_index",
                load(
                    "inverted index",
                    lambda: InvertedIndex.load(config.paths.inverted_store),
                ),
            ),
            encode_query=overrides.get(
                "encode_query",
                lambda text: encode_query_text(text, checkpoint=text_checkpoint),
            ),
            reranker=overrides.get("reranker"),
        )

    def search(
        self,
        query_text: str,
        top_k: int | None = None,
        run_id: str = "cascade_run",
    ) -> list[Candidate]:
        """Run stage A -> B -> C and return a ranked candidate list.

        Args:
            query_text: The description to search for.
            top_k: How many to return. Defaults to `retrieval.rerank_topk`,
                which config sets to the 100 the scoring rewards.
            run_id: Tracing id; ends up inside every `evidence_id`.
        """
        top_k = top_k or self._cfg("rerank_topk", 100)

        coarse = self._coarse(query_text, run_id=run_id)
        if not coarse:
            return []

        grouped = self._group(coarse)

        if self.reranker is None:
            return grouped[:top_k]

        # Stage C is scored under a budget and truncates rather than raising
        # (mfe.online.rerank), so there is no timeout to catch here. Only the
        # head is reranked: reranking is the expensive stage, and R@1 is the
        # only term of the score that distinguishes the head from the tail.
        head = grouped[: self._cfg("rerank_topk", 100)]
        tail = grouped[len(head) :]
        reranked = self._rerank(query_text, head, run_id=run_id)

        return (reranked + tail)[:top_k]

    def _coarse(self, query_text: str, run_id: str = "cascade_run") -> list[Candidate]:
        """Stage A — late fusion of the vector and text arms over the whole store."""
        coarse_topk = self._cfg("coarse_topk", 500)
        result_lists: dict[str, list[Candidate]] = {}

        if self.inverted_index is not None:
            try:
                result_lists["search_text"] = self.inverted_index.search(
                    query_text=query_text, top_k=coarse_topk, run_id=run_id
                )
            except ValueError as err:
                # Query with no indexable term: a miss, not a crash.
                logger.info("text arm returned nothing for %r (%s)", query_text, err)

        vector_hits = self._vector_arm(query_text, coarse_topk, run_id)
        if vector_hits is not None:
            result_lists["vector_search"] = vector_hits

        if not result_lists:
            logger.warning("no store available for query %r; returning nothing", query_text)
            return []

        return weighted_fusion(result_lists, weights=self.fusion_weights, run_id=run_id)

    def _vector_arm(
        self, query_text: str, top_k: int, run_id: str
    ) -> list[Candidate] | None:
        """Encode the query and search Faiss. None when the arm is unavailable.

        A failure here degrades to a BM25-only stage A and says so. That is a
        real loss of recall on descriptive queries, so it is logged at warning
        rather than swallowed — running one-armed without noticing is how a
        team spends a week tuning fusion weights that never applied.
        """
        if self.vector_index is None or self.encode_query is None:
            return None
        try:
            embedding = self.encode_query(query_text)
            return self.vector_index.search(
                query_embedding=embedding.tolist(), top_k=top_k, run_id=run_id
            )
        except Exception as err:  # noqa: BLE001 — degrade, do not abort the query
            logger.warning("vector arm unavailable (%s); stage A is BM25-only", err)
            return None

    def _group(self, candidates: list[Candidate]) -> list[Candidate]:
        """Stage B — dedup + temporal grouping.

        Representatives first, then everything they represent. Under the
        official scoring the tail is not waste: R@20/50/100 read that far down,
        so dropping near-duplicates entirely would trade three terms of the
        score for grid tidiness that only affects the first screen.
        """
        window = self._cfg("temporal_window", 5)
        representatives = dedup(candidates, temporal_window=window)
        chosen = {candidate.keyframe_id for candidate in representatives}
        remainder = [c for c in candidates if c.keyframe_id not in chosen]
        return representatives + remainder

    def _rerank(
        self,
        query_text: str,
        candidates: list[Candidate],
        run_id: str = "cascade_run",
    ) -> list[Candidate]:
        """Stage C — early fusion on original images."""
        return self.reranker.rerank(query=query_text, candidates=candidates, run_id=run_id)

    # --- config plumbing ---------------------------------------------------

    def _cfg(self, name: str, default: Any) -> Any:
        return getattr(getattr(self.config, "retrieval", None), name, default)

    @staticmethod
    def _resolve_weights(config: Any) -> dict[str, float]:
        """Map config's signal names (clip, bm25) onto fusion's tool names."""
        weights = getattr(getattr(config, "retrieval", None), "fusion_weights", None)
        return {
            tool: float(getattr(weights, signal, 1.0)) if weights else 1.0
            for tool, signal in _WEIGHT_KEY.items()
        }
