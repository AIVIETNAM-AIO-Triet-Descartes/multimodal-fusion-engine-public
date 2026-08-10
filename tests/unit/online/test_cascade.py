"""Unit tests for retrieval cascade pipeline (docs/04 §1).

The cascade reads real stores now, so these tests inject fakes rather than
relying on the old self-generated mock candidates.
"""

import numpy as np

from mfe.contracts.tools import Candidate
from mfe.online.cascade import Cascade
from mfe.online.rerank import Reranker


class FakeInverted:
    """Stands in for InvertedIndex.search."""

    def __init__(self, hits):
        self.hits = hits

    def search(self, query_text, top_k, run_id="store"):
        return self.hits[:top_k]


class FakeVector:
    """Stands in for VectorIndex.search."""

    def __init__(self, hits):
        self.hits = hits

    def search(self, query_embedding, top_k, run_id="store", video_id=None):
        return self.hits[:top_k]


def _cand(index, score, tool="search_text", ordinal=0):
    return Candidate(
        keyframe_id=f"L21_V001#{index:03d}",
        video_id="L21_V001",
        score=score,
        evidence_id=f"{tool}:store:{ordinal}",
    )


def test_cascade_skips_stage_c_without_a_reranker():
    """Default = no rerank. A stub scorer at the last stage is worse than none."""
    hits = [_cand(1, 9.0, ordinal=0), _cand(40, 8.0, ordinal=1), _cand(80, 7.0, ordinal=2)]
    cascade = Cascade(inverted_index=FakeInverted(hits))

    results = cascade.search(query_text="blue sky sunset", top_k=5, run_id="test_cascade")

    assert len(results) == 3
    assert not any(c.evidence_id.startswith("rerank:") for c in results)


def test_cascade_runs_stage_c_when_a_reranker_is_injected():
    hits = [_cand(1, 9.0, ordinal=0), _cand(40, 8.0, ordinal=1)]
    cascade = Cascade(inverted_index=FakeInverted(hits), reranker=Reranker(budget_ms=5000))

    results = cascade.search(query_text="blue sky sunset", top_k=5, run_id="test_cascade")

    assert all(c.evidence_id.startswith("rerank:test_cascade:") for c in results)


def test_cascade_fuses_both_arms():
    """Stage A must consult the vector arm too, not just BM25."""
    text_hits = [_cand(1, 9.0, ordinal=0)]
    vector_hits = [_cand(90, 0.8, tool="vector_search", ordinal=0)]

    cascade = Cascade(
        inverted_index=FakeInverted(text_hits),
        vector_index=FakeVector(vector_hits),
        encode_query=lambda text: np.zeros(512, dtype=np.float32),
    )
    results = cascade.search(query_text="red car", top_k=10, run_id="r")

    found = {c.keyframe_id for c in results}
    assert found == {"L21_V001#001", "L21_V001#090"}


def test_cascade_degrades_to_bm25_when_encoder_fails():
    """A broken vector arm must not take the whole query down (docs/04 §1)."""

    def broken_encoder(text):
        raise RuntimeError("no checkpoint on this machine")

    cascade = Cascade(
        inverted_index=FakeInverted([_cand(1, 9.0, ordinal=0)]),
        vector_index=FakeVector([]),
        encode_query=broken_encoder,
    )
    results = cascade.search(query_text="red car", top_k=5, run_id="r")

    assert [c.keyframe_id for c in results] == ["L21_V001#001"]


def test_cascade_keeps_the_tail_for_deep_recall():
    """Stage B must not discard near-duplicates.

    Official PDF §2.2 scores R@{1,5,20,50,100}; dropping the tail throws away
    four of those five terms.
    """
    hits = [_cand(i, 10.0 - i, ordinal=i) for i in range(1, 6)]  # all within window
    cascade = Cascade(inverted_index=FakeInverted(hits))

    results = cascade.search(query_text="q", top_k=100, run_id="r")

    assert len(results) == 5


def test_cascade_returns_empty_without_stores():
    assert Cascade().search(query_text="anything", top_k=5) == []
