"""Unit tests for retrieval cascade pipeline (docs/04 §1)."""

from unittest.mock import MagicMock

from mfe.online.cascade import Cascade


def test_cascade_search_success():
    cascade = Cascade()
    results = cascade.search(query_text="blue sky sunset", top_k=5, run_id="test_cascade")
    assert len(results) > 0
    assert results[0].evidence_id.startswith("rerank:test_cascade:")


def test_cascade_rerank_fallback_on_exception():
    # Mock reranker that throws an exception to test fallback
    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = RuntimeError("GPU memory overflow")

    cascade = Cascade(reranker=mock_reranker)
    results = cascade.search(query_text="red car driving", top_k=5, run_id="test_fallback")

    # Verify fallback executed and returned Stage B candidates smoothly
    assert len(results) > 0
    assert not results[0].evidence_id.startswith("rerank:")
