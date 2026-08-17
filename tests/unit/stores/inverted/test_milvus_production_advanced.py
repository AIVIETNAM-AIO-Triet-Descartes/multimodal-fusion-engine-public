"""Unit tests for advanced MilvusVectorStore multi-vector search and score calibration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mfe.stores.production.config import MilvusConfig
from mfe.stores.production.contracts import StorageHit
from mfe.stores.production.milvus import MilvusVectorStore


def test_calibrate_cosine_score() -> None:
    assert MilvusVectorStore.calibrate_cosine_score(1.0) == 1.0
    assert MilvusVectorStore.calibrate_cosine_score(0.0) == 0.5
    assert MilvusVectorStore.calibrate_cosine_score(-1.0) == 0.0
    assert MilvusVectorStore.calibrate_cosine_score(1.5) == 1.0
    assert MilvusVectorStore.calibrate_cosine_score(-2.0) == 0.0


def test_search_multi_vector_empty() -> None:
    config = MilvusConfig(uri="http://localhost:19530", collection="test_col")
    store = MilvusVectorStore(config, client=MagicMock())
    assert store.search_multi_vector([]) == []


def test_search_multi_vector_weights_mismatch() -> None:
    config = MilvusConfig(uri="http://localhost:19530", collection="test_col", dimension=512)
    store = MilvusVectorStore(config, client=MagicMock())
    vec1 = [0.1] * 512
    vec2 = [0.2] * 512
    with pytest.raises(ValueError, match="weights length"):
        store.search_multi_vector([vec1, vec2], weights=[1.0])


def test_search_multi_vector_aggregation_max() -> None:
    config = MilvusConfig(uri="http://localhost:19530", collection="test_col", dimension=512)
    store = MilvusVectorStore(config, client=MagicMock())

    # Mock search calls
    vec1 = [0.1] * 512
    vec2 = [0.2] * 512

    hit1 = StorageHit(video_id="v1", frame_id=10, score=0.8)
    hit2 = StorageHit(video_id="v2", frame_id=20, score=0.7)
    hit3 = StorageHit(video_id="v1", frame_id=10, score=0.9)  # Higher score for (v1, 10) in query 2

    def mock_search(vec, top_k, video_id=None):
        if vec[0] == 0.1:
            return [hit1, hit2]
        return [hit3]

    store.search = MagicMock(side_effect=mock_search)

    results = store.search_multi_vector(
        [vec1, vec2], weights=[1.0, 1.0], aggregation="max", top_k=5
    )
    assert len(results) == 2
    top = results[0]
    assert top.video_id == "v1"
    assert top.frame_id == 10
    assert top.score == 0.9


def test_search_multi_vector_aggregation_weighted_sum() -> None:
    config = MilvusConfig(uri="http://localhost:19530", collection="test_col", dimension=512)
    store = MilvusVectorStore(config, client=MagicMock())

    vec1 = [0.1] * 512
    vec2 = [0.2] * 512

    hit1 = StorageHit(video_id="v1", frame_id=10, score=0.5)
    hit2 = StorageHit(video_id="v1", frame_id=10, score=0.8)

    def mock_search(vec, top_k, video_id=None):
        if vec[0] == 0.1:
            return [hit1]
        return [hit2]

    store.search = MagicMock(side_effect=mock_search)

    # Weights 2.0 and 1.0 -> 0.5*2.0 + 0.8*1.0 = 1.0 + 0.8 = 1.8
    results = store.search_multi_vector(
        [vec1, vec2], weights=[2.0, 1.0], aggregation="weighted_sum", top_k=5
    )
    assert len(results) == 1
    assert pytest.approx(results[0].score, 0.001) == 1.8


def test_search_multi_vector_aggregation_mean() -> None:
    config = MilvusConfig(uri="http://localhost:19530", collection="test_col", dimension=512)
    store = MilvusVectorStore(config, client=MagicMock())

    vec1 = [0.1] * 512
    vec2 = [0.2] * 512

    hit1 = StorageHit(video_id="v1", frame_id=10, score=0.6)
    hit2 = StorageHit(video_id="v1", frame_id=10, score=0.8)

    def mock_search(vec, top_k, video_id=None):
        if vec[0] == 0.1:
            return [hit1]
        return [hit2]

    store.search = MagicMock(side_effect=mock_search)

    results = store.search_multi_vector(
        [vec1, vec2], aggregation="mean", top_k=5
    )
    assert len(results) == 1
    assert pytest.approx(results[0].score, 0.001) == 0.7


def test_search_multi_vector_aggregation_invalid() -> None:
    config = MilvusConfig(uri="http://localhost:19530", collection="test_col", dimension=512)
    store = MilvusVectorStore(config, client=MagicMock())

    vec1 = [0.1] * 512
    hit1 = StorageHit(video_id="v1", frame_id=10, score=0.6)
    store.search = MagicMock(return_value=[hit1])

    with pytest.raises(ValueError, match="unknown aggregation"):
        store.search_multi_vector([vec1], aggregation="invalid_mode")

