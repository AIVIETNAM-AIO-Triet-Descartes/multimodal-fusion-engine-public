"""Live-service gates. Opt in with MFE_RUN_SEARCH_INTEGRATION=1."""

from __future__ import annotations

import os
import unicodedata
import uuid

import numpy as np
import pytest

if os.getenv("MFE_RUN_SEARCH_INTEGRATION") != "1":
    pytest.skip("live Milvus/Elasticsearch integration is opt-in", allow_module_level=True)

pytest.importorskip("pymilvus")
pytest.importorskip("elasticsearch")

from mfe.stores.production.config import ElasticsearchConfig, MilvusConfig  # noqa: E402
from mfe.stores.production.contracts import StorageRecord  # noqa: E402
from mfe.stores.production.elasticsearch import ElasticsearchTextStore  # noqa: E402
from mfe.stores.production.ingestion import DualStoreIngestor  # noqa: E402
from mfe.stores.production.milvus import MilvusVectorStore  # noqa: E402

RUN_ID = uuid.uuid4().hex
MILVUS_NAME = f"mfe_test_storage_{RUN_ID}"
ELASTIC_NAME = f"mfe-test-storage-{RUN_ID}"


@pytest.fixture(scope="module")
def stores() -> tuple[MilvusVectorStore, ElasticsearchTextStore]:
    assert MILVUS_NAME.startswith("mfe_test_")
    assert ELASTIC_NAME.startswith("mfe-test-")
    vector = MilvusVectorStore(
        MilvusConfig.from_env().model_copy(update={"collection": MILVUS_NAME})
    )
    text = ElasticsearchTextStore(
        ElasticsearchConfig.from_env().model_copy(update={"index": ELASTIC_NAME})
    )
    try:
        vector.ensure_schema()
        text.ensure_schema()
        yield vector, text
    finally:
        if vector.client.has_collection(collection_name=MILVUS_NAME):
            vector.client.drop_collection(collection_name=MILVUS_NAME)
        if text.client.indices.exists(index=ELASTIC_NAME):
            text.client.indices.delete(index=ELASTIC_NAME)


@pytest.mark.integration
def test_real_insert_query_filter_duplicate_and_cross_backend(
    stores: tuple[MilvusVectorStore, ElasticsearchTextStore],
) -> None:
    vector, text = stores
    record = StorageRecord.from_values(
        video_id="__mfe_integration__",
        frame_id=42,
        embedding=[1.0] + [0.0] * 511,
        text="Thành phố Hồ Chí Minh",
        source="caption",
        sv=0,
        ev=100,
    )
    other = StorageRecord.from_values(
        video_id="__mfe_integration__",
        frame_id=43,
        embedding=[0.0, 1.0] + [0.0] * 510,
        text="Một nội dung khác",
        source="caption",
        sv=0,
        ev=100,
    )
    first = DualStoreIngestor(vector, text).ingest([record, other])
    second = DualStoreIngestor(vector, text).ingest([record, other])
    assert first.milvus.succeeded == second.milvus.succeeded == 2
    assert second.milvus.updated == 2
    text.client.indices.refresh(index=text.config.index)
    vector_hits = vector.search(record.embedding or (), video_id=record.video_id)
    text_hits = text.search("thanh pho", video_id=record.video_id, source="caption")
    assert (vector_hits[0].video_id, vector_hits[0].frame_id) == (
        text_hits[0].video_id,
        text_hits[0].frame_id,
    )
    query = np.asarray([1.0] + [0.0] * 511, dtype=np.float32)
    corpus = np.asarray([record.embedding, other.embedding], dtype=np.float32)
    oracle = corpus @ query / (np.linalg.norm(corpus, axis=1) * np.linalg.norm(query))
    assert [hit.frame_id for hit in vector_hits[:2]] == [42, 43]
    assert vector_hits[0].score == pytest.approx(float(oracle[0]), abs=1e-5)

    def analyze(value: str) -> list[str]:
        response = text.client.indices.analyze(
            index=text.config.index, analyzer=text.analyzer_name, text=value
        )
        return [token["token"] for token in response["tokens"]]

    if text.config.analyzer_mode == "dev_fallback":
        assert analyze("THÀNH PHỐ") == analyze("thanh pho") == ["thanh", "pho"]
        assert "2026" in analyze("TP.HCM 2026!")

        # Analyzer v2 removes decomposed combining marks before tokenization;
        # application boundaries also normalize NFC as defence in depth.
        nfd_tokens = analyze(unicodedata.normalize("NFD", "Thành phố"))
        assert nfd_tokens == ["thanh", "pho"]

        # ...and this is the guarantee that actually matters: a decomposed query
        # must still find a document that was written in precomposed form.
        nfd_record = StorageRecord.from_values(
            video_id=record.video_id,
            frame_id=4242,
            text="Thành phố Hồ Chí Minh",
            source="caption",
        )
        text.ingest([nfd_record])
        text.client.indices.refresh(index=text.config.index)
        nfd_hits = text.search(
            unicodedata.normalize("NFD", "Thành phố"),
            video_id=record.video_id,
            source="caption",
        )
        assert any(hit.frame_id == 4242 for hit in nfd_hits), (
            "a decomposed query found nothing — NFC normalization regressed"
        )
    else:
        assert analyze("Thành phố")


@pytest.mark.integration
def test_live_health_and_count_endpoints(
    stores: tuple[MilvusVectorStore, ElasticsearchTextStore],
) -> None:
    vector, text = stores
    assert vector.health().index_ready
    assert text.health().index_ready
    assert vector.count() >= 1
    assert text.count() >= 1
