"""Unit tests for ElasticsearchTextStore upgrades — no live service needed.

Tests verify:
- phrase_boost mode: hybrid must+should query is built correctly
- phrase=True mode: only match_phrase is built
- phrase_boost skipped for single-word queries
- search_multisource aggregation with source weights
- search_by_frame exact filter
- DualStoreIngestor progress callback
- DualStoreIngestor ingest_with_resume skips already-ingested keys
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mfe.stores.production.contracts import IngestReport, StorageHit, StorageRecord
from mfe.stores.production.elasticsearch import (
    SOURCE_SCORE_WEIGHTS,
    ElasticsearchTextStore,
)
from mfe.stores.production.ingestion import DualIngestReport, DualStoreIngestor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _es_config():
    from mfe.stores.production.config import ElasticsearchConfig

    return ElasticsearchConfig(url="http://localhost:9200", index="test_v2")


def _make_store(response_hits: list[dict[str, Any]]) -> tuple[ElasticsearchTextStore, MagicMock]:
    """Return a store wired to a fake client that returns *response_hits*."""
    fake_client = MagicMock()
    fake_response = {"hits": {"hits": response_hits}}
    fake_client.search.return_value = fake_response
    store = ElasticsearchTextStore(config=_es_config(), client=fake_client)
    return store, fake_client


def _hit(vid: str, fid: int, score: float, source: str, text: str = "x") -> dict[str, Any]:
    return {
        "_score": score,
        "_source": {"video_id": vid, "frame_id": fid, "source": source, "text": text},
    }


# ---------------------------------------------------------------------------
# search() — phrase_boost
# ---------------------------------------------------------------------------

class TestSearchPhraseBoost:
    def test_multi_word_query_adds_should_clause(self):
        store, client = _make_store([])
        store.search("áo dài vàng", top_k=5)

        call_kwargs = client.search.call_args[1]
        query = call_kwargs["query"]
        assert "should" in query["bool"], "phrase_boost should add a should clause"
        should = query["bool"]["should"]
        assert any("match_phrase" in clause for clause in should)
        # must clause still uses match (not match_phrase)
        must = query["bool"]["must"]
        assert any("match" in clause for clause in must)

    def test_single_word_no_should_clause(self):
        store, client = _make_store([])
        store.search("người", top_k=5)

        query = client.search.call_args[1]["query"]
        # No space → no phrase boost → bool.should absent
        assert "should" not in query.get("bool", {})

    def test_phrase_true_uses_only_match_phrase(self):
        store, client = _make_store([])
        store.search("áo dài", top_k=5, phrase=True)

        query = client.search.call_args[1]["query"]
        must = query["bool"]["must"]
        assert all("match_phrase" in clause for clause in must)
        assert "should" not in query["bool"]

    def test_phrase_boost_zero_disables_should(self):
        store, client = _make_store([])
        store.search("áo dài", top_k=5, phrase_boost=0.0)

        query = client.search.call_args[1]["query"]
        assert "should" not in query.get("bool", {})

    def test_nfc_normalization_applied(self):
        """NFC applied: NFD input still reaches ES in NFC form."""
        import unicodedata

        raw_nfd = unicodedata.normalize("NFD", "Hồ Chí Minh")
        store, client = _make_store([])
        store.search(raw_nfd, top_k=5)

        # The text sent to ES must be NFC
        query = client.search.call_args[1]["query"]
        must_text = query["bool"]["must"][0]["match"]["text"]
        assert must_text == unicodedata.normalize("NFC", raw_nfd)


# ---------------------------------------------------------------------------
# search_multisource() — aggregation logic
# ---------------------------------------------------------------------------

class TestSearchMultisource:
    def test_aggregates_two_sources_for_same_frame(self):
        """When both OCR and caption match the same frame, keep best weighted."""
        caption_score = 10.0
        ocr_score = 12.0  # Higher raw, but lower weight → caption wins
        hits = [
            _hit("L01_V001", 100, caption_score, "caption", "áo dài vàng"),
            _hit("L01_V001", 100, ocr_score, "ocr", "áo dài"),
        ]
        store, client = _make_store(hits)
        client.search.return_value = {"hits": {"hits": hits}}
        results = store.search_multisource("áo dài", top_k=5)

        # Caption weighted = 10.0 * 1.0 = 10.0
        # OCR weighted    = 12.0 * 0.85 = 10.2 → OCR wins
        assert len(results) == 1
        assert results[0].video_id == "L01_V001"
        assert results[0].frame_id == 100
        assert results[0].source == "ocr"  # higher weighted

    def test_returns_distinct_frames_sorted_by_weighted_score(self):
        hits = [
            _hit("L01_V001", 200, 5.0, "caption"),
            _hit("L01_V001", 300, 9.0, "ocr"),  # weighted=9*0.85=7.65
            _hit("L01_V002", 400, 6.0, "caption"),  # weighted=6*1.0=6.0
        ]
        store, client = _make_store(hits)
        client.search.return_value = {"hits": {"hits": hits}}
        results = store.search_multisource("test", top_k=10)

        assert len(results) == 3
        # OCR frame (7.65) > L01_V002 caption (6.0) > L01_V001 caption (5.0)
        assert results[0].frame_id == 300
        assert results[1].frame_id == 400
        assert results[2].frame_id == 200

    def test_sources_filter_passed_to_es(self):
        store, client = _make_store([])
        store.search_multisource("test", top_k=5, sources=["caption", "ocr"])

        query = client.search.call_args[1]["query"]
        filters = query["bool"]["filter"]
        source_filter = next((f for f in filters if "terms" in f), None)
        assert source_filter is not None
        assert set(source_filter["terms"]["source"]) == {"caption", "ocr"}

    def test_unknown_source_uses_default_weight(self):
        hits = [_hit("V1", 1, 10.0, "new_source_type")]
        store, client = _make_store(hits)
        client.search.return_value = {"hits": {"hits": hits}}
        results = store.search_multisource("test", top_k=5)
        # Should not raise, uses DEFAULT_SOURCE_WEIGHT
        assert len(results) == 1

    def test_top_k_limits_output(self):
        hits = [_hit("V1", i, float(10 - i), "caption") for i in range(10)]
        store, client = _make_store(hits)
        client.search.return_value = {"hits": {"hits": hits}}
        results = store.search_multisource("test", top_k=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# search_by_frame()
# ---------------------------------------------------------------------------

class TestSearchByFrame:
    def test_builds_exact_filter_query(self):
        store, client = _make_store([])
        store.search_by_frame("L01_V001", 500)

        query = client.search.call_args[1]["query"]
        filters = query["bool"]["filter"]
        assert {"term": {"video_id": "L01_V001"}} in filters
        assert {"term": {"frame_id": 500}} in filters

    def test_returns_all_sources_for_frame(self):
        hits = [
            _hit("L01_V001", 500, 1.0, "caption"),
            _hit("L01_V001", 500, 1.0, "ocr"),
            _hit("L01_V001", 500, 1.0, "subtitle"),
        ]
        store, client = _make_store(hits)
        client.search.return_value = {"hits": {"hits": hits}}
        results = store.search_by_frame("L01_V001", 500)

        assert len(results) == 3
        sources = {r.source for r in results}
        assert sources == {"caption", "ocr", "subtitle"}

    def test_raises_on_empty_video_id(self):
        store, client = _make_store([])
        with pytest.raises(ValueError, match="video_id must be non-empty"):
            store.search_by_frame("", 100)


# ---------------------------------------------------------------------------
# DualStoreIngestor — progress_callback
# ---------------------------------------------------------------------------

class TestDualStoreIngestorProgress:
    def _make_ingestor(self) -> tuple[DualStoreIngestor, MagicMock, MagicMock]:
        mock_milvus = MagicMock()
        mock_es = MagicMock()
        mock_milvus.ingest.return_value = IngestReport(
            backend="milvus", attempted=1, inserted=1
        )
        mock_es.ingest.return_value = IngestReport(
            backend="elasticsearch", attempted=1, inserted=1
        )
        ingestor = DualStoreIngestor(vector_store=mock_milvus, text_store=mock_es)
        return ingestor, mock_milvus, mock_es

    def _make_record(self, video_id: str, fid: int) -> StorageRecord:
        return StorageRecord.from_values(
            video_id=video_id,
            frame_id=fid,
            embedding=[0.1] * 512,
            text="áo dài vàng rực rỡ",
            source="caption",
        )

    def test_progress_callback_called_after_each_batch(self):
        ingestor, _, _ = self._make_ingestor()
        records = [self._make_record("V1", i) for i in range(5)]
        calls: list[tuple[int, DualIngestReport]] = []

        ingestor.ingest_batches(records, batch_size=2, progress_callback=lambda n, r: calls.append((n, r)))

        # 5 records, batch_size=2 → 3 flushes (2+2+1)
        assert len(calls) == 3
        assert calls[0][0] == 2
        assert calls[1][0] == 4
        assert calls[2][0] == 5

    def test_no_callback_works_as_before(self):
        ingestor, _, _ = self._make_ingestor()
        records = [self._make_record("V1", i) for i in range(3)]
        report = ingestor.ingest_batches(records, batch_size=2)
        assert report.logical_records == 3


# ---------------------------------------------------------------------------
# DualStoreIngestor — ingest_with_resume
# ---------------------------------------------------------------------------

class TestDualStoreIngestorResume:
    def _make_ingestor(self) -> tuple[DualStoreIngestor, MagicMock, MagicMock]:
        mock_milvus = MagicMock()
        mock_es = MagicMock()
        mock_milvus.ingest.return_value = IngestReport(
            backend="milvus", attempted=1, inserted=1
        )
        mock_es.ingest.return_value = IngestReport(
            backend="elasticsearch", attempted=1, inserted=1
        )
        ingestor = DualStoreIngestor(vector_store=mock_milvus, text_store=mock_es)
        return ingestor, mock_milvus, mock_es

    def _make_record(self, video_id: str, fid: int) -> StorageRecord:
        return StorageRecord.from_values(
            video_id=video_id,
            frame_id=fid,
            embedding=[0.1] * 512,
        )

    def test_skips_already_ingested_keys(self):
        ingestor, mock_milvus, mock_es = self._make_ingestor()
        already_done: set[tuple[str, int]] = {("V1", 0), ("V1", 1)}
        records = [self._make_record("V1", i) for i in range(4)]

        report = ingestor.ingest_with_resume(
            records, batch_size=10, ingested_keys=already_done
        )

        # Only records 2 and 3 should be ingested
        assert report.logical_records == 2

    def test_updates_ingested_keys_set_in_place(self):
        ingestor, _, _ = self._make_ingestor()
        seen: set[tuple[str, int]] = set()
        records = [self._make_record("V1", i) for i in range(3)]

        ingestor.ingest_with_resume(records, ingested_keys=seen)

        assert seen == {("V1", 0), ("V1", 1), ("V1", 2)}

    def test_empty_ingested_keys_defaults_ingest_all(self):
        ingestor, _, _ = self._make_ingestor()
        records = [self._make_record("V1", i) for i in range(3)]
        report = ingestor.ingest_with_resume(records)
        assert report.logical_records == 3


# ---------------------------------------------------------------------------
# SOURCE_SCORE_WEIGHTS sanity
# ---------------------------------------------------------------------------

def test_source_score_weights_ordering():
    """caption must outweigh all others — fundamental design invariant."""
    caption_w = SOURCE_SCORE_WEIGHTS["caption"]
    for src, w in SOURCE_SCORE_WEIGHTS.items():
        if src != "caption":
            assert caption_w >= w, f"caption weight must be >= {src} weight"
