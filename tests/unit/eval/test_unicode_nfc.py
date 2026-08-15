"""Locks NFC normalization across the three boundaries text crosses.

The bug this prevents was found on a live Elasticsearch 9.4.4, not reasoned
about: `asciifolding` folds precomposed characters only, so "Thành phố" analyses
to ["thanh", "pho"] as NFC and to ["thành", "phố"] as NFD. A decomposed query
against a precomposed index returns **zero** hits — not fewer, zero — with no
exception anywhere.

Vietnamese reaches us decomposed more often than one expects: Unikey ships a
"Unicode tổ hợp" output mode, and copy-paste from web or PDF can carry it. So all
three boundaries normalize: index write, query read, and query-set merge.
"""

from __future__ import annotations

import unicodedata

from mfe.eval.merge_queryset import _decomposed_fields, normalize_query
from mfe.eval.official import load_official_queryset
from mfe.stores.production.contracts import StorageRecord

NFC = "Thành phố Hồ Chí Minh"
NFD = unicodedata.normalize("NFD", NFC)


def test_the_two_forms_really_do_differ():
    """Guards the premise: if these ever compare equal, the rest is vacuous."""
    assert NFC != NFD
    assert len(NFD) > len(NFC)
    assert unicodedata.normalize("NFC", NFD) == NFC


class TestStorageRecord:
    def test_text_is_normalized_on_write(self):
        record = StorageRecord.from_values(
            video_id="L21_V001", frame_id=90, text=NFD, source="ocr"
        )
        assert record.text == NFC

    def test_already_nfc_text_is_untouched(self):
        record = StorageRecord.from_values(
            video_id="L21_V001", frame_id=90, text=NFC, source="ocr"
        )
        assert record.text == NFC

    def test_two_forms_of_the_same_text_produce_equal_records(self):
        """The property that matters: form must not change identity."""
        a = StorageRecord.from_values(video_id="L21_V001", frame_id=90, text=NFC, source="ocr")
        b = StorageRecord.from_values(video_id="L21_V001", frame_id=90, text=NFD, source="ocr")
        assert a == b


class TestSearchNormalizes:
    def test_query_is_normalized_before_it_reaches_elasticsearch(self):
        """No live service needed — capture what the client was asked for."""
        captured: dict = {}

        class FakeClient:
            def search(self, **kwargs):
                captured.update(kwargs)
                return {"hits": {"hits": []}}

        from mfe.stores.production.config import ElasticsearchConfig
        from mfe.stores.production.elasticsearch import ElasticsearchTextStore

        store = ElasticsearchTextStore(ElasticsearchConfig(), client=FakeClient())
        store.search(NFD, top_k=5)

        sent = captured["query"]["bool"]["must"][0]["match"]["text"]
        assert sent == NFC, "a decomposed query would have matched nothing"


class TestMergeQueryset:
    def _query(self, **overrides):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        payload = {
            "query_id": "triet_kis_00",
            "type": "KIS",
            "text": NFC,
            "gt_video_id": "L21_V001",
            "gt_frame_range": [0, 100],
        }
        payload.update(overrides)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "triet.jsonl"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return load_official_queryset(path)[0]

    def test_flags_a_decomposed_text_field(self):
        assert _decomposed_fields(self._query(text=NFD)) == ["text"]

    def test_flags_a_decomposed_note(self):
        assert "note" in _decomposed_fields(self._query(note=NFD))

    def test_clean_query_is_not_flagged(self):
        assert _decomposed_fields(self._query()) == []

    def test_normalize_rewrites_the_text(self):
        normalized = normalize_query(self._query(text=NFD))
        assert normalized.text == NFC
        assert _decomposed_fields(normalized) == []

    def test_normalize_returns_the_same_object_when_nothing_changes(self):
        query = self._query()
        assert normalize_query(query) is query

    def test_normalize_preserves_everything_else(self):
        query = self._query(text=NFD)
        normalized = normalize_query(query)
        assert normalized.query_id == query.query_id
        assert normalized.gt_video_id == query.gt_video_id
        assert normalized.gt_frame_range == query.gt_frame_range
