from __future__ import annotations

import unicodedata
from typing import Any

import pytest

from mfe.stores.production.config import ElasticsearchConfig
from mfe.stores.production.contracts import StorageRecord
from mfe.stores.production.elasticsearch import (
    DEV_ANALYZER,
    DEV_CHAR_FILTER,
    ElasticsearchTextStore,
)


def record(video: str = "x", frame: int = 1, source: str = "ocr") -> StorageRecord:
    return StorageRecord(video_id=video, frame_id=frame, text="Thành phố", source=source)


class FakeIndices:
    def __init__(self) -> None:
        self.created: dict[str, Any] | None = None

    def exists(self, **_: Any) -> bool:
        return self.created is not None

    def create(self, **kwargs: Any) -> None:
        self.created = kwargs

    def analyze(self, **_: Any) -> dict[str, Any]:
        return {"tokens": [{"token": "thanh"}]}

    def get_mapping(self, *, index: str) -> dict[str, Any]:
        assert self.created is not None
        return {index: {"mappings": self.created["mappings"]}}


class FakeElasticsearch:
    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.search_kwargs: dict[str, Any] = {}

    def ping(self) -> bool:
        return True

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.search_kwargs = kwargs
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 1.5,
                        "_source": {
                            "video_id": "b",
                            "frame_id": 2,
                            "text": "b",
                            "source": "ocr",
                        },
                    },
                    {
                        "_score": 1.5,
                        "_source": {
                            "video_id": "a",
                            "frame_id": 3,
                            "text": "a",
                            "source": "ocr",
                        },
                    },
                ]
            }
        }


def test_dev_fallback_is_explicit_and_schema_is_strict() -> None:
    client = FakeElasticsearch()
    store = ElasticsearchTextStore(ElasticsearchConfig(), client=client)
    store.ensure_schema()
    definition = client.indices.created
    assert definition is not None
    assert definition["mappings"]["dynamic"] == "strict"
    assert definition["mappings"]["properties"]["text"]["analyzer"] == DEV_ANALYZER
    analysis = definition["settings"]["analysis"]
    assert analysis["analyzer"][DEV_ANALYZER]["char_filter"] == [DEV_CHAR_FILTER]
    assert analysis["char_filter"][DEV_CHAR_FILTER]["pattern"] == r"\p{M}+"


def test_plugin_mode_does_not_claim_or_define_fallback() -> None:
    config = ElasticsearchConfig(analyzer_mode="plugin_required", plugin_analyzer="vi_custom")
    definition = ElasticsearchTextStore(config, client=FakeElasticsearch()).index_definition()
    assert definition["settings"]["analysis"] == {}
    assert definition["mappings"]["properties"]["text"]["analyzer"] == "vi_custom"


def test_bulk_failure_is_visible_in_report() -> None:
    def bulk(_: Any, actions: list[dict[str, Any]], **__: Any) -> Any:
        assert len(actions) == 2
        yield True, {"index": {"status": 201}}
        yield False, {"index": {"status": 429, "error": "busy"}}

    store = ElasticsearchTextStore(ElasticsearchConfig(), client=FakeElasticsearch(), bulk_fn=bulk)
    report = store.ingest([record(), record(frame=2)])
    assert (report.updated, report.failed) == (1, 1)
    assert report.errors[0].retryable
    assert report.errors[0].logical_key == ("x", 2)


def test_duplicate_source_row_fails_before_bulk() -> None:
    called = False

    def bulk(*_: Any, **__: Any) -> Any:
        nonlocal called
        called = True
        return iter(())

    store = ElasticsearchTextStore(ElasticsearchConfig(), client=FakeElasticsearch(), bulk_fn=bulk)
    with pytest.raises(ValueError, match="duplicate"):
        store.ingest([record(), record()])
    assert not called


def test_search_filters_and_stable_tie_break() -> None:
    client = FakeElasticsearch()
    store = ElasticsearchTextStore(ElasticsearchConfig(), client=client)
    hits = store.search("thành phố", video_id="x", source="ocr", phrase=True)
    assert [hit.video_id for hit in hits] == ["a", "b"]
    bool_query = client.search_kwargs["query"]["bool"]
    assert "match_phrase" in bool_query["must"][0]
    assert bool_query["filter"] == [{"term": {"video_id": "x"}}, {"term": {"source": "ocr"}}]


def test_search_normalizes_decomposed_unicode_to_nfc() -> None:
    client = FakeElasticsearch()
    store = ElasticsearchTextStore(ElasticsearchConfig(), client=client)
    store.search(unicodedata.normalize("NFD", "Thành phố"))
    match = client.search_kwargs["query"]["bool"]["must"][0]["match"]
    assert match["text"] == "Thành phố"
