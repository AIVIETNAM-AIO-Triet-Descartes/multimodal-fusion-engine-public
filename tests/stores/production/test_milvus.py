from __future__ import annotations

from typing import Any

import pytest

from mfe.stores.production.config import MilvusConfig
from mfe.stores.production.contracts import StorageRecord
from mfe.stores.production.ids import milvus_primary_key
from mfe.stores.production.milvus import MilvusVectorStore


def record(video: str, frame: int) -> StorageRecord:
    return StorageRecord.from_values(
        video_id=video,
        frame_id=frame,
        embedding=[1.0] + [0.0] * 511,
        sv=0,
        ev=100,
    )


class FakeMilvus:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.search_kwargs: dict[str, Any] = {}

    def has_collection(self, **_: Any) -> bool:
        return True

    def describe_collection(self, **_: Any) -> dict[str, Any]:
        names = MilvusVectorStore.expected_fields()
        return {
            "fields": [
                {
                    "name": name,
                    "type": kind,
                    "params": {"dim": "512"} if name == "embedding" else {},
                }
                for name, kind in names.items()
            ]
        }

    def load_collection(self, **_: Any) -> None:
        pass

    def query(self, *, filter: str, output_fields: list[str], **_: Any) -> list[dict[str, Any]]:
        if output_fields == ["count(*)"]:
            return [{"count(*)": len(self.rows)}]
        return [{"pk": key} for key in self.rows if key in filter]

    def upsert(self, *, data: list[dict[str, Any]], **_: Any) -> None:
        for row in data:
            self.rows[row["pk"]] = row

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.search_kwargs = kwargs
        return [
            [
                {"distance": 0.8, "entity": {"video_id": "b", "frame_id": 2}},
                {"distance": 0.8, "entity": {"video_id": "a", "frame_id": 3}},
            ]
        ]


def test_upsert_is_idempotent_and_reports_existing_rows() -> None:
    client = FakeMilvus()
    store = MilvusVectorStore(MilvusConfig(), client=client)
    first = store.ingest([record("x", 1)])
    second = store.ingest([record("x", 1)])
    assert (first.inserted, first.updated) == (1, 0)
    assert (second.inserted, second.updated) == (0, 1)
    assert store.count() == 1


def test_duplicate_batch_is_rejected_before_mutation() -> None:
    client = FakeMilvus()
    store = MilvusVectorStore(MilvusConfig(), client=client)
    with pytest.raises(ValueError, match="duplicate"):
        store.ingest([record("x", 1), record("x", 1)])
    assert not client.rows


def test_search_uses_cosine_filter_and_deterministic_tie_break() -> None:
    client = FakeMilvus()
    store = MilvusVectorStore(MilvusConfig(), client=client)
    hits = store.search([1.0] + [0.0] * 511, video_id='x" or true')
    assert [hit.video_id for hit in hits] == ["a", "b"]
    assert client.search_kwargs["search_params"]["metric_type"] == "COSINE"
    assert client.search_kwargs["filter"] == 'video_id == "x\\\" or true"'


def test_primary_key_in_payload_is_deterministic() -> None:
    client = FakeMilvus()
    MilvusVectorStore(MilvusConfig(), client=client).ingest([record("x", 7)])
    assert milvus_primary_key("x", 7) in client.rows


def test_transient_upsert_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeMilvus()
    calls = 0
    original = client.upsert

    def flaky(**kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary timeout")
        original(**kwargs)

    client.upsert = flaky  # type: ignore[method-assign]
    monkeypatch.setattr("mfe.stores.production.milvus.time.sleep", lambda _: None)
    report = MilvusVectorStore(MilvusConfig(), client=client).ingest([record("x", 9)])
    assert calls == 2
    assert report.succeeded == 1
