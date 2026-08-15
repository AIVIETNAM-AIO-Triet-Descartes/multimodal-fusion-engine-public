"""Elasticsearch text adapter with explicit analyzer status."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .config import ElasticsearchConfig
from .contracts import HealthStatus, IngestError, IngestReport, StorageHit, StorageRecord
from .ids import elasticsearch_document_id

DEV_ANALYZER = "aic_vi_dev_fallback_v1"


def build_elasticsearch_client(config: ElasticsearchConfig) -> Any:
    try:
        from elasticsearch import Elasticsearch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install mfe[production-stores] to use Elasticsearch") from exc
    kwargs: dict[str, Any] = {"request_timeout": config.request_timeout_seconds}
    if config.api_key is not None:
        kwargs["api_key"] = config.api_key.get_secret_value()
    elif config.username and config.password:
        kwargs["basic_auth"] = (config.username, config.password.get_secret_value())
    return Elasticsearch(config.url, **kwargs)


class ElasticsearchTextStore:
    def __init__(
        self,
        config: ElasticsearchConfig,
        client: Any | None = None,
        bulk_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.client = client or build_elasticsearch_client(config)
        self._bulk_fn = bulk_fn

    @property
    def analyzer_name(self) -> str:
        if self.config.analyzer_mode == "plugin_required":
            return self.config.plugin_analyzer
        return DEV_ANALYZER

    def index_definition(self) -> dict[str, Any]:
        analysis: dict[str, Any] = {}
        if self.config.analyzer_mode == "dev_fallback":
            analysis = {
                "analyzer": {
                    DEV_ANALYZER: {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    }
                }
            }
        return {
            "settings": {"analysis": analysis, "index": {"number_of_shards": 1}},
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "video_id": {"type": "keyword"},
                    "frame_id": {"type": "long"},
                    "text": {
                        "type": "text",
                        "analyzer": self.analyzer_name,
                        "search_analyzer": self.analyzer_name,
                    },
                    "source": {"type": "keyword"},
                    "sv": {"type": "long"},
                    "ev": {"type": "long"},
                    "schema_version": {"type": "keyword"},
                    "text_pipeline_fingerprint": {"type": "keyword"},
                },
            },
        }

    def ensure_schema(self) -> None:
        name = self.config.index
        if not self.client.indices.exists(index=name):
            self.client.indices.create(index=name, **self.index_definition())
        else:
            mapping = self.client.indices.get_mapping(index=name)[name]["mappings"]
            expected = self.index_definition()["mappings"]["properties"]
            actual = mapping.get("properties", {})
            missing = sorted(set(expected) - set(actual))
            wrong = sorted(
                field
                for field, spec in expected.items()
                if actual.get(field, {}).get("type") != spec["type"]
            )
            text_analyzer = actual.get("text", {}).get("analyzer")
            dynamic = mapping.get("dynamic")
            if missing or wrong or text_analyzer != self.analyzer_name or dynamic != "strict":
                raise RuntimeError(
                    "Elasticsearch schema mismatch: "
                    f"missing={missing}, wrong_type={wrong}, dynamic={dynamic!r}, "
                    f"text_analyzer={text_analyzer!r}; "
                    "refusing destructive migration"
                )
        self.client.indices.analyze(
            index=name,
            analyzer=self.analyzer_name,
            text="Thành phố Hồ Chí Minh",
        )

    def ingest(self, records: Iterable[StorageRecord]) -> IngestReport:
        text_records = [record for record in records if record.text is not None]
        keys = [(record.video_id, record.frame_id, record.source) for record in text_records]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate (video_id, frame_id, source) in one Elasticsearch batch")
        if not text_records:
            return IngestReport(backend="elasticsearch", attempted=0)

        actions = [
            {
                "_op_type": "index",
                "_index": self.config.index,
                "_id": elasticsearch_document_id(
                    record.video_id, record.frame_id, record.source or ""
                ),
                "_source": {
                    "video_id": record.video_id,
                    "frame_id": record.frame_id,
                    "text": record.text,
                    "source": record.source,
                    "sv": record.sv,
                    "ev": record.ev,
                    "schema_version": self.config.schema_version,
                    "text_pipeline_fingerprint": self.config.text_pipeline_fingerprint,
                },
            }
            for record in text_records
        ]
        if self._bulk_fn is None:
            try:
                from elasticsearch.helpers import streaming_bulk
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("install mfe[production-stores] to ingest text") from exc
            bulk_fn = streaming_bulk
        else:
            bulk_fn = self._bulk_fn

        errors: list[IngestError] = []
        inserted = 0
        updated = 0
        seen = 0
        for ok, item in bulk_fn(
            self.client,
            actions,
            raise_on_error=False,
            raise_on_exception=False,
            max_retries=3,
            retry_on_status=(429, 502, 503, 504),
        ):
            operation = next(iter(item.values()))
            if ok:
                if operation.get("result") == "created":
                    inserted += 1
                else:
                    updated += 1
                seen += 1
                continue
            status = int(operation.get("status", 0))
            source = actions[seen]["_source"]
            errors.append(
                IngestError(
                    logical_key=(source["video_id"], source["frame_id"]),
                    backend="elasticsearch",
                    message=str(operation.get("error", "bulk item failed"))[:500],
                    retryable=status in {429, 502, 503, 504},
                )
            )
            seen += 1
        return IngestReport(
            backend="elasticsearch",
            attempted=len(actions),
            inserted=inserted,
            updated=updated,
            failed=len(errors),
            errors=tuple(errors),
        )

    def search(
        self,
        text: str,
        *,
        top_k: int = 10,
        video_id: str | None = None,
        source: str | None = None,
        phrase: bool = False,
    ) -> list[StorageHit]:
        if not text.strip():
            raise ValueError("query text must be non-empty")
        if not 1 <= top_k <= 500:
            raise ValueError("top_k must be between 1 and 500")
        filters: list[dict[str, Any]] = []
        if video_id is not None:
            filters.append({"term": {"video_id": video_id}})
        if source is not None:
            filters.append({"term": {"source": source}})
        match_kind = "match_phrase" if phrase else "match"
        response = self.client.search(
            index=self.config.index,
            size=top_k,
            query={"bool": {"must": [{match_kind: {"text": text}}], "filter": filters}},
        )
        hits = [
            StorageHit(
                video_id=item["_source"]["video_id"],
                frame_id=int(item["_source"]["frame_id"]),
                score=float(item["_score"]),
                source=item["_source"].get("source"),
                text=item["_source"].get("text"),
            )
            for item in response["hits"]["hits"]
        ]
        return sorted(hits, key=lambda hit: (-hit.score, hit.video_id, hit.frame_id))

    def count(self) -> int:
        return int(self.client.count(index=self.config.index)["count"])

    def health(self) -> HealthStatus:
        try:
            connection = bool(self.client.ping())
            exists = connection and bool(self.client.indices.exists(index=self.config.index))
            analyzer = "not_checked"
            if exists:
                try:
                    self.client.indices.analyze(
                        index=self.config.index,
                        analyzer=self.analyzer_name,
                        text="Việt Nam",
                    )
                    analyzer = (
                        "dev_fallback_ready"
                        if self.config.analyzer_mode == "dev_fallback"
                        else "plugin_ready"
                    )
                except Exception:
                    analyzer = "failed"
            return HealthStatus(
                backend="elasticsearch",
                connection_ok=connection,
                schema_ready=exists,
                index_ready=exists,
                analyzer_status=analyzer,
                message="ready" if exists and analyzer != "failed" else "not ready",
            )
        except Exception as exc:
            return HealthStatus(
                backend="elasticsearch",
                connection_ok=False,
                schema_ready=False,
                index_ready=False,
                analyzer_status="unavailable",
                message=f"{type(exc).__name__}: connection failed",
            )
