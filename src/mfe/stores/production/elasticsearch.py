"""Elasticsearch text adapter with explicit analyzer status."""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterable
from typing import Any

from .config import ElasticsearchConfig
from .contracts import HealthStatus, IngestError, IngestReport, StorageHit, StorageRecord
from .ids import elasticsearch_document_id

# Source weights for multi-source aggregation, inspired by AIC top-team
# techniques: caption carries full sentence semantics; OCR/subtitle carry
# specific surface forms; object tags are sparse by design.
# These weights are combined at score-merge time — they are NOT query-time
# boosts sent to Elasticsearch (which stays BM25-pure for reproducibility).
SOURCE_SCORE_WEIGHTS: dict[str, float] = {
    "caption": 1.0,
    "ocr": 0.85,
    "subtitle": 0.80,
    "subtitle_partial": 0.70,
    "asr": 0.75,
    "object": 0.60,
}
DEFAULT_SOURCE_WEIGHT: float = 0.70

DEV_ANALYZER = "aic_vi_dev_fallback_v2"
DEV_CHAR_FILTER = "aic_vi_strip_combining_marks_v1"


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
                "char_filter": {
                    DEV_CHAR_FILTER: {
                        "type": "pattern_replace",
                        "pattern": r"\p{M}+",
                        "replacement": "",
                    }
                },
                "analyzer": {
                    DEV_ANALYZER: {
                        "type": "custom",
                        "char_filter": [DEV_CHAR_FILTER],
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
        phrase_boost: float = 2.0,
    ) -> list[StorageHit]:
        """Search text with optional phrase-boost for compound word precision.

        When *phrase_boost* > 1 (default 2.0), a ``match_phrase`` sub-clause is
        added to the ``should`` list with the given boost.  This rewards
        Vietnamese compound words such as "áo dài" or "xe máy" appearing
        together without changing the base ``match`` recall behaviour.

        Technique observed in AIC 2024–2025 top-team solutions; boosts frames
        where the exact phrase appears over frames that only have the individual
        tokens in different fields.
        """
        # Match the normalization `StorageRecord.validate_text` applied on write.
        text = unicodedata.normalize("NFC", text)
        if not text.strip():
            raise ValueError("query text must be non-empty")
        if not 1 <= top_k <= 500:
            raise ValueError("top_k must be between 1 and 500")
        filters: list[dict[str, Any]] = []
        if video_id is not None:
            filters.append({"term": {"video_id": video_id}})
        if source is not None:
            filters.append({"term": {"source": source}})
        if phrase:
            # Caller explicitly wants phrase search — exact mode, no match fallback.
            query: dict[str, Any] = {
                "bool": {
                    "must": [{"match_phrase": {"text": text}}],
                    "filter": filters,
                }
            }
        elif phrase_boost > 1.0 and " " in text:
            # Hybrid: must match (recall) + should phrase (precision boost).
            query = {
                "bool": {
                    "must": [{"match": {"text": text}}],
                    "should": [{"match_phrase": {"text": {"query": text, "boost": phrase_boost}}}],
                    "filter": filters,
                }
            }
        else:
            query = {
                "bool": {
                    "must": [{"match": {"text": text}}],
                    "filter": filters,
                }
            }
        response = self.client.search(
            index=self.config.index,
            size=top_k,
            query=query,
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

    def search_multisource(
        self,
        text: str,
        *,
        top_k: int = 10,
        video_id: str | None = None,
        sources: list[str] | None = None,
        phrase_boost: float = 2.0,
    ) -> list[StorageHit]:
        """Search across multiple sources and aggregate by (video_id, frame_id).

        Retrieves ``top_k * 4`` raw hits from all matching source documents,
        then aggregates per logical frame using source-weighted BM25 scores
        (see ``SOURCE_SCORE_WEIGHTS``).

        This mirrors the technique used by AIC 2024/2025 top teams: a single
        query searches the entire index and the best-matching source per frame
        is selected, weighted by semantic quality (caption > ocr > subtitle >
        object).  The ``source`` field in returned hits is set to the
        highest-scoring source for that frame.

        Args:
            text:        Query string (NFC-normalized before search).
            top_k:       Number of unique frames to return after aggregation.
            video_id:    Optional exact filter.
            sources:     Allowlist of sources; ``None`` searches all sources.
            phrase_boost: See ``search()``.
        """
        text = unicodedata.normalize("NFC", text)
        if not text.strip():
            raise ValueError("query text must be non-empty")
        if not 1 <= top_k <= 500:
            raise ValueError("top_k must be between 1 and 500")

        raw_top_k = min(top_k * 4, 500)
        filters: list[dict[str, Any]] = []
        if video_id is not None:
            filters.append({"term": {"video_id": video_id}})
        if sources is not None:
            filters.append({"terms": {"source": sources}})

        if phrase_boost > 1.0 and " " in text:
            query: dict[str, Any] = {
                "bool": {
                    "must": [{"match": {"text": text}}],
                    "should": [
                        {"match_phrase": {"text": {"query": text, "boost": phrase_boost}}}
                    ],
                    "filter": filters,
                }
            }
        else:
            query = {
                "bool": {
                    "must": [{"match": {"text": text}}],
                    "filter": filters,
                }
            }

        response = self.client.search(
            index=self.config.index,
            size=raw_top_k,
            query=query,
        )

        # Aggregate: for each (video_id, frame_id) keep best weighted score.
        # Weighted score = raw_bm25 * source_weight.  The raw BM25 scale is
        # kept intact so the fusion layer can normalize consistently.
        aggregated: dict[tuple[str, int], dict[str, Any]] = {}
        for item in response["hits"]["hits"]:
            vid = item["_source"]["video_id"]
            fid = int(item["_source"]["frame_id"])
            src = item["_source"].get("source") or ""
            weight = SOURCE_SCORE_WEIGHTS.get(src, DEFAULT_SOURCE_WEIGHT)
            weighted = float(item["_score"]) * weight
            key = (vid, fid)
            if key not in aggregated or weighted > aggregated[key]["weighted"]:
                aggregated[key] = {
                    "video_id": vid,
                    "frame_id": fid,
                    "score": float(item["_score"]),
                    "weighted": weighted,
                    "source": src or None,
                    "text": item["_source"].get("text"),
                }

        ranked = sorted(
            aggregated.values(),
            key=lambda d: (-d["weighted"], d["video_id"], d["frame_id"]),
        )[:top_k]
        return [
            StorageHit(
                video_id=d["video_id"],
                frame_id=d["frame_id"],
                score=d["score"],
                source=d["source"],
                text=d["text"],
            )
            for d in ranked
        ]

    def search_by_frame(
        self,
        video_id: str,
        frame_id: int,
    ) -> list[StorageHit]:
        """Return every indexed text document for a specific (video_id, frame_id).

        Useful for debugging: shows all sources (OCR, caption, subtitle, object)
        that were indexed for a given frame, without scoring.
        """
        if not video_id:
            raise ValueError("video_id must be non-empty")
        response = self.client.search(
            index=self.config.index,
            size=50,
            query={
                "bool": {
                    "filter": [
                        {"term": {"video_id": video_id}},
                        {"term": {"frame_id": frame_id}},
                    ]
                }
            },
        )
        return [
            StorageHit(
                video_id=item["_source"]["video_id"],
                frame_id=int(item["_source"]["frame_id"]),
                score=float(item.get("_score") or 0.0),
                source=item["_source"].get("source"),
                text=item["_source"].get("text"),
            )
            for item in response["hits"]["hits"]
        ]

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
