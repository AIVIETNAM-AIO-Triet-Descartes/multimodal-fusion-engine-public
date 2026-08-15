"""Milvus vector adapter with strict schema checks and idempotent upserts."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any

from .config import MilvusConfig
from .contracts import EMBEDDING_DIM, HealthStatus, IngestReport, StorageHit, StorageRecord
from .ids import milvus_primary_key, milvus_string_literal


def build_milvus_client(config: MilvusConfig) -> Any:
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("install mfe[production-stores] to use Milvus") from exc
    kwargs: dict[str, Any] = {"uri": config.uri}
    if config.token is not None:
        kwargs["token"] = config.token.get_secret_value()
    return MilvusClient(**kwargs)


class MilvusVectorStore:
    def __init__(self, config: MilvusConfig, client: Any | None = None) -> None:
        self.config = config
        self.client = client or build_milvus_client(config)

    def _schema(self) -> tuple[Any, Any]:
        try:
            from pymilvus import DataType
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install mfe[production-stores] to create Milvus schema") from exc
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("video_id", DataType.VARCHAR, max_length=255)
        schema.add_field("frame_id", DataType.INT64)
        schema.add_field("sv", DataType.INT64, nullable=True)
        schema.add_field("ev", DataType.INT64, nullable=True)
        schema.add_field("model_fingerprint", DataType.VARCHAR, max_length=128)
        schema.add_field("schema_version", DataType.VARCHAR, max_length=16)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        indexes = self.client.prepare_index_params()
        # AUTOINDEX is a vector index type; VARCHAR scalar filters use INVERTED.
        indexes.add_index(field_name="video_id", index_type="INVERTED")
        indexes.add_index(
            field_name="embedding",
            index_type=self.config.index_type,
            metric_type=self.config.metric,
        )
        return schema, indexes

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in ("timeout", "temporar", "unavailable", "429", "502", "503", "504")
        )

    def _upsert_with_retry(self, data: list[dict[str, Any]]) -> None:
        for attempt in range(3):
            try:
                self.client.upsert(collection_name=self.config.collection, data=data)
                return
            except Exception as exc:
                if attempt == 2 or not self._is_retryable(exc):
                    raise
                time.sleep(0.1 * (2**attempt))

    @staticmethod
    def expected_fields() -> dict[str, str]:
        return {
            "pk": "VARCHAR",
            "video_id": "VARCHAR",
            "frame_id": "INT64",
            "sv": "INT64",
            "ev": "INT64",
            "model_fingerprint": "VARCHAR",
            "schema_version": "VARCHAR",
            "embedding": "FLOAT_VECTOR",
        }

    def _verify_indexes(self) -> None:
        names = self.client.list_indexes(collection_name=self.config.collection)
        descriptions = [
            self.client.describe_index(
                collection_name=self.config.collection, index_name=index_name
            )
            for index_name in names
        ]
        vector_index = next(
            (item for item in descriptions if item.get("field_name") == "embedding"), None
        )
        if vector_index is None:
            raise RuntimeError("Milvus embedding index is missing")
        index_type = str(vector_index.get("index_type", "")).upper()
        metric = str(vector_index.get("metric_type", "")).upper()
        if index_type != self.config.index_type or metric != self.config.metric:
            raise RuntimeError(
                "Milvus embedding index mismatch: "
                f"index_type={index_type!r}, metric={metric!r}"
            )
        scalar_index = next(
            (item for item in descriptions if item.get("field_name") == "video_id"), None
        )
        if scalar_index is None or str(scalar_index.get("index_type", "")).upper() != "INVERTED":
            raise RuntimeError("Milvus video_id scalar index is missing or not INVERTED")

    def ensure_schema(self) -> None:
        name = self.config.collection
        if not self.client.has_collection(collection_name=name):
            schema, indexes = self._schema()
            self.client.create_collection(
                collection_name=name,
                schema=schema,
                index_params=indexes,
                consistency_level=self.config.consistency_level,
            )
            self._verify_indexes()
            return
        description = self.client.describe_collection(collection_name=name)
        fields = description.get("fields", [])
        actual = {field["name"]: str(field.get("type", "")).upper() for field in fields}
        missing = sorted(set(self.expected_fields()) - set(actual))
        numeric_types = {
            "5": "INT64",
            "21": "VARCHAR",
            "101": "FLOAT_VECTOR",
        }
        wrong = sorted(
            name
            for name, expected in self.expected_fields().items()
            if name in actual
            and not (
                actual[name].endswith(expected)
                or numeric_types.get(actual[name]) == expected
            )
        )
        embedding = next((field for field in fields if field.get("name") == "embedding"), {})
        dimension = embedding.get("params", {}).get("dim")
        if missing or wrong or (dimension is not None and int(dimension) != EMBEDDING_DIM):
            raise RuntimeError(
                "Milvus schema mismatch: "
                f"missing={missing}, wrong_type={wrong}, embedding_dim={dimension!r}; "
                "refusing destructive migration"
            )
        self.client.load_collection(collection_name=name)
        self._verify_indexes()

    def ingest(self, records: Iterable[StorageRecord]) -> IngestReport:
        vector_records = [record for record in records if record.embedding is not None]
        keys = [(record.video_id, record.frame_id) for record in vector_records]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate (video_id, frame_id) in one Milvus batch")
        if not vector_records:
            return IngestReport(backend="milvus", attempted=0)

        primary_keys = [milvus_primary_key(*key) for key in keys]
        existing: set[str] = set()
        try:
            quoted = ", ".join(milvus_string_literal(key) for key in primary_keys)
            rows = self.client.query(
                collection_name=self.config.collection,
                filter=f"pk in [{quoted}]",
                output_fields=["pk"],
            )
            existing = {row["pk"] for row in rows}
        except Exception:  # existence classification is useful but not required for correctness
            existing = set()

        data = [
            {
                "pk": primary_key,
                "video_id": record.video_id,
                "frame_id": record.frame_id,
                "sv": record.sv,
                "ev": record.ev,
                "model_fingerprint": self.config.model_fingerprint,
                "schema_version": self.config.schema_version,
                "embedding": list(record.embedding or ()),
            }
            for primary_key, record in zip(primary_keys, vector_records, strict=True)
        ]
        self._upsert_with_retry(data)
        updated = sum(key in existing for key in primary_keys)
        return IngestReport(
            backend="milvus",
            attempted=len(data),
            inserted=len(data) - updated,
            updated=updated,
        )

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 10,
        video_id: str | None = None,
    ) -> list[StorageHit]:
        query = StorageRecord.from_values(
            video_id="__query__", frame_id=0, embedding=query_embedding
        ).embedding
        if not 1 <= top_k <= 500:
            raise ValueError("top_k must be between 1 and 500")
        filter_expression = ""
        if video_id is not None:
            filter_expression = f"video_id == {milvus_string_literal(video_id)}"
        response = self.client.search(
            collection_name=self.config.collection,
            data=[list(query or ())],
            anns_field="embedding",
            limit=top_k,
            filter=filter_expression,
            output_fields=["video_id", "frame_id"],
            search_params={"metric_type": self.config.metric, "params": {}},
            consistency_level=self.config.consistency_level,
        )
        hits = [
            StorageHit(
                video_id=item["entity"]["video_id"],
                frame_id=int(item["entity"]["frame_id"]),
                score=float(item.get("distance", item.get("score"))),
            )
            for item in (response[0] if response else [])
        ]
        return sorted(hits, key=lambda hit: (-hit.score, hit.video_id, hit.frame_id))

    def count(self) -> int:
        rows = self.client.query(
            collection_name=self.config.collection,
            filter="",
            output_fields=["count(*)"],
        )
        return int(rows[0]["count(*)"])

    def health(self) -> HealthStatus:
        try:
            exists = bool(self.client.has_collection(collection_name=self.config.collection))
            loaded = False
            indexed = False
            if exists:
                state = self.client.get_load_state(collection_name=self.config.collection)
                loaded = str(state.get("state", "")).upper().endswith("LOADED")
                try:
                    self._verify_indexes()
                    indexed = True
                except Exception:
                    indexed = False
            return HealthStatus(
                backend="milvus",
                connection_ok=True,
                schema_ready=exists,
                index_ready=loaded and indexed,
                message="ready" if exists and loaded and indexed else "not ready",
            )
        except Exception as exc:
            return HealthStatus(
                backend="milvus",
                connection_ok=False,
                schema_ready=False,
                index_ready=False,
                message=f"{type(exc).__name__}: connection failed",
            )
