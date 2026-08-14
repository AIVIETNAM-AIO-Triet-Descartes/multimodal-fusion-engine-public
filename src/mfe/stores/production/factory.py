"""Factory for explicitly configured production storage clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ElasticsearchConfig, MilvusConfig
from .elasticsearch import ElasticsearchTextStore
from .milvus import MilvusVectorStore
from .protocols import TextStore, VectorStore


@dataclass(frozen=True)
class ProductionStores:
    vector: VectorStore
    text: TextStore


def build_production_stores(
    *,
    milvus_config: MilvusConfig | None = None,
    elasticsearch_config: ElasticsearchConfig | None = None,
    milvus_client: Any | None = None,
    elasticsearch_client: Any | None = None,
) -> ProductionStores:
    """Build adapters; injected clients keep unit tests and callers deterministic."""
    return ProductionStores(
        vector=MilvusVectorStore(
            milvus_config or MilvusConfig.from_env(), client=milvus_client
        ),
        text=ElasticsearchTextStore(
            elasticsearch_config or ElasticsearchConfig.from_env(),
            client=elasticsearch_client,
        ),
    )
