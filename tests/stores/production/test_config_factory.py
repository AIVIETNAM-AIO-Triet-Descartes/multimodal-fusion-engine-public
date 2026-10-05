from __future__ import annotations

from mfe.stores.production.config import ElasticsearchConfig, MilvusConfig
from mfe.stores.production.factory import build_production_stores


def test_environment_config_masks_secrets(monkeypatch: object) -> None:
    monkeypatch.setenv("MFE_MILVUS_TOKEN", "milvus-secret")  # type: ignore[attr-defined]
    monkeypatch.setenv("MFE_ELASTICSEARCH_API_KEY", "elastic-secret")  # type: ignore[attr-defined]
    milvus = MilvusConfig.from_env()
    elasticsearch = ElasticsearchConfig.from_env()
    assert milvus.token is not None
    assert elasticsearch.api_key is not None
    assert "milvus-secret" not in repr(milvus)
    assert "elastic-secret" not in repr(elasticsearch)


def test_factory_uses_injected_clients_without_connecting() -> None:
    milvus_client = object()
    elasticsearch_client = object()
    stores = build_production_stores(
        milvus_config=MilvusConfig(),
        elasticsearch_config=ElasticsearchConfig(),
        milvus_client=milvus_client,
        elasticsearch_client=elasticsearch_client,
    )
    assert stores.vector.client is milvus_client  # type: ignore[attr-defined]
    assert stores.text.client is elasticsearch_client  # type: ignore[attr-defined]
