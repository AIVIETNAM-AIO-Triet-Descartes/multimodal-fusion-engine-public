"""Environment-driven production-store configuration; contains no secrets."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class MilvusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str = "http://127.0.0.1:19530"
    token: SecretStr | None = None
    collection: str = "aic_keyframe_embeddings_v1"
    dimension: Literal[512] = 512
    metric: Literal["COSINE"] = "COSINE"
    index_type: Literal["AUTOINDEX"] = "AUTOINDEX"
    consistency_level: Literal["Strong"] = "Strong"
    schema_version: Literal["1"] = "1"
    model_fingerprint: str = "clip-vit-b-32-btc-fp16-upcast-fp32"
    timeout_seconds: float = Field(default=20.0, gt=0)

    @classmethod
    def from_env(cls) -> MilvusConfig:
        token = os.getenv("MFE_MILVUS_TOKEN")
        return cls(
            uri=os.getenv("MFE_MILVUS_URI", cls.model_fields["uri"].default),
            token=SecretStr(token) if token else None,
            collection=os.getenv("MFE_MILVUS_COLLECTION", cls.model_fields["collection"].default),
        )


class ElasticsearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = "http://127.0.0.1:9200"
    api_key: SecretStr | None = None
    username: str | None = None
    password: SecretStr | None = None
    index: str = "aic_keyframe_text_v1"
    analyzer_mode: Literal["dev_fallback", "plugin_required"] = "dev_fallback"
    plugin_analyzer: str = "vi_analyzer"
    schema_version: Literal["1"] = "1"
    text_pipeline_fingerprint: str = "aic-vi-dev-fallback-v1"
    request_timeout_seconds: float = Field(default=20.0, gt=0)

    @classmethod
    def from_env(cls) -> ElasticsearchConfig:
        api_key = os.getenv("MFE_ELASTICSEARCH_API_KEY")
        password = os.getenv("MFE_ELASTICSEARCH_PASSWORD")
        return cls(
            url=os.getenv("MFE_ELASTICSEARCH_URL", cls.model_fields["url"].default),
            api_key=SecretStr(api_key) if api_key else None,
            username=os.getenv("MFE_ELASTICSEARCH_USERNAME"),
            password=SecretStr(password) if password else None,
            index=os.getenv("MFE_ELASTICSEARCH_INDEX", cls.model_fields["index"].default),
            analyzer_mode=os.getenv("MFE_ELASTICSEARCH_ANALYZER_MODE", "dev_fallback"),
            plugin_analyzer=os.getenv("MFE_ELASTICSEARCH_PLUGIN_ANALYZER", "vi_analyzer"),
        )
