"""Config loading — docs/01 §3.1.

Everything that could differ between local and Modal lives in a YAML file, not
in code: paths, model names, embed_dim, top-k, fusion weights, submit format.
docs/00 §7: "mọi đường dẫn đọc từ config, không hardcode".

Typed on purpose: a missing key should fail at load, naming the key, rather than
surfacing as a None three call frames into someone else's workstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mfe.contracts.tools import SubmitFormat


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PathsConfig(_Section):
    """Mirrors docs/00 §7 exactly. Artifact locations only — never code."""

    raw: str
    keyframes: str
    segments: str
    vector_store: str
    inverted_store: str
    relational_db: str
    eval_results: str
    throughput: str


class StorageConfig(_Section):
    backend: Literal["local", "modal"] = "local"
    volume: str | None = None


class ComputeConfig(_Section):
    gpu: bool = False
    gpu_type: str | None = None


class ModelsConfig(_Section):
    """A null value disables the module — docs/02 §3 ("cắm-rút").

    `clip` and `detector` are required by docs/02 §3 but may be null here so the
    repo imports before TV1 picks checkpoints; `mfe.offline.ingest` is what
    enforces their presence at run time.
    """

    clip: str | None = None
    detector: str | None = None
    ocr: str | None = None
    metadata: str | None = None


class FusionWeights(_Section):
    """docs/04 §3.1. Weights stay in config so nobody calibrates them deep into
    last year's data (docs/00 §8 risk 4)."""

    clip: float = 1.0
    bm25: float = 1.0


class RetrievalConfig(_Section):
    embed_dim: int = Field(gt=0)
    coarse_topk: int = Field(gt=0, le=500, description="docs/04 §2: top_k <= 500")
    rerank_topk: int = Field(gt=0)
    fusion_weights: FusionWeights = FusionWeights()
    rerank_budget_ms: int = Field(gt=0, description="Over budget -> fall back to coarse")


class SubmitConfig(_Section):
    schema_format: SubmitFormat = "by_timestamp"
    frame_range_window: int = Field(default=25, ge=0)


class EvalConfig(_Section):
    queryset: str
    recall_at: list[int] = Field(default_factory=lambda: [1, 5, 10])


class Config(_Section):
    """The whole of config/local.yaml, validated."""

    paths: PathsConfig
    storage: StorageConfig = StorageConfig()
    compute: ComputeConfig = ComputeConfig()
    models: ModelsConfig = ModelsConfig()
    retrieval: RetrievalConfig
    submit: SubmitConfig = SubmitConfig()
    eval: EvalConfig


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config.

    Owner: Lead. TODO(Lead): implement — read YAML, expand paths relative to the
    repo root so a config works regardless of cwd, then `Config.model_validate`.
    """
    raise NotImplementedError("TODO(Lead): load + validate YAML into Config")
