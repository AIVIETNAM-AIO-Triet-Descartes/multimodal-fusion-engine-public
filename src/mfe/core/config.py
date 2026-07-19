"""Config loading — docs/01 §3.1.

Everything that could differ between local and Modal lives in a YAML file, not
in code: paths, model names, embed_dim, top-k, fusion weights, submit format.
docs/00 §7: "mọi đường dẫn đọc từ config, không hardcode".

Typed on purpose: a missing key should fail at load, naming the key, rather than
surfacing as a None three call frames into someone else's workstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from mfe.contracts.errors import ContractError, ErrorCode
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
    """The whole of config/local.yaml, validated.

    `paths` are absolute after `load_config` resolves them against `repo_root`,
    so the same config works regardless of the process's cwd — a query run from
    `eval/` and an ingest run from the repo root read the same stores.
    """

    paths: PathsConfig
    storage: StorageConfig = StorageConfig()
    compute: ComputeConfig = ComputeConfig()
    models: ModelsConfig = ModelsConfig()
    retrieval: RetrievalConfig
    submit: SubmitConfig = SubmitConfig()
    eval: EvalConfig
    #: Root the paths were resolved against. Injected by load_config, not read
    #: from the YAML — so it never appears in config/*.yaml.
    repo_root: str

    def resolve_path(self, relative: str) -> str:
        """Resolve any other relative path (e.g. eval.queryset) against repo_root.

        Absolute paths pass through unchanged, matching how load_config treats
        the `paths` section.
        """
        if _is_absolute(relative):
            return relative
        return str((Path(self.repo_root) / relative).resolve())


def _is_absolute(path_str: str) -> bool:
    """Whether a config path string is absolute.

    Checks a leading POSIX slash explicitly, not only `Path.is_absolute()`:
    on Windows a Modal-style "/vol/data" has no drive letter, so pathlib reports
    it relative and would wrongly re-root it. Modal is a stub in Sprint 1, but a
    correct check here costs nothing and removes a future foot-gun.
    """
    return (
        path_str.startswith("/")
        or path_str.startswith("\\")
        or (len(path_str) > 1 and path_str[1] == ":")
    )


def _resolve_paths(paths: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Make every relative value under `paths:` absolute against repo_root."""
    resolved: dict[str, Any] = {}
    for key, value in paths.items():
        if isinstance(value, str) and not _is_absolute(value):
            resolved[key] = str((repo_root / value).resolve())
        else:
            resolved[key] = value
    return resolved


def load_config(path: str | Path, repo_root: str | Path | None = None) -> Config:
    """Load and validate a YAML config.

    Owner: Lead.

    `repo_root` anchors relative paths. It defaults to the config file's
    grandparent — `config/local.yaml` sits one level under the repo root — so
    the common case needs no argument. Tests pass it explicitly.

    A missing key or a wrong type fails here, naming the field, rather than
    surfacing as a None deep inside someone else's workstream (docs/01 §3.1).
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT, f"config file not found: {config_path}"
        )

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"config must be a YAML mapping, got {type(raw).__name__}: {config_path}",
        )

    root = Path(repo_root) if repo_root is not None else config_path.resolve().parent.parent
    if "paths" in raw and isinstance(raw["paths"], dict):
        raw["paths"] = _resolve_paths(raw["paths"], root)
    raw["repo_root"] = str(root)

    # pydantic's ValidationError already names the offending field; let it
    # propagate rather than flattening it into a less informative message.
    return Config.model_validate(raw)
