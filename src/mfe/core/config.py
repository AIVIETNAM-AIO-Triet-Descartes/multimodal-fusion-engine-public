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
    """Mirrors docs/00 §7. Input (data/sample/*, provided by the dataset) and
    output (stores/*, we generate). Never code."""

    # Input — dataset extracted into data/sample/ (see data/README.md)
    videos: str
    keyframes: str
    clip_features: str
    map_keyframes: str
    objects: str
    media_info: str
    # Output — what we build
    segments: str
    vector_store: str
    inverted_store: str
    asset_store: str
    relational_db: str
    eval_results: str
    throughput: str
    event_log: str


class StorageConfig(_Section):
    backend: Literal["local", "modal"] = "local"
    volume: str | None = None


class ComputeConfig(_Section):
    gpu: bool = False
    gpu_type: str | None = None


class ModelsConfig(_Section):
    """A null value disables the module — docs/02 §3 ("cắm-rút").

    `clip` and `detector` are required by docs/02 §3 but may be null here so the
    repo imports before Quốc Đạt picks checkpoints; `mfe.offline.ingest` is what
    enforces their presence at run time.
    """

    clip: str | None = None
    detector: str | None = None
    ocr: str | None = None
    metadata: str | None = None
    #: Stage C checkpoint. Null leaves stage C off, which is the safe default:
    #: the last stage is the last word on order, so a reranker that is missing
    #: or wrong is worse than no reranker (docs/triet/report.html điểm 10).
    #: Must be a *different* model from `retrieval.clip_checkpoint` — rescoring
    #: with the same one reproduces stage A exactly.
    reranker: str | None = None


class FeaturesConfig(_Section):
    """Where per-keyframe signals come from — docs/02 §3.

    Sprint 1 uses `provided`: BTC ships `clip-features/*.npy` and `objects/*.json`
    already computed, so no checkpoint is loaded and `models.*` may stay null.
    Switching to `model` is what turns `offline/features/*.py` on; that is a
    later sprint, and `build_enabled_modules` is what enforces the
    required/optional split at that point.
    """

    source: Literal["provided", "model"] = "provided"
    #: Detections below this score are dropped before they reach the Inverted
    #: Index. The provided files carry 100 boxes per keyframe, most of them
    #: noise; indexing all of them would drown the real labels in BM25.
    object_score_min: float = Field(default=0.3, ge=0.0, le=1.0)
    #: Metadata text fed to the Inverted Index, from media-info/*.json.
    metadata_fields: list[str] = Field(default_factory=lambda: ["title", "keywords", "author"])
    #: Append Vietnamese synonyms to each keyframe's object labels at ingest
    #: (mfe.offline.labels_vi). BTC ships English OpenImages names and asks in
    #: Vietnamese, so without this the text arm has no keyframe-level term the
    #: query can ever match. A switch rather than a constant because turning it
    #: off is how the A/B gets measured — and it changes the index, so flipping
    #: it means re-running ingest.
    translate_labels: bool = True


class FusionWeights(BaseModel):
    """docs/04 §3.1. Weights stay in config so nobody calibrates them deep into
    last year's data (docs/00 §8 risk 4).

    One key per stage-A arm (`mfe.online.arms`). Unlike every other section this
    one accepts extra keys, because the arm set is open: merging SOTA_AIC added
    six text arms and more will follow. Names are validated where it can be done
    properly — `Cascade._build_arms` reads a weight per *registered* arm, so a
    typo yields the 1.0 default rather than a crash.

    Weight 0 means "contribute nothing to the ranking", NOT "switch the arm off":
    a zero-weighted arm still admits its candidates into the fused pool. To
    remove an arm entirely, disable it in the registry (`ArmRegistry.set_enabled`).
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    clip: float = 1.0
    bm25: float = 1.0


class FusionConfig(_Section):
    """How stage A combines its arms. See `mfe.online.fusion`.

    `mode` and `normalizer` are separate on purpose. Naming only the mode would
    hide the choice that actually decides the outcome: with a rank-based
    normalizer the two modes differ solely in decay shape, while with `min_max`
    a single outlier score can decide the ranking on its own.
    """

    mode: Literal["rrf", "weighted_sum"] = "rrf"
    normalizer: Literal["rrf", "linear_rank", "min_max"] | None = None
    rrf_k: int = Field(default=60, gt=0)

    def resolved_normalizer(self) -> str:
        """The normalizer to use, defaulting to the one each mode implies."""
        if self.normalizer is not None:
            return self.normalizer
        return "rrf" if self.mode == "rrf" else "linear_rank"


class TranslationConfig(_Section):
    """VI→EN for arms whose model has no Vietnamese — see `mfe.online.translate`.

    `tiers` is deliberately a closed list of *producing* tiers. Two things it
    cannot express, both on purpose:

    * **A "keep the Vietnamese" tier.** There is no such value, so no config can
      ask an English-only tower to read Vietnamese. Out of translations means the
      arm sits out (docs/triet/handoff.md §5.1).
    * **Skipping the hand-written form.** `text_en` on a query always wins; it is
      not a tier and cannot be turned off.

    An empty `tiers` list is legal and means "hand-written only" — the state the
    pilot ran in before this existed.
    """

    enabled: bool = True
    tiers: list[Literal["mt", "llm"]] = Field(default_factory=lambda: ["mt"])
    mt_checkpoint: str = "VietAI/envit5-translation"
    #: Greedy by default: a query set scored twice must give the same two
    #: numbers. Part of the cache key, so raising it invalidates rather than
    #: mixes (`MachineTranslator.model_id`).
    mt_num_beams: int = Field(default=1, ge=1)
    mt_max_new_tokens: int = Field(default=128, gt=0)
    #: Tracked in git, unlike everything under `stores/`: with an LLM tier the
    #: cache *is* what makes two runs comparable, so it has to travel with the
    #: query set rather than being rebuilt per machine.
    cache_path: str = "eval/translations.json"


class RetrievalConfig(_Section):
    embed_dim: int = Field(gt=0)
    coarse_topk: int = Field(gt=0, le=500, description="docs/04 §2: top_k <= 500")
    rerank_topk: int = Field(
        gt=0, description="Also the submit depth; PDF §2.2 allows 100 answers"
    )
    fusion_weights: FusionWeights = FusionWeights()
    fusion: FusionConfig = FusionConfig()
    translation: TranslationConfig = TranslationConfig()
    rerank_budget_ms: int = Field(gt=0, description="Over budget -> truncate, keep coarse order")
    #: How deep stage C reaches. Defaults to the full submission depth: on the
    #: 2026 sample a full-depth rerank moved a correct keyframe from rank 92 to
    #: rank 3, which any shallow head misses entirely. Costs ~12 s/query on CPU
    #: (116 ms/image) — lower it only if TTCA matters more than Final Score.
    rerank_depth: int = Field(default=100, ge=0)
    temporal_window_frames: int = Field(
        default=640, ge=0, description="Stage B grouping distance, in source-video frames"
    )
    dedup_depth: int = Field(
        default=20,
        ge=0,
        description=(
            "How many head slots stage B may reorder for diversity. Beyond this "
            "the fused ranking is left alone — see Cascade.group"
        ),
    )
    clip_checkpoint: str = Field(
        default="clip-ViT-B-32",
        description="Image tower; named in official PDF §3, verified at cosine ~0.999",
    )
    clip_text_checkpoint: str = Field(
        default="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        description="Text tower; same image space, but reads Vietnamese queries",
    )


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
    features: FeaturesConfig = FeaturesConfig()
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

    Owner: Vỹ Triết.

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
