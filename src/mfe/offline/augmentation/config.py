"""Strict configuration for the V1 augmentation experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ExperimentConfig(_Section):
    id: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    parent_bundle: str = Field(min_length=1)
    seed: int = 2026


class TransportConfig(_Section):
    kind: Literal["local", "modal"] = "modal"
    volume: str = "keyframes-btc"
    version: int = Field(default=2, ge=1)
    local_root: str = "augmentation_local"


class DatasetConfig(_Section):
    dev_queryset: str = "eval/queryset_dev.jsonl"
    heldout_queryset: str = "eval/queryset_heldout.jsonl"
    video_inventory: str = "data/augmentation/video_inventory.jsonl"
    extra_pilot_videos: int = Field(default=24, ge=0)


class DetectorConfig(_Section):
    kind: Literal["transnet_v2"] = "transnet_v2"
    factory: str = "transnetv2:TransNetV2"
    checkpoint: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    threshold_grid: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    default_threshold: float = Field(default=0.5, gt=0, lt=1)
    uncertain_below: float = Field(default=0.6, gt=0, le=1)
    micro_shot_seconds: float = Field(default=0.5, gt=0)

    @field_validator("threshold_grid")
    @classmethod
    def _valid_threshold_grid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(not 0 < item < 1 for item in value):
            raise ValueError("threshold_grid must contain values in (0, 1)")
        if tuple(sorted(set(value))) != value:
            raise ValueError("threshold_grid must be strictly increasing")
        return value


class CoverageConfig(_Section):
    target_gap_seconds_grid: tuple[float, ...] = (4.0, 6.0, 8.0)
    k_max_grid: tuple[int, ...] = (2, 3, 4)
    delta_cap_ratio: float = Field(default=0.03, gt=0, le=1)

    @field_validator("target_gap_seconds_grid")
    @classmethod
    def _valid_gap_grid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(item <= 0 for item in value):
            raise ValueError("target gap values must be positive")
        return value

    @field_validator("k_max_grid")
    @classmethod
    def _valid_k_grid(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(item < 1 for item in value):
            raise ValueError("k_max values must be >= 1")
        return value


class DedupConfig(_Section):
    phash_max_distance: int = Field(default=5, ge=0)
    embedding_max_cosine: float = Field(default=0.97, ge=-1, le=1)


class EvaluationConfig(_Section):
    bootstrap_samples: int = Field(default=10_000, ge=100)
    bootstrap_seed: int = 2026
    noninferiority_margin: float = -0.005
    max_latency_ratio: float = Field(default=1.10, ge=1)
    min_oracle_mean_delta: float = Field(default=0.005, ge=0)
    min_oracle_rescues: int = Field(default=2, ge=1)
    heldout_by_type_margin: float = -0.01


class PathsConfig(_Section):
    workdir: str = "augmentation_work/btc_aug_cov_v1"
    base_flat_root: str = "eval/merge/aic_btc_b1_v1"
    base_relational_db: str = "stores/relational/app.db"
    eval_config: str = "config/pilot.yaml"


class AugmentationConfig(_Section):
    experiment: ExperimentConfig
    transport: TransportConfig = TransportConfig()
    dataset: DatasetConfig = DatasetConfig()
    detector: DetectorConfig
    coverage: CoverageConfig = CoverageConfig()
    dedup: DedupConfig = DedupConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    paths: PathsConfig = PathsConfig()

    _repo_root: Path = PrivateAttr(default=Path("."))

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.repo_root / path

    def canonical_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_augmentation_config(path: str | Path) -> AugmentationConfig:
    """Load a strict config and remember its repository-relative root."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: expected a YAML mapping")
    config = AugmentationConfig.model_validate(raw)
    # Config files are kept in ``repo/config``.  Resolve paths from the repo,
    # matching ``mfe.core.config`` without changing that shared module.
    config._repo_root = source.parent.parent
    return config
