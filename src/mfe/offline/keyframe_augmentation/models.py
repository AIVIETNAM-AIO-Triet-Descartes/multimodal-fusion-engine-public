"""Typed artifacts for the keyframe augmentation protocol."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mfe.contracts.ids import build_keyframe_id, parse_keyframe_id


class _Artifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EvidenceLabel(StrEnum):
    MEASURED = "MEASURED"
    DERIVED = "DERIVED"
    HYPOTHESIS = "HYPOTHESIS"
    UNVERIFIED = "UNVERIFIED"


class CoverageClass(StrEnum):
    C0 = "C0"
    C1 = "C1"
    CT = "CT"
    CE = "CE"
    COVERED = "COVERED"


class AugmentationConfig(_Artifact):
    protocol_version: str = "btc-augmentation-v1"
    dataset_version: str
    split_version: str
    seed: int = 20260827
    guardrail_ratio: float = Field(default=0.03, gt=0.0, le=1.0)
    target_gap_seconds: float = Field(default=4.0, gt=0.0)
    k_max: int = Field(default=6, ge=1)
    micro_shot_seconds: float = Field(default=0.8, ge=0.0)
    interior_margin_seconds: float = Field(default=0.25, ge=0.0)
    interior_margin_ratio: float = Field(default=0.10, ge=0.0, lt=0.5)
    detector_confidence_min: float = Field(default=0.50, ge=0.0, le=1.0)
    target_gap_grid: list[float] = Field(default_factory=lambda: [2.0, 4.0, 6.0])
    k_max_grid: list[int] = Field(default_factory=lambda: [2, 4, 6])
    non_inferiority_margin: float = Field(default=0.0, ge=0.0)
    max_p95_latency_increase_ratio: float = Field(default=0.10, ge=0.0)
    bootstrap_samples: int = Field(default=2000, ge=100)
    phash_max_distance: int | None = Field(default=None, ge=0, le=64)

    @model_validator(mode="after")
    def _grid_is_valid(self) -> AugmentationConfig:
        if not self.target_gap_grid or any(value <= 0 for value in self.target_gap_grid):
            raise ValueError("target_gap_grid must contain positive values")
        if not self.k_max_grid or any(value < 1 for value in self.k_max_grid):
            raise ValueError("k_max_grid must contain positive integers")
        return self


class BtcFrame(_Artifact):
    video_id: str
    ordinal: int = Field(ge=0)
    frame_idx: int = Field(ge=0)
    pts_time: float = Field(ge=0.0)
    fps: float = Field(gt=0.0)
    keyframe_id: str

    @model_validator(mode="after")
    def _identity_agrees(self) -> BtcFrame:
        video_id, frame_idx = parse_keyframe_id(self.keyframe_id)
        if (video_id, frame_idx) != (self.video_id, self.frame_idx):
            raise ValueError("BTC keyframe identity does not match video_id/frame_idx")
        return self

    @classmethod
    def build(
        cls,
        *,
        video_id: str,
        ordinal: int,
        frame_idx: int,
        pts_time: float,
        fps: float,
    ) -> BtcFrame:
        return cls(
            video_id=video_id,
            ordinal=ordinal,
            frame_idx=frame_idx,
            pts_time=pts_time,
            fps=fps,
            keyframe_id=build_keyframe_id(video_id, frame_idx),
        )


class ShotProposal(_Artifact):
    video_id: str
    shot_id: str
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    # Canonical literal FPS from map-keyframes/Gate 0.  Frame endpoints are
    # decoder integer counters; this value is only used for durations/margins.
    fps: float = Field(gt=0.0)
    decoder_fps: float | None = Field(default=None, gt=0.0)
    detector: Literal["transnetv2"] = "transnetv2"
    detector_fingerprint: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    audit_status: Literal["auto", "confirmed", "transition", "error"] = "auto"

    @model_validator(mode="after")
    def _end_not_before_start(self) -> ShotProposal:
        if self.end_frame < self.start_frame:
            raise ValueError("shot endpoint is inclusive and must be >= start_frame")
        return self

    @property
    def duration_seconds(self) -> float:
        return (self.end_frame - self.start_frame + 1) / self.fps


class ShotAudit(_Artifact):
    video_id: str
    shot_id: str
    start_frame: int
    end_frame: int
    duration_seconds: float
    coverage_class: CoverageClass
    btc_count: int = Field(ge=0)
    k_target: int = Field(ge=0)
    deficit: int = Field(ge=0)
    max_blind_gap_seconds: float = Field(ge=0.0)
    eligible: bool
    reason: str
    detector_confidence: float | None = None
    audit_status: str


class CandidateFrame(_Artifact):
    arm: Literal["B1", "B2", "B3", "B4"]
    video_id: str
    shot_id: str
    frame_idx: int = Field(ge=0)
    pts_time: float = Field(ge=0.0)
    fps: float = Field(gt=0.0)
    selector: Literal["shot_center", "farthest", "dynamic_density", "uniform_random"]
    coverage_class: Literal["C0", "C1"]
    priority: tuple[float, float, float, str, int]
    detector_fingerprint: str
    audit_status: str
    frame_idx_source: Literal["decoder_counter"] = "decoder_counter"

    @model_validator(mode="after")
    def _coordinate_roundtrip(self) -> CandidateFrame:
        if math.floor(self.pts_time * self.fps) != self.frame_idx:
            raise ValueError("candidate PTS must floor-roundtrip under literal FPS")
        return self

    @property
    def keyframe_id(self) -> str:
        return build_keyframe_id(self.video_id, self.frame_idx)


class ProvenanceRow(_Artifact):
    keyframe_id: str
    video_id: str
    frame_idx: int = Field(ge=0)
    pts_time: float = Field(ge=0.0)
    ordinal: int = Field(ge=0)
    origin: Literal["btc", "supplemental"]
    selector: str
    shot_id: str | None
    parent_bundle: str
    run_id: str
    decoder_fingerprint: str
    config_hash: str
    image_sha256: str
    phash_distance: int | None = None
    embedding_similarity: float | None = None
    audit_status: str
    decoder_pts_time: float | None = None
    frame_idx_source: Literal["map-keyframes.csv", "decoder_counter"]
    pts_time_source: Literal[
        "map-keyframes.csv", "decoder_pts", "synthetic_literal_fps_interval"
    ]
    mapping_previous_keyframe_id: str | None = None
    mapping_next_keyframe_id: str | None = None
    interpolation_frame_error: float | None = None
    mapping_audit_status: Literal["PASS", "WARNING_OPEN_BRACKET", "WARNING_INTERPOLATION"]

    @model_validator(mode="after")
    def _identity_agrees(self) -> ProvenanceRow:
        if build_keyframe_id(self.video_id, self.frame_idx) != self.keyframe_id:
            raise ValueError("provenance keyframe_id disagrees with coordinates")
        return self
