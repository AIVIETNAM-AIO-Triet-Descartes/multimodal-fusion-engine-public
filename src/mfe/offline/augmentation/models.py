"""Pilot-only records for BTC keyframe augmentation.

These records are intentionally not added to :mod:`mfe.contracts.records`.
The research decision requires a provenance sidecar first; a shared-contract
migration is allowed only after the policy passes Gate 2.
"""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mfe.contracts.ids import build_keyframe_id, parse_keyframe_id, parse_shot_id


class _PilotRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class CoverageClass(StrEnum):
    """Coverage classification from the approved research protocol."""

    C0 = "C0"
    C1 = "C1"
    CT = "CT"
    CE = "CE"
    OK = "OK"


class Selector(StrEnum):
    """Frame-selection policy recorded in the delta provenance."""

    SHOT_CENTER = "shot_center"
    FARTHEST = "farthest"
    RANDOM = "random"
    ORACLE_TARGET = "oracle_target"
    DAKE = "dake"


class AuditStatus(StrEnum):
    """Whether a record may proceed without a manual override."""

    AUTO_ACCEPTED = "auto_accepted"
    MANUAL_ACCEPTED = "manual_accepted"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class ShotProposal(_PilotRecord):
    """One non-overlapping, half-open TransNetV2 shot proposal."""

    video_id: str
    shot_id: str
    start_frame: int = Field(ge=0)
    end_frame_exclusive: int = Field(gt=0)
    start_pts_time: float = Field(ge=0)
    end_pts_time: float = Field(gt=0)
    boundary_confidence: float = Field(ge=0, le=1)
    detector_fingerprint: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_interval_and_identity(self) -> ShotProposal:
        if self.end_frame_exclusive <= self.start_frame:
            raise ValueError("shot frame interval must be non-empty and half-open")
        if self.end_pts_time <= self.start_pts_time:
            raise ValueError("shot PTS interval must be strictly increasing")
        parsed_video, _ = parse_shot_id(self.shot_id)
        if parsed_video != self.video_id:
            raise ValueError(
                f"shot_id {self.shot_id!r} encodes {parsed_video!r}, "
                f"not {self.video_id!r}"
            )
        return self

    @property
    def duration_seconds(self) -> float:
        return self.end_pts_time - self.start_pts_time


class BtcFrame(_PilotRecord):
    """The small BTC mapping surface required by coverage analysis."""

    keyframe_id: str
    video_id: str
    frame_idx: int = Field(ge=0)
    pts_time: float = Field(ge=0)

    @model_validator(mode="after")
    def _validate_key(self) -> BtcFrame:
        parsed_video, parsed_frame = parse_keyframe_id(self.keyframe_id)
        if (parsed_video, parsed_frame) != (self.video_id, self.frame_idx):
            raise ValueError("BTC keyframe identity disagrees with video_id/frame_idx")
        return self


class CoverageAuditRow(_PilotRecord):
    """Coverage decision for one shot and one density configuration."""

    video_id: str
    shot_id: str
    start_frame: int = Field(ge=0)
    end_frame_exclusive: int = Field(gt=0)
    start_pts_time: float = Field(ge=0)
    end_pts_time: float = Field(gt=0)
    detector_confidence: float = Field(ge=0, le=1)
    btc_keyframe_ids: tuple[str, ...] = ()
    btc_count: int = Field(ge=0)
    max_temporal_gap: float = Field(ge=0)
    coverage_class: CoverageClass
    target_gap_seconds: float = Field(gt=0)
    k_target: int = Field(ge=1)
    k_max: int = Field(ge=1)
    deficit: int = Field(ge=0)
    audit_status: AuditStatus
    audit_reason: str = ""

    @model_validator(mode="after")
    def _validate_counts(self) -> CoverageAuditRow:
        if self.btc_count != len(self.btc_keyframe_ids):
            raise ValueError("btc_count must equal len(btc_keyframe_ids)")
        for keyframe_id in self.btc_keyframe_ids:
            parsed_video, _ = parse_keyframe_id(keyframe_id)
            if parsed_video != self.video_id:
                raise ValueError("BTC keyframe id belongs to another video")
        if self.k_target > self.k_max:
            raise ValueError("k_target may not exceed k_max")
        expected_deficit = max(0, self.k_target - self.btc_count)
        if self.deficit != expected_deficit:
            raise ValueError(
                f"deficit must equal max(0, k_target - btc_count)={expected_deficit}"
            )
        if self.end_frame_exclusive <= self.start_frame:
            raise ValueError("coverage row has an empty frame interval")
        if self.end_pts_time <= self.start_pts_time:
            raise ValueError("coverage row has a non-positive PTS interval")
        return self

    @property
    def eligible(self) -> bool:
        eligible_classes = {CoverageClass.C0, CoverageClass.C1}
        accepted_statuses = {AuditStatus.AUTO_ACCEPTED, AuditStatus.MANUAL_ACCEPTED}
        return self.coverage_class in eligible_classes and self.audit_status in accepted_statuses


class CandidateRow(_PilotRecord):
    """A source-video frame proposed for the delta candidate pool."""

    keyframe_id: str
    video_id: str
    frame_idx: int = Field(ge=0)
    literal_fps: float = Field(gt=0)
    pts_time: float = Field(ge=0)
    decoder_pts_time: float = Field(ge=0)
    frame_idx_source: str = Field(default="decoder_counter", pattern="^decoder_counter$")
    pts_time_source: str = Field(
        default="synthetic_literal_fps_interval",
        pattern="^synthetic_literal_fps_interval$",
    )
    shot_id: str
    selector: Selector
    coverage_class: CoverageClass
    selection_round: int = Field(ge=1)
    distance_to_nearest_btc: float = Field(ge=0)
    detector_confidence: float = Field(ge=0, le=1)
    decoder_fingerprint: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    audit_status: AuditStatus = AuditStatus.AUTO_ACCEPTED

    @model_validator(mode="after")
    def _validate_identity(self) -> CandidateRow:
        expected = build_keyframe_id(self.video_id, self.frame_idx)
        if self.keyframe_id != expected:
            raise ValueError(f"candidate keyframe_id must be {expected!r}")
        shot_video, _ = parse_shot_id(self.shot_id)
        if shot_video != self.video_id:
            raise ValueError("candidate shot_id belongs to another video")
        if self.coverage_class not in {CoverageClass.C0, CoverageClass.C1} and (
            self.selector != Selector.ORACLE_TARGET
        ):
            raise ValueError("non-oracle candidates may only come from C0/C1")
        if math.floor(self.pts_time * self.literal_fps) != self.frame_idx:
            raise ValueError("candidate pts_time must floor-roundtrip to frame_idx")
        return self


class OracleTargetRow(_PilotRecord):
    """A deliberately labelled frame used only by the DEV oracle."""

    query_id: str = Field(min_length=1)
    event_index: int = Field(ge=0)
    video_id: str
    frame_range_start: int = Field(ge=0)
    frame_range_end: int = Field(ge=0)
    target: CandidateRow

    @model_validator(mode="after")
    def _validate_target(self) -> OracleTargetRow:
        if self.frame_range_end < self.frame_range_start:
            raise ValueError("oracle frame range must be inclusive and non-empty")
        if self.target.video_id != self.video_id:
            raise ValueError("oracle target belongs to another video")
        if not self.frame_range_start <= self.target.frame_idx <= self.frame_range_end:
            raise ValueError("oracle target lies outside its GT frame range")
        if self.target.selector != Selector.ORACLE_TARGET:
            raise ValueError("oracle target candidate must use selector=oracle_target")
        return self


class AugmentationManifestRow(_PilotRecord):
    """Required provenance sidecar row for a selected supplemental frame."""

    keyframe_id: str
    video_id: str
    frame_idx: int = Field(ge=0)
    literal_fps: float = Field(gt=0)
    pts_time: float = Field(ge=0)
    decoder_pts_time: float = Field(ge=0)
    frame_idx_source: str = Field(pattern="^decoder_counter$")
    pts_time_source: str = Field(pattern="^synthetic_literal_fps_interval$")
    ordinal: int = Field(ge=0)
    origin: str = Field(pattern=r"^(btc|supplemental)$")
    selector: Selector
    shot_id: str
    parent_bundle: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    decoder_fingerprint: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phash_distance: float | None = Field(default=None, ge=0)
    embedding_similarity: float | None = Field(default=None, ge=-1, le=1)
    audit_status: AuditStatus
    coverage_class: CoverageClass
    selection_round: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_identity(self) -> AugmentationManifestRow:
        if self.keyframe_id != build_keyframe_id(self.video_id, self.frame_idx):
            raise ValueError("manifest keyframe_id disagrees with video_id/frame_idx")
        if math.floor(self.pts_time * self.literal_fps) != self.frame_idx:
            raise ValueError("manifest pts_time must floor-roundtrip to frame_idx")
        return self


class VideoInventoryRow(_PilotRecord):
    """Canonical input inventory used by Gate 0 cohort locking."""

    video_id: str
    source_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(gt=0)
    fps: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    btc_keyframes: int = Field(ge=0)
    map_path: str = Field(min_length=1)
    map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def programme(self) -> str:
        return self.video_id.split("_", 1)[0]

    @property
    def btc_density(self) -> float:
        return self.btc_keyframes / self.duration_seconds
