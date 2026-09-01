"""Read-only contract for oracle artifacts produced by Trọng's materializer.

The Quang branch intentionally keeps its older candidate schema for existing
experiments.  Notebook 18 consumes already-frozen research artifacts whose PTS
contract is decoder-authoritative, so parsing them through that older schema
would either fail or silently reinterpret provenance.  These small models form
an explicit compatibility boundary without changing either producer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mfe.contracts.ids import build_keyframe_id
from mfe.offline.augmentation.io import canonical_hash


class OracleArtifactCandidate(BaseModel):
    """One decoder-resolved source frame from notebook 17."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    keyframe_id: str
    video_id: str
    frame_idx: int = Field(ge=0)
    pts_time: float = Field(ge=0)
    pts_time_source: Literal["decoder_pts"]
    selection_pts_time: float | None = Field(default=None, ge=0)
    shot_id: str
    selector: Literal["oracle_target"]
    coverage_class: Literal["C0", "C1"]
    selection_round: int = Field(ge=1)
    distance_to_nearest_btc: float = Field(ge=0)
    detector_confidence: float = Field(ge=0, le=1)
    decoder_fingerprint: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_status: Literal["auto_accepted", "manual_accepted"]

    @model_validator(mode="after")
    def _identity_matches_source_frame(self) -> OracleArtifactCandidate:
        expected = build_keyframe_id(self.video_id, self.frame_idx)
        if self.keyframe_id != expected:
            raise ValueError(f"oracle artifact keyframe_id must be {expected!r}")
        if not self.shot_id.startswith(f"{self.video_id}#s"):
            raise ValueError("oracle artifact shot_id belongs to another video")
        return self


class OracleArtifactTarget(BaseModel):
    """A DEV query/event target linked to one materialized source frame."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    query_id: str = Field(min_length=1)
    event_index: int = Field(ge=0)
    video_id: str
    frame_range_start: int = Field(ge=0)
    frame_range_end: int = Field(ge=0)
    target: OracleArtifactCandidate

    @model_validator(mode="after")
    def _target_is_inside_inclusive_interval(self) -> OracleArtifactTarget:
        if self.frame_range_end < self.frame_range_start:
            raise ValueError("oracle interval must be inclusive and non-empty")
        if self.target.video_id != self.video_id:
            raise ValueError("oracle target belongs to another video")
        if not self.frame_range_start <= self.target.frame_idx <= self.frame_range_end:
            raise ValueError("oracle target lies outside its GT interval")
        return self


def validate_oracle_completion_marker(
    marker: Mapping[str, Any],
    *,
    run_id: str,
    worker: str,
    video_id: str,
    config_hash: str,
    oracle_candidates_sha256: str,
    observed_remote_hashes: Mapping[str, str | None],
) -> bool:
    """Accept only a lineage-matched marker whose remote artifact hashes agree."""

    payload = dict(marker)
    claimed = payload.pop("marker_payload_hash", None)
    if claimed != canonical_hash(payload):
        return False
    expected = {
        "status": "COMPLETE",
        "run_id": run_id,
        "worker": worker,
        "video_id": video_id,
        "config_hash": config_hash,
        "oracle_candidates_sha256": oracle_candidates_sha256,
        "btc_mutated": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return False
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    return all(observed_remote_hashes.get(path) == digest for path, digest in artifacts.items())
