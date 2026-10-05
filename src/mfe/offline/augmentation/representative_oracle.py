"""Deterministic sampling and lineage contracts for the O1-S lower-bound scan."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mfe.contracts.ids import build_keyframe_id
from mfe.offline.augmentation.io import canonical_hash
from mfe.offline.augmentation.oracle_artifacts import OracleArtifactTarget


class RepresentativeMembership(BaseModel):
    """One query/event whose inclusive GT interval contains a sampled frame."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    query_id: str = Field(min_length=1)
    event_index: int = Field(ge=0)
    frame_range_start: int = Field(ge=0)
    frame_range_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _interval_is_valid(self) -> RepresentativeMembership:
        if self.frame_range_end < self.frame_range_start:
            raise ValueError("representative interval must be inclusive and non-empty")
        return self


class RepresentativeScanRequest(BaseModel):
    """One source-frame coordinate to decode once for one or more DEV events."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    keyframe_id: str
    video_id: str
    frame_idx: int = Field(ge=0)
    memberships: tuple[RepresentativeMembership, ...] = Field(min_length=1)
    sampling_policy: Literal["inclusive_uniform_integer_lb64"]

    @model_validator(mode="after")
    def _identity_and_memberships_match(self) -> RepresentativeScanRequest:
        expected = build_keyframe_id(self.video_id, self.frame_idx)
        if self.keyframe_id != expected:
            raise ValueError(f"representative keyframe_id must be {expected!r}")
        identities = [(item.query_id, item.event_index) for item in self.memberships]
        if identities != sorted(set(identities)):
            raise ValueError("representative memberships must be unique and sorted")
        if any(
            not item.frame_range_start <= self.frame_idx <= item.frame_range_end
            for item in self.memberships
        ):
            raise ValueError("representative frame lies outside a linked GT interval")
        return self


class RepresentativeResolvedSample(BaseModel):
    """Decoder-authoritative O1-S sample with deterministic image provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    request: RepresentativeScanRequest
    decoder_pts_time: float = Field(ge=0)
    literal_fps: float = Field(gt=0)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    jpeg_name: str = Field(pattern=r"^[0-9]{9}\.jpg$")
    frame_idx_source: Literal["decoder_counter"] = "decoder_counter"
    pts_time_source: Literal["decoder_pts"] = "decoder_pts"

    @model_validator(mode="after")
    def _jpeg_name_matches_frame(self) -> RepresentativeResolvedSample:
        expected = f"{self.request.frame_idx:09d}.jpg"
        if self.jpeg_name != expected:
            raise ValueError(f"representative JPEG name must be {expected!r}")
        return self


def inclusive_uniform_indices(
    start_frame: int,
    end_frame: int,
    *,
    max_samples: int,
) -> tuple[int, ...]:
    """Sample an inclusive integer interval without float rounding.

    Intervals no longer than ``max_samples`` are a full scan. Longer intervals
    retain both endpoints and use lower integer positions between them. Such a
    scan is explicitly a lower bound and must never be reported as full O1-S.
    """

    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("inclusive frame interval is invalid")
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2 to preserve both endpoints")
    length = end_frame - start_frame + 1
    count = min(length, max_samples)
    if count == 1:
        return (start_frame,)
    span = end_frame - start_frame
    indices = tuple(
        start_frame + (sample_index * span) // (count - 1)
        for sample_index in range(count)
    )
    if indices[0] != start_frame or indices[-1] != end_frame:
        raise AssertionError("inclusive sampler dropped an endpoint")
    if len(indices) != len(set(indices)):
        raise AssertionError("inclusive sampler produced duplicate frame indices")
    return indices


def build_representative_scan_plan(
    targets: Iterable[OracleArtifactTarget | Mapping[str, Any]],
    *,
    max_samples_per_event: int = 64,
) -> list[RepresentativeScanRequest]:
    """Union deterministic per-event samples so overlapping events decode once."""

    by_frame: dict[tuple[str, int], set[tuple[str, int, int, int]]] = defaultdict(set)
    seen_events: set[tuple[str, int]] = set()
    for value in targets:
        target = (
            value
            if isinstance(value, OracleArtifactTarget)
            else OracleArtifactTarget.model_validate(value)
        )
        event_identity = (target.query_id, target.event_index)
        if event_identity in seen_events:
            raise ValueError(f"duplicate oracle target event: {event_identity}")
        seen_events.add(event_identity)
        membership = (
            target.query_id,
            target.event_index,
            target.frame_range_start,
            target.frame_range_end,
        )
        for frame_idx in inclusive_uniform_indices(
            target.frame_range_start,
            target.frame_range_end,
            max_samples=max_samples_per_event,
        ):
            by_frame[(target.video_id, frame_idx)].add(membership)

    requests = []
    for (video_id, frame_idx), memberships in sorted(by_frame.items()):
        requests.append(
            RepresentativeScanRequest(
                keyframe_id=build_keyframe_id(video_id, frame_idx),
                video_id=video_id,
                frame_idx=frame_idx,
                memberships=tuple(
                    RepresentativeMembership(
                        query_id=query_id,
                        event_index=event_index,
                        frame_range_start=start,
                        frame_range_end=end,
                    )
                    for query_id, event_index, start, end in sorted(memberships)
                ),
                sampling_policy="inclusive_uniform_integer_lb64",
            )
        )
    return requests


def representative_scan_plan_hash(
    requests: Iterable[RepresentativeScanRequest | Mapping[str, Any]],
) -> str:
    """Hash the coordinate/membership plan independently of decode outputs."""

    payload = [
        (
            row.model_dump(mode="json")
            if isinstance(row, RepresentativeScanRequest)
            else RepresentativeScanRequest.model_validate(row).model_dump(mode="json")
        )
        for row in requests
    ]
    payload.sort(key=lambda row: (row["video_id"], row["frame_idx"]))
    return canonical_hash(payload)


def partition_representative_scan(
    requests: Sequence[RepresentativeScanRequest],
    runners: Sequence[str],
    video_costs: Mapping[str, int],
) -> dict[str, list[RepresentativeScanRequest]]:
    """Keep videos intact while balancing source bytes across two or more runners."""

    runner_names = sorted(set(runners))
    if not runner_names or len(runner_names) != len(runners):
        raise ValueError("runners must be non-empty and unique")
    by_video: dict[str, list[RepresentativeScanRequest]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for request in requests:
        key = (request.video_id, request.frame_idx)
        if key in seen:
            raise ValueError(f"duplicate representative request: {key}")
        seen.add(key)
        if int(video_costs.get(request.video_id, 0)) <= 0:
            raise ValueError(f"missing/invalid source cost: {request.video_id}")
        by_video[request.video_id].append(request)

    totals = {runner: 0 for runner in runner_names}
    assignments = {runner: [] for runner in runner_names}
    for video_id in sorted(by_video, key=lambda value: (-int(video_costs[value]), value)):
        runner = min(runner_names, key=lambda value: (totals[value], value))
        assignments[runner].extend(sorted(by_video[video_id], key=lambda row: row.frame_idx))
        totals[runner] += int(video_costs[video_id])
    return assignments


def validate_scan_completion_marker(
    marker: Mapping[str, Any],
    *,
    run_id: str,
    runner: str,
    video_id: str,
    plan_hash: str,
    observed_remote_hashes: Mapping[str, str | None],
) -> bool:
    """Accept only a complete lower-bound scan marker with verified artifacts."""

    payload = dict(marker)
    claimed = payload.pop("marker_payload_hash", None)
    if claimed != canonical_hash(payload):
        return False
    expected = {
        "status": "COMPLETE",
        "run_id": run_id,
        "runner": runner,
        "video_id": video_id,
        "scan_plan_hash": plan_hash,
        "sampling_policy": "inclusive_uniform_integer_lb64",
        "heldout_accessed": False,
        "btc_mutated": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return False
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    return all(observed_remote_hashes.get(path) == digest for path, digest in artifacts.items())


__all__ = [
    "RepresentativeMembership",
    "RepresentativeResolvedSample",
    "RepresentativeScanRequest",
    "build_representative_scan_plan",
    "inclusive_uniform_indices",
    "partition_representative_scan",
    "representative_scan_plan_hash",
    "validate_scan_completion_marker",
]
