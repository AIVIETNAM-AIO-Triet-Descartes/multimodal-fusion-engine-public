"""Fail-closed corpus lock and temporal-coverage audit for BTC shot proposals.

This module intentionally performs no network I/O.  A notebook or CLI stages
the small producer artifacts and canonical BTC mappings locally, then calls the
pure functions here.  Temporal coverage is diagnostic evidence only; it cannot
establish retrieval headroom or an official-score improvement.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mfe.offline.keyframe_augmentation.common import sha256_file, sha256_json
from mfe.offline.keyframe_augmentation.gate0 import read_and_validate_mapping
from mfe.offline.keyframe_augmentation.gate1 import classify_shot, load_shot_proposals
from mfe.offline.keyframe_augmentation.models import AugmentationConfig, CoverageClass
from mfe.offline.source import VideoEntry


def byte_balanced_slices(
    video_ids: Sequence[str], count: int, catalogue: Mapping[str, VideoEntry]
) -> list[list[str]]:
    """Reproduce the producer notebook's deterministic worker partition."""
    if count < 1:
        raise ValueError("worker count must be positive")
    missing = sorted(set(video_ids) - set(catalogue))
    if missing:
        raise ValueError(f"videos missing from catalogue: {missing[:5]}")
    buckets: list[list[str]] = [[] for _ in range(count)]
    totals = [0] * count
    for video_id in sorted(video_ids, key=lambda value: (-catalogue[value].size, value)):
        index = min(range(count), key=lambda value: (totals[value], value))
        buckets[index].append(video_id)
        totals[index] += catalogue[video_id].size
    return [sorted(bucket) for bucket in buckets]


def expected_worker_assignments(
    *,
    producer_config: Mapping[str, Any],
    inventory: Mapping[str, Any],
    catalogue: Mapping[str, VideoEntry],
) -> dict[str, list[str]]:
    """Build the exact 12 producer slices from the pinned inventory/config."""
    assignment_key = str(producer_config["source"]["assignment_key"])
    assignments = inventory[assignment_key]
    result: dict[str, list[str]] = {}
    for member in producer_config["members"]:
        workers = list(producer_config["workers"][member])
        owner_videos = sorted(
            video_id
            for zip_name in assignments[member]
            for video_id in inventory["zips"][zip_name]["videos"]
        )
        slices = byte_balanced_slices(owner_videos, len(workers), catalogue)
        result.update(dict(zip(workers, slices, strict=True)))
    flattened = [video_id for videos in result.values() for video_id in videos]
    if len(flattened) != len(set(flattened)):
        raise ValueError("producer worker assignments overlap")
    return result


def validate_corpus_index(
    *,
    producer_config: Mapping[str, Any],
    expected: Mapping[str, Sequence[str]],
    manifests: Mapping[str, Mapping[str, Any]],
    markers: Mapping[str, Mapping[str, Any]],
    producer_config_sha256: str,
) -> dict[str, Any]:
    """Validate all manifests and marker payloads before reading shot data."""
    expected_workers = set(expected)
    if set(manifests) != expected_workers:
        raise ValueError(
            f"manifest workers mismatch: missing={sorted(expected_workers - set(manifests))}, "
            f"extra={sorted(set(manifests) - expected_workers)}"
        )
    expected_owner = {
        video_id: worker for worker, videos in expected.items() for video_id in videos
    }
    if set(markers) != set(expected_owner):
        raise ValueError(
            f"marker inventory mismatch: missing={len(set(expected_owner) - set(markers))}, "
            f"extra={len(set(markers) - set(expected_owner))}"
        )

    manifest_hashes: dict[str, str] = {}
    verified_by_manifest: dict[str, Mapping[str, Any]] = {}
    for worker, assigned in expected.items():
        manifest = dict(manifests[worker])
        payload_hash = manifest.pop("manifest_payload_hash", None)
        if payload_hash != sha256_json(manifest):
            raise ValueError(f"{worker}: invalid manifest_payload_hash")
        if manifest.get("run_id") != producer_config["run_id"]:
            raise ValueError(f"{worker}: wrong run_id")
        if manifest.get("worker") != worker:
            raise ValueError(f"{worker}: worker identity mismatch")
        if manifest.get("config_sha256") != producer_config_sha256:
            raise ValueError(f"{worker}: producer config hash mismatch")
        verified = list(manifest.get("verified", []))
        if manifest.get("assigned_count") != len(assigned):
            raise ValueError(f"{worker}: assigned_count mismatch")
        if (
            manifest.get("verified_complete_count") != len(assigned)
            or len(verified) != len(assigned)
        ):
            raise ValueError(f"{worker}: incomplete final manifest")
        actual_ids = [str(row["video_id"]) for row in verified]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(assigned):
            raise ValueError(f"{worker}: verified video inventory mismatch")
        for row in verified:
            video_id = str(row["video_id"])
            if video_id in verified_by_manifest:
                raise ValueError(f"duplicate manifest verification for {video_id}")
            verified_by_manifest[video_id] = row
        manifest_hashes[worker] = str(payload_hash)

    detector_fingerprints: set[str] = set()
    marker_payload_hashes: dict[str, str] = {}
    for video_id, marker_value in markers.items():
        marker = dict(marker_value)
        payload_hash = marker.pop("marker_payload_hash", None)
        if payload_hash != sha256_json(marker):
            raise ValueError(f"{video_id}: invalid marker_payload_hash")
        expected_worker = expected_owner[video_id]
        if marker.get("status") != "COMPLETE" or marker.get("video_id") != video_id:
            raise ValueError(f"{video_id}: marker is not a matching COMPLETE marker")
        if marker.get("worker") != expected_worker:
            raise ValueError(f"{video_id}: marker worker mismatch")
        if marker.get("config_sha256") != producer_config_sha256:
            raise ValueError(f"{video_id}: marker config hash mismatch")
        manifest_row = verified_by_manifest[video_id]
        for key in ("marker_payload_hash", "shots_sha256", "report_sha256"):
            marker_value_for_key = payload_hash if key == "marker_payload_hash" else marker.get(key)
            if manifest_row.get(key) != marker_value_for_key:
                raise ValueError(f"{video_id}: manifest/marker {key} mismatch")
        detector_fingerprints.add(
            sha256_json(
                {
                    "commit": marker.get("transnet_commit"),
                    "source": marker.get("transnet_source_sha256"),
                    "weights": marker.get("transnet_weights_sha256"),
                    "threshold": marker.get("threshold"),
                }
            )
        )
        marker_payload_hashes[video_id] = str(payload_hash)

    return {
        "status": "PASS",
        "expected_workers": len(expected),
        "validated_manifests": len(manifests),
        "expected_videos": len(expected_owner),
        "validated_markers": len(markers),
        "missing_videos": [],
        "duplicate_videos": [],
        "manifest_payload_hashes": manifest_hashes,
        "marker_payload_hashes_sha256": sha256_json(marker_payload_hashes),
        "detector_fingerprints": sorted(detector_fingerprints),
    }


def audit_video(
    *,
    video_id: str,
    shots_path: str | Path,
    mapping_path: str | Path,
    marker: Mapping[str, Any],
    config: AugmentationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one pair of artifacts and return per-video and per-shot rows."""
    shots_source = Path(shots_path)
    mapping_source = Path(mapping_path)
    if sha256_file(shots_source) != marker.get("shots_sha256"):
        raise ValueError(f"{video_id}: shots SHA-256 mismatch")
    if sha256_file(mapping_source) != marker.get("btc_mapping_sha256"):
        raise ValueError(f"{video_id}: BTC mapping SHA-256 mismatch")
    shots = load_shot_proposals(shots_source)
    if not shots or any(shot.video_id != video_id for shot in shots):
        raise ValueError(f"{video_id}: empty or cross-video shot artifact")
    if len(shots) != marker.get("shot_count"):
        raise ValueError(f"{video_id}: shot_count mismatch")
    if shots[0].start_frame != 0:
        raise ValueError(f"{video_id}: first shot does not start at frame zero")
    for previous, current in zip(shots, shots[1:], strict=False):
        if current.start_frame != previous.end_frame + 1:
            raise ValueError(f"{video_id}: shot timeline has a gap or overlap")
    total_frames = shots[-1].end_frame + 1
    btc_frames, mapping_report = read_and_validate_mapping(
        mapping_source, video_id=video_id, total_frames=total_frames
    )
    literal_fps = float(mapping_report["literal_fps"])
    if any(abs(shot.fps - literal_fps) > 1e-9 for shot in shots):
        raise ValueError(f"{video_id}: shot/mapping literal FPS mismatch")
    if abs(float(marker.get("literal_fps")) - literal_fps) > 1e-9:
        raise ValueError(f"{video_id}: marker/mapping literal FPS mismatch")

    audits = [
        classify_shot(
            shot,
            btc_frames,
            target_gap_seconds=config.target_gap_seconds,
            k_max=config.k_max,
            config=config,
        )
        for shot in shots
    ]
    counts = Counter(audit.coverage_class.value for audit in audits)
    shot_rows = [audit.model_dump(mode="json") for audit in audits]
    per_video = {
        "video_id": video_id,
        "literal_fps": literal_fps,
        "total_frames": total_frames,
        "shot_count": len(shots),
        "btc_csv_rows": mapping_report["csv_rows"],
        "btc_unique_frames": mapping_report["unique_frames"],
        "btc_duplicate_aliases": len(mapping_report["duplicate_aliases"]),
        "coverage_counts": {
            coverage.value: counts.get(coverage.value, 0) for coverage in CoverageClass
        },
        "eligible_shots": counts.get("C0", 0) + counts.get("C1", 0),
        "max_blind_gap_seconds": max(row["max_blind_gap_seconds"] for row in shot_rows),
        "shots_sha256": sha256_file(shots_source),
        "btc_mapping_sha256": sha256_file(mapping_source),
    }
    per_video["audit_payload_hash"] = sha256_json(per_video)
    return per_video, shot_rows


def build_summary(
    *, per_video_rows: Sequence[Mapping[str, Any]], config: AugmentationConfig
) -> dict[str, Any]:
    """Aggregate coverage without promoting a temporal diagnostic to a decision."""
    counts: Counter[str] = Counter()
    total_shots = 0
    for row in per_video_rows:
        total_shots += int(row["shot_count"])
        counts.update(row["coverage_counts"])
    return {
        "schema_version": 1,
        "status": "AUDIT_COMPLETE",
        "evidence_label": "MEASURED",
        "decision_authority": "diagnostic_only",
        "video_count": len(per_video_rows),
        "shot_count": total_shots,
        "coverage_counts": {
            coverage.value: counts.get(coverage.value, 0) for coverage in CoverageClass
        },
        "eligible_shots": counts.get("C0", 0) + counts.get("C1", 0),
        "target_gap_seconds": config.target_gap_seconds,
        "k_max": config.k_max,
        "rounding_contract": {
            "frame_from_pts": "floor",
            "hard_ratio_budget": "floor",
            "k_target": "ceil",
            "min_gap": "ceil",
        },
        "oracle_status": {
            "O1-T": "MEASURED_BY_THIS_AUDIT",
            "O1-S": "BLOCKED_FEATURE_ARTIFACTS_REQUIRED",
            "O1-R": "BLOCKED_QUERY_RANK_TRACES_REQUIRED",
        },
        "prohibited_inference": "Temporal coverage alone cannot establish NO_HEADROOM.",
    }
