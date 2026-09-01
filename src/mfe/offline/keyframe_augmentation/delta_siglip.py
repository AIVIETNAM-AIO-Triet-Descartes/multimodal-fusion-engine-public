"""Pure helpers for resumable supplemental-frame decode and SigLIP work."""

from __future__ import annotations

import io
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mfe.offline.keyframe_augmentation.common import sha256_file, sha256_json


def validate_siglip_baseline_manifest(
    manifest: Mapping[str, Any],
    *,
    vector_sha256: str,
    checkpoint: str,
    dimension: int,
    batch_size: int,
) -> dict[str, Any]:
    """Validate the stored SigLIP artifact without inventing legacy stage metadata."""
    artifact = (manifest.get("artifacts") or {}).get("siglip_vectors")
    if not isinstance(artifact, Mapping):
        raise ValueError("baseline manifest has no siglip_vectors artifact")
    params = artifact.get("params") or {}
    checks = {
        "sha256": artifact.get("sha256") == vector_sha256,
        "model": artifact.get("model") == checkpoint,
        "dimension": artifact.get("dim") == dimension,
        "tower": params.get("tower") == "image",
        "batch_size": params.get("batch_size") == batch_size,
    }
    stage = (manifest.get("stages") or {}).get("n4_siglip")
    stage_environment = stage.get("env") if isinstance(stage, Mapping) else None
    if not isinstance(stage_environment, Mapping):
        stage_environment = None
    return {
        "artifact_contract_status": "PASS" if all(checks.values()) else "BLOCKED",
        "artifact_contract_checks": checks,
        "stage_environment_status": (
            "AVAILABLE" if stage_environment is not None else "UNVERIFIED_LEGACY_MANIFEST"
        ),
        "stage_environment": dict(stage_environment or {}),
    }


def partition_feature_worklist(
    rows: Sequence[Mapping[str, Any]],
    worker_assignments: Mapping[str, Sequence[str]],
) -> dict[str, list[dict[str, Any]]]:
    """Partition a frozen union worklist by its video's producer worker."""
    owner = {
        video_id: worker
        for worker, video_ids in worker_assignments.items()
        for video_id in video_ids
    }
    if len(owner) != sum(len(video_ids) for video_ids in worker_assignments.values()):
        raise ValueError("worker video assignments overlap")
    result: dict[str, list[dict[str, Any]]] = {
        worker: [] for worker in sorted(worker_assignments)
    }
    seen: set[tuple[str, int]] = set()
    for value in rows:
        row = dict(value)
        key = (str(row["video_id"]), int(row["frame_idx"]))
        if key in seen:
            raise ValueError(f"duplicate feature work item: {key}")
        seen.add(key)
        worker = owner.get(key[0])
        if worker is None:
            raise ValueError(f"feature work video has no worker assignment: {key[0]}")
        expected_hash = row.pop("work_item_hash", None)
        if expected_hash != sha256_json(row):
            raise ValueError(f"invalid feature work item hash: {key}")
        row["work_item_hash"] = expected_hash
        result[worker].append(row)
    for worker in result:
        result[worker].sort(key=lambda row: (row["video_id"], row["frame_idx"]))
    return result


def partition_feature_worklist_by_runner(
    rows: Sequence[Mapping[str, Any]],
    runners: Sequence[str],
    video_costs: Mapping[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """Keep videos intact while greedily balancing source-byte cost across runners."""
    runner_names = sorted(set(runners))
    if not runner_names or len(runner_names) != len(runners):
        raise ValueError("runners must be non-empty and unique")
    by_video: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for value in rows:
        row = dict(value)
        video_id = str(row["video_id"])
        key = (video_id, int(row["frame_idx"]))
        if key in seen:
            raise ValueError(f"duplicate feature work item: {key}")
        seen.add(key)
        expected_hash = row.pop("work_item_hash", None)
        if expected_hash != sha256_json(row):
            raise ValueError(f"invalid feature work item hash: {key}")
        if video_id not in video_costs or int(video_costs[video_id]) <= 0:
            raise ValueError(f"missing/invalid source cost: {video_id}")
        row["work_item_hash"] = expected_hash
        by_video.setdefault(video_id, []).append(row)

    totals = {runner: 0 for runner in runner_names}
    assignments = {runner: [] for runner in runner_names}
    ordered_videos = sorted(
        by_video,
        key=lambda video_id: (-int(video_costs[video_id]), video_id),
    )
    for video_id in ordered_videos:
        runner = min(runner_names, key=lambda name: (totals[name], name))
        assignments[runner].extend(by_video[video_id])
        totals[runner] += int(video_costs[video_id])
    for runner in assignments:
        assignments[runner].sort(key=lambda row: (row["video_id"], row["frame_idx"]))
    return assignments


def write_deterministic_image_tar(
    image_paths: Mapping[str, str | Path], output: str | Path
) -> Path:
    """Archive JPEGs with fixed metadata so reruns have an identical SHA-256."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    with tarfile.open(temporary, "w") as archive:
        for name, source_value in sorted(image_paths.items()):
            if Path(name).name != name or not name.lower().endswith(".jpg"):
                raise ValueError(f"unsafe/non-JPEG archive name: {name}")
            source = Path(source_value)
            payload = source.read_bytes()
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    temporary.replace(target)
    return target


def image_manifest(
    *, video_id: str, rows: Sequence[Mapping[str, Any]], image_dir: str | Path
) -> dict[str, Any]:
    """Bind every frozen coordinate to one decoded JPEG hash."""
    root = Path(image_dir)
    images = []
    for row in sorted(rows, key=lambda value: int(value["frame_idx"])):
        frame_idx = int(row["frame_idx"])
        path = root / f"{frame_idx:09d}.jpg"
        if not path.is_file():
            raise ValueError(f"missing decoded frame: {video_id}/{frame_idx}")
        images.append(
            {
                "keyframe_id": row["keyframe_id"],
                "video_id": video_id,
                "frame_idx": frame_idx,
                "pts_time": row["pts_time"],
                "fps": row["fps"],
                "jpeg_name": path.name,
                "jpeg_sha256": sha256_file(path),
                "work_item_hash": row["work_item_hash"],
            }
        )
    manifest = {
        "schema_version": 1,
        "video_id": video_id,
        "frame_count": len(images),
        "frame_idx_source": "decoder_counter",
        "images": images,
    }
    manifest["manifest_payload_hash"] = sha256_json(manifest)
    return manifest
