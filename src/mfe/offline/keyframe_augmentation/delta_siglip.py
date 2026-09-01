"""Pure helpers for resumable supplemental-frame decode and SigLIP work."""

from __future__ import annotations

import io
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mfe.offline.keyframe_augmentation.common import sha256_file, sha256_json


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
