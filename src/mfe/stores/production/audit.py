"""Read-only dataset contract and count reconciliation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_COLUMNS = ("n", "pts_time", "fps", "frame_idx")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_by_stem(root: Path, tree_names: tuple[str, ...], stem: str, suffix: str) -> Path:
    candidates = [
        path
        for tree in tree_names
        if (root / tree).is_dir()
        for path in (root / tree).rglob(f"{stem}{suffix}")
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one {suffix} for {stem} under {tree_names}; found {len(candidates)}"
        )
    return candidates[0]


def _one_directory(root: Path, tree: str, video_id: str) -> Path:
    candidates = [path for path in (root / tree).rglob(video_id) if path.is_dir()]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one directory for {video_id} under {tree}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def audit_sample(root: Path) -> dict[str, Any]:
    mapping_root = root / "map-keyframes"
    if not mapping_root.is_dir():
        raise FileNotFoundError(f"missing map-keyframes directory: {mapping_root}")
    results: list[dict[str, Any]] = []
    total_rows = 0
    for csv_path in sorted(mapping_root.rglob("*.csv")):
        video_id = csv_path.stem
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
                raise ValueError(
                    f"{csv_path}: columns must be exactly {REQUIRED_COLUMNS}, "
                    f"got {reader.fieldnames}"
                )
            rows = list(reader)
        if not rows:
            raise ValueError(f"{csv_path}: empty mapping")
        fps_values = [float(row["fps"]) for row in rows]
        if not all(math.isfinite(fps) and fps > 0 for fps in fps_values):
            raise ValueError(f"{csv_path}: fps must be finite and positive")
        if any(fps != fps_values[0] for fps in fps_values):
            raise ValueError(f"{csv_path}: fps changes within a video")
        n_values = [int(row["n"]) for row in rows]
        if n_values != list(range(n_values[0], n_values[0] + len(rows))):
            raise ValueError(f"{csv_path}: n must be contiguous and strictly increasing")
        pts_values = [float(row["pts_time"]) for row in rows]
        if not all(math.isfinite(pts) and pts >= 0 for pts in pts_values):
            raise ValueError(f"{csv_path}: pts_time must be finite and non-negative")
        if any(
            current < previous
            for previous, current in zip(pts_values, pts_values[1:], strict=False)
        ):
            raise ValueError(f"{csv_path}: pts_time must be monotonic")
        frame_values = [int(row["frame_idx"]) for row in rows]
        duplicate_frame_count = len(frame_values) - len(set(frame_values))
        mismatches = [
            index
            for index, (pts, frame) in enumerate(zip(pts_values, frame_values, strict=True))
            if math.floor(pts * fps_values[0]) != frame
        ]
        feature_path = _one_by_stem(
            root, ("clip-features", "clip-features-32"), video_id, ".npy"
        )
        embeddings = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if embeddings.shape != (len(rows), 512):
            raise ValueError(
                f"{feature_path}: shape {embeddings.shape} != ({len(rows)}, 512)"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError(f"{feature_path}: embedding contains NaN or infinity")
        norms = np.linalg.norm(np.asarray(embeddings, dtype=np.float32), axis=1)
        video_path = _one_by_stem(root, ("videos",), video_id, ".mp4")
        media_info_path = _one_by_stem(root, ("media-info",), video_id, ".json")
        media_info = json.loads(media_info_path.read_text(encoding="utf-8"))
        if not isinstance(media_info, dict):
            raise ValueError(f"{media_info_path}: media info must be a JSON object")
        keyframe_files = sorted(
            path
            for path in _one_directory(root, "keyframes", video_id).iterdir()
            if path.is_file()
        )
        object_files = sorted(
            path
            for path in _one_directory(root, "objects", video_id).iterdir()
            if path.is_file() and path.suffix.lower() == ".json"
        )
        if len(keyframe_files) != len(rows) or len(object_files) != len(rows):
            raise ValueError(
                f"{video_id}: mapping/keyframe/object counts are "
                f"{len(rows)}/{len(keyframe_files)}/{len(object_files)}"
            )
        for object_path in object_files:
            payload = json.loads(object_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"{object_path}: object artifact must be a JSON object")
        total_rows += len(rows)
        results.append(
            {
                "video_id": video_id,
                "rows": len(rows),
                "fps": fps_values[0],
                "frame_formula_mismatches": mismatches,
                "duplicate_frame_count": duplicate_frame_count,
                "embedding_dtype": str(embeddings.dtype),
                "embedding_shape": list(embeddings.shape),
                "keyframe_count": len(keyframe_files),
                "object_count": len(object_files),
                "video_bytes": video_path.stat().st_size,
                "embedding_norm_min": float(norms.min()),
                "embedding_norm_max": float(norms.max()),
                "mapping_sha256": sha256(csv_path),
                "embedding_sha256": sha256(feature_path),
                "video_sha256": sha256(video_path),
                "media_info_sha256": sha256(media_info_path),
            }
        )
    if not results:
        raise ValueError("no mapping CSV files found")
    return {
        "root": str(root.resolve()),
        "video_count": len(results),
        "vector_rows": total_rows,
        "frame_formula_mismatch_count": sum(
            len(item["frame_formula_mismatches"]) for item in results
        ),
        "duplicate_frame_count": sum(item["duplicate_frame_count"] for item in results),
        "videos": results,
    }


def audit_indexing_core(
    root: Path, *, validate_object_json: bool = False
) -> dict[str, Any]:
    """Audit every available mapping/vector/keyframe/object artifact without mutation.

    Unlike ``audit_sample``, this inventory does not require videos or media-info and reports
    row anomalies instead of stopping at the first one. It is intended for incomplete BTC
    handoffs where the indexing artifacts arrived before the original media.
    """
    mapping_root = root / "map-keyframes"
    feature_root = root / "clip-features"
    if not mapping_root.is_dir() or not feature_root.is_dir():
        raise FileNotFoundError("map-keyframes and clip-features directories are required")

    feature_paths: dict[str, list[Path]] = {}
    for path in feature_root.rglob("*.npy"):
        feature_paths.setdefault(path.stem, []).append(path)
    keyframe_paths: dict[str, list[Path]] = {}
    for path in (root / "keyframes").rglob("*"):
        if path.is_dir():
            keyframe_paths.setdefault(path.name, []).append(path)
    object_paths: dict[str, list[Path]] = {}
    for path in (root / "objects").rglob("*"):
        if path.is_dir():
            object_paths.setdefault(path.name, []).append(path)

    videos: list[dict[str, Any]] = []
    totals = {
        "mapping_rows": 0,
        "valid_computed_frame_rows": 0,
        "blank_frame_idx_rows": 0,
        "invalid_frame_idx_rows": 0,
        "provided_formula_mismatches": 0,
        "computed_frame_duplicates": 0,
        "provided_frame_duplicates": 0,
        "n_sequence_anomalies": 0,
        "fps_change_videos": 0,
        "pts_nonmonotonic_videos": 0,
        "missing_feature_files": 0,
        "wrong_feature_shapes": 0,
        "unexpected_feature_dtypes": 0,
        "nonfinite_feature_files": 0,
        "zero_vector_rows": 0,
        "embedding_norm_min": None,
        "embedding_norm_max": None,
        "keyframe_count_mismatches": 0,
        "object_count_mismatches": 0,
        "invalid_object_json_files": 0,
    }
    mapping_files = sorted(mapping_root.rglob("*.csv"))
    for csv_path in mapping_files:
        video_id = csv_path.stem
        anomalies: list[str] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = tuple(reader.fieldnames or ())
            rows = list(reader)
        if columns != REQUIRED_COLUMNS:
            anomalies.append(f"columns={columns!r}")

        fps_values: list[float] = []
        pts_values: list[float] = []
        n_values: list[int] = []
        provided_frames: list[int] = []
        computed_frames: list[int] = []
        blank_frames = 0
        invalid_frames = 0
        mismatch_count = 0
        for row_index, row in enumerate(rows):
            try:
                n_values.append(int(row.get("n", "")))
            except (TypeError, ValueError):
                anomalies.append(f"invalid n at row {row_index + 2}")
            try:
                fps = float(row.get("fps", ""))
                pts = float(row.get("pts_time", ""))
            except (TypeError, ValueError):
                anomalies.append(f"invalid fps/pts at row {row_index + 2}")
                continue
            if not math.isfinite(fps) or fps <= 0 or not math.isfinite(pts) or pts < 0:
                anomalies.append(f"non-positive/non-finite fps/pts at row {row_index + 2}")
                continue
            fps_values.append(fps)
            pts_values.append(pts)
            computed = math.floor(pts * fps)
            computed_frames.append(computed)
            raw_frame = (row.get("frame_idx") or "").strip()
            if not raw_frame:
                blank_frames += 1
                continue
            try:
                provided = int(raw_frame)
            except ValueError:
                invalid_frames += 1
                continue
            provided_frames.append(provided)
            mismatch_count += provided != computed

        fps_changes = len(set(fps_values)) > 1
        n_sequence_anomaly = n_values != list(
            range(n_values[0], n_values[0] + len(n_values))
        ) if n_values else bool(rows)
        pts_nonmonotonic = any(
            current < previous
            for previous, current in zip(pts_values, pts_values[1:], strict=False)
        )
        computed_duplicates = len(computed_frames) - len(set(computed_frames))
        provided_duplicates = len(provided_frames) - len(set(provided_frames))
        if fps_changes:
            anomalies.append("fps_changes")
        if n_sequence_anomaly:
            anomalies.append("n_sequence_anomaly")
        if pts_nonmonotonic:
            anomalies.append("pts_nonmonotonic")
        if blank_frames:
            anomalies.append(f"blank_frame_idx_rows={blank_frames}")
        if invalid_frames:
            anomalies.append(f"invalid_frame_idx_rows={invalid_frames}")
        if mismatch_count:
            anomalies.append(f"provided_formula_mismatches={mismatch_count}")
        if computed_duplicates:
            anomalies.append(f"computed_frame_duplicates={computed_duplicates}")
        if provided_duplicates:
            anomalies.append(f"provided_frame_duplicates={provided_duplicates}")

        feature_matches = feature_paths.get(video_id, [])
        feature_shape: list[int] | None = None
        feature_dtype: str | None = None
        feature_finite: bool | None = None
        feature_norm_min: float | None = None
        feature_norm_max: float | None = None
        zero_vector_rows = 0
        feature_checksum: str | None = None
        if len(feature_matches) != 1:
            totals["missing_feature_files"] += 1
            anomalies.append(f"feature_file_count={len(feature_matches)}")
        else:
            feature_path = feature_matches[0]
            embeddings = np.load(feature_path, mmap_mode="r", allow_pickle=False)
            feature_shape = list(embeddings.shape)
            feature_dtype = str(embeddings.dtype)
            feature_finite = bool(np.isfinite(embeddings).all())
            feature_checksum = sha256(feature_path)
            if embeddings.shape != (len(rows), 512):
                totals["wrong_feature_shapes"] += 1
                anomalies.append(f"feature_shape={embeddings.shape!r}")
            if not feature_finite:
                totals["nonfinite_feature_files"] += 1
                anomalies.append("nonfinite_embedding")
            if feature_dtype not in {"float16", "float32"}:
                totals["unexpected_feature_dtypes"] += 1
                anomalies.append(f"feature_dtype={feature_dtype}")
            if embeddings.ndim == 2 and embeddings.shape[1] == 512 and feature_finite:
                norms = np.linalg.norm(np.asarray(embeddings, dtype=np.float32), axis=1)
                feature_norm_min = float(norms.min()) if len(norms) else None
                feature_norm_max = float(norms.max()) if len(norms) else None
                zero_vector_rows = int(np.count_nonzero(norms == 0.0))
                totals["zero_vector_rows"] += zero_vector_rows
                current_min = totals["embedding_norm_min"]
                current_max = totals["embedding_norm_max"]
                if feature_norm_min is not None:
                    totals["embedding_norm_min"] = (
                        feature_norm_min
                        if current_min is None
                        else min(current_min, feature_norm_min)
                    )
                if feature_norm_max is not None:
                    totals["embedding_norm_max"] = (
                        feature_norm_max
                        if current_max is None
                        else max(current_max, feature_norm_max)
                    )

        keyframe_dirs = keyframe_paths.get(video_id, [])
        object_dirs = object_paths.get(video_id, [])
        keyframe_count = (
            sum(path.is_file() for path in keyframe_dirs[0].iterdir())
            if len(keyframe_dirs) == 1 and keyframe_dirs[0].is_dir()
            else -1
        )
        object_count = (
            sum(
                path.is_file() and path.suffix.lower() == ".json"
                for path in object_dirs[0].iterdir()
            )
            if len(object_dirs) == 1 and object_dirs[0].is_dir()
            else -1
        )
        if keyframe_count != len(rows):
            totals["keyframe_count_mismatches"] += 1
            anomalies.append(f"keyframe_count={keyframe_count}")
        if object_count != len(rows):
            totals["object_count_mismatches"] += 1
            anomalies.append(f"object_count={object_count}")
        invalid_object_json = 0
        if validate_object_json and object_count >= 0:
            for object_path in object_dirs[0].iterdir():
                if not object_path.is_file() or object_path.suffix.lower() != ".json":
                    continue
                try:
                    payload = json.loads(object_path.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        invalid_object_json += 1
                except (OSError, UnicodeError, json.JSONDecodeError):
                    invalid_object_json += 1
            if invalid_object_json:
                anomalies.append(f"invalid_object_json={invalid_object_json}")
        totals["invalid_object_json_files"] += invalid_object_json

        totals["mapping_rows"] += len(rows)
        totals["valid_computed_frame_rows"] += len(computed_frames)
        totals["blank_frame_idx_rows"] += blank_frames
        totals["invalid_frame_idx_rows"] += invalid_frames
        totals["provided_formula_mismatches"] += mismatch_count
        totals["computed_frame_duplicates"] += computed_duplicates
        totals["provided_frame_duplicates"] += provided_duplicates
        totals["n_sequence_anomalies"] += int(n_sequence_anomaly)
        totals["fps_change_videos"] += int(fps_changes)
        totals["pts_nonmonotonic_videos"] += int(pts_nonmonotonic)
        videos.append(
            {
                "video_id": video_id,
                "rows": len(rows),
                "fps_values": sorted(set(fps_values)),
                "fps_changes": fps_changes,
                "n_sequence_anomaly": n_sequence_anomaly,
                "pts_nonmonotonic": pts_nonmonotonic,
                "blank_frame_idx_rows": blank_frames,
                "invalid_frame_idx_rows": invalid_frames,
                "provided_formula_mismatches": mismatch_count,
                "computed_frame_duplicates": computed_duplicates,
                "provided_frame_duplicates": provided_duplicates,
                "duplicate_computed_frame_ids": sorted(
                    frame
                    for frame in set(computed_frames)
                    if computed_frames.count(frame) > 1
                ),
                "feature_shape": feature_shape,
                "feature_dtype": feature_dtype,
                "feature_finite": feature_finite,
                "feature_norm_min": feature_norm_min,
                "feature_norm_max": feature_norm_max,
                "zero_vector_rows": zero_vector_rows,
                "mapping_sha256": sha256(csv_path),
                "feature_sha256": feature_checksum,
                "keyframe_count": keyframe_count,
                "object_count": object_count,
                "invalid_object_json_files": invalid_object_json,
                "anomalies": anomalies,
            }
        )

    available_mapping_ids = {path.stem for path in mapping_files}
    feature_ids = set(feature_paths)
    keyframe_ids = set(keyframe_paths)
    object_ids = set(object_paths)
    return {
        "root": str(root.resolve()),
        "mapping_video_count": len(mapping_files),
        "feature_video_count": len(feature_ids),
        "keyframe_video_count": len(keyframe_ids),
        "object_video_count": len(object_ids),
        "orphan_feature_ids": sorted(feature_ids - available_mapping_ids),
        "orphan_keyframe_ids": sorted(keyframe_ids - available_mapping_ids),
        "orphan_object_ids": sorted(object_ids - available_mapping_ids),
        "totals": totals,
        "anomalous_video_count": sum(bool(video["anomalies"]) for video in videos),
        "videos": videos,
    }
