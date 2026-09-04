"""Gate 0: lock the dataset and verify the coordinate/split contracts."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from mfe.contracts.ids import build_keyframe_id, frame_idx_from_pts, parse_keyframe_id
from mfe.offline.keyframe_augmentation.common import sha256_file, sha256_json
from mfe.offline.keyframe_augmentation.models import AugmentationConfig, BtcFrame

REQUIRED_MAPPING_COLUMNS = {"n", "pts_time", "fps", "frame_idx"}


def audit_mapping_math(mapping_paths: list[str | Path]) -> dict[str, Any]:
    """Measure formula behaviour while keeping CSV ``frame_idx`` canonical.

    This audit computes expected values only to count disagreements; it never
    writes them back or substitutes them for the source column.
    """
    rows = 0
    populated = 0
    blank_frame_idx = 0
    floor_mismatches = 0
    round_mismatches = 0
    duplicate_adjacent_pairs = 0
    decreasing_adjacent_pairs = 0
    fps_rows: Counter[str] = Counter()
    fps_videos: Counter[str] = Counter()
    floor_mismatches_by_fps: Counter[str] = Counter()
    round_mismatches_by_fps: Counter[str] = Counter()
    round_video_ratios_by_fps: dict[str, list[float]] = {}
    files: list[dict[str, Any]] = []
    for raw_path in sorted((Path(path).resolve() for path in mapping_paths), key=str):
        video_fps: set[float] = set()
        previous_frame: int | None = None
        file_rows = 0
        file_populated = 0
        file_round_mismatches = 0
        with raw_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = REQUIRED_MAPPING_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{raw_path}: missing mapping columns: {sorted(missing)}")
            for line_number, row in enumerate(reader, 2):
                rows += 1
                file_rows += 1
                raw_frame = (row.get("frame_idx") or "").strip()
                if not raw_frame:
                    blank_frame_idx += 1
                    previous_frame = None
                    continue
                try:
                    pts_time = float(row["pts_time"])
                    literal_fps = float(row["fps"])
                    csv_frame_idx = int(raw_frame)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{raw_path}:{line_number}: invalid numeric row") from error
                populated += 1
                file_populated += 1
                fps_key = format(literal_fps, "g")
                fps_rows[fps_key] += 1
                video_fps.add(literal_fps)
                floor_wrong = math.floor(pts_time * literal_fps) != csv_frame_idx
                floor_mismatches += floor_wrong
                floor_mismatches_by_fps[fps_key] += floor_wrong
                # round is measured as a counterfactual only; production never uses it for ID.
                round_wrong = round(pts_time * literal_fps) != csv_frame_idx
                round_mismatches += round_wrong
                file_round_mismatches += round_wrong
                round_mismatches_by_fps[fps_key] += round_wrong
                if previous_frame is not None:
                    duplicate_adjacent_pairs += csv_frame_idx == previous_frame
                    decreasing_adjacent_pairs += csv_frame_idx < previous_frame
                previous_frame = csv_frame_idx
        if len(video_fps) != 1:
            raise ValueError(f"{raw_path}: expected one literal FPS, got {sorted(video_fps)}")
        file_fps_key = format(next(iter(video_fps)), "g")
        fps_videos[file_fps_key] += 1
        if file_populated:
            round_video_ratios_by_fps.setdefault(file_fps_key, []).append(
                file_round_mismatches / file_populated
            )
        files.append(
            {
                "path": str(raw_path),
                "sha256": sha256_file(raw_path),
                "rows": file_rows,
                "populated": file_populated,
                "literal_fps": float(file_fps_key),
                "round_mismatches": file_round_mismatches,
            }
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "files": len(files),
        "rows": rows,
        "populated_frame_idx_rows": populated,
        "blank_frame_idx_rows": blank_frame_idx,
        "floor_mismatches": floor_mismatches,
        "round_mismatches": round_mismatches,
        "round_mismatch_ratio": round_mismatches / populated if populated else None,
        "duplicate_adjacent_pairs": duplicate_adjacent_pairs,
        "decreasing_adjacent_pairs": decreasing_adjacent_pairs,
        "fps_rows": dict(sorted(fps_rows.items())),
        "fps_videos": dict(sorted(fps_videos.items())),
        "floor_mismatches_by_fps": dict(sorted(floor_mismatches_by_fps.items())),
        "round_mismatches_by_fps": dict(sorted(round_mismatches_by_fps.items())),
        "round_mismatch_ratio_by_fps": {
            fps: round_mismatches_by_fps[fps] / count for fps, count in sorted(fps_rows.items())
        },
        "round_video_ratio_range_by_fps": {
            fps: {"min": min(values), "max": max(values)}
            for fps, values in sorted(round_video_ratios_by_fps.items())
        },
        "coordinate_source": "frame_idx copied from map-keyframes CSV; formulas are audit-only",
        "files_manifest_hash": sha256_json(files),
        "evidence": "MEASURED",
    }
    report["run_hash"] = sha256_json(report)
    return report


def _query_identity(row: dict[str, Any], *, source: Path, line_number: int) -> tuple[str, str]:
    query_id = row.get("query_id")
    video_id = row.get("gt_video_id")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError(f"{source}:{line_number}: query_id must be a non-empty string")
    if not isinstance(video_id, str) or not video_id:
        raise ValueError(f"{source}:{line_number}: gt_video_id must be a non-empty string")
    return query_id, video_id


def inspect_split(path: str | Path, *, expected_split: str) -> dict[str, Any]:
    source = Path(path)
    query_ids: list[str] = []
    video_ids: list[str] = []
    query_types: Counter[str] = Counter()
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            if row.get("split") != expected_split:
                raise ValueError(
                    f"{source}:{line_number}: split={row.get('split')!r}, "
                    f"expected {expected_split!r}"
                )
            query_id, video_id = _query_identity(row, source=source, line_number=line_number)
            query_ids.append(query_id)
            video_ids.append(video_id)
            query_types[str(row.get("type", "UNKNOWN"))] += 1

    duplicates = sorted(query_id for query_id, count in Counter(query_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"{source}: duplicate query_id(s): {duplicates[:10]}")
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "queries": len(query_ids),
        "query_ids": sorted(query_ids),
        "videos": sorted(set(video_ids)),
        "query_types": dict(sorted(query_types.items())),
    }


def verify_split_isolation(dev_path: str | Path, heldout_path: str | Path) -> dict[str, Any]:
    dev = inspect_split(dev_path, expected_split="dev")
    heldout = inspect_split(heldout_path, expected_split="heldout")
    query_overlap = sorted(set(dev["query_ids"]) & set(heldout["query_ids"]))
    video_overlap = sorted(set(dev["videos"]) & set(heldout["videos"]))
    if query_overlap or video_overlap:
        raise ValueError(
            "DEV/held-out leakage: "
            f"query_overlap={query_overlap[:10]}, video_overlap={video_overlap[:10]}"
        )
    return {"dev": dev, "heldout": heldout, "query_overlap": [], "video_overlap": []}


def _probe_video(path: Path) -> dict[str, Any]:
    try:
        import cv2  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError("Gate 0 video probing requires opencv-python") from error

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"invalid video probe for {path}: fps={fps}, frames={total_frames}, "
            f"size={width}x{height}"
        )
    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": total_frames / fps,
    }


def read_and_validate_mapping(
    mapping_path: str | Path,
    *,
    video_id: str,
    total_frames: int | None = None,
) -> tuple[list[BtcFrame], dict[str, Any]]:
    source = Path(mapping_path)
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_MAPPING_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source}: missing mapping columns: {sorted(missing)}")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError(f"{source}: mapping is empty")

    frames: list[BtcFrame] = []
    fps_values: set[float] = set()
    ordinals: set[int] = set()
    seen_frames: dict[int, BtcFrame] = {}
    duplicate_aliases: list[dict[str, int]] = []
    previous_pts = -math.inf
    previous_ordinal = -1
    for line_number, row in enumerate(raw_rows, 2):
        try:
            ordinal = int(row["n"])
            pts_time = float(row["pts_time"])
            fps = float(row["fps"])
            # CSV is authoritative.  The floor computation below is an
            # independent validation and never becomes the stored identity.
            frame_idx = int(row["frame_idx"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{source}:{line_number}: invalid numeric mapping row") from error
        if ordinal < 0 or pts_time < 0 or fps <= 0 or frame_idx < 0:
            raise ValueError(f"{source}:{line_number}: coordinates must be non-negative; fps > 0")
        if ordinal in ordinals:
            raise ValueError(f"{source}:{line_number}: duplicate ordinal n={ordinal}")
        if ordinal <= previous_ordinal or pts_time < previous_pts:
            raise ValueError(f"{source}:{line_number}: mapping must be ordered by n and pts_time")
        expected_frame = frame_idx_from_pts(pts_time, fps)
        if frame_idx != expected_frame:
            raise ValueError(
                f"{source}:{line_number}: frame_idx={frame_idx} != "
                f"floor(pts_time={pts_time} * literal fps={fps})={expected_frame}"
            )
        if total_frames is not None and frame_idx >= total_frames:
            raise ValueError(
                f"{source}:{line_number}: frame_idx={frame_idx} outside [0,{total_frames})"
            )
        keyframe_id = build_keyframe_id(video_id, frame_idx)
        if parse_keyframe_id(keyframe_id) != (video_id, frame_idx):
            raise AssertionError("keyframe ID round-trip failed")
        frame = BtcFrame.build(
            video_id=video_id,
            ordinal=ordinal,
            frame_idx=frame_idx,
            pts_time=pts_time,
            fps=fps,
        )
        if frame_idx in seen_frames:
            duplicate_aliases.append(
                {
                    "frame_idx": frame_idx,
                    "kept_n": seen_frames[frame_idx].ordinal,
                    "dropped_n": ordinal,
                }
            )
        else:
            seen_frames[frame_idx] = frame
            frames.append(frame)
        fps_values.add(fps)
        ordinals.add(ordinal)
        previous_pts = pts_time
        previous_ordinal = ordinal

    if len(fps_values) != 1:
        raise ValueError(f"{source}: mapping mixes literal FPS values: {sorted(fps_values)}")
    return frames, {
        "csv_rows": len(raw_rows),
        "unique_frames": len(frames),
        "duplicate_aliases": duplicate_aliases,
        "literal_fps": next(iter(fps_values)),
        "coordinate_source": "map-keyframes.csv",
        "first_frame_idx": frames[0].frame_idx,
        "last_frame_idx": frames[-1].frame_idx,
    }


def build_dataset_lock(
    *,
    config: AugmentationConfig,
    config_path: str | Path,
    map_dir: str | Path,
    video_dir: str | Path,
    dev_path: str | Path,
    heldout_path: str | Path,
    video_ids: list[str] | None = None,
    hash_videos: bool = True,
) -> tuple[dict[str, Any], dict[str, list[BtcFrame]]]:
    mapping_root = Path(map_dir)
    video_root = Path(video_dir)
    ids = video_ids or sorted(path.stem for path in mapping_root.glob("*.csv"))
    if not ids:
        raise ValueError(f"no mapping CSV found in {mapping_root}")

    split = verify_split_isolation(dev_path, heldout_path)
    config_payload = config.model_dump(mode="json")
    config_hash = sha256_json(config_payload)
    records: list[dict[str, Any]] = []
    mappings: dict[str, list[BtcFrame]] = {}
    for video_id in ids:
        mapping_path = mapping_root / f"{video_id}.csv"
        video_path = video_root / f"{video_id}.mp4"
        if not mapping_path.is_file() or not video_path.is_file():
            raise ValueError(f"{video_id}: missing mapping or video")
        video_probe = _probe_video(video_path)
        frames, mapping_stats = read_and_validate_mapping(
            mapping_path,
            video_id=video_id,
            total_frames=video_probe["total_frames"],
        )
        literal_fps = mapping_stats["literal_fps"]
        fps_delta = abs(float(video_probe["fps"]) - literal_fps)
        if fps_delta > 0.05:
            raise ValueError(
                f"{video_id}: decoder FPS {video_probe['fps']} disagrees with "
                f"mapping literal FPS {literal_fps}"
            )
        mappings[video_id] = frames
        record = {
            "video_id": video_id,
            "mapping_path": str(mapping_path.resolve()),
            "mapping_sha256": sha256_file(mapping_path),
            "video_path": str(video_path.resolve()),
            "video_sha256": sha256_file(video_path) if hash_videos else None,
            "video_hash_status": "MEASURED" if hash_videos else "UNVERIFIED",
            "probe": video_probe,
            "mapping": mapping_stats,
            "contract_status": "PASS",
        }
        records.append(record)

    content = {
        "schema_version": 1,
        "protocol_version": config.protocol_version,
        "dataset_version": config.dataset_version,
        "split_version": config.split_version,
        "config_path": str(Path(config_path).resolve()),
        "config_hash": config_hash,
        "split": split,
        "videos": records,
    }
    content["dataset_hash"] = sha256_json(content)
    content["run_id"] = f"kfaug-{content['dataset_hash'][:12]}-{config_hash[:12]}"
    content["evidence"] = {
        "dataset_hash": "MEASURED" if hash_videos else "UNVERIFIED",
        "mapping_contract": "MEASURED",
        "split_isolation": "MEASURED",
    }
    return content, mappings
