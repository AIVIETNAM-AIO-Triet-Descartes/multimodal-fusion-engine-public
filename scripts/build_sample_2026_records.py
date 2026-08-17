"""Build reproducible production-store JSONL from an AIC ``sample_2026`` tree.

The source tree is read-only. Duplicate logical keys fail closed by default; the
explicit ``quarantine-first`` policy exists only for local smoke testing and is
fully recorded in the generated manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mfe.stores.production.config import MilvusConfig
from mfe.stores.production.contracts import StorageRecord

BUILDER_VERSION = "1"
BASELINE_EMBEDDING_DIM = 512
TEXT_SOURCE = "objects"


@dataclass(frozen=True)
class RecordSpec:
    video_id: str
    frame_id: int
    sv: int
    ev: int
    embedding: np.ndarray
    object_path: Path
    n: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(root: Path, paths: set[Path]) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return {
        "algorithm": "sha256(relative_path_nul_file_sha256)",
        "sha256": digest.hexdigest(),
        "file_count": len(paths),
        "total_bytes": total_bytes,
    }


def repository_state() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {
        "commit": commit.stdout.strip() or None,
        "dirty": bool(status.stdout.strip()),
    }


def select_object_text(
    path: Path,
    *,
    minimum_score: float,
    maximum_labels: int,
    fallback: Literal["top1", "error"],
) -> tuple[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        labels = payload["detection_class_entities"]
        scores = payload["detection_scores"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise ValueError(f"invalid object artifact: {path}: {exc}") from exc
    if not isinstance(labels, list) or not isinstance(scores, list) or len(labels) != len(scores):
        raise ValueError(f"object labels/scores must be equal-length lists: {path}")

    best_by_label: dict[str, float] = {}
    for raw_label, raw_score in zip(labels, scores, strict=True):
        label = str(raw_label).strip()
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid object score in {path}: {raw_score!r}") from exc
        if not label or not math.isfinite(score):
            continue
        if not 0 <= score <= 1:
            raise ValueError(f"object score must be between 0 and 1 in {path}: {score}")
        best_by_label[label] = max(best_by_label.get(label, -math.inf), score)
    ranked = sorted(best_by_label.items(), key=lambda item: (-item[1], item[0]))
    selected = [item for item in ranked if item[1] >= minimum_score][:maximum_labels]
    used_fallback = False
    if not selected:
        if fallback == "error" or not ranked:
            raise ValueError(
                f"no object label reaches minimum score {minimum_score} in {path}"
            )
        selected = ranked[:1]
        used_fallback = True
    return " ".join(label for label, _ in selected), {
        "selected_labels": len(selected),
        "top_score": ranked[0][1],
        "used_fallback": used_fallback,
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def collect_records(
    root: Path,
    *,
    duplicate_policy: Literal["error", "quarantine-first"],
) -> tuple[list[RecordSpec], list[dict[str, Any]], set[Path], dict[str, int]]:
    mapping_root = root / "map-keyframes"
    feature_root = root / "clip-features"
    object_root = root / "objects"
    for required in (mapping_root, feature_root, object_root):
        if not required.is_dir():
            raise FileNotFoundError(f"required sample_2026 tree is missing: {required}")

    mapping_paths = sorted(mapping_root.glob("*.csv"))
    if not mapping_paths:
        raise FileNotFoundError(f"no mapping CSV files found under {mapping_root}")

    kept: dict[tuple[str, int], RecordSpec] = {}
    duplicates: list[dict[str, Any]] = []
    source_paths: set[Path] = set()
    source_rows_by_video: dict[str, int] = {}
    for csv_path in mapping_paths:
        video_id = csv_path.stem
        feature_path = feature_root / f"{video_id}.npy"
        if not feature_path.is_file():
            raise FileNotFoundError(f"missing CLIP feature file: {feature_path}")
        source_paths.update((csv_path, feature_path))
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        embeddings = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if embeddings.shape != (len(rows), BASELINE_EMBEDDING_DIM):
            raise ValueError(
                f"{video_id}: feature shape {embeddings.shape} != "
                f"({len(rows)}, {BASELINE_EMBEDDING_DIM})"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError(f"{video_id}: CLIP feature array contains NaN or infinity")
        if np.any(np.linalg.norm(np.asarray(embeddings, dtype=np.float32), axis=1) == 0):
            raise ValueError(f"{video_id}: CLIP feature array contains a zero vector")

        parsed: list[tuple[int, int]] = []
        previous_pts = -math.inf
        fps_values: set[float] = set()
        for position, row in enumerate(rows, start=1):
            try:
                n = int(row["n"])
                pts_time = float(row["pts_time"])
                fps = float(row["fps"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{csv_path}:{position + 1}: invalid n/pts_time/fps") from exc
            if (
                n != position
                or not math.isfinite(pts_time)
                or pts_time < 0
                or not math.isfinite(fps)
                or fps <= 0
            ):
                raise ValueError(f"{csv_path}:{position + 1}: invalid sequence/time/fps")
            if pts_time < previous_pts:
                raise ValueError(f"{csv_path}:{position + 1}: pts_time is not monotonic")
            previous_pts = pts_time
            fps_values.add(fps)
            frame_id = math.floor(pts_time * fps)
            provided = (row.get("frame_idx") or "").strip()
            if provided and int(provided) != frame_id:
                raise ValueError(
                    f"{csv_path}:{position + 1}: frame_idx {provided} != computed {frame_id}"
                )
            parsed.append((n, frame_id))
        if len(fps_values) != 1:
            raise ValueError(f"{video_id}: fps must be constant inside one mapping CSV")
        if not parsed:
            raise ValueError(f"{video_id}: mapping CSV is empty")

        sv = min(frame_id for _, frame_id in parsed)
        ev = max(frame_id for _, frame_id in parsed)
        source_rows_by_video[video_id] = len(parsed)
        for array_index, (n, frame_id) in enumerate(parsed):
            object_path = object_root / video_id / f"{n:03d}.json"
            if not object_path.is_file():
                raise FileNotFoundError(f"missing object artifact: {object_path}")
            source_paths.add(object_path)
            vector = np.asarray(embeddings[array_index], dtype=np.float32).copy()
            spec = RecordSpec(video_id, frame_id, sv, ev, vector, object_path, n)
            key = (video_id, frame_id)
            previous = kept.get(key)
            if previous is None:
                kept[key] = spec
                continue
            duplicates.append(
                {
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "kept_n": previous.n,
                    "quarantined_n": n,
                    "embedding_cosine": _cosine(previous.embedding, vector),
                    "reason": "duplicate_logical_key",
                }
            )

    if duplicates and duplicate_policy == "error":
        raise ValueError(
            f"found {len(duplicates)} duplicate (video_id, frame_id) rows; "
            "fix the extraction/mapping contract or explicitly use "
            "--duplicate-policy quarantine-first for a local smoke test"
        )
    return list(kept.values()), duplicates, source_paths, source_rows_by_video


def _temporary_text_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(raw_path)


def build_records(
    root: Path,
    output: Path,
    manifest_path: Path,
    *,
    duplicate_policy: Literal["error", "quarantine-first"] = "error",
    object_min_score: float = 0.2,
    object_max_labels: int = 10,
    object_fallback: Literal["top1", "error"] = "top1",
    overwrite: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    manifest_path = manifest_path.resolve()
    if output == manifest_path:
        raise ValueError("output JSONL and manifest paths must differ")
    output_inside_source = root == output or root in output.parents
    manifest_inside_source = root == manifest_path or root in manifest_path.parents
    if output_inside_source or manifest_inside_source:
        raise ValueError("generated files must be outside the read-only sample_2026 tree")
    if not 0 <= object_min_score <= 1:
        raise ValueError("object-min-score must be between 0 and 1")
    if not 1 <= object_max_labels <= 100:
        raise ValueError("object-max-labels must be between 1 and 100")
    for destination in (output, manifest_path):
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing generated file: {destination}")

    records, duplicates, source_paths, source_rows_by_video = collect_records(
        root, duplicate_policy=duplicate_policy
    )
    records.sort(key=lambda item: (item.video_id, item.frame_id, item.n))
    output_tmp = _temporary_text_path(output)
    manifest_tmp = _temporary_text_path(manifest_path)
    label_counts: list[int] = []
    fallback_count = 0
    records_by_video: dict[str, int] = {}
    try:
        with output_tmp.open("w", encoding="utf-8", newline="\n") as stream:
            for spec in records:
                text, text_audit = select_object_text(
                    spec.object_path,
                    minimum_score=object_min_score,
                    maximum_labels=object_max_labels,
                    fallback=object_fallback,
                )
                label_counts.append(text_audit["selected_labels"])
                fallback_count += int(text_audit["used_fallback"])
                record = StorageRecord.from_values(
                    video_id=spec.video_id,
                    frame_id=spec.frame_id,
                    sv=spec.sv,
                    ev=spec.ev,
                    embedding=spec.embedding,
                    text=text,
                    source=TEXT_SOURCE,
                )
                stream.write(
                    json.dumps(
                        record.model_dump(mode="json", exclude_none=True),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                records_by_video[spec.video_id] = records_by_video.get(spec.video_id, 0) + 1

        output_sha256 = sha256_file(output_tmp)
        manifest: dict[str, Any] = {
            "builder": "scripts/build_sample_2026_records.py",
            "builder_version": BUILDER_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "repository": repository_state(),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "competition_year": 2026,
            "source_root": str(root),
            "source_fingerprint": source_fingerprint(root, source_paths),
            "output_path": str(output),
            "output_bytes": output_tmp.stat().st_size,
            "output_sha256": output_sha256,
            "video_count": len(records_by_video),
            "source_rows": sum(source_rows_by_video.values()),
            "records_written": len(records),
            "source_rows_by_video": dict(sorted(source_rows_by_video.items())),
            "records_by_video": dict(sorted(records_by_video.items())),
            "embedding": {
                "dimension": BASELINE_EMBEDDING_DIM,
                "dtype_written": "float32 JSON numbers",
                "model_fingerprint": MilvusConfig().model_fingerprint,
            },
            "text": {
                "source": TEXT_SOURCE,
                "minimum_detection_score": object_min_score,
                "maximum_labels": object_max_labels,
                "fallback": object_fallback,
                "fallback_records": fallback_count,
                "selected_labels_min": min(label_counts),
                "selected_labels_median": statistics.median(label_counts),
                "selected_labels_max": max(label_counts),
                "unicode_normalization": "NFC",
            },
            "duplicate_policy": duplicate_policy,
            "quarantined_duplicate_rows": len(duplicates),
            "duplicates": duplicates,
        }
        manifest_tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        output_tmp.replace(output)
        manifest_tmp.replace(manifest_path)
        return manifest
    finally:
        output_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--duplicate-policy",
        choices=("error", "quarantine-first"),
        default="error",
    )
    parser.add_argument("--object-min-score", type=float, default=0.2)
    parser.add_argument("--object-max-labels", type=int, default=10)
    parser.add_argument("--object-fallback", choices=("top1", "error"), default="top1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build_records(
        args.root,
        args.output,
        args.manifest,
        duplicate_policy=args.duplicate_policy,
        object_min_score=args.object_min_score,
        object_max_labels=args.object_max_labels,
        object_fallback=args.object_fallback,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
