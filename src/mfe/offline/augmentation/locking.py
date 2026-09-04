"""Gate-0 dataset/query/model lock and deterministic pilot cohort selection."""

from __future__ import annotations

import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from mfe.offline.augmentation.config import AugmentationConfig
from mfe.offline.augmentation.io import read_jsonl, sha256_file
from mfe.offline.augmentation.models import VideoInventoryRow


def load_inventory(path: str | Path, *, verify_files: bool = True) -> list[VideoInventoryRow]:
    rows = [VideoInventoryRow.model_validate(item) for item in read_jsonl(path)]
    ids = [row.video_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("video inventory contains duplicate video_id")
    if verify_files:
        for row in rows:
            video = Path(row.source_path)
            mapping = Path(row.map_path)
            if not video.is_file() or not mapping.is_file():
                raise FileNotFoundError(f"{row.video_id}: source video/map is unavailable")
            if video.stat().st_size != row.byte_size or sha256_file(video) != row.sha256:
                raise ValueError(f"{row.video_id}: source video fingerprint changed")
            if sha256_file(mapping) != row.map_sha256:
                raise ValueError(f"{row.video_id}: map-keyframes fingerprint changed")
    return rows


def queryset_video_ids(path: str | Path) -> set[str]:
    return {
        str(row["gt_video_id"])
        for row in read_jsonl(path)
        if row.get("gt_video_id")
    }


def _quartile(value: float, ordered_values: list[float]) -> int:
    if len(ordered_values) <= 1:
        return 0
    rank = sum(item < value for item in ordered_values)
    return min(3, (4 * rank) // len(ordered_values))


def stratified_extra_sample(
    inventory: list[VideoInventoryRow],
    *,
    excluded: set[str],
    count: int,
    seed: int,
) -> list[str]:
    """Round-robin four-factor strata so one programme cannot dominate."""

    candidates = [row for row in inventory if row.video_id not in excluded]
    if count >= len(candidates):
        return sorted(row.video_id for row in candidates)
    durations = sorted(row.duration_seconds for row in candidates)
    densities = sorted(row.btc_density for row in candidates)
    strata: dict[tuple, list[str]] = defaultdict(list)
    for row in candidates:
        fps_bucket = round(row.fps, 2)
        key = (
            row.programme,
            fps_bucket,
            _quartile(row.duration_seconds, durations),
            _quartile(row.btc_density, densities),
        )
        strata[key].append(row.video_id)
    rng = random.Random(seed)
    for values in strata.values():
        rng.shuffle(values)
    selected: list[str] = []
    keys = sorted(strata)
    while len(selected) < count:
        progress = False
        for key in keys:
            if strata[key]:
                selected.append(strata[key].pop())
                progress = True
                if len(selected) == count:
                    break
        if not progress:
            break
    return selected


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Gate 0 requires a Git checkout with a committed HEAD") from error


def build_gate0_lock(
    config: AugmentationConfig,
    inventory: list[VideoInventoryRow],
    *,
    parent_manifest_sha256: str,
) -> dict[str, Any]:
    dev_path = config.resolve(config.dataset.dev_queryset)
    heldout_path = config.resolve(config.dataset.heldout_queryset)
    dev_videos = queryset_video_ids(dev_path)
    heldout_videos = queryset_video_ids(heldout_path)
    overlap = dev_videos & heldout_videos
    if overlap:
        raise ValueError(f"DEV and held-out share videos: {sorted(overlap)}")
    inventory_by_id = {row.video_id: row for row in inventory}
    missing = (dev_videos | heldout_videos) - set(inventory_by_id)
    if missing:
        raise ValueError(f"query videos missing from inventory: {sorted(missing)}")
    extras = stratified_extra_sample(
        inventory,
        excluded=dev_videos | heldout_videos,
        count=config.dataset.extra_pilot_videos,
        seed=config.experiment.seed,
    )
    cohort = sorted(dev_videos | set(extras))
    locked_video_ids = sorted(set(cohort) | heldout_videos)
    rows = [
        inventory_by_id[video_id].model_dump(mode="json")
        for video_id in locked_video_ids
    ]
    return {
        "schema_version": 1,
        "experiment_id": config.experiment.id,
        "parent_bundle": config.experiment.parent_bundle,
        "parent_manifest_sha256": parent_manifest_sha256,
        "git_commit": _git_commit(config.repo_root),
        "config_hash": config.canonical_hash(),
        "dev_queryset": str(dev_path),
        "dev_queryset_sha256": sha256_file(dev_path),
        "heldout_queryset": str(heldout_path),
        "heldout_queryset_sha256": sha256_file(heldout_path),
        "dev_videos": sorted(dev_videos),
        "heldout_videos_excluded": sorted(heldout_videos),
        "extra_pilot_videos": extras,
        "cohort": cohort,
        "inventory": rows,
    }


def read_lock(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value.get("config_hash"):
        raise ValueError(f"{path}: invalid Gate-0 lock")
    return value
