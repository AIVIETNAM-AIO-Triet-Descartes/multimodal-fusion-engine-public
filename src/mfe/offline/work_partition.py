"""Validated executor-level partitions inside one member's shard assignment.

The shard plan remains the ownership contract.  This module only lets one owner
run disjoint subsets on multiple compute backends; it never creates new authors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WorkerSlice:
    """One worker's ZIP/video subset, still owned by ``member``."""

    member: str
    worker: str
    zip_names: tuple[str, ...]
    videos: tuple[str, ...]


def load_worker_slice(
    path: str | Path,
    plan: Any,
    *,
    member: str,
    worker: str,
    expected_shards: str | None = None,
) -> WorkerSlice:
    """Load and validate a complete, pairwise-disjoint worker partition.

    Validation is deliberately against ``plan.zips_for(member)`` rather than a
    second copied ownership table.  A worker overlay that omits, duplicates, or
    steals one ZIP is rejected before any model or output writer is created.
    """

    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    configured_member = str(raw.get("member", ""))
    if configured_member != member:
        raise ValueError(
            f"{path}: partition danh cho {configured_member!r}, khong phai {member!r}"
        )

    configured_shards = str(raw.get("shards", ""))
    if expected_shards is not None and configured_shards != expected_shards:
        raise ValueError(
            f"{path}: shards={configured_shards!r}, cho doi {expected_shards!r}"
        )

    workers = raw.get("workers")
    if not isinstance(workers, dict) or not workers:
        raise ValueError(f"{path}: thieu workers")
    if worker not in workers:
        raise ValueError(f"{path}: worker {worker!r} khong ton tai; co {sorted(workers)}")

    owner_pairs = list(plan.zips_for(member))
    owner_names = [name for name, _url in owner_pairs]
    owner_set = set(owner_names)
    if not owner_names:
        raise ValueError(f"{member!r} khong co ZIP nao trong shard plan")

    claimed_by: dict[str, str] = {}
    groups: dict[str, tuple[str, ...]] = {}
    for worker_name, body in workers.items():
        if not isinstance(body, dict) or not isinstance(body.get("zips"), list):
            raise ValueError(f"{path}: workers.{worker_name}.zips phai la list")
        names = tuple(str(name) for name in body["zips"])
        if len(names) != len(set(names)):
            raise ValueError(f"{path}: worker {worker_name!r} lap ZIP trong chinh no")
        groups[str(worker_name)] = names
        for name in names:
            previous = claimed_by.get(name)
            if previous is not None:
                raise ValueError(
                    f"{path}: {name} bi giao cho ca {previous!r} va {worker_name!r}"
                )
            claimed_by[name] = str(worker_name)

    claimed = set(claimed_by)
    foreign = sorted(claimed - owner_set)
    missing = sorted(owner_set - claimed)
    if foreign:
        raise ValueError(
            f"{path}: ZIP khong thuoc phan {member}: {foreign}. "
            "Khong duoc cham vao phan cua Quang."
        )
    if missing:
        raise ValueError(f"{path}: chua giao het ZIP cua {member}: {missing}")

    selected = set(groups[worker])
    ordered_names = tuple(name for name in owner_names if name in selected)
    videos = tuple(
        video_id
        for name in ordered_names
        for video_id in plan.videos_in(name)
    )
    owner_videos = set(plan.videos_for(member))
    unexpected_videos = sorted(set(videos) - owner_videos)
    if unexpected_videos:
        raise ValueError(
            f"{path}: worker {worker!r} co video ngoai phan {member}: "
            f"{unexpected_videos[:6]}"
        )

    return WorkerSlice(
        member=member,
        worker=worker,
        zip_names=ordered_names,
        videos=videos,
    )
