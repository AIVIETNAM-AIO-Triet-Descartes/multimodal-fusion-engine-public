"""Build an immutable SQLite base-plus-delta snapshot for A/B evaluation."""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from mfe.contracts.ids import pts_time_to_timestamp_ms
from mfe.mapping.relational import RelationalMap
from mfe.offline.augmentation.io import canonical_hash, sha256_file
from mfe.offline.augmentation.models import CandidateRow


def build_relational_snapshot(
    base_db: str | Path,
    output_db: str | Path,
    candidates: list[CandidateRow],
    *,
    overwrite: bool = False,
    allow_preexisting_mapping_problems: bool = False,
) -> dict[str, Any]:
    """Clone BTC SQLite, insert delta, and rebuild per-video temporal ordinals."""

    source = Path(base_db)
    target = Path(output_db)
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite snapshot: {target}")

    with RelationalMap(str(source)) as relational:
        baseline_problems = relational.verify_mapping()
    if baseline_problems and not allow_preexisting_mapping_problems:
        raise RuntimeError(f"base mapping audit failed: {baseline_problems[:5]}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)

    source_connection = sqlite3.connect(str(source))
    target_connection = sqlite3.connect(str(target))
    try:
        source_connection.backup(target_connection)
    finally:
        source_connection.close()

    affected = sorted({item.video_id for item in candidates})
    candidate_ids = [item.keyframe_id for item in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        target_connection.close()
        target.unlink(missing_ok=True)
        raise ValueError("candidate list contains duplicate keyframe ids")
    try:
        with target_connection:
            for video_id in affected:
                video = target_connection.execute(
                    "SELECT total_frames, duration_ms FROM videos WHERE video_id = ?",
                    (video_id,),
                ).fetchone()
                if video is None:
                    raise ValueError(f"candidate video is absent from base snapshot: {video_id}")
                selected = [item for item in candidates if item.video_id == video_id]
                for item in selected:
                    if item.frame_idx >= int(video[0]):
                        raise ValueError(f"{item.keyframe_id}: frame is outside source video")
                    if pts_time_to_timestamp_ms(item.pts_time) > int(video[1]):
                        raise ValueError(f"{item.keyframe_id}: PTS is outside source video")

                base_rows = target_connection.execute(
                    """
                    SELECT keyframe_id, frame_idx, pts_time
                    FROM keyframes WHERE video_id = ?
                    """,
                    (video_id,),
                ).fetchall()
                base_ids = {str(row[0]) for row in base_rows}
                duplicates = base_ids & {item.keyframe_id for item in selected}
                if duplicates:
                    raise ValueError(
                        f"delta duplicates protected BTC ids: {sorted(duplicates)[:3]}"
                    )
                combined = [
                    (float(row[2]), int(row[1]), 0, str(row[0]), None)
                    for row in base_rows
                ]
                combined.extend(
                    (item.pts_time, item.frame_idx, 1, item.keyframe_id, item)
                    for item in selected
                )
                combined.sort()
                # Move existing ordinals out of the way before assigning their
                # final positions; the schema has UNIQUE(video_id, keyframe_index).
                offset = len(combined) + len(base_rows) + 1
                target_connection.execute(
                    "UPDATE keyframes SET keyframe_index = keyframe_index + ? WHERE video_id = ?",
                    (offset, video_id),
                )
                for ordinal, (_, _, _, keyframe_id, candidate) in enumerate(combined):
                    if candidate is None:
                        target_connection.execute(
                            "UPDATE keyframes SET keyframe_index = ? WHERE keyframe_id = ?",
                            (ordinal, keyframe_id),
                        )
                    else:
                        target_connection.execute(
                            """
                            INSERT INTO keyframes
                                (keyframe_id, video_id, keyframe_index, frame_idx,
                                 pts_time, timestamp_ms, shot_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                candidate.keyframe_id,
                                candidate.video_id,
                                ordinal,
                                candidate.frame_idx,
                                candidate.pts_time,
                                pts_time_to_timestamp_ms(candidate.pts_time),
                                candidate.shot_id,
                            ),
                        )
        integrity = target_connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = target_connection.execute("PRAGMA foreign_key_check").fetchall()
        row_count = int(target_connection.execute("SELECT COUNT(*) FROM keyframes").fetchone()[0])
    except Exception:
        target_connection.close()
        target.unlink(missing_ok=True)
        raise
    target_connection.close()
    if integrity != "ok" or foreign_keys:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"snapshot SQLite audit failed: integrity={integrity}, foreign_keys={foreign_keys[:3]}"
        )
    with RelationalMap(str(target)) as relational:
        problems = relational.verify_mapping()
    baseline_counter = Counter(baseline_problems)
    snapshot_counter = Counter(problems)
    introduced_problems = list((snapshot_counter - baseline_counter).elements())
    removed_problems = list((baseline_counter - snapshot_counter).elements())
    if introduced_problems or removed_problems:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            "snapshot mapping audit changed the frozen baseline: "
            f"introduced={introduced_problems[:5]}, removed={removed_problems[:5]}"
        )
    return {
        "base_sha256": sha256_file(source),
        "snapshot_sha256": sha256_file(target),
        "delta_rows": len(candidates),
        "total_rows": row_count,
        "affected_videos": affected,
        "baseline_mapping_problem_count": len(baseline_problems),
        "baseline_mapping_problems_sha256": canonical_hash(baseline_problems),
        "snapshot_mapping_problem_count": len(problems),
        "introduced_mapping_problem_count": len(introduced_problems),
        "removed_mapping_problem_count": len(removed_problems),
    }
