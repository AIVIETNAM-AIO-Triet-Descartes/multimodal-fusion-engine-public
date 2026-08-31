"""Fail-closed bridge from Trọng's runtime artifacts to Quang's research contract.

The runtime package is allowed to produce detector, selector, materialization,
feature, and score artifacts.  It is not allowed to declare scientific GO on
its own.  This module checks the non-negotiable acceptance conditions without
copying either implementation's selector logic.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from mfe.offline.keyframe_augmentation.diagnostics import compare_feature_parity
from mfe.offline.keyframe_augmentation.oracle import (
    FailureStage,
    blocked_oracle_row,
    validate_oracle_rows,
)

ARMS = ("B1", "B2", "B3", "B4")
WORKER_ASSIGNMENTS = {
    "triet": ("triet1", "triet2"),
    "trong": ("trong1", "trong2"),
    "quang": ("quang1", "quang2", "quang3", "quang4"),
    "bao": ("bao1", "bao2"),
    "dat": ("dat1", "dat2"),
}


def validate_worker_assignment(member: str, worker: str) -> None:
    """Reject an owner/worker typo before an expensive Kaggle run."""

    allowed = WORKER_ASSIGNMENTS.get(member)
    if allowed is None or worker not in allowed:
        raise ValueError(f"worker {worker!r} is not assigned to member {member!r}")


def validate_video_assignment(
    assignment_config: str | Path, member: str, video_id: str
) -> list[str]:
    """Verify one smoke video belongs to the member's canonical ZIP assignment."""

    source = Path(assignment_config)
    book = yaml.safe_load(source.read_text(encoding="utf-8"))
    assigned_zips = list(book["assignment"][member])
    assigned_videos = {
        item for zip_name in assigned_zips for item in book["zips"][zip_name]["videos"]
    }
    if video_id not in assigned_videos:
        raise ValueError(f"video {video_id!r} is not assigned to member {member!r}")
    return assigned_zips


def validate_primary_equal_budget(
    *,
    base_keyframes: int,
    cap_ratio: float,
    arm_rows: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Prove the primary B1--B4 comparison uses the locked hard budget.

    The primary experiment is invalid when any arm underfills.  We deliberately
    do not shrink every arm to the smallest pool because that changes the
    pre-registered treatment and hides selector feasibility failures.
    """

    if base_keyframes < 0 or not 0 < cap_ratio <= 1:
        raise ValueError("base_keyframes must be non-negative and cap_ratio in (0, 1]")
    if set(arm_rows) != set(ARMS):
        raise ValueError(f"arm_rows must contain exactly {ARMS}")
    hard_budget = math.floor(cap_ratio * base_keyframes)
    materialized = {arm: list(arm_rows[arm]) for arm in ARMS}
    counts = {arm: len(rows) for arm, rows in materialized.items()}
    duplicate_ids = {
        arm: sorted(
            key
            for key in {str(row["keyframe_id"]) for row in rows}
            if sum(str(item["keyframe_id"]) == key for item in rows) > 1
        )
        for arm, rows in materialized.items()
    }
    underfilled = sorted(arm for arm, count in counts.items() if count < hard_budget)
    over_budget = sorted(arm for arm, count in counts.items() if count > hard_budget)
    exact = not underfilled and not over_budget and not any(duplicate_ids.values())
    report = {
        "schema_version": 1,
        "base_keyframes": base_keyframes,
        "cap_ratio": cap_ratio,
        "locked_hard_delta_budget": hard_budget,
        "B0_delta_count": 0,
        "selected_counts": counts,
        "underfilled_arms": underfilled,
        "over_budget_arms": over_budget,
        "duplicate_keyframe_ids": duplicate_ids,
        "primary_equal_budget_pass": exact,
        "matched_underfill_secondary_only": True,
        "status": "PASS" if exact else "BLOCKED_PRIMARY_EQUAL_BUDGET",
    }
    if not exact:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


def build_blocked_oracle_evidence(
    events: Iterable[Mapping[str, Any]],
    *,
    provenance_hashes: Mapping[str, str],
    failure_stage: FailureStage = "representation",
) -> list[dict[str, Any]]:
    """Emit honest O1 evidence rows when source features/ranks are unavailable."""

    rows = [
        blocked_oracle_row(
            dict(event),
            provenance_hashes=dict(provenance_hashes),
            failure_stage=failure_stage,
        ).model_dump(mode="json")
        for event in events
    ]
    validate_oracle_rows(rows)
    return rows


def validate_feature_parity_acceptance(
    btc_fingerprint: Mapping[str, Any],
    supplemental_fingerprint: Mapping[str, Any],
    *,
    canary_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require both exact metadata parity and a frozen numeric canary."""

    report = compare_feature_parity(
        dict(btc_fingerprint),
        dict(supplemental_fingerprint),
        canary_evidence=None if canary_evidence is None else dict(canary_evidence),
    )
    if report["status"] != "MEASURED":
        raise RuntimeError(
            "feature parity is not MEASURED: "
            + json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
    return report


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    """Write a small integration artifact without exposing a partial file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(target)
    return target


__all__ = [
    "ARMS",
    "WORKER_ASSIGNMENTS",
    "build_blocked_oracle_evidence",
    "validate_feature_parity_acceptance",
    "validate_primary_equal_budget",
    "validate_video_assignment",
    "validate_worker_assignment",
    "write_json_atomic",
]
