"""Proof and worklist builders for a frozen equal-budget B1--B4 selection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from mfe.offline.keyframe_augmentation.common import sha256_json
from mfe.offline.keyframe_augmentation.models import BtcFrame, CandidateFrame

ARMS = ("B1", "B2", "B3", "B4")
EXPECTED_SELECTORS = {
    "B1": "shot_center",
    "B2": "farthest",
    "B3": "dynamic_density",
    "B4": "uniform_random",
}


def validate_frozen_selection(
    *,
    selected: Mapping[str, Sequence[CandidateFrame]],
    mappings: Mapping[str, Sequence[BtcFrame]],
    budget_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove exact identity dedup, BTC protection and primary equal budget."""
    if set(selected) != set(ARMS):
        raise ValueError(f"selection arms must be exactly {ARMS}")
    budget = int(budget_report["locked_hard_delta_budget"])
    if budget < 1:
        raise ValueError("locked hard delta budget must be positive")
    if not budget_report.get("primary_equal_budget_pass"):
        raise ValueError(
            f"primary equal-budget selection is blocked: {budget_report.get('underfilled_arms')}"
        )

    protected = {
        (video_id, frame.frame_idx)
        for video_id, frames in mappings.items()
        for frame in frames
    }
    arm_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for arm in ARMS:
        rows = list(selected[arm])
        counts[arm] = len(rows)
        if len(rows) != budget:
            raise ValueError(f"{arm}: selected {len(rows)} != locked budget {budget}")
        identities = [(row.video_id, row.frame_idx) for row in rows]
        if len(identities) != len(set(identities)):
            raise ValueError(f"{arm}: duplicate candidate identity")
        collisions = sorted(set(identities) & protected)
        if collisions:
            raise ValueError(f"{arm}: candidate collides with protected BTC: {collisions[:3]}")
        if any(row.arm != arm for row in rows):
            raise ValueError(f"{arm}: candidate arm field mismatch")
        if any(row.selector != EXPECTED_SELECTORS[arm] for row in rows):
            raise ValueError(f"{arm}: selector contract mismatch")
        if any(row.coverage_class not in {"C0", "C1"} for row in rows):
            raise ValueError(f"{arm}: candidate outside C0/C1 eligibility")
        arm_hashes[arm] = sha256_json([row.model_dump(mode="json") for row in rows])

    return {
        "schema_version": 1,
        "status": "PASS",
        "baseline": {"arm": "B0", "delta_count": 0, "policy": "BTC_ONLY"},
        "locked_hard_delta_budget": budget,
        "selected_counts": counts,
        "all_arms_equal_budget": len(set(counts.values())) == 1,
        "exact_btc_collisions": 0,
        "within_arm_duplicate_identities": 0,
        "eligibility": "C0_C1_ONLY",
        "selector_by_arm": EXPECTED_SELECTORS,
        "candidate_payload_hashes": arm_hashes,
        "feature_parity_status": "BLOCKED_PENDING_FEATURE_EXTRACTION",
        "image_near_duplicate_status": "BLOCKED_PENDING_DECODED_IMAGES",
        "official_score_status": "UNVERIFIED",
    }


def build_feature_worklist(
    selected: Mapping[str, Sequence[CandidateFrame]],
) -> list[dict[str, Any]]:
    """Deduplicate decode/feature work across arms without changing arm membership."""
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in ARMS:
        for candidate in selected[arm]:
            key = (candidate.video_id, candidate.frame_idx)
            row = merged.setdefault(
                key,
                {
                    "keyframe_id": candidate.keyframe_id,
                    "video_id": candidate.video_id,
                    "frame_idx": candidate.frame_idx,
                    "pts_time": candidate.pts_time,
                    "fps": candidate.fps,
                    "origin": "supplemental",
                    "frame_idx_source": "decoder_counter",
                    "pts_time_source": "synthetic_literal_fps_interval",
                    "arms": [],
                    "selectors": [],
                    "shot_ids": [],
                    "coverage_classes": [],
                    "detector_fingerprint": candidate.detector_fingerprint,
                },
            )
            if row["fps"] != candidate.fps or row["pts_time"] != candidate.pts_time:
                raise ValueError(f"candidate coordinate mismatch across arms: {key}")
            row["arms"].append(arm)
            row["selectors"].append(candidate.selector)
            row["shot_ids"].append(candidate.shot_id)
            row["coverage_classes"].append(candidate.coverage_class)
    rows = []
    for row in merged.values():
        for field in ("arms", "selectors", "shot_ids", "coverage_classes"):
            row[field] = sorted(set(row[field]))
        row["work_item_hash"] = sha256_json(row)
        rows.append(row)
    return sorted(rows, key=lambda row: (row["video_id"], row["frame_idx"]))


def crowding_diagnostics(
    selected: Mapping[str, Sequence[CandidateFrame]],
) -> dict[str, Any]:
    """Report concentration only; do not silently alter canonical selectors."""
    report: dict[str, Any] = {"schema_version": 1, "decision_authority": "diagnostic_only"}
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        counts = Counter(row.video_id for row in selected[arm])
        total = sum(counts.values())
        shares = [count / total for count in counts.values()] if total else []
        by_arm[arm] = {
            "candidate_count": total,
            "videos_with_candidates": len(counts),
            "max_candidates_per_video": max(counts.values(), default=0),
            "max_video_share": max(shares, default=0.0),
            "video_hhi": sum(share * share for share in shares),
            "top_videos": [
                {"video_id": video_id, "count": count}
                for video_id, count in sorted(
                    counts.items(), key=lambda item: (-item[1], item[0])
                )[:20]
            ],
        }
    report["arms"] = by_arm
    report["next_gate"] = (
        "Measure origin exposure in DEV rank traces; do not add a quota before factorized A/B."
    )
    return report
