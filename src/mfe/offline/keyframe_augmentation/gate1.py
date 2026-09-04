"""Gate 1: TransNetV2 coverage audit, DEV tuning, selectors and oracles."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mfe.offline.keyframe_augmentation.common import (
    pts_time_for_frame_idx,
    read_jsonl,
    sha256_json,
)
from mfe.offline.keyframe_augmentation.models import (
    AugmentationConfig,
    BtcFrame,
    CandidateFrame,
    CoverageClass,
    ShotAudit,
    ShotProposal,
)


def load_shot_proposals(path: str | Path) -> list[ShotProposal]:
    proposals = [ShotProposal.model_validate(row) for row in read_jsonl(path)]
    seen: set[tuple[str, str]] = set()
    for proposal in proposals:
        key = (proposal.video_id, proposal.shot_id)
        if key in seen:
            raise ValueError(f"duplicate shot proposal: {key}")
        seen.add(key)
    return sorted(proposals, key=lambda shot: (shot.video_id, shot.start_frame, shot.shot_id))


def load_query_events(path: str | Path) -> list[dict[str, Any]]:
    """Flatten DEV queries into strict event intervals without touching held-out."""
    events: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("split") != "dev":
                raise ValueError(f"{path}:{line_number}: Gate 1 may only read split=dev")
            query_id = str(row["query_id"])
            video_id = str(row["gt_video_id"])
            query_type = str(row.get("type", "UNKNOWN"))
            if query_type == "TRAKE":
                intervals = row.get("gt_events_strict") or row.get("gt_events")
            else:
                interval = row.get("gt_frame_range_strict") or row.get("gt_frame_range")
                intervals = [interval]
            if not isinstance(intervals, list):
                raise ValueError(f"{path}:{line_number}: ground-truth intervals missing")
            for event_index, interval in enumerate(intervals):
                if not isinstance(interval, list) or len(interval) != 2:
                    raise ValueError(f"{path}:{line_number}: invalid interval {interval!r}")
                start, end = int(interval[0]), int(interval[1])
                if start < 0 or end < start:
                    raise ValueError(f"{path}:{line_number}: invalid interval [{start},{end}]")
                events.append(
                    {
                        "event_id": f"{query_id}:{event_index}",
                        "query_id": query_id,
                        "query_type": query_type,
                        "video_id": video_id,
                        "start_frame": start,
                        "end_frame": end,
                    }
                )
    return events


def _max_blind_gap_frames(shot: ShotProposal, btc_frames: list[int]) -> int:
    points = [shot.start_frame, *sorted(btc_frames), shot.end_frame]
    return max(
        (right - left for left, right in zip(points, points[1:], strict=False)),
        default=0,
    )


def classify_shot(
    shot: ShotProposal,
    btc_frames: list[BtcFrame],
    *,
    target_gap_seconds: float,
    k_max: int,
    config: AugmentationConfig,
) -> ShotAudit:
    in_shot = [
        frame.frame_idx
        for frame in btc_frames
        if shot.start_frame <= frame.frame_idx <= shot.end_frame
    ]
    duration = shot.duration_seconds
    k_target = min(k_max, max(1, math.ceil(duration / target_gap_seconds)))
    deficit = max(0, k_target - len(in_shot))
    max_gap = _max_blind_gap_frames(shot, in_shot) / shot.fps

    if shot.audit_status == "error":
        coverage_class = CoverageClass.CE
        reason = "human/detector audit marked error"
    elif shot.audit_status == "transition" or duration < config.micro_shot_seconds:
        coverage_class = CoverageClass.CT
        reason = "micro-shot or audited transition"
    elif shot.audit_status != "confirmed" and (
        shot.confidence is None or shot.confidence < config.detector_confidence_min
    ):
        coverage_class = CoverageClass.CE
        reason = "detector confidence missing/below locked threshold"
    elif not in_shot:
        coverage_class = CoverageClass.C0
        reason = "no protected BTC frame in inclusive shot"
    elif deficit > 0 or max_gap > target_gap_seconds:
        coverage_class = CoverageClass.C1
        reason = "BTC exists but count or measured temporal gap is under target"
    else:
        coverage_class = CoverageClass.COVERED
        reason = "BTC count and measured temporal gaps meet target"

    return ShotAudit(
        video_id=shot.video_id,
        shot_id=shot.shot_id,
        start_frame=shot.start_frame,
        end_frame=shot.end_frame,
        duration_seconds=duration,
        coverage_class=coverage_class,
        btc_count=len(in_shot),
        k_target=k_target,
        deficit=deficit,
        max_blind_gap_seconds=max_gap,
        eligible=coverage_class in {CoverageClass.C0, CoverageClass.C1},
        reason=reason,
        detector_confidence=shot.confidence,
        audit_status=shot.audit_status,
    )


def _interior(shot: ShotProposal, config: AugmentationConfig) -> tuple[int, int] | None:
    # The seconds margin is a minimum exclusion, hence ceil.  The ratio is a
    # hard cap that must not exclude more than the configured share, hence
    # floor.  Their minimum preserves a usable interior for short shots.
    seconds_margin = math.ceil(config.interior_margin_seconds * shot.fps)
    ratio_cap = math.floor(
        (shot.end_frame - shot.start_frame + 1) * config.interior_margin_ratio
    )
    margin = min(
        seconds_margin,
        ratio_cap,
    )
    start = shot.start_frame + margin
    end = shot.end_frame - margin
    return (start, end) if start <= end else None


def _candidate_priority(audit: ShotAudit, shot: ShotProposal, frame_idx: int) -> tuple:
    confirmed = 1.0 if shot.audit_status == "confirmed" else float(shot.confidence or 0.0)
    return (
        confirmed,
        audit.duration_seconds,
        audit.max_blind_gap_seconds,
        shot.video_id,
        frame_idx,
    )


def _candidate(
    arm: str,
    selector: str,
    shot: ShotProposal,
    audit: ShotAudit,
    frame_idx: int,
) -> CandidateFrame:
    return CandidateFrame(
        arm=arm,
        video_id=shot.video_id,
        shot_id=shot.shot_id,
        frame_idx=frame_idx,
        pts_time=pts_time_for_frame_idx(frame_idx, shot.fps),
        fps=shot.fps,
        selector=selector,
        coverage_class=audit.coverage_class.value,
        priority=_candidate_priority(audit, shot, frame_idx),
        detector_fingerprint=shot.detector_fingerprint,
        audit_status=shot.audit_status,
    )


def _farthest_frame(
    start: int,
    end: int,
    anchors: list[int],
    *,
    prefer_center_when_empty: bool,
) -> int | None:
    if start > end:
        return None
    if not anchors:
        return (start + end) // 2 if prefer_center_when_empty else start
    relevant = sorted({value for value in anchors if start <= value <= end})
    points = {start, end}
    for left, right in zip(relevant, relevant[1:], strict=False):
        points.add((left + right) // 2)
        points.add((left + right + 1) // 2)
    if relevant:
        points.add(start)
        points.add(end)
    choices = [point for point in points if point not in anchors]
    if not choices:
        return None
    return min(choices, key=lambda point: (-min(abs(point - anchor) for anchor in anchors), point))


def generate_candidate_pools(
    proposals: list[ShotProposal],
    mappings: dict[str, list[BtcFrame]],
    *,
    config: AugmentationConfig,
    target_gap_seconds: float | None = None,
    k_max: int | None = None,
) -> tuple[list[ShotAudit], dict[str, list[CandidateFrame]]]:
    gap = target_gap_seconds or config.target_gap_seconds
    maximum = k_max or config.k_max
    pools: dict[str, list[CandidateFrame]] = {arm: [] for arm in ("B1", "B2", "B3", "B4")}
    audits: list[ShotAudit] = []
    rng = random.Random(config.seed)

    for shot in proposals:
        btc = mappings.get(shot.video_id)
        if btc is None:
            raise ValueError(f"shot {shot.shot_id} has no Gate 0 mapping")
        audit = classify_shot(
            shot,
            btc,
            target_gap_seconds=gap,
            k_max=maximum,
            config=config,
        )
        audits.append(audit)
        if not audit.eligible:
            continue
        bounds = _interior(shot, config)
        if bounds is None:
            audits[-1] = audit.model_copy(
                update={
                    "coverage_class": CoverageClass.CT,
                    "eligible": False,
                    "reason": "interior is empty after transition margin",
                }
            )
            continue
        start, end = bounds
        btc_indices = [
            frame.frame_idx
            for frame in btc
            if shot.start_frame <= frame.frame_idx <= shot.end_frame
        ]
        protected = {frame.frame_idx for frame in btc}

        if audit.coverage_class == CoverageClass.C0:
            center = (start + end) // 2
            if center not in protected:
                pools["B1"].append(_candidate("B1", "shot_center", shot, audit, center))

        farthest = _farthest_frame(
            start,
            end,
            btc_indices,
            prefer_center_when_empty=True,
        )
        if farthest is not None and farthest not in protected:
            pools["B2"].append(_candidate("B2", "farthest", shot, audit, farthest))

        selected: list[int] = []
        anchors = list(btc_indices)
        while len(anchors) + len(selected) < maximum:
            current = [*anchors, *selected]
            current_gap = _max_blind_gap_frames(shot, current) / shot.fps
            count_short = len(current) < audit.k_target
            if selected and not count_short and current_gap <= gap:
                break
            frame_idx = _farthest_frame(
                start,
                end,
                current,
                prefer_center_when_empty=True,
            )
            if frame_idx is None or frame_idx in protected or frame_idx in selected:
                break
            selected.append(frame_idx)
        for frame_idx in selected:
            pools["B3"].append(_candidate("B3", "dynamic_density", shot, audit, frame_idx))

        available = [frame for frame in range(start, end + 1) if frame not in protected]
        random_count = min(len(selected), len(available))
        for frame_idx in sorted(rng.sample(available, random_count)):
            pools["B4"].append(_candidate("B4", "uniform_random", shot, audit, frame_idx))

    for arm in pools:
        unique: dict[tuple[str, int], CandidateFrame] = {}
        for candidate in pools[arm]:
            unique.setdefault((candidate.video_id, candidate.frame_idx), candidate)
        pools[arm] = sorted(unique.values(), key=_sort_key)
    return audits, pools


def _sort_key(candidate: CandidateFrame) -> tuple:
    confirmed, duration, blind_gap, video_id, frame_idx = candidate.priority
    return (-confirmed, -duration, -blind_gap, video_id, frame_idx)


def equalize_budget(
    pools: dict[str, list[CandidateFrame]],
    *,
    btc_count: int,
    guardrail_ratio: float,
) -> tuple[dict[str, list[CandidateFrame]], dict[str, Any]]:
    requested = math.floor(btc_count * guardrail_ratio)
    arms = ("B1", "B2", "B3", "B4")
    selected = {arm: list(pools[arm][:requested]) for arm in arms}
    selected_counts = {arm: len(selected[arm]) for arm in arms}
    deficits = {arm: requested - count for arm, count in selected_counts.items()}
    underfilled_arms = sorted(arm for arm, deficit in deficits.items() if deficit > 0)
    matched_underfill_budget = min(selected_counts.values(), default=0)
    return selected, {
        "btc_count": btc_count,
        "guardrail_ratio": guardrail_ratio,
        "requested_delta_budget": requested,
        "locked_hard_delta_budget": requested,
        "common_delta_budget": requested if not underfilled_arms else None,
        "selected_counts": selected_counts,
        "deficits": deficits,
        "underfilled_arms": underfilled_arms,
        "primary_equal_budget_pass": not underfilled_arms,
        "matched_underfill_budget": matched_underfill_budget,
        "matched_underfill_authority": "secondary_diagnostic_only",
        "pool_sizes": {arm: len(pools[arm]) for arm in sorted(pools)},
        "underfilled": bool(underfilled_arms),
        "guardrail_status": "PASS",
        "interpretation": (
            "primary equal-budget A/B is valid"
            if not underfilled_arms
            else "primary A/B is blocked; do not hide underfill by shrinking all arms"
        ),
    }


def covered_events(
    events: list[dict[str, Any]],
    mappings: dict[str, list[BtcFrame]],
    candidates: list[CandidateFrame] | None = None,
) -> set[str]:
    by_video: dict[str, set[int]] = {
        video_id: {frame.frame_idx for frame in frames} for video_id, frames in mappings.items()
    }
    for candidate in candidates or []:
        by_video.setdefault(candidate.video_id, set()).add(candidate.frame_idx)
    return {
        event["event_id"]
        for event in events
        if any(
            event["start_frame"] <= frame_idx <= event["end_frame"]
            for frame_idx in by_video.get(event["video_id"], set())
        )
    }


def tune_dynamic_density(
    *,
    proposals: list[ShotProposal],
    mappings: dict[str, list[BtcFrame]],
    events: list[dict[str, Any]],
    config: AugmentationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    btc_count = sum(len(frames) for frames in mappings.values())
    baseline = covered_events(events, mappings)
    rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for gap in sorted(set(config.target_gap_grid)):
        for k_max in sorted(set(config.k_max_grid)):
            _, pools = generate_candidate_pools(
                proposals,
                mappings,
                config=config,
                target_gap_seconds=gap,
                k_max=k_max,
            )
            budget = math.floor(btc_count * config.guardrail_ratio)
            chosen = pools["B3"][:budget]
            coverage = covered_events(events, mappings, chosen)
            row = {
                "target_gap_seconds": gap,
                "k_max": k_max,
                "pool_count": len(pools["B3"]),
                "selected_count": len(chosen),
                "covered_events": len(coverage),
                "rescued_events": len(coverage - baseline),
                "coverage_rate": len(coverage) / len(events) if events else None,
                "official_final_score": "UNVERIFIED",
                "decision_authority": "diagnostic_only",
                "is_locked_starting_point": (
                    gap == config.target_gap_seconds and k_max == config.k_max
                ),
            }
            rows.append(row)
            if row["is_locked_starting_point"]:
                selected = row
    if selected is None:
        raise ValueError("target_gap_seconds/k_max must be present in their DEV grids")
    # Coverage is diagnostic only.  Parameter selection remains at the locked
    # starting point until equal-budget official DEV runs choose a winner.
    return selected, rows


def build_oracles(
    events: list[dict[str, Any]],
    mappings: dict[str, list[BtcFrame]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build coordinate oracles without pretending they measured retrieval.

    O1-T is only a *temporal-coverage* diagnostic and therefore contains events
    with no BTC frame in the strict interval.  O1-S covers every event and asks
    a later feature-parity run to find the best decodable source frame inside
    the complete inclusive GT interval.  O1-R is a later force-rank ceiling.
    Zero O1-T rows never implies zero semantic or ranking headroom.
    """
    baseline = covered_events(events, mappings)
    o1_t: list[dict[str, Any]] = []
    o1_s: list[dict[str, Any]] = []
    for event in events:
        # Integer midpoint deliberately chooses the lower of the two central
        # frames.  It is a discrete selector, not a seconds->frame conversion.
        midpoint = (event["start_frame"] + event["end_frame"]) // 2
        o1_s.append(
            {
                **event,
                "candidate_start_frame": event["start_frame"],
                "candidate_end_frame": event["end_frame"],
                "reference_midpoint_frame": midpoint,
                "oracle": "O1-S",
                "evidence": "DERIVED",
                "note": (
                    "decode every source frame in the inclusive GT interval, apply delta "
                    "feature parity, and retain the best real retrieval result"
                ),
            }
        )
        if event["event_id"] not in baseline:
            o1_t.append(
                {
                    **event,
                    "frame_idx": midpoint,
                    "oracle": "O1-T",
                    "evidence": "DERIVED",
                    "note": "temporal-coverage candidate; retrieval result remains UNVERIFIED",
                }
            )
    query_events: dict[str, list[str]] = defaultdict(list)
    for event in events:
        query_events[event["query_id"]].append(event["event_id"])
    baseline_queries = sum(
        1 for ids in query_events.values() if all(event_id in baseline for event_id in ids)
    )
    temporal_headroom = len(events) - len(baseline)
    return {"O1-T": o1_t, "O1-S": o1_s}, {
        "events": len(events),
        "baseline_covered_events": len(baseline),
        "baseline_fully_covered_queries": baseline_queries,
        "o1_t_candidates": len(o1_t),
        "o1_t_scope": "temporal_missing_slice_only",
        "o1_t_no_candidate_events": len(events) - len(o1_t),
        "o1_t_no_candidate_reason": "btc_touches_interval_or_no_temporal_candidate",
        "o1_t_end_to_end": "UNVERIFIED",
        "o1_s_event_ranges": len(o1_s),
        "o1_s_scope": "every_DEV_event_inclusive_gt_interval",
        "o1_s_best_source_frame_end_to_end": "UNVERIFIED",
        "o1_r_force_rank_ceiling_events": len(events),
        "o1_r_scope": "force_ground_truth_at_retrieval_fusion_final_reranker",
        "o1_r_force_rank_end_to_end": "UNVERIFIED",
        "temporal_coverage_headroom_events": temporal_headroom,
        "semantic_representative_headroom": "UNVERIFIED",
        "ranking_headroom": "UNVERIFIED",
        "oracle_evidence_schema": (
            "mfe.offline.keyframe_augmentation.oracle.OracleEvidenceRow"
        ),
        "oracle_evidence_status": "BLOCKED_MISSING_FEATURE_AND_RANK_ARTIFACTS",
        "research_stop_requires_official_ab": True,
        "decision_authority": "diagnostic_only",
    }


def run_gate1(
    *,
    config: AugmentationConfig,
    proposals: list[ShotProposal],
    mappings: dict[str, list[BtcFrame]],
    dev_events: list[dict[str, Any]],
    dataset_hash: str,
) -> tuple[
    dict[str, Any],
    dict[str, list[CandidateFrame]],
    dict[str, list[dict[str, Any]]],
]:
    selected, tuning_rows = tune_dynamic_density(
        proposals=proposals,
        mappings=mappings,
        events=dev_events,
        config=config,
    )
    audits, pools = generate_candidate_pools(
        proposals,
        mappings,
        config=config,
        target_gap_seconds=selected["target_gap_seconds"],
        k_max=selected["k_max"],
    )
    equalized, budget = equalize_budget(
        pools,
        btc_count=sum(len(frames) for frames in mappings.values()),
        guardrail_ratio=config.guardrail_ratio,
    )
    oracle_manifests, oracle = build_oracles(dev_events, mappings)
    class_counts = Counter(audit.coverage_class.value for audit in audits)
    report = {
        "schema_version": 1,
        "dataset_hash": dataset_hash,
        "config_hash": sha256_json(config.model_dump(mode="json")),
        "tuned_on_split": "dev",
        "heldout_accessed": False,
        "selected_parameters": selected,
        "parameter_selection_status": "UNVERIFIED_PENDING_OFFICIAL_DEV_AB",
        "tuning_grid": tuning_rows,
        "shot_classes": dict(sorted(class_counts.items())),
        "shots": [audit.model_dump(mode="json") for audit in audits],
        "budget": budget,
        "oracle": oracle,
        "transnet_audit": {
            "proposals": len(proposals),
            "fingerprints": sorted({shot.detector_fingerprint for shot in proposals}),
            "ct_ce_require_human_review": sum(
                audit.coverage_class in {CoverageClass.CT, CoverageClass.CE} for audit in audits
            ),
            "evidence": "MEASURED",
        },
        "decision": (
            "PROCEED_TO_REPRESENTATIVE_ORACLE"
            if budget["primary_equal_budget_pass"]
            else "BLOCKED_PRIMARY_AB_UNDERFILL"
        ),
        "evidence": {
            "coverage_and_budget": "DERIVED",
            "official_score_uplift": "UNVERIFIED",
            "transnet_quality_on_aic": "UNVERIFIED until CT/CE human audit",
        },
    }
    report["run_hash"] = sha256_json(report)
    return report, equalized, oracle_manifests
