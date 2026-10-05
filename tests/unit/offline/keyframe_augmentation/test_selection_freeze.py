from __future__ import annotations

import pytest

from mfe.offline.keyframe_augmentation.gate1 import equalize_budget
from mfe.offline.keyframe_augmentation.models import BtcFrame, CandidateFrame
from mfe.offline.keyframe_augmentation.selection_freeze import (
    build_feature_worklist,
    crowding_diagnostics,
    validate_frozen_selection,
)


def _btc(frame_idx: int) -> BtcFrame:
    return BtcFrame.build(
        video_id="V", ordinal=frame_idx, frame_idx=frame_idx,
        pts_time=(frame_idx + 0.5) / 10.0, fps=10.0,
    )


def _candidate(arm: str, frame_idx: int) -> CandidateFrame:
    selectors = {
        "B1": "shot_center", "B2": "farthest",
        "B3": "dynamic_density", "B4": "uniform_random",
    }
    return CandidateFrame(
        arm=arm,
        video_id="V",
        shot_id=f"V#s{frame_idx}",
        frame_idx=frame_idx,
        pts_time=(frame_idx + 0.5) / 10.0,
        fps=10.0,
        selector=selectors[arm],
        coverage_class="C0",
        priority=(0.9, 2.0, 2.0, "V", frame_idx),
        detector_fingerprint="transnetv2:test",
        audit_status="auto",
    )


def test_freeze_proves_equal_budget_and_builds_deduplicated_worklist() -> None:
    pools = {
        arm: [_candidate(arm, frame_idx) for frame_idx in (10, 20, 30)]
        for arm in ("B1", "B2", "B3", "B4")
    }
    selected, budget = equalize_budget(pools, btc_count=100, guardrail_ratio=0.02)

    proof = validate_frozen_selection(
        selected=selected, mappings={"V": [_btc(1)]}, budget_report=budget
    )
    worklist = build_feature_worklist(selected)

    assert proof["locked_hard_delta_budget"] == 2
    assert proof["selected_counts"] == {"B1": 2, "B2": 2, "B3": 2, "B4": 2}
    assert len(worklist) == 2
    assert worklist[0]["arms"] == ["B1", "B2", "B3", "B4"]
    assert crowding_diagnostics(selected)["arms"]["B1"]["candidate_count"] == 2


def test_freeze_rejects_btc_collision() -> None:
    pools = {arm: [_candidate(arm, 10)] for arm in ("B1", "B2", "B3", "B4")}
    selected, budget = equalize_budget(pools, btc_count=100, guardrail_ratio=0.01)

    with pytest.raises(ValueError, match="collides with protected BTC"):
        validate_frozen_selection(
            selected=selected, mappings={"V": [_btc(10)]}, budget_report=budget
        )


def test_freeze_rejects_underfilled_primary_ab() -> None:
    pools = {arm: [_candidate(arm, 10)] for arm in ("B1", "B2", "B3", "B4")}
    _selected, budget = equalize_budget(pools, btc_count=100, guardrail_ratio=0.02)

    with pytest.raises(ValueError, match="equal-budget selection is blocked"):
        validate_frozen_selection(
            selected=pools, mappings={"V": []}, budget_report=budget
        )
