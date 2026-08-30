from __future__ import annotations

import pytest

from mfe.offline.keyframe_augmentation.gate1 import (
    _interior,
    build_oracles,
    equalize_budget,
    generate_candidate_pools,
    run_gate1,
)
from mfe.offline.keyframe_augmentation.models import (
    AugmentationConfig,
    BtcFrame,
    CandidateFrame,
    CoverageClass,
    ShotProposal,
)


def _config() -> AugmentationConfig:
    return AugmentationConfig(
        dataset_version="test",
        split_version="test-split",
        seed=7,
        guardrail_ratio=1.0,
        target_gap_seconds=2.0,
        k_max=4,
        target_gap_grid=[2.0, 4.0],
        k_max_grid=[2, 4],
        bootstrap_samples=100,
    )


def _frame(frame_idx: int) -> BtcFrame:
    return BtcFrame.build(
        video_id="V",
        ordinal=frame_idx + 1,
        frame_idx=frame_idx,
        pts_time=frame_idx / 10.0,
        fps=10.0,
    )


def _shot(number: int, start: int, end: int, **updates) -> ShotProposal:
    values = {
        "video_id": "V",
        "shot_id": f"V#s{number:02d}",
        "start_frame": start,
        "end_frame": end,
        "fps": 10.0,
        "detector_fingerprint": "transnetv2:test",
        "confidence": 0.9,
    }
    values.update(updates)
    return ShotProposal(**values)


def test_transnet_audit_classifies_c0_c1_ct_ce_and_covered() -> None:
    mappings = {"V": [_frame(150), _frame(210), _frame(270), _frame(420), _frame(440)]}
    proposals = [
        _shot(0, 0, 99),
        _shot(1, 100, 299),
        _shot(2, 300, 304),
        _shot(3, 305, 399, confidence=0.2),
        _shot(4, 410, 449),
    ]

    audits, _ = generate_candidate_pools(proposals, mappings, config=_config())

    assert [audit.coverage_class for audit in audits] == [
        CoverageClass.C0,
        CoverageClass.C1,
        CoverageClass.CT,
        CoverageClass.CE,
        CoverageClass.COVERED,
    ]
    assert audits[1].max_blind_gap_seconds == 6.0


def test_b0_b4_selectors_are_deterministic_equal_budget_and_btc_protected() -> None:
    config = _config()
    mappings = {"V": [_frame(150), _frame(210), _frame(270)]}
    proposals = [_shot(0, 0, 99), _shot(1, 100, 299)]

    _, first = generate_candidate_pools(proposals, mappings, config=config)
    _, second = generate_candidate_pools(proposals, mappings, config=config)
    equalized, budget = equalize_budget(first, btc_count=3, guardrail_ratio=1.0)

    assert first == second
    assert budget["locked_hard_delta_budget"] == 3
    assert budget["matched_underfill_budget"] == 1
    assert budget["primary_equal_budget_pass"] is False
    assert budget["common_delta_budget"] is None
    assert {len(rows) for rows in equalized.values()} != {1}
    protected = {150, 210, 270}
    assert all(row.frame_idx not in protected for rows in equalized.values() for row in rows)
    assert set(equalized) == {"B1", "B2", "B3", "B4"}


def test_oracles_expose_temporal_and_representative_work_without_claiming_uplift() -> None:
    events = [
        {
            "event_id": "q1:0",
            "query_id": "q1",
            "query_type": "KIS",
            "video_id": "V",
            "start_frame": 40,
            "end_frame": 60,
        }
    ]

    manifests, report = build_oracles(events, {"V": [_frame(0)]})

    assert manifests["O1-T"][0]["frame_idx"] == 50
    assert manifests["O1-S"][0]["candidate_start_frame"] == 40
    assert manifests["O1-S"][0]["candidate_end_frame"] == 60
    assert report["temporal_coverage_headroom_events"] == 1
    assert report["o1_t_end_to_end"] == "UNVERIFIED"
    assert report["o1_s_best_source_frame_end_to_end"] == "UNVERIFIED"
    assert report["o1_r_force_rank_end_to_end"] == "UNVERIFIED"


def test_temporal_coverage_does_not_suppress_representative_oracle() -> None:
    events = [
        {
            "event_id": "q1:0",
            "query_id": "q1",
            "query_type": "KIS",
            "video_id": "V",
            "start_frame": 40,
            "end_frame": 60,
        }
    ]

    manifests, report = build_oracles(events, {"V": [_frame(50)]})

    assert manifests["O1-T"] == []
    assert len(manifests["O1-S"]) == 1
    assert report["temporal_coverage_headroom_events"] == 0
    assert report["semantic_representative_headroom"] == "UNVERIFIED"
    assert report["ranking_headroom"] == "UNVERIFIED"


@pytest.mark.parametrize(
    ("fps", "expected_margin"),
    [(25.0, 7), (30.0, 8), (29.97, 8), (26.44, 7)],
)
def test_interior_margin_ceil_seconds_floor_ratio_cap(
    fps: float, expected_margin: int
) -> None:
    config = _config().model_copy(
        update={"interior_margin_seconds": 0.25, "interior_margin_ratio": 0.10}
    )
    shot = _shot(0, 0, 99, fps=fps)

    assert _interior(shot, config) == (expected_margin, 99 - expected_margin)


def test_hard_budget_floors_ratio_and_never_exceeds_cap() -> None:
    pools = {arm: [] for arm in ("B1", "B2", "B3", "B4")}
    _, report = equalize_budget(pools, btc_count=101, guardrail_ratio=0.03)

    assert report["requested_delta_budget"] == 3
    assert report["locked_hard_delta_budget"] == 3
    assert report["underfilled_arms"] == ["B1", "B2", "B3", "B4"]
    assert report["matched_underfill_authority"] == "secondary_diagnostic_only"


def test_primary_equal_budget_passes_only_when_every_arm_fills_locked_budget() -> None:
    pools = {arm: list(range(5)) for arm in ("B1", "B2", "B3", "B4")}

    selected, report = equalize_budget(pools, btc_count=101, guardrail_ratio=0.03)

    assert report["locked_hard_delta_budget"] == 3
    assert report["primary_equal_budget_pass"] is True
    assert report["underfilled_arms"] == []
    assert report["common_delta_budget"] == 3
    assert {len(rows) for rows in selected.values()} == {3}


def test_candidate_rejects_pts_that_does_not_floor_to_decoder_frame() -> None:
    with pytest.raises(ValueError, match="floor-roundtrip"):
        CandidateFrame(
            arm="B3",
            video_id="V",
            shot_id="V#s00",
            frame_idx=1,
            pts_time=0.019,
            fps=30.0,
            selector="dynamic_density",
            coverage_class="C0",
            priority=(1.0, 1.0, 1.0, "V", 1),
            detector_fingerprint="transnetv2:test",
            audit_status="confirmed",
        )


def test_gate1_tunes_only_from_supplied_dev_events_and_is_deterministic() -> None:
    config = _config()
    mappings = {"V": [_frame(150), _frame(210), _frame(270)]}
    proposals = [_shot(0, 0, 99), _shot(1, 100, 299)]
    events = [
        {
            "event_id": "q1:0",
            "query_id": "q1",
            "query_type": "KIS",
            "video_id": "V",
            "start_frame": 45,
            "end_frame": 55,
        }
    ]

    first, arms_first, _ = run_gate1(
        config=config,
        proposals=proposals,
        mappings=mappings,
        dev_events=events,
        dataset_hash="d" * 64,
    )
    second, arms_second, _ = run_gate1(
        config=config,
        proposals=proposals,
        mappings=mappings,
        dev_events=events,
        dataset_hash="d" * 64,
    )

    assert first["run_hash"] == second["run_hash"]
    assert arms_first == arms_second
    assert first["heldout_accessed"] is False
    assert first["tuned_on_split"] == "dev"
