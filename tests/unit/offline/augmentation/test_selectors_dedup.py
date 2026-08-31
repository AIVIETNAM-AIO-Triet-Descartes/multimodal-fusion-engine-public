from __future__ import annotations

import numpy as np

from mfe.offline.augmentation.dedup import ProtectedReference, protected_dedup
from mfe.offline.augmentation.models import (
    AuditStatus,
    BtcFrame,
    CandidateRow,
    CoverageAuditRow,
    CoverageClass,
    Selector,
)
from mfe.offline.augmentation.selectors import (
    apply_global_cap,
    counts_by_shot,
    match_random_control,
    select_b3,
    select_b4,
)


def audit(shot_number: int, coverage: CoverageClass, deficit: int = 2):
    start = shot_number * 10
    return CoverageAuditRow(
        video_id="L21_V001",
        shot_id=f"L21_V001#s{shot_number:02d}",
        start_frame=start,
        end_frame_exclusive=start + 10,
        start_pts_time=float(start),
        end_pts_time=float(start + 10),
        detector_confidence=0.9,
        btc_keyframe_ids=(),
        btc_count=0,
        max_temporal_gap=10.0,
        coverage_class=coverage,
        target_gap_seconds=4.0,
        k_target=2,
        k_max=3,
        deficit=deficit,
        audit_status=AuditStatus.AUTO_ACCEPTED,
    )


def candidate(frame_idx: int, shot_number: int, selection_round: int = 1):
    return CandidateRow(
        keyframe_id=f"L21_V001#{frame_idx:06d}",
        video_id="L21_V001",
        frame_idx=frame_idx,
        literal_fps=1.0,
        pts_time=float(frame_idx),
        decoder_pts_time=float(frame_idx),
        shot_id=f"L21_V001#s{shot_number:02d}",
        selector=Selector.FARTHEST,
        coverage_class=CoverageClass.C1,
        selection_round=selection_round,
        distance_to_nearest_btc=5.0,
        detector_confidence=0.9,
        decoder_fingerprint="decoder-v1",
        config_hash="config-v1",
    )


def test_b3_dynamic_quota_and_b4_have_exactly_matched_shot_budgets():
    rows = [audit(0, CoverageClass.C0), audit(1, CoverageClass.C1)]
    timeline = {index: float(index) for index in range(20)}
    btc_row = BtcFrame(
        keyframe_id="L21_V001#000015",
        video_id="L21_V001",
        frame_idx=15,
        pts_time=15.0,
    )
    btc_by_shot = {"L21_V001#s00": [], "L21_V001#s01": [btc_row]}
    common = {
        "decoder_fingerprints": {"L21_V001": "decoder-v1"},
        "literal_fps_by_video": {"L21_V001": 1.0},
        "config_hash": "config-v1",
    }

    b3 = select_b3(rows, {"L21_V001": timeline}, btc_by_shot, **common)
    b4 = select_b4(
        b3,
        rows,
        {"L21_V001": timeline},
        btc_by_shot,
        **common,
        seed=2026,
    )

    assert counts_by_shot(b3) == {"L21_V001#s00": 2, "L21_V001#s01": 2}
    assert counts_by_shot(b4) == counts_by_shot(b3)
    assert all(item.selector == Selector.RANDOM for item in b4)
    assert not ({item.keyframe_id for item in b4} & {btc_row.keyframe_id})


def test_selector_separates_decoder_pts_from_floor_safe_mapping_pts() -> None:
    row = CoverageAuditRow(
        video_id="L21_V001",
        shot_id="L21_V001#s00",
        start_frame=7670,
        end_frame_exclusive=7679,
        start_pts_time=306.8,
        end_pts_time=307.12,
        detector_confidence=0.9,
        btc_keyframe_ids=(),
        btc_count=0,
        max_temporal_gap=0.32,
        coverage_class=CoverageClass.C0,
        target_gap_seconds=6.0,
        k_target=1,
        k_max=3,
        deficit=1,
        audit_status=AuditStatus.AUTO_ACCEPTED,
    )
    timeline = {index: index / 25.0 for index in range(7670, 7679)}
    timeline[7674] = 306.96  # binary float floors to 7673 when multiplied by 25

    selected = select_b3(
        [row],
        {"L21_V001": timeline},
        {"L21_V001#s00": []},
        literal_fps_by_video={"L21_V001": 25.0},
        decoder_fingerprints={"L21_V001": "decoder-v1"},
        config_hash="config-v1",
    )

    assert selected[0].frame_idx == 7674
    assert selected[0].decoder_pts_time == 306.96
    assert selected[0].pts_time == (7674.5 / 25.0)


def test_global_cap_is_round_first_and_never_exceeds_three_percent():
    rows = [audit(index, CoverageClass.C1) for index in range(4)]
    proposed = [
        candidate(index * 10 + round_, index, round_)
        for round_ in (1, 2)
        for index in range(4)
    ]

    selected = apply_global_cap(
        proposed,
        rows,
        base_keyframes=100,
        cap_ratio=0.03,
    )

    assert len(selected) == 3
    assert all(item.selection_round == 1 for item in selected)


def test_dedup_protects_btc_and_compares_delta_only_inside_same_shot():
    first = candidate(2, 0)
    second = candidate(12, 1)
    protected = [
        ProtectedReference(
            keyframe_id="L21_V001#000001",
            shot_id="L21_V001#s00",
            phash=0,
            embedding=np.array([1.0, 0.0]),
        )
    ]

    result = protected_dedup(
        [first, second],
        protected,
        {first.keyframe_id: 1, second.keyframe_id: 1},
        {
            first.keyframe_id: np.array([1.0, 0.0]),
            second.keyframe_id: np.array([1.0, 0.0]),
        },
        phash_max_distance=1,
        embedding_max_cosine=0.97,
    )

    assert [item.keyframe_id for item in result.kept] == [second.keyframe_id]
    assert result.dropped[0].duplicate_of == "L21_V001#000001"
    assert result.dropped[0].reason == "phash"


def test_random_control_is_matched_again_after_dedup():
    treatment = [candidate(1, 0), candidate(2, 0, 2), candidate(11, 1)]
    random_pool = [
        candidate(3, 0),
        candidate(4, 0, 2),
        candidate(5, 0, 3),
        candidate(13, 1),
        candidate(14, 1, 2),
    ]

    matched = match_random_control(treatment, random_pool)

    assert counts_by_shot(matched) == counts_by_shot(treatment)
    assert [item.frame_idx for item in matched] == [3, 4, 13]
