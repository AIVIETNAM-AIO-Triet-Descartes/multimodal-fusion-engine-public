from __future__ import annotations

import pytest

from mfe.offline.augmentation.io import canonical_hash
from mfe.offline.augmentation.oracle_artifacts import (
    OracleArtifactCandidate,
    OracleArtifactTarget,
)
from mfe.offline.augmentation.representative_oracle import (
    RepresentativeResolvedSample,
    build_representative_scan_plan,
    inclusive_uniform_indices,
    partition_representative_scan,
    representative_scan_plan_hash,
    validate_scan_completion_marker,
)


def _target(
    query_id: str,
    *,
    event_index: int = 0,
    video_id: str = "L21_V001",
    start: int = 10,
    end: int = 20,
) -> OracleArtifactTarget:
    frame_idx = (start + end) // 2
    return OracleArtifactTarget(
        query_id=query_id,
        event_index=event_index,
        video_id=video_id,
        frame_range_start=start,
        frame_range_end=end,
        target=OracleArtifactCandidate(
            keyframe_id=f"{video_id}#{frame_idx:06d}",
            video_id=video_id,
            frame_idx=frame_idx,
            pts_time=frame_idx / 25,
            pts_time_source="decoder_pts",
            shot_id=f"{video_id}#s000001",
            selector="oracle_target",
            coverage_class="C1",
            selection_round=1,
            distance_to_nearest_btc=1.0,
            detector_confidence=1.0,
            decoder_fingerprint="decoder",
            config_hash="a" * 64,
            audit_status="auto_accepted",
        ),
    )


def test_inclusive_sampler_full_scans_short_and_one_frame_intervals() -> None:
    assert inclusive_uniform_indices(7, 7, max_samples=64) == (7,)
    assert inclusive_uniform_indices(7, 10, max_samples=64) == (7, 8, 9, 10)


def test_inclusive_sampler_preserves_endpoints_without_float_rounding() -> None:
    sampled = inclusive_uniform_indices(100, 10_000, max_samples=64)
    assert len(sampled) == 64
    assert sampled[0] == 100
    assert sampled[-1] == 10_000
    assert sampled == tuple(sorted(set(sampled)))


@pytest.mark.parametrize(
    ("start", "end", "max_samples"),
    [(-1, 2, 64), (2, 1, 64), (1, 2, 1)],
)
def test_inclusive_sampler_rejects_invalid_contracts(
    start: int, end: int, max_samples: int
) -> None:
    with pytest.raises(ValueError):
        inclusive_uniform_indices(start, end, max_samples=max_samples)


def test_scan_plan_unions_overlapping_events_and_is_deterministic() -> None:
    targets = [
        _target("q2", start=10, end=12),
        _target("q1", start=11, end=13),
    ]
    first = build_representative_scan_plan(targets)
    second = build_representative_scan_plan(reversed(targets))
    assert first == second
    assert [row.frame_idx for row in first] == [10, 11, 12, 13]
    shared = next(row for row in first if row.frame_idx == 11)
    assert [(item.query_id, item.event_index) for item in shared.memberships] == [
        ("q1", 0),
        ("q2", 0),
    ]
    assert representative_scan_plan_hash(first) == representative_scan_plan_hash(second)


def test_scan_plan_rejects_duplicate_query_event_identity() -> None:
    with pytest.raises(ValueError, match="duplicate oracle target event"):
        build_representative_scan_plan([_target("q"), _target("q")])


def test_partition_keeps_each_video_on_exactly_one_runner() -> None:
    requests = build_representative_scan_plan(
        [
            _target("q1", video_id="L21_V001", start=1, end=2),
            _target("q2", video_id="L21_V002", start=3, end=4),
            _target("q3", video_id="L21_V003", start=5, end=6),
        ]
    )
    assignments = partition_representative_scan(
        requests,
        ("quang", "trong"),
        {"L21_V001": 100, "L21_V002": 80, "L21_V003": 20},
    )
    owner = {}
    for runner, rows in assignments.items():
        for row in rows:
            assert owner.setdefault(row.video_id, runner) == runner
    assert set(owner) == {"L21_V001", "L21_V002", "L21_V003"}


def test_resolved_sample_requires_decoder_identity_and_matching_jpeg_name() -> None:
    request = build_representative_scan_plan([_target("q", start=17, end=17)])[0]
    sample = RepresentativeResolvedSample(
        request=request,
        decoder_pts_time=17 / 25,
        literal_fps=25.0,
        image_sha256="b" * 64,
        jpeg_name="000000017.jpg",
    )
    assert sample.frame_idx_source == "decoder_counter"
    assert sample.pts_time_source == "decoder_pts"
    with pytest.raises(ValueError, match="JPEG name"):
        RepresentativeResolvedSample(
            request=request,
            decoder_pts_time=17 / 25,
            literal_fps=25.0,
            image_sha256="b" * 64,
            jpeg_name="000000018.jpg",
        )


def test_scan_completion_marker_fails_closed_on_remote_hash_drift() -> None:
    marker = {
        "status": "COMPLETE",
        "run_id": "o1s-scan-v1",
        "runner": "quang",
        "video_id": "L21_V001",
        "scan_plan_hash": "a" * 64,
        "sampling_policy": "inclusive_uniform_integer_lb64",
        "heldout_accessed": False,
        "btc_mutated": False,
        "artifacts": {"/research/a": "b" * 64},
    }
    marker["marker_payload_hash"] = canonical_hash(marker)
    kwargs = {
        "run_id": "o1s-scan-v1",
        "runner": "quang",
        "video_id": "L21_V001",
        "plan_hash": "a" * 64,
    }
    assert validate_scan_completion_marker(
        marker,
        observed_remote_hashes={"/research/a": "b" * 64},
        **kwargs,
    )
    assert not validate_scan_completion_marker(
        marker,
        observed_remote_hashes={"/research/a": "c" * 64},
        **kwargs,
    )
