from __future__ import annotations

import csv
import sys
import types

import pytest

from mfe.offline.augmentation.coverage import classify_coverage, read_btc_mapping
from mfe.offline.augmentation.io import sha256_path
from mfe.offline.augmentation.models import AuditStatus, BtcFrame, CoverageClass, ShotProposal
from mfe.offline.augmentation.transnet import (
    TransNetV2Adapter,
    proposals_from_scores,
    verify_shot_partition,
)


def shot(*, confidence: float = 1.0, end_frame: int = 10, end_pts: float = 10.0):
    return ShotProposal(
        video_id="L21_V001",
        shot_id="L21_V001#s00",
        start_frame=0,
        end_frame_exclusive=end_frame,
        start_pts_time=0.0,
        end_pts_time=end_pts,
        boundary_confidence=confidence,
        detector_fingerprint="detector-v1",
    )


def btc(frame_idx: int, pts_time: float) -> BtcFrame:
    return BtcFrame(
        keyframe_id=f"L21_V001#{frame_idx:06d}",
        video_id="L21_V001",
        frame_idx=frame_idx,
        pts_time=pts_time,
    )


def test_saved_model_tree_hash_covers_relative_names_and_weight_bytes(tmp_path):
    model = tmp_path / "saved-model"
    variables = model / "variables"
    variables.mkdir(parents=True)
    (model / "saved_model.pb").write_bytes(b"graph")
    weights = variables / "variables.data-00000-of-00001"
    weights.write_bytes(b"weights-v1")
    (variables / "variables.index").write_bytes(b"index")

    first = sha256_path(model)
    weights.write_bytes(b"weights-v2")
    second = sha256_path(model)

    assert first != second
    assert sha256_path(model) == second


def test_transnet_adapter_passes_saved_model_directory_to_official_factory(
    tmp_path, monkeypatch
):
    model = tmp_path / "saved-model"
    (model / "variables").mkdir(parents=True)
    (model / "saved_model.pb").write_bytes(b"graph")
    (model / "variables" / "variables.index").write_bytes(b"index")
    (model / "variables" / "variables.data-00000-of-00001").write_bytes(b"weights")
    received = {}

    class FakeTransNetV2:
        def __init__(self, *, model_dir):
            received["model_dir"] = model_dir

    fake_module = types.ModuleType("fake_transnetv2")
    fake_module.TransNetV2 = FakeTransNetV2
    monkeypatch.setitem(sys.modules, "fake_transnetv2", fake_module)

    adapter = TransNetV2Adapter(
        "fake_transnetv2:TransNetV2", model, sha256_path(model)
    )

    assert received["model_dir"] == str(model)
    assert isinstance(adapter.model, FakeTransNetV2)


def test_transition_runs_form_a_complete_half_open_partition():
    scores = [0.0, 0.8, 0.9, 0.0, 0.0, 0.0]
    pts = tuple(float(index) for index in range(len(scores)))

    proposals = proposals_from_scores(
        "L21_V001", scores, pts, 6.0, 0.5, "transnet-sha"
    )

    assert [(item.start_frame, item.end_frame_exclusive) for item in proposals] == [
        (0, 2),
        (2, 6),
    ]
    assert [item.boundary_confidence for item in proposals] == [0.9, 0.9]
    assert verify_shot_partition(proposals, 6) == []


def test_coverage_classes_fail_closed_for_transition_and_uncertain_shots():
    common = {
        "target_gap_seconds": 4.0,
        "k_max": 3,
        "micro_shot_seconds": 0.5,
        "uncertain_below": 0.6,
    }
    c0 = classify_coverage(shot(), [], **common)
    c1 = classify_coverage(shot(), [btc(5, 5.0)], **common)
    ok = classify_coverage(
        shot(end_frame=6, end_pts=6.0),
        [btc(2, 2.0), btc(4, 4.0)],
        **common,
    )
    uncertain = classify_coverage(shot(confidence=0.59), [], **common)
    micro = classify_coverage(
        shot(confidence=0.1, end_frame=1, end_pts=0.25), [], **common
    )

    assert (c0.coverage_class, c0.deficit, c0.eligible) == (CoverageClass.C0, 3, True)
    assert (c1.coverage_class, c1.deficit, c1.eligible) == (CoverageClass.C1, 2, True)
    assert (ok.coverage_class, ok.deficit) == (CoverageClass.OK, 0)
    assert uncertain.coverage_class == CoverageClass.CE
    assert uncertain.audit_status == AuditStatus.NEEDS_REVIEW
    assert micro.coverage_class == CoverageClass.CT
    assert not micro.eligible


def test_mapping_reader_copies_literal_frame_and_pts_and_collapses_identity(tmp_path):
    mapping = tmp_path / "L21_V001.csv"
    with mapping.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["n", "frame_idx", "pts_time"])
        writer.writeheader()
        writer.writerow({"n": 1, "frame_idx": 17, "pts_time": "0.56789"})
        writer.writerow({"n": 2, "frame_idx": 17, "pts_time": "0.60000"})
        writer.writerow({"n": 3, "frame_idx": 31, "pts_time": "1.03437"})

    rows = read_btc_mapping("L21_V001", mapping)

    assert [(row.frame_idx, row.pts_time) for row in rows] == [
        (17, pytest.approx(0.56789)),
        (31, pytest.approx(1.03437)),
    ]
    assert [row.keyframe_id for row in rows] == [
        "L21_V001#000017",
        "L21_V001#000031",
    ]
