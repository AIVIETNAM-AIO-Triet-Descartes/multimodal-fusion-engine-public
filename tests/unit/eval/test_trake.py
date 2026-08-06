"""Unit tests for TRAKE metric score evaluation.

Ref: docs/metrics-btc.html §4 (PDF §2.1.3).
"""

import pytest

from mfe.contracts.errors import ContractError
from mfe.eval.metrics.trake import (
    TRAKEEvent,
    TRAKEGroundTruth,
    TRAKESubmission,
    score_trake,
)


def test_official_pdf_fixture() -> None:
    """Official PDF test fixture from BTC specs §2.1.3.

    GT: video L10_V010, 4 events: [95,105], [145,155], [195,205], [245,255]
    Submit: (L10_V010, [101, 156, 203, 251])
    Matches:
      - 101 in [95,105] -> Match
      - 156 in [145,155] -> Fail
      - 203 in [195,205] -> Match
      - 251 in [245,255] -> Match
    Expected score: 3 / 4 = 0.75
    """
    gt = TRAKEGroundTruth(
        video_id="L10_V010",
        events=[
            TRAKEEvent(95, 105),
            TRAKEEvent(145, 155),
            TRAKEEvent(195, 205),
            TRAKEEvent(245, 255),
        ],
    )
    submission = TRAKESubmission(
        video_id="L10_V010",
        frame_ids=[101, 156, 203, 251],
    )

    score = score_trake(submission, gt)
    assert score == pytest.approx(0.75)


def test_perfect_match() -> None:
    """All submitted frame_ids fall inside their respective event windows -> 1.0."""
    gt = TRAKEGroundTruth(
        video_id="L01_V001",
        events=[
            TRAKEEvent(10, 20),
            TRAKEEvent(30, 40),
        ],
    )
    submission = TRAKESubmission(
        video_id="L01_V001",
        frame_ids=[15, 35],
    )
    assert score_trake(submission, gt) == 1.0


def test_zero_match_same_video() -> None:
    """Same video_id, but none of the frame_ids fall inside their event windows -> 0.0."""
    gt = TRAKEGroundTruth(
        video_id="L01_V001",
        events=[
            TRAKEEvent(10, 20),
            TRAKEEvent(30, 40),
        ],
    )
    submission = TRAKESubmission(
        video_id="L01_V001",
        frame_ids=[25, 50],
    )
    assert score_trake(submission, gt) == 0.0


def test_wrong_video_id() -> None:
    """Wrong video_id returns 0.0 immediately regardless of frame matches."""
    gt = TRAKEGroundTruth(
        video_id="L01_V001",
        events=[
            TRAKEEvent(10, 20),
            TRAKEEvent(30, 40),
        ],
    )
    submission = TRAKESubmission(
        video_id="L02_V999",
        frame_ids=[15, 35],
    )
    assert score_trake(submission, gt) == 0.0


def test_inclusive_boundaries() -> None:
    """frame_id equal to start_frame or end_frame must count as a match."""
    gt = TRAKEGroundTruth(
        video_id="L05_V005",
        events=[
            TRAKEEvent(100, 200),
            TRAKEEvent(300, 400),
        ],
    )
    # 100 == start_frame, 400 == end_frame
    submission = TRAKESubmission(
        video_id="L05_V005",
        frame_ids=[100, 400],
    )
    assert score_trake(submission, gt) == 1.0


def test_mismatched_event_count_raises_error() -> None:
    """Submitting fewer or more frame_ids than events in GT raises ContractError."""
    gt = TRAKEGroundTruth(
        video_id="L01_V001",
        events=[
            TRAKEEvent(10, 20),
            TRAKEEvent(30, 40),
        ],
    )
    submission_too_few = TRAKESubmission(
        video_id="L01_V001",
        frame_ids=[15],
    )
    submission_too_many = TRAKESubmission(
        video_id="L01_V001",
        frame_ids=[15, 35, 50],
    )

    with pytest.raises(ContractError, match="Mismatched event count"):
        score_trake(submission_too_few, gt)

    with pytest.raises(ContractError, match="Mismatched event count"):
        score_trake(submission_too_many, gt)


def test_invalid_dataclass_inputs() -> None:
    """Invalid input types or start_frame > end_frame raise ContractError."""
    with pytest.raises(ContractError, match="cannot be greater than end_frame"):
        TRAKEEvent(start_frame=100, end_frame=50)

    with pytest.raises(ContractError, match="start_frame must be an integer"):
        TRAKEEvent(start_frame="100", end_frame=200)  # type: ignore

    with pytest.raises(ContractError, match="frame_ids.*must be an integer"):
        TRAKESubmission(video_id="L01_V001", frame_ids=[15, "35"])  # type: ignore

    with pytest.raises(ContractError, match="video_id must be a non-empty string"):
        TRAKEGroundTruth(video_id="", events=[TRAKEEvent(10, 20)])

    with pytest.raises(ContractError, match="must contain a non-empty list of events"):
        TRAKEGroundTruth(video_id="L01_V001", events=[])
