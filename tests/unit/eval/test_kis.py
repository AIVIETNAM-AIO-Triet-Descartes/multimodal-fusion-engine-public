"""Textual KIS scorer — docs/metrics-btc.html §2 (PDF §2.1.1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mfe.eval.metrics.kis import KISGroundTruth, KISSubmission, score_kis


def _ground_truth() -> KISGroundTruth:
    return KISGroundTruth(video_id="L01_V001", start_frame_id=500, end_frame_id=510)


@pytest.mark.parametrize(
    ("video_id", "frame_id", "expected"),
    [
        ("L01_V001", 505, 1.0),  # PDF: đúng cả hai
        ("L01_V001", 600, 0.0),  # PDF: ngoài range
        ("L02_V003", 505, 0.0),  # PDF: sai video
    ],
)
def test_pdf_kis_examples(video_id: str, frame_id: int, expected: float) -> None:
    submission = KISSubmission(video_id=video_id, frame_id=frame_id)
    assert score_kis(submission, _ground_truth()) == expected


@pytest.mark.parametrize("frame_id", [500, 510])
def test_interval_is_closed_at_both_ends(frame_id: int) -> None:
    """`[s, e]` is read as a closed interval — the assumption of §2.

    Both boundary frames score 1.0. If BTC later specifies a half-open interval,
    this test is the thing that fails, which is the point of writing it down.
    """
    submission = KISSubmission(video_id="L01_V001", frame_id=frame_id)
    assert score_kis(submission, _ground_truth()) == 1.0


@pytest.mark.parametrize("frame_id", [499, 511])
def test_just_outside_the_interval_misses(frame_id: int) -> None:
    submission = KISSubmission(video_id="L01_V001", frame_id=frame_id)
    assert score_kis(submission, _ground_truth()) == 0.0


def test_single_frame_ground_truth() -> None:
    """s == e is a legal one-frame answer, not an empty interval."""
    gt = KISGroundTruth(video_id="L01_V001", start_frame_id=500, end_frame_id=500)
    assert score_kis(KISSubmission(video_id="L01_V001", frame_id=500), gt) == 1.0
    assert score_kis(KISSubmission(video_id="L01_V001", frame_id=501), gt) == 0.0


def test_video_id_comparison_is_exact() -> None:
    """Case and whitespace differences are misses, not near-misses.

    BTC ids are machine-generated. A submission that differs by case came from a
    bug in our pipeline, and normalising it here would hide that bug.
    """
    gt = _ground_truth()
    for wrong in ("l01_v001", " L01_V001", "L01_V001 "):
        assert score_kis(KISSubmission(video_id=wrong, frame_id=505), gt) == 0.0


# --- input that must fail loudly rather than score 0.0 ----------------------


@pytest.mark.parametrize("bad", ["505", 505.0, True, None, -1])
def test_invalid_frame_id_is_rejected(bad: object) -> None:
    """A malformed frame_id must not be coerced into a plausible miss.

    `True` matters most: bool subclasses int, so without strict validation it
    would score as frame 1 and return an entirely believable 0.0.
    """
    with pytest.raises(ValidationError):
        KISSubmission(video_id="L01_V001", frame_id=bad)


@pytest.mark.parametrize("bad", ["", 123, None])
def test_invalid_video_id_is_rejected(bad: object) -> None:
    with pytest.raises(ValidationError):
        KISSubmission(video_id=bad, frame_id=505)


def test_missing_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        KISSubmission(video_id="L01_V001")


def test_extra_field_is_rejected() -> None:
    """A stray field usually means a record from another scorer leaked in."""
    with pytest.raises(ValidationError):
        KISSubmission(video_id="L01_V001", frame_id=505, answer="màu xanh")


def test_inverted_ground_truth_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        KISGroundTruth(video_id="L01_V001", start_frame_id=510, end_frame_id=500)


def test_wrong_record_types_raise() -> None:
    gt = _ground_truth()
    with pytest.raises(TypeError):
        score_kis({"video_id": "L01_V001", "frame_id": 505}, gt)
    with pytest.raises(TypeError):
        score_kis(KISSubmission(video_id="L01_V001", frame_id=505), object())


def test_records_are_immutable() -> None:
    submission = KISSubmission(video_id="L01_V001", frame_id=505)
    with pytest.raises(ValidationError):
        submission.frame_id = 506
