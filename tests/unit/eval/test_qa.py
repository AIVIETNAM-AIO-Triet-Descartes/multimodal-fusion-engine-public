"""Unit tests for Quang's isolated AIC 2026 Q&A scorer."""

from __future__ import annotations

import itertools
import unicodedata

import pytest
from pydantic import ValidationError

from mfe.eval.qa import QAGroundTruth, QASubmission, answer_match, score_qa

_NUMBER_WORDS_0_TO_20 = (
    "kh\u00f4ng",
    "m\u1ed9t",
    "hai",
    "ba",
    "b\u1ed1n",
    "n\u0103m",
    "s\u00e1u",
    "b\u1ea3y",
    "t\u00e1m",
    "ch\u00edn",
    "m\u01b0\u1eddi",
    "m\u01b0\u1eddi m\u1ed9t",
    "m\u01b0\u1eddi hai",
    "m\u01b0\u1eddi ba",
    "m\u01b0\u1eddi b\u1ed1n",
    "m\u01b0\u1eddi l\u0103m",
    "m\u01b0\u1eddi s\u00e1u",
    "m\u01b0\u1eddi b\u1ea3y",
    "m\u01b0\u1eddi t\u00e1m",
    "m\u01b0\u1eddi ch\u00edn",
    "hai m\u01b0\u01a1i",
)


def _ground_truth() -> QAGroundTruth:
    return QAGroundTruth(
        video_id="L05_V005",
        start_frame_id=800,
        end_frame_id=900,
        answer="m\u00e0u xanh",
    )


@pytest.mark.parametrize(
    ("video_id", "frame_id", "answer", "expected"),
    [
        ("L05_V005", 888, "m\u00e0u xanh", 1.0),
        ("L05_V005", 888, "m\u00e0u tr\u1eafng", 0.0),
        ("L06_V007", 888, "m\u00e0u xanh", 0.0),
    ],
)
def test_pdf_qa_examples(video_id, frame_id, answer, expected):
    submission = QASubmission(video_id=video_id, frame_id=frame_id, answer=answer)
    assert score_qa(submission, _ground_truth()) == expected


@pytest.mark.parametrize("frame_id", [800, 900])
def test_frame_range_is_inclusive(frame_id):
    submission = QASubmission(video_id="L05_V005", frame_id=frame_id, answer="m\u00e0u xanh")
    assert score_qa(submission, _ground_truth()) == 1.0


@pytest.mark.parametrize("frame_id", [799, 901])
def test_frame_outside_range_scores_zero(frame_id):
    submission = QASubmission(video_id="L05_V005", frame_id=frame_id, answer="m\u00e0u xanh")
    assert score_qa(submission, _ground_truth()) == 0.0


@pytest.mark.parametrize(
    ("video_ok", "frame_ok", "answer_ok"),
    list(itertools.product([False, True], repeat=3)),
)
def test_score_truth_table_requires_all_three_conditions(video_ok, frame_ok, answer_ok):
    submission = QASubmission(
        video_id="L05_V005" if video_ok else "L06_V007",
        frame_id=888 if frame_ok else 901,
        answer="m\u00e0u xanh" if answer_ok else "m\u00e0u tr\u1eafng",
    )
    expected = 1.0 if video_ok and frame_ok and answer_ok else 0.0
    assert score_qa(submission, _ground_truth()) == expected


@pytest.mark.parametrize("number", range(21))
def test_vietnamese_number_words_match_digits_in_both_directions(number):
    word = _NUMBER_WORDS_0_TO_20[number]
    digit = str(number)
    assert answer_match(digit, word) is True
    assert answer_match(word, digit) is True


def test_answer_match_normalizes_case_whitespace_and_wrapping_punctuation():
    assert answer_match("  M\u00c0U\tXANH! ", "m\u00e0u xanh") is True


def test_answer_match_normalizes_composed_and_decomposed_unicode():
    composed = "m\u00e0u xanh"
    decomposed = unicodedata.normalize("NFD", composed)
    assert answer_match(composed, decomposed) is True


@pytest.mark.parametrize("unicode_digit", ["\uff15", "\u0665"])
def test_answer_match_normalizes_unicode_decimal_digits(unicode_digit):
    assert answer_match(unicode_digit, "n\u0103m") is True


def test_answer_match_rejects_different_meanings():
    assert answer_match("m\u00e0u xanh", "m\u00e0u tr\u1eafng") is False


def test_answer_match_does_not_remove_vietnamese_diacritics():
    assert answer_match("m\u00e0u", "m\u00e1u") is False


@pytest.mark.parametrize(
    ("answer", "ground_truth"),
    [("xanh-\u0111\u1ecf", "xanh \u0111\u1ecf"), ("50%", "50")],
)
def test_answer_match_preserves_meaning_bearing_internal_or_symbol_characters(answer, ground_truth):
    assert answer_match(answer, ground_truth) is False


def test_answer_match_does_not_rewrite_numbers_inside_arbitrary_phrases():
    assert answer_match("5 ng\u01b0\u1eddi", "n\u0103m ng\u01b0\u1eddi") is False


@pytest.mark.parametrize("invalid", ["", "   ", "!!!"])
def test_answer_match_rejects_empty_normalized_input(invalid):
    with pytest.raises(ValueError, match="answer must"):
        answer_match(invalid, "m\u00e0u xanh")


def test_answer_match_rejects_non_string_input():
    with pytest.raises(TypeError, match="answer must be a string"):
        answer_match(None, "m\u00e0u xanh")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("5", "N\u0103m"),
        ("M\u00c0U XANH!", "m\u00e0u xanh"),
        ("m\u00e0u xanh", "m\u00e0u tr\u1eafng"),
        ("m\u00e0u", "m\u00e1u"),
    ],
)
def test_answer_match_is_symmetric(left, right):
    assert answer_match(left, right) == answer_match(right, left)


def test_answer_match_is_deterministic_and_returns_bool():
    outcomes = [answer_match("  N\u0102M! ", "5") for _ in range(20)]
    assert outcomes == [True] * 20
    assert all(type(outcome) is bool for outcome in outcomes)


def test_score_qa_always_returns_float_for_valid_inputs():
    hit = QASubmission(video_id="L05_V005", frame_id=888, answer="m\u00e0u xanh")
    miss = QASubmission(video_id="L05_V005", frame_id=901, answer="m\u00e0u xanh")
    assert type(score_qa(hit, _ground_truth())) is float
    assert type(score_qa(miss, _ground_truth())) is float


@pytest.mark.parametrize(
    "overrides",
    [
        {"video_id": "   "},
        {"answer": "   "},
        {"frame_id": -1},
        {"frame_id": 888.0},
        {"unexpected": "field"},
    ],
)
def test_submission_contract_rejects_invalid_input(overrides):
    values = {"video_id": "L05_V005", "frame_id": 888, "answer": "m\u00e0u xanh"}
    values.update(overrides)
    with pytest.raises(ValidationError):
        QASubmission(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"video_id": ""},
        {"answer": ""},
        {"start_frame_id": -1},
        {"end_frame_id": 900.0},
        {"start_frame_id": 901, "end_frame_id": 900},
        {"unexpected": "field"},
    ],
)
def test_ground_truth_contract_rejects_invalid_input(overrides):
    values = {
        "video_id": "L05_V005",
        "start_frame_id": 800,
        "end_frame_id": 900,
        "answer": "m\u00e0u xanh",
    }
    values.update(overrides)
    with pytest.raises(ValidationError):
        QAGroundTruth(**values)


def test_contracts_are_frozen():
    submission = QASubmission(video_id="L05_V005", frame_id=888, answer="m\u00e0u xanh")
    with pytest.raises(ValidationError):
        submission.frame_id = 889
