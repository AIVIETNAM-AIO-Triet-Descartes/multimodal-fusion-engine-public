"""Unit tests for Quang's isolated AIC 2026 Q&A scorer."""

from __future__ import annotations

import itertools
import unicodedata

import pytest
from pydantic import ValidationError

import mfe.eval.metrics.qa as qa_module
from mfe.eval.metrics.qa import QAGroundTruth, QASubmission, answer_match, score_qa

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


def test_single_frame_range_is_valid_and_inclusive():
    ground_truth = QAGroundTruth(
        video_id="L05_V005",
        start_frame_id=888,
        end_frame_id=888,
        answer="màu xanh",
    )
    submission = QASubmission(
        video_id="L05_V005",
        frame_id=888,
        answer="màu xanh",
    )
    assert score_qa(submission, ground_truth) == 1.0


def test_frame_zero_is_not_shifted_or_clamped():
    ground_truth = QAGroundTruth(
        video_id="L05_V005",
        start_frame_id=0,
        end_frame_id=0,
        answer="màu xanh",
    )
    submission = QASubmission(
        video_id="L05_V005",
        frame_id=0,
        answer="màu xanh",
    )
    assert score_qa(submission, ground_truth) == 1.0


def test_arbitrarily_large_integer_frame_is_scored_without_coercion():
    large_frame = 10**100
    ground_truth = QAGroundTruth(
        video_id="L05_V005",
        start_frame_id=large_frame,
        end_frame_id=large_frame,
        answer="màu xanh",
    )
    submission = QASubmission(
        video_id="L05_V005",
        frame_id=large_frame,
        answer="màu xanh",
    )
    assert score_qa(submission, ground_truth) == 1.0


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


@pytest.mark.parametrize(
    "whitespace_variant",
    ["màu\txanh", "màu\nxanh", "màu\r\nxanh", "màu\u00a0xanh"],
)
def test_answer_match_collapses_supported_unicode_whitespace(whitespace_variant):
    assert answer_match(whitespace_variant, "màu xanh") is True


def test_answer_match_normalizes_composed_and_decomposed_unicode():
    composed = "m\u00e0u xanh"
    decomposed = unicodedata.normalize("NFD", composed)
    assert answer_match(composed, decomposed) is True


@pytest.mark.parametrize("unicode_digit", ["\uff15", "\u0665"])
def test_answer_match_normalizes_unicode_decimal_digits(unicode_digit):
    assert answer_match(unicode_digit, "n\u0103m") is True


def test_answer_match_normalization_is_idempotent():
    normalized = qa_module._normalize_answer("  NĂM! ")
    assert normalized == "5"
    assert qa_module._normalize_answer(normalized) == normalized


@pytest.mark.parametrize(
    "punctuation", [".", ",", "!", "?", ":", ";", '"', "'", "(", ")", "[", "]"]
)
def test_answer_match_removes_only_supported_boundary_punctuation(punctuation):
    assert answer_match(f"{punctuation}màu xanh{punctuation}", "màu xanh") is True


def test_answer_match_rejects_different_meanings():
    assert answer_match("m\u00e0u xanh", "m\u00e0u tr\u1eafng") is False


def test_answer_match_does_not_remove_vietnamese_diacritics():
    assert answer_match("m\u00e0u", "m\u00e1u") is False


@pytest.mark.parametrize(
    ("left", "right"),
    [("năm", "nam"), ("màu", "mau"), ("màu", "máu")],
)
def test_answer_match_preserves_vietnamese_diacritic_distinctions(left, right):
    assert answer_match(left, right) is False


@pytest.mark.parametrize(
    ("answer", "ground_truth"),
    [("xanh-\u0111\u1ecf", "xanh \u0111\u1ecf"), ("50%", "50")],
)
def test_answer_match_preserves_meaning_bearing_internal_or_symbol_characters(answer, ground_truth):
    assert answer_match(answer, ground_truth) is False


def test_answer_match_does_not_rewrite_numbers_inside_arbitrary_phrases():
    assert answer_match("5 ng\u01b0\u1eddi", "n\u0103m ng\u01b0\u1eddi") is False


@pytest.mark.parametrize("numeric_form", ["05", "5.0", "+5", "-5", "5,0", "5/5"])
def test_answer_match_does_not_parse_unsupported_numeric_forms(numeric_form):
    assert answer_match(numeric_form, "năm") is False


def test_answer_match_normalizes_each_unicode_decimal_digit_without_number_parsing():
    assert answer_match("２０", "20") is True
    assert answer_match("２０", "hai mươi") is True


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("xanh dương", "xanh lam"),
        ("màu xanh", "chiếc ly màu xanh"),
        ("đỏ xanh", "xanh đỏ"),
    ],
)
def test_answer_match_does_not_guess_synonyms_substrings_or_word_order(left, right):
    assert answer_match(left, right) is False


@pytest.mark.parametrize("invalid", ["\u200b", "màu\u200bxanh", "\x00", "màu\x00xanh"])
def test_answer_match_rejects_hidden_control_or_format_characters(invalid):
    with pytest.raises(ValueError, match="control or format"):
        answer_match(invalid, "màu xanh")


def test_answer_match_handles_a_very_long_valid_answer_deterministically():
    long_answer = "x" * 100_000
    assert answer_match(long_answer, long_answer) is True


@pytest.mark.parametrize("invalid", ["", "   ", "!!!"])
def test_answer_match_rejects_empty_normalized_input(invalid):
    with pytest.raises(ValueError, match="answer must"):
        answer_match(invalid, "m\u00e0u xanh")


def test_answer_match_rejects_non_string_input():
    with pytest.raises(TypeError, match="answer must be a string"):
        answer_match(None, "m\u00e0u xanh")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("left", "right"),
    [(str(number), word) for number, word in enumerate(_NUMBER_WORDS_0_TO_20)]
    + [
        ("M\u00c0U XANH!", "m\u00e0u xanh"),
        ("màu\u00a0xanh", "màu xanh"),
        ("５", "năm"),
        ("m\u00e0u xanh", "m\u00e0u tr\u1eafng"),
        ("m\u00e0u", "m\u00e1u"),
        ("05", "năm"),
        ("xanh dương", "xanh lam"),
    ],
)
def test_answer_match_is_symmetric(left, right):
    assert answer_match(left, right) == answer_match(right, left)


def test_answer_match_is_deterministic_and_returns_bool():
    outcomes = [answer_match("  N\u0102M! ", "5") for _ in range(20)]
    assert outcomes == [True] * 20
    assert all(type(outcome) is bool for outcome in outcomes)
    assert type(answer_match("màu xanh", "màu trắng")) is bool


def test_score_qa_always_returns_float_for_valid_inputs():
    hit = QASubmission(video_id="L05_V005", frame_id=888, answer="m\u00e0u xanh")
    miss = QASubmission(video_id="L05_V005", frame_id=901, answer="m\u00e0u xanh")
    assert type(score_qa(hit, _ground_truth())) is float
    assert type(score_qa(miss, _ground_truth())) is float


def test_score_qa_does_not_mutate_models_or_number_aliases():
    submission = QASubmission(video_id="L05_V005", frame_id=888, answer="Năm")
    ground_truth = QAGroundTruth(
        video_id="L05_V005",
        start_frame_id=800,
        end_frame_id=900,
        answer="5",
    )
    submission_before = submission.model_dump()
    ground_truth_before = ground_truth.model_dump()
    aliases_before = qa_module._VIETNAMESE_NUMBERS_0_TO_20

    assert score_qa(submission, ground_truth) == 1.0
    assert submission.model_dump() == submission_before
    assert ground_truth.model_dump() == ground_truth_before
    assert qa_module._VIETNAMESE_NUMBERS_0_TO_20 is aliases_before


@pytest.mark.parametrize(
    "wrong_video_id",
    [
        "l05_v005",
        "L05_V005.mp4",
        " L05_V005",
        "L05_V005 ",
        "L05_V00５",
        "L06_V007",
    ],
)
def test_video_id_comparison_is_exact_without_normalization(wrong_video_id):
    submission = QASubmission(
        video_id=wrong_video_id,
        frame_id=888,
        answer="màu xanh",
    )
    assert submission.video_id == wrong_video_id
    assert score_qa(submission, _ground_truth()) == 0.0


@pytest.mark.parametrize(
    "invalid_frame",
    [None, True, False, 888.0, float("nan"), float("inf"), "888"],
)
def test_submission_rejects_non_strict_frame_types(invalid_frame):
    with pytest.raises(ValidationError):
        QASubmission(video_id="L05_V005", frame_id=invalid_frame, answer="màu xanh")


@pytest.mark.parametrize(
    "invalid_boundary",
    [None, True, False, 800.0, float("nan"), float("inf"), "800"],
)
@pytest.mark.parametrize("field_name", ["start_frame_id", "end_frame_id"])
def test_ground_truth_rejects_non_strict_boundary_types(field_name, invalid_boundary):
    values = {
        "video_id": "L05_V005",
        "start_frame_id": 800,
        "end_frame_id": 900,
        "answer": "màu xanh",
    }
    values[field_name] = invalid_boundary
    with pytest.raises(ValidationError):
        QAGroundTruth(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"video_id": "   "},
        {"answer": "   "},
        {"answer": "!!!"},
        {"answer": "\u200b"},
        {"answer": "màu\x00xanh"},
        {"video_id": "L05_V005\x00"},
        {"video_id": 5},
        {"answer": 5},
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
        {"answer": "!!!"},
        {"answer": "\u200b"},
        {"answer": "màu\x00xanh"},
        {"video_id": "L05_V005\u200b"},
        {"video_id": None},
        {"answer": None},
        {"start_frame_id": -1},
        {"end_frame_id": -1},
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


def test_contracts_reject_missing_fields_and_non_mapping_objects():
    with pytest.raises(ValidationError):
        QASubmission(video_id="L05_V005", frame_id=888)
    with pytest.raises(ValidationError):
        QAGroundTruth(
            video_id="L05_V005",
            start_frame_id=800,
            end_frame_id=900,
        )
    with pytest.raises(ValidationError):
        QASubmission.model_validate(None)
    with pytest.raises(ValidationError):
        QAGroundTruth.model_validate(None)


def test_score_qa_rejects_invalid_model_objects_with_clear_errors():
    submission = QASubmission(video_id="L05_V005", frame_id=888, answer="màu xanh")
    with pytest.raises(TypeError, match="submission must be a QASubmission"):
        score_qa(None, _ground_truth())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="gt must be a QAGroundTruth"):
        score_qa(submission, None)  # type: ignore[arg-type]
