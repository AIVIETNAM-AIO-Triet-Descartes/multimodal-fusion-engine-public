"""Deterministic per-answer Q&A scoring for AIC 2026.

This module intentionally scores one answer only. Ranking thresholds, R@k, and
Final Score aggregation belong to the evaluation harness owned by the Lead.
"""

from __future__ import annotations

import unicodedata
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

__all__ = ["QAGroundTruth", "QASubmission", "answer_match", "score_qa"]

_NonBlankString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, strict=True),
]
_FrameId = Annotated[int, Field(ge=0, strict=True)]

# Deliberately conservative: strip only common wrapping punctuation. Meaning-
# bearing symbols such as %, +, -, /, and internal punctuation are preserved.
_BOUNDARY_PUNCTUATION = ".,!?;:\u2026\"'\u201c\u201d\u2018\u2019()[]{}<>"

# Whole-answer aliases required by the BTC example ("5" or "N\u0103m"). The
# matcher does not rewrite number words inside arbitrary phrases because doing
# so without an accepted-answer contract can introduce false positives.
_VIETNAMESE_NUMBERS_0_TO_20 = (
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
_NUMBER_WORD_TO_DIGIT = {
    word: str(number) for number, word in enumerate(_VIETNAMESE_NUMBERS_0_TO_20)
}


class _QAModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class QASubmission(_QAModel):
    """One Q&A answer submitted for scoring."""

    video_id: _NonBlankString
    frame_id: _FrameId
    answer: _NonBlankString


class QAGroundTruth(_QAModel):
    """One accepted video/frame interval and answer for a Q&A query."""

    video_id: _NonBlankString
    start_frame_id: _FrameId
    end_frame_id: _FrameId
    answer: _NonBlankString

    @model_validator(mode="after")
    def _range_is_ordered(self) -> QAGroundTruth:
        if self.start_frame_id > self.end_frame_id:
            raise ValueError("start_frame_id must be <= end_frame_id")
        return self


def _normalize_decimal_digits(text: str) -> str:
    normalized: list[str] = []
    for character in text:
        decimal_value = unicodedata.decimal(character, None)
        normalized.append(str(decimal_value) if decimal_value is not None else character)
    return "".join(normalized)


def _normalize_answer(answer: str) -> str:
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")

    # NFKC handles canonical and compatibility-equivalent forms (for example,
    # composed/decomposed Vietnamese text and full-width decimal digits).
    normalized = unicodedata.normalize("NFKC", answer).casefold()
    normalized = _normalize_decimal_digits(normalized)
    normalized = " ".join(normalized.split())
    normalized = normalized.strip(_BOUNDARY_PUNCTUATION)
    normalized = " ".join(normalized.split())
    if not normalized:
        raise ValueError("answer must contain non-punctuation text")

    return _NUMBER_WORD_TO_DIGIT.get(normalized, normalized)


def answer_match(a: str, gt_a: str) -> bool:
    """Match two Q&A answers using the deterministic v1 policy.

    Supported equivalences are NFKC Unicode normalization, case folding,
    whitespace collapsing, conservative wrapping-punctuation removal, Unicode
    decimal-digit normalization, and whole-answer Vietnamese number words 0-20.

    This is not a general semantic judge: it deliberately does not remove
    Vietnamese diacritics, fuzzy-match, substring-match, or guess synonyms and
    paraphrases. The separate interface lets the Lead replace it later with an
    accepted-answer set or a calibrated judge without changing ``score_qa``.

    Raises ``TypeError`` for non-string inputs and ``ValueError`` for answers
    that become empty after safe normalization.
    """

    return bool(_normalize_answer(a) == _normalize_answer(gt_a))


def score_qa(submission: QASubmission, gt: QAGroundTruth) -> float:
    """Return the official binary R-Score for one valid Q&A submission.

    A hit requires the exact video id, an inclusive source-frame match in
    ``[start_frame_id, end_frame_id]``, and an answer accepted by
    :func:`answer_match`. R@k and Final Score are intentionally out of scope.
    """

    is_correct = (
        submission.video_id == gt.video_id
        and gt.start_frame_id <= submission.frame_id <= gt.end_frame_id
        and answer_match(submission.answer, gt.answer)
    )
    return 1.0 if is_correct else 0.0
