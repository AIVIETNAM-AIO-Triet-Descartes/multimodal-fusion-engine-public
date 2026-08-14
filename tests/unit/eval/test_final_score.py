"""R@k and Final Score — docs/metrics-btc.html §5 (PDF §2.2)."""

from __future__ import annotations

import pytest

from mfe.contracts.errors import ContractError
from mfe.eval.metrics.final_score import (
    DEFAULT_K,
    MAX_ANSWERS_PER_QUERY,
    compute_final_score,
    evaluate_query,
    r_at_k,
    score_answers,
)
from mfe.eval.metrics.kis import KISGroundTruth, KISSubmission
from mfe.eval.metrics.qa import QAGroundTruth, QASubmission
from mfe.eval.metrics.trake import (
    TRAKEEvent,
    TRAKEGroundTruth,
    TRAKESubmission,
)


def test_pdf_worked_example() -> None:
    """The one fully worked example in the PDF, reproduced exactly.

    100 answers; r_1 = 0.5, r_3 = 0.8 (the best), r_15 = 0.6, the rest lower.
    R@1 = 0.5 and R@5 = R@20 = R@50 = R@100 = 0.8 -> (0.5 + 0.8*4)/5 = 0.74.
    """
    scores = [0.0] * 100
    scores[0] = 0.5
    scores[2] = 0.8
    scores[14] = 0.6

    assert r_at_k(scores, 1) == 0.5
    assert r_at_k(scores, 5) == 0.8
    assert r_at_k(scores, 20) == 0.8
    assert r_at_k(scores, 50) == 0.8
    assert r_at_k(scores, 100) == 0.8
    assert compute_final_score(scores) == pytest.approx(0.74)


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(1, 1.0), (5, 0.8), (6, 0.6), (20, 0.6), (21, 0.4), (100, 0.2)],
)
def test_rank_is_what_the_formula_rewards(rank: int, expected: float) -> None:
    """One perfect answer, scored at every position it could land in.

    Same answer, same retrieval set — only the rank differs. The score falls in
    steps at the five thresholds, not smoothly: ranks 6 and 20 are worth exactly
    the same, while moving 6 -> 5 gains 0.2. That is where reranking effort pays.
    """
    scores = [0.0] * (rank - 1) + [1.0] + [0.0] * (100 - rank)
    assert compute_final_score(scores) == pytest.approx(expected)


def test_short_submission_loses_the_free_terms() -> None:
    """Positions 6-100 cost nothing, so not filling them only ever loses score."""
    only_five_wrong = [0.0] * 5
    assert compute_final_score(only_five_wrong) == 0.0
    # The same five, plus a hit at rank 40 that a 5-answer submission threw away.
    with_tail = [0.0] * 39 + [1.0]
    assert compute_final_score(with_tail) == pytest.approx(0.4)


def test_empty_submission_scores_zero() -> None:
    assert r_at_k([], 1) == 0.0
    assert compute_final_score([]) == 0.0


def test_fewer_answers_than_k_is_not_an_error() -> None:
    """R@100 over 3 answers is the max of those 3 — what BTC would compute."""
    assert r_at_k([0.0, 0.6, 0.2], 100) == 0.6


def test_k_must_be_positive() -> None:
    with pytest.raises(ContractError):
        r_at_k([1.0], 0)


def test_r_score_outside_unit_interval_is_rejected() -> None:
    """Guards against a scorer returning a raw count instead of a fraction."""
    with pytest.raises(ContractError):
        r_at_k([1.5], 1)
    with pytest.raises(ContractError):
        r_at_k([-0.1], 1)


def test_default_k_matches_the_pdf() -> None:
    assert DEFAULT_K == (1, 5, 20, 50, 100)


# --- dispatch across the three query types ---------------------------------


def test_kis_end_to_end() -> None:
    gt = KISGroundTruth(video_id="L01_V001", start_frame_id=500, end_frame_id=510)
    submissions = [
        KISSubmission(video_id="L02_V003", frame_id=505),  # sai video
        KISSubmission(video_id="L01_V001", frame_id=600),  # ngoài range
        KISSubmission(video_id="L01_V001", frame_id=505),  # đúng, rank 3
    ]
    result = evaluate_query(submissions, gt, query_id="q001")

    assert result.query_type == "KIS"
    assert result.r_scores == [0.0, 0.0, 1.0]
    assert result.first_hit_rank == 3
    assert result.r_at_k[1] == 0.0
    assert result.r_at_k[5] == 1.0
    assert result.final_score == pytest.approx(0.8)


def test_qa_end_to_end() -> None:
    gt = QAGroundTruth(
        video_id="L05_V005", start_frame_id=800, end_frame_id=900, answer="màu xanh"
    )
    submissions = [
        QASubmission(video_id="L05_V005", frame_id=888, answer="màu trắng"),
        QASubmission(video_id="L05_V005", frame_id=888, answer="màu xanh"),
    ]
    result = evaluate_query(submissions, gt)
    assert result.query_type == "QA"
    assert result.r_scores == [0.0, 1.0]
    assert result.final_score == pytest.approx(0.8)


def test_trake_partial_credit_propagates() -> None:
    """TRAKE is the only type with fractional R-Scores — max must keep them."""
    gt = TRAKEGroundTruth(
        video_id="L10_V010",
        events=[
            TRAKEEvent(95, 105),
            TRAKEEvent(145, 155),
            TRAKEEvent(195, 205),
            TRAKEEvent(245, 255),
        ],
    )
    submissions = [
        TRAKESubmission(video_id="L10_V010", frame_ids=[101, 156, 203, 251]),  # 0.75
        TRAKESubmission(video_id="L99_V999", frame_ids=[101, 150, 203, 251]),  # 0.0
    ]
    result = evaluate_query(submissions, gt)
    assert result.query_type == "TRAKE"
    assert result.r_scores == [0.75, 0.0]
    assert result.r_at_k[1] == 0.75
    assert result.final_score == pytest.approx(0.75)


def test_over_the_answer_cap_is_rejected() -> None:
    gt = KISGroundTruth(video_id="L01_V001", start_frame_id=500, end_frame_id=510)
    too_many = [
        KISSubmission(video_id="L01_V001", frame_id=i)
        for i in range(MAX_ANSWERS_PER_QUERY + 1)
    ]
    with pytest.raises(ContractError):
        score_answers(too_many, gt)


def test_malformed_answer_raises_instead_of_scoring_zero() -> None:
    """A broken submission builder must not read as a weak retriever."""
    gt = TRAKEGroundTruth(video_id="L10_V010", events=[TRAKEEvent(95, 105)])
    with pytest.raises(ContractError):
        # Two frames for a one-event query: a bug, not a miss.
        score_answers([TRAKESubmission(video_id="L10_V010", frame_ids=[101, 102])], gt)


def test_mismatched_submission_type_raises() -> None:
    gt = KISGroundTruth(video_id="L01_V001", start_frame_id=500, end_frame_id=510)
    with pytest.raises(TypeError):
        score_answers([TRAKESubmission(video_id="L01_V001", frame_ids=[505])], gt)


def test_unknown_ground_truth_type_raises() -> None:
    with pytest.raises(ContractError):
        score_answers([], object())
