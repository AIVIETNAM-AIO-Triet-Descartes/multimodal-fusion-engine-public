"""R@k and Final Score — the official aggregation, docs/metrics-btc.html §5.

Formula (PDF §2.2). A team submits up to 100 ranked answers per query::

    R@k         = max( R-Score(r_1), ..., R-Score(r_k) )     k in {1,5,20,50,100}
    Final Score = (1/5) * sum over those five k

Two things follow from `max`, and they drive how the harness should be read:

* **Rank matters, presence does not.** The same correct answer at position 1
  scores 1.00 and at position 6 scores 0.80. Four of the five terms are already
  won once the answer is anywhere in the top 5, so the marginal value of
  re-ranking within the tail is nearly zero, while moving the first hit from
  rank 6 to rank 1 is worth 0.2 of the total.
* **Submitting fewer than 100 answers throws away free score.** Positions 6-100
  cost nothing and can only raise `R@20`, `R@50`, `R@100`. This is why
  `retrieval.rerank_topk` is 100 in `config/local.yaml`.

Layering: this module never inspects a keyframe. It calls the three per-answer
scorers (`score_kis` / `score_qa` / `score_trake`) written by Đạt, Quang and
Trọng, and does arithmetic on the floats they return.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.eval.metrics.kis import KISGroundTruth, score_kis
from mfe.eval.metrics.qa import QAGroundTruth, score_qa
from mfe.eval.metrics.trake import TRAKEGroundTruth, score_trake

__all__ = (
    "DEFAULT_K",
    "MAX_ANSWERS_PER_QUERY",
    "QueryScore",
    "compute_final_score",
    "evaluate_query",
    "r_at_k",
    "score_answers",
)

#: The five thresholds of PDF §2.2. Not tunable — changing them changes what
#: "Final Score" means, so an experiment that wants other k values should read
#: `QueryScore.r_at_k` rather than pass a different tuple here.
DEFAULT_K: tuple[int, ...] = (1, 5, 20, 50, 100)

#: PDF §2.2 caps a submission at 100 answers. Exceeding it is not clipped
#: silently: BTC would judge the first 100, so a harness that scored 150 would
#: report a number the real leaderboard cannot produce.
MAX_ANSWERS_PER_QUERY = 100


class QueryScore(BaseModel):
    """Final Score for one query, with the five terms that produced it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str | None = None
    query_type: str
    n_answers: int
    r_scores: list[float]
    r_at_k: dict[int, float]
    final_score: float

    @property
    def first_hit_rank(self) -> int | None:
        """1-based rank of the first answer with a non-zero R-Score.

        This, not the Final Score itself, is the number to look at when tuning:
        it says exactly which of the five terms are being lost.
        """
        for rank, score in enumerate(self.r_scores, 1):
            if score > 0.0:
                return rank
        return None


def score_answers(
    submissions: Sequence[object], gt: KISGroundTruth | QAGroundTruth | TRAKEGroundTruth
) -> list[float]:
    """Score a ranked answer list against one ground truth, in submitted order.

    Dispatch is on the type of `gt`, not on a string tag, so a KIS answer paired
    with a TRAKE ground truth fails inside the per-answer scorer instead of
    being scored by the wrong formula.

    Exceptions from the per-answer scorers are **not** swallowed. A malformed
    answer (wrong record type, or a TRAKE answer with the wrong number of
    frames) is a bug in whatever built the submission; converting it to `0.0`
    here would make it indistinguishable from a genuine miss and let a broken
    submission builder read as a weak retriever.
    """
    if isinstance(gt, KISGroundTruth):
        scorer = score_kis
    elif isinstance(gt, QAGroundTruth):
        scorer = score_qa
    elif isinstance(gt, TRAKEGroundTruth):
        scorer = score_trake
    else:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"unknown ground-truth type {type(gt).__name__}; expected KIS, QA or TRAKE",
        )

    if len(submissions) > MAX_ANSWERS_PER_QUERY:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"{len(submissions)} answers submitted; BTC accepts at most "
            f"{MAX_ANSWERS_PER_QUERY} per query (PDF §2.2)",
        )

    # The record type is not checked here: each per-answer scorer already
    # rejects a submission of the wrong type, and duplicating that check would
    # give the same mistake two different error messages.
    return [float(scorer(submission, gt)) for submission in submissions]


def r_at_k(r_scores: Sequence[float], k: int) -> float:
    """Best R-Score among the first `k` answers — PDF §2.2.

    Not to be confused with :func:`mfe.eval.metrics.retrieval.recall_at_k`, which
    answers "is the ground-truth keyframe id in the top k" for our internal
    retrieval benchmark. This one aggregates R-Scores already computed by the
    official per-answer formula, and is what the leaderboard uses.

    Fewer than `k` answers is not an error — it scores the max over what was
    actually submitted, exactly as BTC would. An empty list scores `0.0`.
    """
    if k <= 0:
        raise ContractError(ErrorCode.INVALID_ARGUMENT, f"k must be > 0, got {k}")
    window = r_scores[:k]
    if not window:
        return 0.0
    for score in window:
        if not 0.0 <= score <= 1.0:
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT, f"R-Score must be in [0.0, 1.0], got {score}"
            )
    return float(max(window))


def compute_final_score(
    r_scores: Sequence[float], ks: Sequence[int] = DEFAULT_K
) -> float:
    """Mean of R@k over the five official thresholds — PDF §2.2.

    Takes R-Scores rather than submissions so the aggregation can be tested
    against the PDF's worked example without constructing any records, and so
    it is reusable by whatever produced those scores.
    """
    if not ks:
        raise ContractError(ErrorCode.INVALID_ARGUMENT, "ks must not be empty")
    return sum(r_at_k(r_scores, k) for k in ks) / len(ks)


def evaluate_query(
    submissions: Sequence[object],
    gt: KISGroundTruth | QAGroundTruth | TRAKEGroundTruth,
    query_id: str | None = None,
    ks: Sequence[int] = DEFAULT_K,
) -> QueryScore:
    """Score one query end to end: per-answer -> R@k -> Final Score."""
    query_type = {
        KISGroundTruth: "KIS",
        QAGroundTruth: "QA",
        TRAKEGroundTruth: "TRAKE",
    }.get(type(gt), type(gt).__name__)

    r_scores = score_answers(submissions, gt)
    per_k = {k: r_at_k(r_scores, k) for k in ks}
    return QueryScore(
        query_id=query_id,
        query_type=query_type,
        n_answers=len(r_scores),
        r_scores=r_scores,
        r_at_k=per_k,
        final_score=sum(per_k.values()) / len(per_k),
    )
