"""Metrics — one import site for two different families of number.

**Official BTC scoring** (`kis`, `qa`, `trake`, `final_score`): reproduces the
formulas of PDF §2.1-2.2 exactly. Each per-answer scorer is a pure function of
`(submission, ground_truth)`; `final_score` aggregates them into R@k and Final
Score. This is what the leaderboard computes.

**Internal retrieval benchmark** (`retrieval`): Recall@K over keyframe ids,
TTCA, latency, and the run summary of docs/01 §2. These measure our own pipeline
during development and have no BTC counterpart.

Keeping both under one package is deliberate — they get compared in the same
report — but the two `@k` functions are *not* interchangeable, so they keep
different names: `r_at_k` (official, max over R-Scores) and `recall_at_k`
(internal, id membership in the top k).

Note for anyone moving files here: `metrics` was a module and became a package.
A `metrics.py` left beside this directory would be shadowed by it and silently
stop being imported, which is exactly how `run.py` briefly lost `summarize`.
"""

from mfe.eval.metrics.final_score import (
    DEFAULT_K,
    MAX_ANSWERS_PER_QUERY,
    QueryScore,
    compute_final_score,
    evaluate_query,
    r_at_k,
    score_answers,
)
from mfe.eval.metrics.kis import KISGroundTruth, KISSubmission, score_kis
from mfe.eval.metrics.qa import QAGroundTruth, QASubmission, answer_match, score_qa
from mfe.eval.metrics.retrieval import (
    DEFAULT_NEIGHBOR_TOLERANCE,
    QueryRecord,
    compute_ttca,
    recall_at_k,
    summarize,
)
from mfe.eval.metrics.trake import (
    TRAKEEvent,
    TRAKEGroundTruth,
    TRAKESubmission,
    score_trake,
)

__all__ = [
    # --- official BTC scoring -------------------------------------------------
    "DEFAULT_K",
    "MAX_ANSWERS_PER_QUERY",
    "KISGroundTruth",
    "KISSubmission",
    "QAGroundTruth",
    "QASubmission",
    "QueryScore",
    "TRAKEEvent",
    "TRAKEGroundTruth",
    "TRAKESubmission",
    "answer_match",
    "compute_final_score",
    "evaluate_query",
    "r_at_k",
    "score_answers",
    "score_kis",
    "score_qa",
    "score_trake",
    # --- internal retrieval benchmark ----------------------------------------
    "DEFAULT_NEIGHBOR_TOLERANCE",
    "QueryRecord",
    "compute_ttca",
    "recall_at_k",
    "summarize",
]
