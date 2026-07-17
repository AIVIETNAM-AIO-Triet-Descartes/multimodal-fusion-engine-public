"""Metrics — defined once here, referenced everywhere else. docs/00 §6.

    Recall@K        Is the right answer in the top K?          Lead (harness)
    TTCA            Query received -> answer ready to submit    Lead + TV3
    Latency         One retrieval/rerank call                   TV3
    Throughput      Seconds per hour of video (offline)         TV1 (offline/throughput.py)
    Mapping acc.    % of submits with the right ID/frame/ts     TV4
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class QueryRecord(BaseModel):
    """One line of `eval/queryset.jsonl` — docs/01 §2, docs/05 §4.

    Produced by TV4, who understands the mapping and ground truth best. This is
    the TV4 <-> Lead sync point in docs/01 §1: get it wrong and the harness
    cannot score anything.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    text: str
    gt_keyframe_ids: list[str]
    gt_video_id: str
    type: str  # KIS first; multi-step QA later (docs/05 §4)


def recall_at_k(ranked: list[str], ground_truth: list[str], k: int) -> float:
    """Is any ground-truth keyframe in the top K? docs/00 §6.

    Owner: Lead. TODO(Lead): implement.

    Decide and document what counts as a hit, because docs/context/buoi-1.md §7
    says BTC may treat a *range* of contiguous frames as correct — so an exact
    keyframe_id match may be stricter than the real scoring, and would
    understate the system. Neighbour tolerance needs the Relational Map, not
    string equality.
    """
    raise NotImplementedError("TODO(Lead): recall@k")


def compute_ttca(events: list[dict]) -> float:
    """Seconds from receiving a query to having a correct answer ready. docs/00 §6.

    Owner: Lead. TODO(Lead): implement — from QUERY_RECEIVED to ANSWER_READY in
    the event log (`mfe.core.logging`).

    This, not vector-search latency, is the number to optimize
    (docs/context/buoi-2.md §2). It includes the human: a model that lifts
    Recall but slows the operator can lower the total score (docs/04 §5).

    Use the monotonic timestamps in the log, not wall clock.
    """
    raise NotImplementedError("TODO(Lead): TTCA from event log")


def summarize(rows: list[dict]) -> dict:
    """Aggregate a run: median TTCA, mean Recall@K by type, valid-submit share.

    Owner: Lead. TODO(Lead): implement.

    Median TTCA, not mean — docs/01 §2 asks for median, and one query where the
    operator gave up would drag a mean into meaninglessness.
    """
    raise NotImplementedError("TODO(Lead): aggregate per-query rows")
