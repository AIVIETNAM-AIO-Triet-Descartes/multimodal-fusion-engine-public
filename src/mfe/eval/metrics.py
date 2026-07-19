"""Metrics — defined once here, referenced everywhere else. docs/00 §6.

    Recall@K        Is the right answer in the top K?          Vỹ Triết (harness)
    TTCA            Query received -> answer ready to submit    Vỹ Triết + Phạm Trọng
    Latency         One retrieval/rerank call                   Phạm Trọng
    Throughput      Seconds per hour of video (offline)         Quốc Đạt (offline/throughput.py)
    Mapping acc.    % of submits with the right ID/frame/ts     Nhật Bảo
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class QueryRecord(BaseModel):
    """One line of `eval/queryset.jsonl` — docs/01 §2, docs/05 §4.

    Produced by Nhật Bảo, who understands the mapping and ground truth best. This is
    the Nhật Bảo <-> Vỹ Triết sync point in docs/01 §1: get it wrong and the harness
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

    Owner: Vỹ Triết. TODO(Vỹ Triết): implement.

    Decide and document what counts as a hit, because docs/context/buoi-1.md §7
    says BTC may treat a *range* of contiguous frames as correct — so an exact
    keyframe_id match may be stricter than the real scoring, and would
    understate the system. Neighbour tolerance needs the Relational Map, not
    string equality.
    """
    raise NotImplementedError("TODO(Vỹ Triết): recall@k")


def compute_ttca(events: list[dict]) -> float:
    """Seconds from receiving a query to having a correct answer ready. docs/00 §6.

    Owner: Vỹ Triết. TODO(Vỹ Triết): implement — from QUERY_RECEIVED to ANSWER_READY in
    the event log (`mfe.core.logging`).

    This, not vector-search latency, is the number to optimize
    (docs/context/buoi-2.md §2). It includes the human: a model that lifts
    Recall but slows the operator can lower the total score (docs/04 §5).

    Use the monotonic timestamps in the log, not wall clock.
    """
    raise NotImplementedError("TODO(Vỹ Triết): TTCA from event log")


def summarize(rows: list[dict]) -> dict:
    """Aggregate a run: median TTCA, mean Recall@K by type, valid-submit share.

    Owner: Vỹ Triết. TODO(Vỹ Triết): implement.

    Median TTCA, not mean — docs/01 §2 asks for median, and one query where the
    operator gave up would drag a mean into meaninglessness.
    """
    raise NotImplementedError("TODO(Vỹ Triết): aggregate per-query rows")
