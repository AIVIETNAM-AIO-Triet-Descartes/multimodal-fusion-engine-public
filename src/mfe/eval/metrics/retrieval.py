"""Metrics — defined once here, referenced everywhere else. docs/00 §6.

    Recall@K        Is the right answer in the top K?          Vỹ Triết (harness)
    TTCA            Query received -> answer ready to submit    Vỹ Triết + Phạm Trọng
    Latency         One retrieval/rerank call                   Phạm Trọng
    Throughput      Seconds per hour of video (offline)         Quốc Đạt (offline/throughput.py)
    Mapping acc.    % of submits with the right ID/frame/ts     Nhật Bảo
"""

from __future__ import annotations

import statistics

from pydantic import BaseModel, ConfigDict

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.ids import parse_keyframe_id
from mfe.core.logging import EventType

#: How many keyframes either side of a ground-truth keyframe still count as a
#: hit. docs/context/buoi-1.md §7: BTC may treat a *range* of contiguous frames
#: as correct, so exact keyframe_id equality would understate the system.
#: 0 = exact match only, which is what Sprint 1 reports.
DEFAULT_NEIGHBOR_TOLERANCE = 0


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

    @property
    def has_ground_truth(self) -> bool:
        """Whether this query can be scored at all.

        The Sprint 1 queryset ships with empty `gt_keyframe_ids` — they can only
        be filled once ingest has produced real ids. Scoring an unfilled query as
        a miss would report a Recall of 0 that says nothing about retrieval, so
        the harness counts these separately instead of averaging them in.
        """
        return bool(self.gt_keyframe_ids)


def recall_at_k(
    ranked: list[str],
    ground_truth: list[str],
    k: int,
    tolerance: int = DEFAULT_NEIGHBOR_TOLERANCE,
) -> float:
    """Is any ground-truth keyframe in the top K? docs/00 §6. Owner: Vỹ Triết.

    Returns 1.0 or 0.0 — for KIS there is one right answer, so "how many of the
    relevant items did we retrieve" collapses to "did we get it".

    What counts as a hit is a decision, not an implementation detail. With
    `tolerance=0` it is exact `keyframe_id` equality. With `tolerance=n` a
    keyframe of the same video whose `keyframe_index` is within n also counts,
    because docs/context/buoi-1.md §7 says BTC may accept a contiguous range —
    so exact matching can report a miss on an answer that would have scored.

    The tolerance is read off the id itself rather than the Relational Map:
    `keyframe_index` proximity is what "adjacent keyframe" means here, and going
    through the map would make a pure function depend on an open store.
    """
    if k <= 0:
        raise ContractError(ErrorCode.INVALID_ARGUMENT, f"k must be > 0, got {k}")
    if not ground_truth:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            "recall_at_k needs at least one ground-truth keyframe_id",
        )

    top = ranked[:k]
    if any(candidate in ground_truth for candidate in top):
        return 1.0
    if tolerance <= 0:
        return 0.0

    targets: list[tuple[str, int]] = []
    for keyframe_id in ground_truth:
        try:
            targets.append(parse_keyframe_id(keyframe_id))
        except ContractError:
            continue

    for candidate in top:
        try:
            video_id, index = parse_keyframe_id(candidate)
        except ContractError:
            continue
        for target_video, target_index in targets:
            if video_id == target_video and abs(index - target_index) <= tolerance:
                return 1.0
    return 0.0


def compute_ttca(events: list[dict]) -> float | None:
    """Seconds from receiving a query to having a correct answer ready. docs/00 §6.

    Owner: Vỹ Triết. Measured from QUERY_RECEIVED to ANSWER_READY in the event
    log (`mfe.core.logging`), using the monotonic timestamps — a wall-clock
    adjustment mid-session would corrupt the number.

    This, not vector-search latency, is what to optimize
    (docs/context/buoi-2.md §2). It includes the human: a model that lifts
    Recall but slows the operator can lower the total score (docs/04 §5).

    Returns None when the query never reached ANSWER_READY. That is a real
    outcome — the operator gave up — and it must not enter a median as if it
    had merely been fast.
    """
    started_at: float | None = None
    for event in events:
        name = event.get("event")
        if name == EventType.QUERY_RECEIVED:
            started_at = event.get("ts_monotonic")
        elif name == EventType.ANSWER_READY and started_at is not None:
            ended_at = event.get("ts_monotonic")
            if ended_at is None:
                return None
            return float(ended_at) - float(started_at)
    return None


def _mean_recall(rows: list[dict], keys: list[str]) -> dict[str, float]:
    return {
        key: round(statistics.fmean([row[key] for row in rows if key in row]), 4)
        for key in keys
        if any(key in row for row in rows)
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate a run: median TTCA, mean Recall@K by type, valid-submit share.

    Owner: Vỹ Triết.

    Median TTCA, not mean — docs/01 §2 asks for median, and one query where the
    operator gave up would drag a mean into meaninglessness. Queries with no
    TTCA are excluded from the median and counted separately, so a run where
    half the queries never resolved cannot read as a fast run.
    """
    empty = {
        "queries": 0,
        "scored_queries": 0,
        "crashed": 0,
        "crashed_query_ids": [],
        "recall": {},
        "recall_by_type": {},
        "ttca_s": {"median": None, "resolved": 0, "unresolved": 0},
        "latency_ms": {"median": None},
        "submit_valid_share": None,
    }
    if not rows:
        return empty

    crashed = [row for row in rows if row.get("error")]
    scored = [row for row in rows if row.get("scored")]

    recall_keys = sorted(
        {key for row in scored for key in row if key.startswith("recall_at_")},
        key=lambda key: int(key.rsplit("_", 1)[1]),
    )

    recall_by_type: dict[str, dict[str, float]] = {}
    for query_type in sorted({row.get("type", "unknown") for row in scored}):
        subset = [row for row in scored if row.get("type", "unknown") == query_type]
        recall_by_type[query_type] = _mean_recall(subset, recall_keys)

    ttca_values = [row["ttca_s"] for row in rows if row.get("ttca_s") is not None]
    latencies = [row["latency_ms"] for row in rows if row.get("latency_ms") is not None]
    submit_flags = [row["submit_valid"] for row in rows if "submit_valid" in row]

    return {
        "queries": len(rows),
        "scored_queries": len(scored),
        "crashed": len(crashed),
        "crashed_query_ids": [row.get("query_id") for row in crashed],
        "recall": _mean_recall(scored, recall_keys),
        "recall_by_type": recall_by_type,
        "ttca_s": {
            "median": round(statistics.median(ttca_values), 4) if ttca_values else None,
            "resolved": len(ttca_values),
            "unresolved": len(rows) - len(ttca_values),
        },
        "latency_ms": {
            "median": round(statistics.median(latencies), 3) if latencies else None,
        },
        "submit_valid_share": (
            round(sum(1 for flag in submit_flags if flag) / len(submit_flags), 4)
            if submit_flags
            else None
        ),
    }
