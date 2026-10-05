"""Coordinate ascent over fusion weights — step B3 of ke-hoach-merge-sota.md §3.4b.

Weights are measured, never argued. The one time this project set them by
reasoning from a per-arm table, the result was **worse** than leaving every arm
at 1.0 (0.585/0.452 against 0.607/0.474), because RRF reads *rank* and not score
magnitude, so "this arm is more precise, give it more weight" does not convert
into a better ranking. `config/pilot.yaml` carries that measurement so nobody
retries the argument.

Two things this module does that a naive sweep would not:

**It retrieves once and re-fuses many times.** Fusion weights change nothing
about what the arms return — only how their lists are combined. So every arm is
queried once per query, and the several hundred weight vectors are scored
against that cached evidence. A 10-arm × 6-value × 3-round ascent is ~180
evaluations; paying Elasticsearch and Milvus for each would make the search
unaffordable, and an unaffordable search is how people end up guessing instead.

**It sweeps OFF as well as 0.** This is not the grid §3.4b names. `weight = 0`
does not remove an arm — its candidates still enter the fused pool
(`weighted_fusion`) — so the two settings are genuinely different configurations
and both belong in the sweep.

Measured, the difference is narrower than "weight 0 does not disable the arm"
suggests, and it is worth writing down precisely because the loose version
invites the wrong conclusion. Fusing `good=[11,12,13,1000]` with
`harmful=[21,22,23]`:

    harmful w=1    [11, 21, 12, 22, 13, 23, 1000]   answer at rank 7
    harmful w=0    [11, 12, 13, 1000, 21, 22, 23]   answer at rank 4
    harmful OFF    [11, 12, 13, 1000]               answer at rank 4

A zero-weighted arm contributes no score, so it **cannot outrank anything a
weighted arm scored** — it appends to the tail. What it still changes is the
*contents* of the submitted list, which matters at the 100-answer cap. So for an
arm like `ocr_sviptr` (rank 1.0 alone on 3/9, noise on the other six, ablation
mean rank 2.7 → 2.4) the fix the ascent finds is usually `0.0`, with OFF tying
it. Keep OFF in the grid anyway: it is the only setting that expresses "this arm
is not part of the system", and a tie between them is information, not noise.

The ascent is **order-dependent and greedy** — it finds a local optimum, not the
best assignment. That is the intended trade (§3.4b: "vài chục lần chạy, không
phải grid search tổ hợp"), but it means the arm order is part of the result, so
it is fixed and recorded rather than left to a set's iteration order.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mfe.contracts.tools import Candidate
from mfe.eval.metrics.final_score import MAX_ANSWERS_PER_QUERY, QueryScore, evaluate_query
from mfe.eval.official import (
    AnyQuery,
    QAQuery,
    TRAKEQuery,
    to_kis_submissions,
    to_qa_submissions,
    to_trake_submissions,
)
from mfe.online.fusion import weighted_fusion

logger = logging.getLogger(__name__)

#: The grid §3.4b B3 names. `OFF` is ours — see the module docstring.
WEIGHT_GRID: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)

#: Sentinel for "take this arm out of the pool entirely", as distinct from 0.0.
OFF = "off"

Setting = float | str


@dataclass(frozen=True)
class QueryEvidence:
    """What every arm returned for one query, retrieved once.

    `per_event` holds one result-list-per-arm mapping per event: exactly one for
    KIS and QA, N for TRAKE. Keeping the shape uniform means the scorer below has
    one code path rather than a branch that only TRAKE exercises — and TRAKE is
    the type most likely to be under-tested, since it is 3 of 9 pilot queries and
    the two that score 0.0.
    """

    query: AnyQuery
    per_event: list[dict[str, list[Candidate]]]

    @property
    def arm_names(self) -> set[str]:
        return {name for event in self.per_event for name in event}


def evidence_key(config_path: str, queryset: str, coarse_topk: int, en_source: str) -> str:
    """Vân tay của một lượt truy hồi — thứ làm cache hết hạn khi cần.

    Cố ý **không** gồm trọng số: đó chính là thứ cache này cho phép quét miễn phí.
    """
    return f"{config_path}|{queryset}|{coarse_topk}|{en_source}"


def load_evidence_cache(path: str | Path, queries, key: str) -> list[QueryEvidence] | None:
    """Dựng lại `QueryEvidence` từ đĩa, hoặc None nếu cache không dùng được.

    Trả None (chứ không ném) khi vân tay lệch hoặc thiếu câu: cache cũ là chuyện
    thường và truy hồi lại là đúng phản ứng. Nhưng nó phải **nói ra** — một cache
    lặng lẽ khớp nhầm cho ra một bảng số đo của kho khác, và không có gì trong
    bảng đó lộ ra điều ấy.
    """
    path = Path(path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("key") != key:
        logger.warning("cache %s la cua luot khac (%s) — truy hoi lai", path, payload.get("key"))
        return None
    stored = payload.get("queries", {})
    missing = [q.query_id for q in queries if q.query_id not in stored]
    if missing:
        logger.warning("cache %s thieu %d cau — truy hoi lai", path, len(missing))
        return None

    evidences = []
    for query in queries:
        per_event = [
            {
                arm: [
                    Candidate(
                        keyframe_id=keyframe_id,
                        video_id=keyframe_id.split("#")[0],
                        score=float(score),
                        evidence_id=f"{arm}:cache:{rank}",
                    )
                    for rank, (keyframe_id, score) in enumerate(rows, start=1)
                ]
                for arm, rows in event.items()
            }
            for event in stored[query.query_id]
        ]
        evidences.append(QueryEvidence(query=query, per_event=per_event))
    logger.info("cache %s: dung lai %d cau, khong truy hoi lai", path, len(evidences))
    return evidences


def save_evidence_cache(path: str | Path, evidences: Sequence[QueryEvidence], key: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": key,
        "queries": {
            ev.query.query_id: [
                {arm: [[c.keyframe_id, c.score] for c in cands] for arm, cands in event.items()}
                for event in ev.per_event
            ]
            for ev in evidences
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info("da ghi cache truy hoi: %s", path)


@dataclass(frozen=True)
class Assignment:
    """One candidate configuration: a weight per arm, or OFF."""

    settings: dict[str, Setting]

    def weights(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in self.settings.items()
            if value != OFF
        }

    def disabled(self) -> set[str]:
        return {name for name, value in self.settings.items() if value == OFF}

    def with_change(self, arm: str, value: Setting) -> Assignment:
        return Assignment({**self.settings, arm: value})

    def describe(self) -> str:
        return ", ".join(
            f"{name}={'OFF' if value == OFF else value}"
            for name, value in sorted(self.settings.items())
        )


@dataclass(frozen=True)
class Measurement:
    """What one assignment scored. `wide` is the objective BTC actually reports."""

    wide: float
    strict: float | None
    #: Mean rank of the first correct answer, over the queries that found one.
    #: The handoff's tables are in this unit, and it moves when Final Score does
    #: not: with 9 queries, Final Score saturates as soon as a hit lands in the
    #: top 5, so two clearly different configurations can tie at 0.711.
    mean_first_hit_rank: float | None
    hits: int
    n_queries: int

    def objective(self, kind: str = "wide") -> float:
        if kind == "strict":
            return self.strict if self.strict is not None else 0.0
        if kind == "both":
            return (self.wide + (self.strict if self.strict is not None else 0.0)) / 2
        return self.wide

    def summary(self) -> str:
        strict = f"{self.strict:.3f}" if self.strict is not None else "-"
        rank = (
            f"{self.mean_first_hit_rank:.1f}"
            if self.mean_first_hit_rank is not None
            else "-"
        )
        return (
            f"RONG {self.wide:.3f}  HEP {strict}  hangTB {rank}  "
            f"hit {self.hits}/{self.n_queries}"
        )


def fuse_and_group(
    result_lists: dict[str, list[Candidate]],
    assignment: Assignment,
    grouper,
    normalizer: str,
    rrf_k: int,
    top_k: int,
    run_id: str = "tune",
) -> list[Candidate]:
    """Re-run everything downstream of retrieval for one assignment.

    A disabled arm is dropped from `result_lists` before fusion rather than
    zero-weighted — that *is* the difference the OFF setting exists to express.
    """
    kept = {
        name: candidates
        for name, candidates in result_lists.items()
        if name not in assignment.disabled()
    }
    if not kept:
        return []
    fused = weighted_fusion(
        kept,
        weights=assignment.weights(),
        run_id=run_id,
        normalizer=normalizer,
        rrf_k=rrf_k,
    )
    return grouper(fused)[:top_k]


def score_evidence(
    evidence: QueryEvidence,
    assignment: Assignment,
    relational,
    grouper,
    normalizer: str,
    rrf_k: int,
    top_k: int,
) -> tuple[QueryScore, QueryScore | None]:
    """Score one query under one assignment, against both ground-truth windows."""
    ranked = [
        fuse_and_group(
            event, assignment, grouper, normalizer, rrf_k, top_k, run_id=evidence.query.query_id
        )
        for event in evidence.per_event
    ]

    query = evidence.query
    if isinstance(query, TRAKEQuery):
        submissions: list[object] = list(
            to_trake_submissions(ranked, relational, limit=top_k)
        )
    elif isinstance(query, QAQuery):
        submissions = list(
            to_qa_submissions(ranked[0], relational, query.answer(), limit=top_k)
        )
    else:
        submissions = list(to_kis_submissions(ranked[0], relational, limit=top_k))

    wide = evaluate_query(submissions, query.ground_truth(), query_id=query.query_id)
    strict_gt = query.ground_truth_strict()
    strict = (
        evaluate_query(submissions, strict_gt, query_id=query.query_id)
        if strict_gt is not None
        else None
    )
    return wide, strict


def measure(
    evidences: Sequence[QueryEvidence],
    assignment: Assignment,
    relational,
    grouper,
    normalizer: str = "rrf",
    rrf_k: int = 60,
    top_k: int = MAX_ANSWERS_PER_QUERY,
) -> Measurement:
    """Mean Final Score over the query set for one assignment."""
    wides: list[float] = []
    stricts: list[float] = []
    ranks: list[int] = []
    for evidence in evidences:
        wide, strict = score_evidence(
            evidence, assignment, relational, grouper, normalizer, rrf_k, top_k
        )
        wides.append(wide.final_score)
        if strict is not None:
            stricts.append(strict.final_score)
        if wide.first_hit_rank is not None:
            ranks.append(wide.first_hit_rank)

    return Measurement(
        wide=sum(wides) / len(wides) if wides else 0.0,
        strict=sum(stricts) / len(stricts) if stricts else None,
        mean_first_hit_rank=sum(ranks) / len(ranks) if ranks else None,
        hits=len(ranks),
        n_queries=len(evidences),
    )


@dataclass
class Step:
    """One arm's sweep: everything tried, and what was kept."""

    round_index: int
    arm: str
    tried: list[tuple[Setting, Measurement]] = field(default_factory=list)
    before: Setting = 1.0
    after: Setting = 1.0

    @property
    def changed(self) -> bool:
        return self.before != self.after


@dataclass
class AscentResult:
    baseline: Measurement
    best: Measurement
    assignment: Assignment
    steps: list[Step]
    rounds_run: int
    objective: str


def _pick_best(
    incumbent: Setting,
    tried: list[tuple[Setting, Measurement]],
    objective: str,
) -> Setting:
    """Best setting, with ties broken conservatively.

    Two tie-breaks, both to keep 9 pilot queries from producing false movement:

    1. **The incumbent wins a tie.** Final Score saturates — a hit anywhere in
       the top 5 already scores 0.8 — so exact ties are common and a naive `max`
       would wander between equally-good values run to run.
    2. Otherwise the value closest to 1.0 wins, and OFF loses every tie it is in.
       Removing an arm on a tie is a real change to the system justified by no
       measured gain.
    """
    best_score = max(measurement.objective(objective) for _, measurement in tried)
    winners = [
        setting
        for setting, measurement in tried
        if measurement.objective(objective) == best_score
    ]
    if incumbent in winners:
        return incumbent
    numeric = [value for value in winners if value != OFF]
    if numeric:
        return min(numeric, key=lambda value: (abs(float(value) - 1.0), float(value)))
    return winners[0]


def coordinate_ascent(
    evidences: Sequence[QueryEvidence],
    start: Assignment,
    arm_order: Sequence[str],
    relational,
    grouper,
    grid: Sequence[float] = WEIGHT_GRID,
    allow_off: bool = True,
    rounds: int = 3,
    objective: str = "wide",
    normalizer: str = "rrf",
    rrf_k: int = 60,
    top_k: int = MAX_ANSWERS_PER_QUERY,
    on_step=None,
) -> AscentResult:
    """Sweep one arm at a time, keep the best, repeat — §3.4b B3.

    Stops early when a whole round changes nothing, which is the honest signal
    that the local optimum is reached; running the remaining rounds would only
    re-measure the same numbers.
    """
    settings: Sequence[Setting] = ([OFF, *grid] if allow_off else list(grid))

    def evaluate(assignment: Assignment) -> Measurement:
        return measure(
            evidences, assignment, relational, grouper, normalizer, rrf_k, top_k
        )

    baseline = evaluate(start)
    current = start
    current_score = baseline
    steps: list[Step] = []
    rounds_run = 0

    for round_index in range(1, rounds + 1):
        rounds_run = round_index
        changed_this_round = False
        for arm in arm_order:
            incumbent = current.settings.get(arm, 1.0)
            step = Step(round_index=round_index, arm=arm, before=incumbent)
            for value in settings:
                trial = current.with_change(arm, value)
                measurement = (
                    current_score if value == incumbent else evaluate(trial)
                )
                step.tried.append((value, measurement))
            chosen = _pick_best(incumbent, step.tried, objective)
            step.after = chosen
            steps.append(step)
            if on_step is not None:
                on_step(step)
            if chosen != incumbent:
                changed_this_round = True
                current = current.with_change(arm, chosen)
                current_score = next(
                    measurement for value, measurement in step.tried if value == chosen
                )
        if not changed_this_round:
            logger.info("round %d changed nothing; stopping", round_index)
            break

    return AscentResult(
        baseline=baseline,
        best=current_score,
        assignment=current,
        steps=steps,
        rounds_run=rounds_run,
        objective=objective,
    )


def ablation(
    evidences: Sequence[QueryEvidence],
    start: Assignment,
    arm_order: Sequence[str],
    relational,
    grouper,
    **kwargs: Any,
) -> list[tuple[str, Measurement]]:
    """Full set, then the full set minus each arm in turn.

    Answers "is this arm worth keeping", which is a different question from
    "where does this arm rank the answer alone" — and on this data the two give
    opposite answers for `ocr_sviptr` (rank 1.0 alone, negative contribution in
    the fusion). Reported as `[("<full>", …), ("-arm", …), …]`.
    """
    rows = [("<full>", measure(evidences, start, relational, grouper, **kwargs))]
    for arm in arm_order:
        without = start.with_change(arm, OFF)
        rows.append((f"-{arm}", measure(evidences, without, relational, grouper, **kwargs)))
    return rows
