"""Score a real retrieval run with the official BTC metrics.

    python -m mfe.eval.official --config config/sample2026.yaml \
        --queryset eval/queryset_btc_2026.jsonl

This is the piece that connects the two halves the team built separately:
`online/cascade.py` produces ranked `Candidate`s, and
`eval/metrics/{kis,qa,trake}.py` score `(submission, ground_truth)` pairs. They
speak different languages, and this module is the translation:

    Candidate(keyframe_id="L23_V001#008")
        -> RelationalMap.get_keyframe  -> frame_idx 375
        -> KISSubmission(video_id="L23_V001", frame_id=375)

Two things are deliberate about that arrow.

**`frame_idx` is read, never recomputed.** The keyframe's real source frame
comes out of `map-keyframes/*.csv` at ingest time and is stored on the row. Our
own docs once carried three mutually inconsistent worked examples of the
frame/timestamp formula (adr/0002); deriving at submit time is exactly how that
class of bug reaches a scoreboard.

**BTC calls the field `frame_id`, we call it `frame_idx`.** The rename happens
here, once, at the boundary — not inside the scorers, which use BTC's name, and
not inside the Relational Map, which uses the dataset's.

`mfe.eval.run` remains the internal benchmark (Recall@K over keyframe ids, TTCA,
exit criteria). This module answers a different question: what would the
leaderboard say. Keep both — the internal one is far more sensitive during
development, because Final Score saturates at 0.8 as soon as anything correct
lands in the top 5.

Two ground-truth windows, and why one is not enough
----------------------------------------------------
Every query carries a **scene span** (`gt_frame_range`) and, optionally, an
**anchor frame** (`gt_anchor_frame`). Both are source-video frame numbers, so
neither depends on `keyframe_id` — but they behave very differently when the
keyframe extraction changes, and that is the point.

* **Wide** = the scene span: every frame in which the answer is visible.
  A scene runs a few seconds, so *any* reasonable extraction samples inside it.
  This score is therefore comparable across extraction strategies, which is what
  you want when asking "did retrieval improve".
* **Strict** = `anchor ± STRICT_HALF_WINDOW`, the ~11-frame window BTC actually
  uses (PDF §2.1.1). It is extraction-*dependent* on purpose: if our keyframes
  are too sparse to land in an 11-frame window, we genuinely lose those points
  at the competition, and this is the number that says so.

Measured, which is why both exist: with ground truth written as `anchor ± 5`
against BTC's provided keyframes, re-extracting at every 30 frames — three times
denser than the provided set — leaves **3 of 4 windows unreachable**. A one-window
harness would have reported that denser extraction as a regression.

A wide score far above the strict score means "we find the moment, our sampling
is too coarse to prove it". Those are different problems with different fixes.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.tools import Candidate
from mfe.core.cli import setup_console
from mfe.core.config import Config, load_config
from mfe.eval.metrics.final_score import (
    MAX_ANSWERS_PER_QUERY,
    QueryScore,
    evaluate_query,
)
from mfe.eval.metrics.kis import KISGroundTruth, KISSubmission
from mfe.eval.metrics.qa import QAGroundTruth, QASubmission
from mfe.eval.metrics.trake import (
    TRAKEEvent,
    TRAKEGroundTruth,
    TRAKESubmission,
)
from mfe.mapping.relational import RelationalMap
from mfe.online.cascade import Cascade

logger = logging.getLogger(__name__)

_FrameId = Annotated[int, Field(ge=0)]


# --------------------------------------------------------------------------
# Query set — one JSON object per line, discriminated by `type`
# --------------------------------------------------------------------------


#: `[start, end]` inclusive, in source-video frames. One field instead of two
#: because a hand-written query set is edited by people: `gt_frame_range` cannot
#: be filled in with start and end swapped the way two sibling keys can.
_FrameRange = tuple[_FrameId, _FrameId]


class _Query(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    #: Who wrote it. Also the key `merge_queryset` uses to check that a query
    #: only refers to videos in that person's own sample.
    author: str | None = None
    #: "dev" or "heldout". Must agree with the directory the line came from —
    #: `merge_queryset` refuses a line whose `split` contradicts its location,
    #: because a held-out query that says "dev" is one that gets debugged.
    split: Literal["dev", "heldout"] | None = None
    note: str | None = Field(
        default=None, description="How the ground truth was established"
    )


#: Half-width of the strict window built around an anchor frame. BTC's own
#: worked example scores `505` against `[500, 510]` (PDF §2.1.1) — 11 frames —
#: and §2.1.3 calls TRAKE windows "thường dưới 10 frame". So 5 is not a knob to
#: tune; it is what the competition looks like. Derived from the anchor rather
#: than written per query so the two windows can never disagree by hand.
STRICT_HALF_WINDOW = 5


def _check_range(query_id: str, frame_range: _FrameRange) -> _FrameRange:
    start, end = frame_range
    if start > end:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"{query_id}: gt_frame_range start {start} is after end {end}",
        )
    return frame_range


def _strict_range(
    query_id: str, frame_range: _FrameRange, anchor: int
) -> _FrameRange:
    """The BTC-sized window around `anchor`, checked against the wide one.

    An anchor outside its own scene span is a contradiction in the annotation:
    one of the two numbers describes a different moment than the other, and
    scoring either would silently believe the wrong one.
    """
    start, end = frame_range
    if not start <= anchor <= end:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"{query_id}: anchor frame {anchor} is outside its own scene span "
            f"[{start}, {end}]",
        )
    return (max(0, anchor - STRICT_HALF_WINDOW), anchor + STRICT_HALF_WINDOW)


class KISQuery(_Query):
    """One Textual KIS query and its accepted answer interval."""

    type: Literal["KIS"]
    text: str
    gt_video_id: str
    #: The **scene span**: every frame in which the answer is visible. Wide on
    #: purpose — see the module docstring on the two windows.
    gt_frame_range: _FrameRange
    #: The one frame the annotator is certain about, used to derive the strict
    #: window. Optional; without it the query is only scored on the wide window.
    gt_anchor_frame: _FrameId | None = None

    def ground_truth(self) -> KISGroundTruth:
        start, end = _check_range(self.query_id, self.gt_frame_range)
        return KISGroundTruth(
            video_id=self.gt_video_id, start_frame_id=start, end_frame_id=end
        )

    def ground_truth_strict(self) -> KISGroundTruth | None:
        if self.gt_anchor_frame is None:
            return None
        start, end = _strict_range(
            self.query_id, self.gt_frame_range, self.gt_anchor_frame
        )
        return KISGroundTruth(
            video_id=self.gt_video_id, start_frame_id=start, end_frame_id=end
        )


class QAQuery(_Query):
    """One Q&A query: retrieve the frame *and* state the answer."""

    type: Literal["QA"]
    #: The question as BTC would pose it.
    text: str
    #: What we actually send to the retriever. A BTC Q&A query is a *question*
    #: ("mặc áo dài màu gì"), but CLIP was trained on captions, not questions —
    #: on this sample the question form pushed the correct keyframe out of the
    #: top 100 while a descriptive rewrite of the same scene put it in. Rewriting
    #: the question into a description is the agent layer's job (Buổi 3); until
    #: that exists the rewrite is written down here rather than pretended away.
    #: Left unset, the question itself is used.
    retrieval_text: str | None = None
    #: What our system answers. Sprint 1 has no VQA reader — that is the Buổi 3
    #: agent layer — so when this is left unset the harness submits `gt_answer`
    #: itself. That deliberately scores the retrieval half alone, because the
    #: reading half does not exist yet: a 0.0 caused by having no reader would
    #: be indistinguishable from a 0.0 caused by finding the wrong frame.
    #: **The number is therefore an upper bound on QA** until a reader lands.
    #: Whoever writes the query set should leave it unset.
    system_answer: str | None = None
    gt_video_id: str
    #: Scene span — see `KISQuery.gt_frame_range`.
    gt_frame_range: _FrameRange
    gt_anchor_frame: _FrameId | None = None
    gt_answer: str

    def answer(self) -> str:
        return self.system_answer if self.system_answer is not None else self.gt_answer

    def _at(self, frame_range: _FrameRange) -> QAGroundTruth:
        start, end = frame_range
        return QAGroundTruth(
            video_id=self.gt_video_id,
            start_frame_id=start,
            end_frame_id=end,
            answer=self.gt_answer,
        )

    def ground_truth(self) -> QAGroundTruth:
        return self._at(_check_range(self.query_id, self.gt_frame_range))

    def ground_truth_strict(self) -> QAGroundTruth | None:
        if self.gt_anchor_frame is None:
            return None
        return self._at(
            _strict_range(self.query_id, self.gt_frame_range, self.gt_anchor_frame)
        )


class TRAKEQuery(_Query):
    """One TRAKE query: a described sequence, and N accepted intervals."""

    type: Literal["TRAKE"]
    #: The query as BTC would pose it — **one** sentence describing the whole
    #: sequence, not a list. This is what the system actually receives.
    text: str
    #: Our decomposition of `text` into one search string per event, in temporal
    #: order. Optional, and optional on purpose: splitting a sequence
    #: description into its steps is the agent layer's job (Buổi 3), and until
    #: that exists the split is written down here rather than pretended away.
    #: Left unset, every event searches with `text`, which finds the right video
    #: but gives the assembler almost nothing to tell the events apart — the
    #: honest floor for TRAKE, and a number to beat rather than to report.
    #: Whoever writes the query set may leave it unset.
    event_texts: list[str] | None = None
    gt_video_id: str
    #: Scene span per event, in temporal order. Its length defines N.
    gt_events: list[_FrameRange] = Field(min_length=1)
    #: One anchor frame per event, same order and length as `gt_events`.
    #: Optional; without it the query is only scored on the wide windows.
    gt_anchor_frames: list[_FrameId] | None = None

    def searches(self) -> list[str]:
        """One search string per event — the decomposition, or `text` repeated."""
        if self.event_texts:
            if len(self.event_texts) != len(self.gt_events):
                raise ContractError(
                    ErrorCode.INVALID_ARGUMENT,
                    f"{self.query_id}: {len(self.event_texts)} event text(s) but "
                    f"{len(self.gt_events)} ground-truth interval(s)",
                )
            return list(self.event_texts)
        return [self.text] * len(self.gt_events)

    def ground_truth(self) -> TRAKEGroundTruth:
        self.searches()  # raises on a length mismatch before anything is scored
        events = []
        for index, frame_range in enumerate(self.gt_events, 1):
            start, end = _check_range(f"{self.query_id} event {index}", frame_range)
            events.append(TRAKEEvent(start, end))
        return TRAKEGroundTruth(video_id=self.gt_video_id, events=events)

    def ground_truth_strict(self) -> TRAKEGroundTruth | None:
        if not self.gt_anchor_frames:
            return None
        if len(self.gt_anchor_frames) != len(self.gt_events):
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                f"{self.query_id}: {len(self.gt_anchor_frames)} anchor frame(s) but "
                f"{len(self.gt_events)} event(s) — one anchor per event, in order",
            )
        events = []
        for index, (frame_range, anchor) in enumerate(
            zip(self.gt_events, self.gt_anchor_frames, strict=True), 1
        ):
            start, end = _strict_range(
                f"{self.query_id} event {index}",
                _check_range(f"{self.query_id} event {index}", frame_range),
                anchor,
            )
            events.append(TRAKEEvent(start, end))
        return TRAKEGroundTruth(video_id=self.gt_video_id, events=events)


AnyQuery = KISQuery | QAQuery | TRAKEQuery


def load_official_queryset(path: str | Path) -> list[AnyQuery]:
    """Read the BTC-format query set, failing loudly on a malformed line."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"official queryset not found: {source}")

    adapter = _query_adapter()
    queries: list[AnyQuery] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("//"):
            continue
        try:
            queries.append(adapter.validate_json(raw))
        except Exception as err:
            raise ValueError(f"{source}:{line_number} is not a valid query: {err}") from err
    return queries


def _query_adapter():
    from pydantic import TypeAdapter

    return TypeAdapter(Annotated[AnyQuery, Field(discriminator="type")])


# --------------------------------------------------------------------------
# Candidate -> official submission
# --------------------------------------------------------------------------


def candidate_frame_id(candidate: Candidate, relational: RelationalMap) -> int:
    """The source-video frame BTC scores, read off the Relational Map.

    Raises rather than guessing when the keyframe is unknown — docs/05 §2,
    "nếu keyframe_id không có trong Relational Map -> lỗi, không đoán".
    """
    return relational.get_keyframe(candidate.keyframe_id).frame_idx


def to_kis_submissions(
    candidates: list[Candidate],
    relational: RelationalMap,
    limit: int = MAX_ANSWERS_PER_QUERY,
) -> list[KISSubmission]:
    return [
        KISSubmission(
            video_id=candidate.video_id,
            frame_id=candidate_frame_id(candidate, relational),
        )
        for candidate in candidates[:limit]
    ]


def to_qa_submissions(
    candidates: list[Candidate],
    relational: RelationalMap,
    answer: str,
    limit: int = MAX_ANSWERS_PER_QUERY,
) -> list[QASubmission]:
    return [
        QASubmission(
            video_id=candidate.video_id,
            frame_id=candidate_frame_id(candidate, relational),
            answer=answer,
        )
        for candidate in candidates[:limit]
    ]


def to_trake_submissions(
    per_event: list[list[Candidate]],
    relational: RelationalMap,
    limit: int = MAX_ANSWERS_PER_QUERY,
) -> list[TRAKESubmission]:
    """Assemble N per-event ranked lists into whole-video TRAKE answers.

    Baseline strategy, and only that. TRAKE retrieval proper — deciding *what*
    to search for each event — is out of scope (docs/metrics-btc.html §6). This
    is the assembly step that turns N independent searches into answers of the
    shape BTC scores.

        1. A TRAKE answer names one video, so candidates are grouped by video
           and each video yields at most one answer.
        2. Within a video the N frames must be strictly increasing: the events
           are a sequence, so event j+1 cannot happen before event j. The
           assignment maximising total candidate score under that constraint is
           found by dynamic programming over (event, candidate).
        3. A video with no valid increasing assignment is dropped rather than
           padded — a padded frame scores as a wrong guess and is
           indistinguishable from a real attempt.
        4. Videos are ranked by mean candidate score across the events.

    Step 2 is DP rather than greedy for a reason measured on this sample: the
    greedy version took the best-scoring candidate for event 1 first, which for
    L23_V001 was keyframe 031, and that single pick made every later event
    unsatisfiable — discarding the correct keyframes for events 2 and 4, which
    retrieval *had* found. Greedy dropped the only video that could score.
    """
    videos = {candidate.video_id for ranked in per_event for candidate in ranked}
    n_events = len(per_event)

    scored: list[tuple[float, TRAKESubmission]] = []
    for video_id in videos:
        # Per event: this video's candidates as (frame_id, score), by frame.
        options: list[list[tuple[int, float]]] = []
        for ranked in per_event:
            picks = sorted(
                (candidate_frame_id(c, relational), c.score)
                for c in ranked
                if c.video_id == video_id
            )
            options.append(picks)
        if any(not picks for picks in options):
            logger.debug("%s dropped: event with no candidate in this video", video_id)
            continue

        # best[i] = (total score, chosen frames) for event j ending at option i.
        best: list[tuple[float, list[int]] | None] = [
            (score, [frame]) for frame, score in options[0]
        ]
        for event_index in range(1, n_events):
            nxt: list[tuple[float, list[int]] | None] = []
            for frame, score in options[event_index]:
                feasible = [
                    entry
                    for entry, (previous_frame, _) in zip(
                        best, options[event_index - 1], strict=True
                    )
                    if entry is not None and previous_frame < frame
                ]
                if not feasible:
                    nxt.append(None)
                    continue
                total, frames = max(feasible, key=lambda entry: entry[0])
                nxt.append((total + score, [*frames, frame]))
            best = nxt

        complete = [entry for entry in best if entry is not None]
        if not complete:
            logger.debug("%s dropped: no strictly increasing assignment", video_id)
            continue
        total, frames = max(complete, key=lambda entry: entry[0])
        scored.append(
            (total / n_events, TRAKESubmission(video_id=video_id, frame_ids=frames))
        )

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [submission for _, submission in scored[:limit]]


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


class QueryOutcome(BaseModel):
    """One query scored against both windows — the same answers, two rulers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Scene span. Independent of how keyframes were extracted, so this is the
    #: number to compare across extraction changes.
    wide: QueryScore
    #: BTC-sized window around the anchor frame. None when the query carries no
    #: anchor. Deliberately extraction-*dependent*: it answers "is our sampling
    #: dense enough to score at the competition", which is a real constraint.
    strict: QueryScore | None = None

    @property
    def query_id(self) -> str | None:
        return self.wide.query_id

    @property
    def query_type(self) -> str:
        return self.wide.query_type


class OfficialRunResult(BaseModel):
    """One run of the official metrics over a query set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    per_query: list[QueryOutcome]
    mean_final_score: float
    #: Mean over the queries that carry an anchor. None when none do.
    mean_final_score_strict: float | None = None
    #: How many queries the strict mean is over — without it the number is
    #: unreadable, since a strict mean over 2 of 115 queries says nothing.
    strict_queries: int = 0
    by_type: dict[str, float]


def score_one(
    query: AnyQuery, cascade: Cascade, relational: RelationalMap, top_k: int
) -> QueryOutcome:
    """Search once, score against both windows.

    The search runs **once**: the answers we would submit do not depend on which
    ruler we hold them against. Running it twice would also pay stage C's ~12 s
    twice, and would let the two scores drift apart on a non-deterministic tie.
    """
    if isinstance(query, TRAKEQuery):
        per_event = [
            cascade.search(query_text=text, top_k=top_k, run_id=query.query_id)
            for text in query.searches()
        ]
        submissions: list[object] = list(
            to_trake_submissions(per_event, relational, limit=top_k)
        )
    else:
        search_text = (
            query.retrieval_text or query.text
            if isinstance(query, QAQuery)
            else query.text
        )
        candidates = cascade.search(
            query_text=search_text, top_k=top_k, run_id=query.query_id
        )
        if isinstance(query, QAQuery):
            submissions = list(
                to_qa_submissions(candidates, relational, query.answer(), limit=top_k)
            )
        else:
            submissions = list(to_kis_submissions(candidates, relational, limit=top_k))

    strict_gt = query.ground_truth_strict()
    return QueryOutcome(
        wide=evaluate_query(submissions, query.ground_truth(), query_id=query.query_id),
        strict=(
            evaluate_query(submissions, strict_gt, query_id=query.query_id)
            if strict_gt is not None
            else None
        ),
    )


def run(
    config_path: str = "config/sample2026.yaml",
    queryset_path: str | None = None,
    out_dir: str | None = None,
) -> OfficialRunResult:
    """Score a whole query set with the official metrics and write the report."""
    config: Config = load_config(config_path)
    queryset_path = queryset_path or "eval/queryset_btc_2026.jsonl"
    out_dir = out_dir or config.paths.eval_results
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    queries = load_official_queryset(queryset_path)
    top_k = min(config.retrieval.rerank_topk, MAX_ANSWERS_PER_QUERY)

    relational = RelationalMap(config.paths.relational_db)
    try:
        relational.init_schema()
        # No `reranker=None` override: stage C is a config decision now
        # (`models.reranker`), and pinning it off here is how the harness spent
        # a run reporting that a working reranker changed nothing.
        cascade = Cascade.from_config(config)
        results = [score_one(query, cascade, relational, top_k) for query in queries]
    finally:
        relational.close()

    by_type: dict[str, float] = {}
    for query_type in sorted({outcome.query_type for outcome in results}):
        subset = [o.wide.final_score for o in results if o.query_type == query_type]
        by_type[query_type] = sum(subset) / len(subset)

    with_anchor = [o.strict.final_score for o in results if o.strict is not None]
    result = OfficialRunResult(
        per_query=results,
        mean_final_score=(
            sum(o.wide.final_score for o in results) / len(results) if results else 0.0
        ),
        mean_final_score_strict=(
            sum(with_anchor) / len(with_anchor) if with_anchor else None
        ),
        strict_queries=len(with_anchor),
        by_type=by_type,
    )

    report = Path(out_dir) / "official-score.json"
    report.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_report(queries, result, report)
    return result


def _print_report(
    queries: list[AnyQuery], result: OfficialRunResult, report: Path
) -> None:
    by_id = {query.query_id: query for query in queries}
    header = (
        f"{'query_id':<16} {'type':<6} {'answers':>7} {'hit@':>5}  "
        f"{'R@1':>5} {'R@5':>5} {'R@20':>5} {'R@50':>5} {'R@100':>5}  "
        f"{'RONG':>6} {'HEP':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for outcome in result.per_query:
        score = outcome.wide
        rank = score.first_hit_rank
        cells = " ".join(f"{score.r_at_k[k]:>5.2f}" for k in (1, 5, 20, 50, 100))
        strict = (
            f"{outcome.strict.final_score:>6.3f}" if outcome.strict is not None else "     -"
        )
        print(
            f"{score.query_id or '':<16} {score.query_type:<6} {score.n_answers:>7} "
            f"{(str(rank) if rank else '-'):>5}  {cells}  "
            f"{score.final_score:>6.3f} {strict}"
        )

    print("\nFinal Score theo loai truy van (cua so RONG):")
    for query_type, value in result.by_type.items():
        print(f"  {query_type:<6} {value:.3f}")
    print(f"  {'MEAN':<6} {result.mean_final_score:.3f}")

    print("\nHai cot diem do hai thu khac nhau:")
    print(
        "  RONG = ca canh chua dap an. Khong phu thuoc cach trich keyframe,"
        " nen dung de so sanh giua cac lan doi cach trich."
    )
    if result.mean_final_score_strict is None:
        print(
            "  HEP  = cua so co BTC quanh anchor frame. Chua truy van nao co "
            "`gt_anchor_frame`, nen cot nay trong."
        )
    else:
        print(
            f"  HEP  = cua so ~11 frame quanh anchor, co BTC that. MEAN "
            f"{result.mean_final_score_strict:.3f} tren "
            f"{result.strict_queries}/{len(result.per_query)} truy van co anchor."
        )
        print(
            "         HEP thap hon nhieu so voi RONG = mat do keyframe chua du "
            "day, khong phai retrieval kem."
        )

    print("\nTruy van:")
    for outcome in result.per_query:
        score = outcome.wide
        query = by_id.get(score.query_id or "")
        if query is None:
            continue
        text = query.text
        if isinstance(query, TRAKEQuery) and query.event_texts:
            text += "   [" + " | ".join(query.event_texts) + "]"
        print(f"  [{score.query_type}] {score.query_id}: {text}")
        if query.note:
            print(f"        ground truth: {query.note}")
    print(f"\nBao cao: {report}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mfe-official",
        description="Score a retrieval run with the official BTC metrics (PDF §2).",
    )
    parser.add_argument("--config", default="config/sample2026.yaml")
    parser.add_argument("--queryset", default=None)
    parser.add_argument("--out", default=None)
    return parser


def main() -> None:
    setup_console()
    args = build_parser().parse_args()
    run(config_path=args.config, queryset_path=args.queryset, out_dir=args.out)


if __name__ == "__main__":
    main()
