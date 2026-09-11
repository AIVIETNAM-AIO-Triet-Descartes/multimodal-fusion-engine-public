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

Two ground-truth windows, both annotated by hand
------------------------------------------------
Every query carries a **scene span** (`gt_frame_range`) and, optionally, a
**strict window** (`gt_frame_range_strict`) nested inside it. Both are
source-video frame numbers read off the video itself, never off a keyframe grid.

* **Scene** = every frame in which the answer is visible, bounded by where the
  shot starts and ends. A scene runs seconds, so any reasonable extraction
  samples inside it — which makes this score comparable across extraction
  strategies, the number to watch when asking "did retrieval improve".
* **Strict** = the tight window the annotator believes BTC would give for *this*
  query. Not derived, and deliberately not a fixed width.

Why the strict width has to be annotated, not computed
-------------------------------------------------------
It is tempting to define strict as `anchor ± 5`, because PDF §2.1.1's worked
example scores `505` against `[500, 510]`. This module used to do exactly that,
and it was wrong twice over.

**The width follows the query, not a constant.** BTC's window has to be wide
enough to cover the frames that genuinely answer the question. A query that many
adjacent frames satisfy earns a wider window; one pinned to a single instant
earns a narrow one. §2.1.3's "thường dưới 10 frame" describes TRAKE because
*actions are short*, not because 10 is a universal number. Freezing one width
across every query encodes a guess as if it were the rule.

**The anchor must come from the video, not from `map-keyframes`.** The earlier
design read the anchor out of BTC's CSV, which meant the strict window was
centred on a keyframe *we happen to hold* — the test set quietly shaped by the
system under test. A window built that way always contains exactly one keyframe,
so it can never expose a sampling problem, and it rewards an extraction for
landing near BTC's choice rather than near the real moment.

The test set has to behave like BTC: watch the video, find the moment, place the
window. If no keyframe in the index falls inside it, that is a finding about the
**extraction**, and the right response is to fix the extraction — not to widen
the ground truth until the system passes.

Reading the two columns
------------------------
Scene high, strict low: the right shot is surfaced but not the right frame — a
precision problem, which is what reranking is for. Both near zero: retrieval
missed the scene outright.
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
from mfe.online import dante
from mfe.online.cascade import Cascade
from mfe.online.dante import EventHit

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
    #: English form of the query, for arms whose model has no Vietnamese:
    #: SigLIP SO400M's text tower, CLAP, ColPali. Optional — omit it and those
    #: arms sit out with a logged reason rather than searching in a language
    #: they cannot read. Cross-lingual arms (BAAI/bge-m3) never need it.
    text_en: str | None = None
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


def _check_range(query_id: str, frame_range: _FrameRange, label: str) -> _FrameRange:
    start, end = frame_range
    if start > end:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"{query_id}: {label} start {start} is after end {end}",
        )
    return frame_range


def _check_nested(
    query_id: str, wide: _FrameRange, strict: _FrameRange, label: str = ""
) -> _FrameRange:
    """The strict window must sit inside the scene it belongs to.

    Both are annotated by hand from the same viewing, so a strict window poking
    outside its scene means one of the two was read off the wrong moment. Left
    unchecked, the two columns of the report would disagree for a reason nobody
    could see from the numbers.
    """
    where = f"{query_id}{label}"
    _check_range(where, wide, "scene range")
    _check_range(where, strict, "strict range")
    if strict[0] < wide[0] or strict[1] > wide[1]:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"{where}: strict window {list(strict)} is not inside its scene "
            f"{list(wide)}",
        )
    return strict


class KISQuery(_Query):
    """One Textual KIS query and its accepted answer interval."""

    type: Literal["KIS"]
    text: str
    gt_video_id: str
    #: The **scene span**: every frame in which the answer is visible.
    gt_frame_range: _FrameRange
    #: The tight window, annotated by watching the video — not derived. See the
    #: module docstring for why its width has to vary with the query.
    #: Optional; without it the query is only scored on the scene window.
    gt_frame_range_strict: _FrameRange | None = None

    def ground_truth(self) -> KISGroundTruth:
        start, end = _check_range(self.query_id, self.gt_frame_range, "gt_frame_range")
        return KISGroundTruth(
            video_id=self.gt_video_id, start_frame_id=start, end_frame_id=end
        )

    def ground_truth_strict(self) -> KISGroundTruth | None:
        if self.gt_frame_range_strict is None:
            return None
        start, end = _check_nested(
            self.query_id, self.gt_frame_range, self.gt_frame_range_strict
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
    #: Tight window, annotated — see `KISQuery.gt_frame_range_strict`.
    gt_frame_range_strict: _FrameRange | None = None
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
        return self._at(
            _check_range(self.query_id, self.gt_frame_range, "gt_frame_range")
        )

    def ground_truth_strict(self) -> QAGroundTruth | None:
        if self.gt_frame_range_strict is None:
            return None
        return self._at(
            _check_nested(
                self.query_id, self.gt_frame_range, self.gt_frame_range_strict
            )
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
    #: Tight window per event, same order and length as `gt_events`. This is
    #: where BTC's "thường dưới 10 frame" belongs: TRAKE windows are narrow
    #: because the *actions* are short, not because 10 is a universal width.
    #: Optional; without it the query is only scored on the scene windows.
    gt_events_strict: list[_FrameRange] | None = None

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
            start, end = _check_range(
                self.query_id, frame_range, f"gt_events[{index}]"
            )
            events.append(TRAKEEvent(start, end))
        return TRAKEGroundTruth(video_id=self.gt_video_id, events=events)

    def ground_truth_strict(self) -> TRAKEGroundTruth | None:
        if not self.gt_events_strict:
            return None
        if len(self.gt_events_strict) != len(self.gt_events):
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT,
                f"{self.query_id}: {len(self.gt_events_strict)} strict window(s) but "
                f"{len(self.gt_events)} event(s) — one per event, in the same order",
            )
        events = []
        for index, (wide, strict) in enumerate(
            zip(self.gt_events, self.gt_events_strict, strict=True), 1
        ):
            start, end = _check_nested(
                self.query_id, wide, strict, f" event {index}"
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

    **One exception, and it declares itself.** A frame the operator scrubbed to in
    the submission desk is not in BTC's catalogue at all, so the lookup can only
    fail for it. Such a candidate carries its own `frame_idx`, and carrying it is
    the whole point: the override is explicit per candidate, so a genuinely
    missing keyframe still raises exactly as before instead of being papered over
    by a blanket "parse the id" fallback.
    """
    if candidate.frame_idx is not None:
        return candidate.frame_idx
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


def to_event_hits(
    candidates: list[Candidate], relational: RelationalMap
) -> list[EventHit]:
    """`Candidate` -> `dante.EventHit`, reading both coordinates off the map.

    DANTE needs two different numbers for the same keyframe and they must not be
    confused: `ordinal` (`keyframe_index`) is what its temporal penalty measures
    distance on, `frame_id` is what BTC scores. `mfe.contracts.ids` is emphatic
    about the difference; this is the one place both are looked up together.
    """
    hits: list[EventHit] = []
    for candidate in candidates:
        row = relational.get_keyframe(candidate.keyframe_id)
        hits.append(
            EventHit(
                keyframe_id=candidate.keyframe_id,
                video_id=candidate.video_id,
                ordinal=row.keyframe_index,
                frame_id=row.frame_idx,
                score=float(candidate.score),
            )
        )
    return hits


def to_trake_submissions(
    per_event: list[list[Candidate]],
    relational: RelationalMap,
    limit: int = MAX_ANSWERS_PER_QUERY,
    *,
    penalty: float = dante.PENALTY,
) -> list[TRAKESubmission]:
    """Assemble N per-event ranked lists into whole-video TRAKE answers, by DANTE.

    `mfe.online.dante` implements the algorithm (arXiv:2512.13169 §4.3); this is
    the translation between our `Candidate` and its `EventHit`, and back into the
    submission shape BTC scores.

    **What changed, and why.** The previous version here was a DP too, but with
    no temporal penalty: it maximised the total candidate score under a
    strictly-increasing constraint and nothing else. That treats "event 2 happens
    two keyframes after event 1" and "event 2 happens twenty minutes later" as
    equally good, which is exactly what a TRAKE query is *not* asking. DANTE adds
    `λ·(t − τ)`, which prefers a nearby continuation without forbidding a distant
    one — so a genuinely sparse action sequence still assembles.

    The three properties the old version had are kept, because they are right:
    one answer per video, videos ranked by score, and a video with no increasing
    assignment **dropped rather than padded** — a padded frame scores as a wrong
    guess and is indistinguishable from a real attempt.
    """
    alignments = dante.align(
        [to_event_hits(ranked, relational) for ranked in per_event],
        penalty=penalty,
        limit=limit,
    )
    return [
        TRAKESubmission(video_id=item.video_id, frame_ids=item.frame_ids)
        for item in alignments
    ]


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
    #: anchor. Answers "did we rank the exact keyframe the annotator picked",
    #: not "is our sampling dense enough" — the anchor *is* a keyframe, so the
    #: window always contains one and sparseness cannot show up here.
    strict: QueryScore | None = None

    @property
    def query_id(self) -> str | None:
        return self.wide.query_id

    @property
    def query_type(self) -> str:
        return self.wide.query_type


class RunProvenance(BaseModel):
    """What produced these numbers — §3.6 of ke-hoach-merge-sota.md.

    Two scores run with different arms enabled, or a different fusion mode, are
    two numbers that cannot be compared, and nothing in the report used to say
    which was which. That is not hypothetical: `eval/results_pilot/official-score.json`
    held 0.663/0.574 (a `weighted_sum` run) while `config/pilot.yaml` said
    `mode: rrf`, whose measured pair is 0.711/0.533. Whoever read the file next
    would have taken the wrong baseline and attributed the gap to their change.

    Recorded from the *live registry* rather than from config, so an arm that
    config asks for but that could not open its store shows up here as
    unavailable instead of being reported as if it had voted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    config_path: str
    queryset_path: str
    n_queries: int
    top_k: int
    fusion_mode: str
    #: What each arm's scores were actually converted to before weighting. Worth
    #: recording separately from `fusion_mode`: the mode picks a default, and a
    #: config may override it, so the mode alone does not determine the run.
    fusion_normalizer: str
    rrf_k: int
    #: One row per registered arm: `name`, `status` ("on" / "off" /
    #: "unavailable — reason"), and `weight`. Named after the field §3.6 asks for.
    tiers_enabled: list[dict[str, object]]
    #: Where each query's English form came from, e.g. `{"manual": 7, "mt": 2}`;
    #: `"none"` counts queries that got none, whose English-only arms sat out.
    #: Filled after scoring, so it reports what happened rather than what was
    #: configured — SigLIP on hand-written English (rank 2.0) and SigLIP on
    #: machine English are different measurements wearing the same arm name.
    translation_used: dict[str, int] = Field(default_factory=dict)


class OfficialRunResult(BaseModel):
    """One run of the official metrics over a query set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Optional so that a result assembled in a test need not fake a cascade.
    provenance: RunProvenance | None = None
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
            cascade.search(
                query_text=text,
                top_k=top_k,
                run_id=query.query_id,
                query_text_en=query.text_en,
            )
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
            query_text=search_text,
            top_k=top_k,
            run_id=query.query_id,
            query_text_en=query.text_en,
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


def describe_run(
    cascade: Cascade,
    config: Config,
    config_path: str,
    queryset_path: str,
    n_queries: int,
    top_k: int,
) -> RunProvenance:
    """Snapshot the settings a score depends on, taken after the arms are built."""
    fusion = config.retrieval.fusion
    return RunProvenance(
        config_path=config_path,
        queryset_path=queryset_path,
        n_queries=n_queries,
        top_k=top_k,
        fusion_mode=fusion.mode,
        fusion_normalizer=fusion.resolved_normalizer(),
        rrf_k=fusion.rrf_k,
        tiers_enabled=[
            {"name": row["name"], "status": row["status"], "weight": row["weight"]}
            for row in cascade.arms.report()
        ],
        translation_used=dict(getattr(cascade.translator, "usage", {}) or {}),
    )


def run(
    config_path: str = "config/sample2026.yaml",
    queryset_path: str | None = None,
    out_dir: str | None = None,
    flat_root: str | None = None,
) -> OfficialRunResult:
    """Score a whole query set with the official metrics and write the report.

    `flat_root` chạy trên **bản gộp phẳng** thay vì Elasticsearch + Milvus. Đó là
    đường duy nhất chạy được trên laptop (CLAUDE.md §4 đã chốt bỏ cả hai kho), và
    nó là đường phải dùng để **công bố trên heldout**: chỉnh trọng số trên dev ở
    đường phẳng rồi công bố ở một đường khác là công bố một con số khác.
    """
    config: Config = load_config(config_path)
    queryset_path = queryset_path or "eval/queryset_btc_2026.jsonl"
    out_dir = out_dir or config.paths.eval_results
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    queries = load_official_queryset(queryset_path)
    top_k = min(config.retrieval.rerank_topk, MAX_ANSWERS_PER_QUERY)

    relational = RelationalMap(config.paths.relational_db)
    cascade: Cascade | None = None
    try:
        relational.init_schema()
        # No `reranker=None` override: stage C is a config decision now
        # (`models.reranker`), and pinning it off here is how the harness spent
        # a run reporting that a working reranker changed nothing.
        if flat_root:
            from mfe.online.flat_arms import build_flat_cascade  # noqa: PLC0415

            cascade = build_flat_cascade(config, flat_root)
        else:
            cascade = Cascade.from_config(config)
        results = [score_one(query, cascade, relational, top_k) for query in queries]
        # After scoring, not before: `translation_used` reports what actually
        # happened to each query, which is only known once they have all run.
        provenance = describe_run(
            cascade,
            config=config,
            config_path=config_path,
            queryset_path=queryset_path,
            n_queries=len(queries),
            top_k=top_k,
        )
    finally:
        # Persist any translation paid for this run, including on the way out of
        # a failure: the entries are correct regardless of what went wrong later,
        # and the next run should not buy them twice.
        if cascade is not None and cascade.translator is not None:
            cascade.translator.save_cache()
        relational.close()

    by_type: dict[str, float] = {}
    for query_type in sorted({outcome.query_type for outcome in results}):
        subset = [o.wide.final_score for o in results if o.query_type == query_type]
        by_type[query_type] = sum(subset) / len(subset)

    with_anchor = [o.strict.final_score for o in results if o.strict is not None]
    result = OfficialRunResult(
        provenance=provenance,
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
    if is_heldout(queries):
        # The file gets the same treatment as the console: aggregates only. Hiding
        # per-query behaviour on screen while writing it to disk would be theatre —
        # the temptation is the file, not the print. Re-run after the config is
        # frozen if a per-query audit is ever needed.
        payload: dict = {
            "split": "heldout",
            "provenance": provenance.model_dump(),
            "n_queries": len(result.per_query),
            "mean_final_score": result.mean_final_score,
            "mean_final_score_strict": result.mean_final_score_strict,
            "strict_queries": result.strict_queries,
            "by_type": result.by_type,
            "withheld": (
                "per_query bi luoc CO Y — ky luat hop den, "
                "docs/task/bo-test-dev-heldout.md §7"
            ),
        }
    else:
        payload = result.model_dump()
    report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_report(queries, result, report)
    return result


def _print_provenance(provenance: RunProvenance) -> None:
    """Print what this run was, above the numbers it produced."""
    print(
        f"\ncau hinh: {provenance.config_path}  ·  truy van: {provenance.queryset_path} "
        f"({provenance.n_queries})  ·  top_k {provenance.top_k}"
    )
    normalizer = provenance.fusion_normalizer
    detail = f", k={provenance.rrf_k}" if normalizer == "rrf" else ""
    print(f"fusion: mode={provenance.fusion_mode}, normalizer={normalizer}{detail}")
    if provenance.translation_used:
        source = ", ".join(
            f"{tier}={count}" for tier, count in sorted(provenance.translation_used.items())
        )
        print(f"text_en lay tu: {source}")
    on = [row for row in provenance.tiers_enabled if row["status"] == "on"]
    print(f"arm dang chay ({len(on)}/{len(provenance.tiers_enabled)}):")
    for row in provenance.tiers_enabled:
        mark = "+" if row["status"] == "on" else "-"
        print(f"  {mark} {str(row['name']):<18} w={row['weight']:<5} {row['status']}")


def is_heldout(queries: list[AnyQuery]) -> bool:
    """Whether this query set is held out, and therefore may only be *scored*.

    Read off `split` on the queries themselves rather than the file path, so the
    rule holds however the harness is invoked.
    """
    return any(query.split == "heldout" for query in queries)


def _print_heldout_report(result: OfficialRunResult, report: Path) -> None:
    """Aggregate numbers only — the whole point of a held-out set.

    Kỷ luật hộp đen (docs/task/bo-test-dev-heldout.md §7): on held-out data we may
    know **what the BTC metric says** and nothing else. No per-query rows, no first
    hit ranks, no R@k curve, no query text. Those are behaviour, and reading
    behaviour is how a held-out set quietly turns into a second dev set — after
    which nothing is left to measure generalisation with.

    Every number withheld here is available on `eval/dev/*`, which is where tuning
    belongs.
    """
    print("\n=== HELD-OUT: chi bao cao diem tong hop ===")
    if result.provenance is not None:
        _print_provenance(result.provenance)
    print(f"\nSo truy van: {len(result.per_query)}")
    print("\nFinal Score theo loai truy van (cua so RONG):")
    for query_type, value in result.by_type.items():
        print(f"  {query_type:<6} {value:.3f}")
    print(f"  {'MEAN':<6} {result.mean_final_score:.3f}")
    if result.mean_final_score_strict is not None:
        print(
            f"\nMEAN cua so HEP: {result.mean_final_score_strict:.3f} "
            f"tren {result.strict_queries}/{len(result.per_query)} truy van co cua so hep."
        )
    print(
        "\nDA AN CO Y: bang tung cau, hang trung binh, R@k tung cau, va noi dung truy van.\n"
        "Muon xem hanh vi de sua he thong thi quay ve bo dev — do la ranh gioi lam cho\n"
        "con so heldout con y nghia. Chi tiet van nam trong file JSON de kiem toan sau\n"
        "khi da CHOT cau hinh, khong phai de dieu chinh no."
    )
    print(f"\nBao cao: {report}\n")


def _print_report(
    queries: list[AnyQuery], result: OfficialRunResult, report: Path
) -> None:
    if is_heldout(queries):
        _print_heldout_report(result, report)
        return

    by_id = {query.query_id: query for query in queries}
    if result.provenance is not None:
        _print_provenance(result.provenance)
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
            f"  HEP  = cua so 11 frame quanh anchor. Hoi 'co xep dung CAI KEYFRAME "
            f"do len cao khong', chat hon RONG. MEAN "
            f"{result.mean_final_score_strict:.3f} tren "
            f"{result.strict_queries}/{len(result.per_query)} truy van co anchor."
        )
        print(
            "         RONG cao ma HEP thap = tim dung canh, chua dung khung hinh "
            "-> van de do chinh xac, rerank chua toi. Ca hai ~0 = truot han canh."
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
    parser.add_argument(
        "--flat-root",
        default=None,
        help="Chay tren kho phang (vd stores/flat) thay vi Elasticsearch + Milvus.",
    )
    return parser


def _load_env_file(path: str = ".env") -> None:
    """Đưa `.env` vào môi trường, **không** đè lên biến đã đặt sẵn.

    `retrieval.rerank_tiers` có thể gọi những tầng đọc thẳng `os.environ`:
    `vlm` cần `GEMINI_API_KEY`, `hf_cross` cần `HF_TOKEN`. Thiếu bước này thì
    tầng **báo đúng** ("chua co GEMINI_API_KEY") và lượt chạy vẫn xong, in ra
    một điểm số trông bình thường — nghĩa là "tầng không giúp gì" đọc giống hệt
    "tầng không chạy". Đây là đường **NỘP BÀI**, nên hậu quả nặng hơn hẳn một
    bảng đo lệch: nó tạo ra một file nộp không hề rerank mà không ai biết.

    `measure_rerank.py` đã mất một lượt đo vì đúng chuyện này (CLAUDE.md §5), và
    bản sửa hôm đó chỉ vá cho script đo — đường nộp vẫn thiếu cho tới 28-08.
    """
    import os  # noqa: PLC0415

    from mfe.offline.transport import load_dotenv  # noqa: PLC0415

    for key, value in load_dotenv(path).items():
        os.environ.setdefault(key, value)


def main() -> None:
    setup_console()
    _load_env_file()
    args = build_parser().parse_args()
    run(
        config_path=args.config,
        queryset_path=args.queryset,
        out_dir=args.out,
        flat_root=args.flat_root,
    )


if __name__ == "__main__":
    main()
