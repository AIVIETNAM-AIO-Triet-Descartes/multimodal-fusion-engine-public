"""Official-score A/B runner and frozen Go/No-Go rules."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from mfe.core.config import load_config
from mfe.eval.official import (
    MAX_ANSWERS_PER_QUERY,
    KISQuery,
    QAQuery,
    QueryOutcome,
    TRAKEQuery,
    evaluate_query,
    load_official_queryset,
    score_one,
    to_kis_submissions,
    to_qa_submissions,
    to_trake_submissions,
)
from mfe.mapping.relational import RelationalMap
from mfe.offline.augmentation.models import OracleTargetRow
from mfe.offline.augmentation.oracle import force_rank
from mfe.offline.augmentation.overlay import load_overlay_flat_bundle
from mfe.online.flat_arms import build_flat_cascade


@dataclass(frozen=True)
class BootstrapInterval:
    mean_delta: float
    lower_95: float
    upper_95: float


def paired_bootstrap_interval(
    baseline: list[float],
    treatment: list[float],
    *,
    samples: int = 10_000,
    seed: int = 2026,
) -> BootstrapInterval:
    """Paired non-parametric 95% interval over per-query score deltas."""

    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("baseline/treatment must have the same non-zero length")
    if samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    delta = np.asarray(treatment, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    means = delta[indices].mean(axis=1)
    return BootstrapInterval(
        mean_delta=float(delta.mean()),
        lower_95=float(np.quantile(means, 0.025)),
        upper_95=float(np.quantile(means, 0.975)),
    )


@dataclass(frozen=True)
class VariantRun:
    name: str
    query_ids: tuple[str, ...]
    query_types: tuple[str, ...]
    scores: tuple[float, ...]
    latency_ms: tuple[float, ...]

    @property
    def mean_score(self) -> float:
        return statistics.fmean(self.scores) if self.scores else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latency_ms:
            return 0.0
        return float(np.quantile(np.asarray(self.latency_ms), 0.95))

    @property
    def by_type(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for query_type in sorted(set(self.query_types)):
            values = [
                score
                for score, item_type in zip(self.scores, self.query_types, strict=True)
                if item_type == query_type
            ]
            out[query_type] = statistics.fmean(values)
        return out


def run_overlay_variant(
    *,
    name: str,
    config_path: str,
    queryset_path: str,
    relational_db: str,
    base_flat_root: str,
    delta_flat_root: str,
    include_ids: list[str],
    latency_repeats: int = 1,
) -> VariantRun:
    """Run official scoring on a base-plus-allowlist overlay.

    The first run supplies the score.  Repeats are used only for latency and
    must produce byte-equivalent score objects; otherwise the experiment is not
    reproducible enough for a causal A/B comparison.
    """

    if latency_repeats < 1:
        raise ValueError("latency_repeats must be >= 1")
    config = load_config(config_path)
    queries = load_official_queryset(queryset_path)
    bundle = load_overlay_flat_bundle(base_flat_root, delta_flat_root, include_ids)
    cascade = build_flat_cascade(config, base_flat_root, flat_bundle=bundle)
    relational = RelationalMap(relational_db)
    relational.init_schema()
    query_ids: list[str] = []
    query_types: list[str] = []
    scores: list[float] = []
    latencies: list[float] = []
    top_k = min(config.retrieval.rerank_topk, MAX_ANSWERS_PER_QUERY)
    try:
        for query in queries:
            observed = []
            timings = []
            for _ in range(latency_repeats):
                started = time.perf_counter()
                outcome = score_one(query, cascade, relational, top_k)
                timings.append((time.perf_counter() - started) * 1000.0)
                observed.append(outcome)
            first = observed[0].model_dump(mode="json")
            if any(item.model_dump(mode="json") != first for item in observed[1:]):
                raise RuntimeError(f"query {query.query_id} is non-deterministic across repeats")
            query_ids.append(query.query_id)
            query_types.append(query.type)
            scores.append(observed[0].wide.final_score)
            latencies.append(statistics.median(timings))
    finally:
        close = getattr(cascade, "close", None)
        if callable(close):
            close()
        relational.close()
    return VariantRun(
        name=name,
        query_ids=tuple(query_ids),
        query_types=tuple(query_types),
        scores=tuple(scores),
        latency_ms=tuple(latencies),
    )


def _score_one_forced(
    query,
    cascade,
    relational: RelationalMap,
    top_k: int,
    targets: dict[int, OracleTargetRow],
) -> QueryOutcome:
    """Official scoring with a deliberately injected O1-R target."""

    if isinstance(query, TRAKEQuery):
        per_event = []
        for event_index, text in enumerate(query.searches()):
            candidates = cascade.search(
                query_text=text,
                top_k=top_k,
                run_id=query.query_id,
                query_text_en=query.text_en,
            )
            if event_index in targets:
                candidates = force_rank(
                    candidates,
                    targets[event_index],
                    run_id=query.query_id,
                    top_k=top_k,
                )
            per_event.append(candidates)
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
        if 0 in targets:
            candidates = force_rank(
                candidates,
                targets[0],
                run_id=query.query_id,
                top_k=top_k,
            )
        if isinstance(query, QAQuery):
            submissions = list(
                to_qa_submissions(candidates, relational, query.answer(), limit=top_k)
            )
        elif isinstance(query, KISQuery):
            submissions = list(to_kis_submissions(candidates, relational, limit=top_k))
        else:  # pragma: no cover - discriminated queryset prevents this
            raise TypeError(type(query))
    strict_gt = query.ground_truth_strict()
    return QueryOutcome(
        wide=evaluate_query(submissions, query.ground_truth(), query_id=query.query_id),
        strict=(
            evaluate_query(submissions, strict_gt, query_id=query.query_id)
            if strict_gt is not None
            else None
        ),
    )


def run_force_rank_variant(
    *,
    name: str,
    config_path: str,
    queryset_path: str,
    relational_db: str,
    base_flat_root: str,
    targets: list[OracleTargetRow],
) -> VariantRun:
    """Run O1-R while leaving retrieval/index artifacts unchanged."""

    config = load_config(config_path)
    queries = load_official_queryset(queryset_path)
    cascade = build_flat_cascade(config, base_flat_root)
    relational = RelationalMap(relational_db)
    relational.init_schema()
    by_query: dict[str, dict[int, OracleTargetRow]] = {}
    for target in targets:
        by_query.setdefault(target.query_id, {})[target.event_index] = target
    query_ids: list[str] = []
    query_types: list[str] = []
    scores: list[float] = []
    latencies: list[float] = []
    top_k = min(config.retrieval.rerank_topk, MAX_ANSWERS_PER_QUERY)
    try:
        for query in queries:
            started = time.perf_counter()
            outcome = _score_one_forced(
                query,
                cascade,
                relational,
                top_k,
                by_query.get(query.query_id, {}),
            )
            latencies.append((time.perf_counter() - started) * 1000.0)
            query_ids.append(query.query_id)
            query_types.append(query.type)
            scores.append(outcome.wide.final_score)
    finally:
        close = getattr(cascade, "close", None)
        if callable(close):
            close()
        relational.close()
    return VariantRun(
        name=name,
        query_ids=tuple(query_ids),
        query_types=tuple(query_types),
        scores=tuple(scores),
        latency_ms=tuple(latencies),
    )


def _aligned(*runs: VariantRun) -> None:
    if not runs:
        return
    expected = runs[0].query_ids
    if any(run.query_ids != expected for run in runs[1:]):
        raise ValueError("variant query order/identity does not match baseline")


def oracle_gate(
    baseline: VariantRun,
    oracle: VariantRun,
    *,
    min_mean_delta: float,
    min_rescues: int,
) -> dict[str, Any]:
    _aligned(baseline, oracle)
    deltas = [right - left for left, right in zip(baseline.scores, oracle.scores, strict=True)]
    rescued = sum(delta > 0 for delta in deltas)
    regressed = sum(delta < 0 for delta in deltas)
    mean_delta = statistics.fmean(deltas) if deltas else 0.0
    reasons = []
    if rescued < min_rescues:
        reasons.append(f"rescued {rescued} < {min_rescues}")
    if mean_delta < min_mean_delta:
        reasons.append(f"mean delta {mean_delta:.6f} < {min_mean_delta:.6f}")
    if regressed > rescued:
        reasons.append(f"regressions {regressed} > rescues {rescued}")
    return {
        "go": not reasons,
        "mean_delta": mean_delta,
        "rescued": rescued,
        "regressed": regressed,
        "reasons": reasons,
    }


def dev_gate(
    baseline: VariantRun,
    b3: VariantRun,
    b4: VariantRun,
    *,
    target_query_ids: set[str],
    delta_frames: int,
    base_frames: int,
    cap_ratio: float,
    noninferiority_margin: float,
    max_latency_ratio: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Apply every frozen balanced Gate-2 criterion."""

    _aligned(baseline, b3, b4)
    interval = paired_bootstrap_interval(
        list(baseline.scores),
        list(b3.scores),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    target_deltas = [
        right - left
        for query_id, left, right in zip(
            baseline.query_ids, baseline.scores, b3.scores, strict=True
        )
        if query_id in target_query_ids
    ]
    target_delta = statistics.fmean(target_deltas) if target_deltas else 0.0
    delta_ratio = delta_frames / base_frames if base_frames else 1.0
    latency_ratio = (
        b3.p95_latency_ms / baseline.p95_latency_ms
        if baseline.p95_latency_ms > 0
        else float("inf")
    )
    rescued_natural = sum(
        right > left
        for left, right in zip(baseline.scores, b3.scores, strict=True)
    )
    reasons: list[str] = []
    if b3.mean_score <= baseline.mean_score:
        reasons.append("B3 mean Final Score does not beat B0")
    if b3.mean_score <= b4.mean_score:
        reasons.append("B3 mean Final Score does not beat matched B4")
    if interval.lower_95 < noninferiority_margin:
        reasons.append(
            f"LCB95 {interval.lower_95:.6f} < margin {noninferiority_margin:.6f}"
        )
    if not target_deltas or target_delta <= 0:
        reasons.append("coverage-gap target subset has no positive mean delta")
    if rescued_natural < 1:
        reasons.append("B3 rescues no natural DEV query")
    if delta_ratio > cap_ratio:
        reasons.append(f"delta ratio {delta_ratio:.4%} > cap {cap_ratio:.4%}")
    if latency_ratio > max_latency_ratio:
        reasons.append(f"p95 latency ratio {latency_ratio:.3f} > {max_latency_ratio:.3f}")
    return {
        "go": not reasons,
        "b0_mean": baseline.mean_score,
        "b3_mean": b3.mean_score,
        "b4_mean": b4.mean_score,
        "bootstrap": interval.__dict__,
        "target_mean_delta": target_delta,
        "rescued_natural": rescued_natural,
        "delta_frames": delta_frames,
        "delta_ratio": delta_ratio,
        "latency_ratio": latency_ratio,
        "reasons": reasons,
    }


def heldout_gate(
    baseline: VariantRun,
    b3: VariantRun,
    *,
    by_type_margin: float,
    max_latency_ratio: float,
) -> dict[str, Any]:
    """Aggregate-only held-out decision; never returns per-query fields."""

    _aligned(baseline, b3)
    latency_ratio = (
        b3.p95_latency_ms / baseline.p95_latency_ms
        if baseline.p95_latency_ms > 0
        else float("inf")
    )
    by_type_delta = {
        key: b3.by_type[key] - baseline.by_type[key]
        for key in sorted(set(baseline.by_type) & set(b3.by_type))
    }
    reasons: list[str] = []
    if b3.mean_score <= baseline.mean_score:
        reasons.append("held-out overall B3 does not beat B0")
    for key, delta in by_type_delta.items():
        if delta < by_type_margin:
            reasons.append(f"held-out {key} delta {delta:.6f} < {by_type_margin:.6f}")
    if latency_ratio > max_latency_ratio:
        reasons.append(f"held-out p95 latency ratio {latency_ratio:.3f} is too high")
    return {
        "split": "heldout",
        "go": not reasons,
        "n_queries": len(baseline.query_ids),
        "b0_mean": baseline.mean_score,
        "b3_mean": b3.mean_score,
        "by_type_delta": by_type_delta,
        "latency_ratio": latency_ratio,
        "withheld": "per-query results intentionally omitted",
        "reasons": reasons,
    }
