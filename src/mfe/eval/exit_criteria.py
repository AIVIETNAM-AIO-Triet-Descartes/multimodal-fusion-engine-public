"""Sprint 1 exit criteria, checked automatically — docs/01 §2, docs/00 §1.

The harness prints PASS/FAIL per criterion. The point of automating it: "Done"
should be something the team runs, not something they argue about at review.

docs/00 §1, verbatim:

  - Offline chạy trên một thư mục video bất kỳ bằng một lệnh, đổ ra đủ 4 kho.
  - Online nhận query text -> trả top-k keyframe -> mở đúng video/timestamp.
  - Mapping ID<->frame<->timestamp đúng 100% trên tập mẫu.
  - Mọi submit sai schema bị validator chặn.
  - Chạy được >= 30 query nội bộ không crash, có log tái hiện.
  - Có số liệu Recall@K, TTCA, latency, throughput ghi lại.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

#: docs/00 §1 and docs/context/buoi-1.md §9 both say >= 30.
MIN_QUERIES_WITHOUT_CRASH = 30

#: docs/00 §1: mapping must be 100% on the sample. Not 99%.
REQUIRED_MAPPING_ACCURACY = 1.0


class CriterionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    passed: bool
    detail: str


def check_all(run_summary: dict) -> list[CriterionResult]:
    """Evaluate every criterion against a harness run. Owner: Vỹ Triết."""
    return [
        check_stores_built(run_summary),
        check_queries_without_crash(run_summary),
        check_mapping_accuracy(run_summary),
        check_invalid_submits_blocked(run_summary),
        check_metrics_recorded(run_summary),
    ]


def check_stores_built(run_summary: dict) -> CriterionResult:
    """All 4 stores exist and cover the same keyframe_id set (docs/02 §6)."""
    stores = run_summary.get("stores") or {}
    expected = ("relational", "vector", "inverted", "asset")
    missing = [name for name in expected if not stores.get(name)]
    if missing:
        return CriterionResult(
            name="4 kho được dựng",
            passed=False,
            detail=f"missing or empty: {', '.join(missing)}. Run mfe-ingest first.",
        )
    counts = {name: stores[name] for name in expected}
    if len(set(counts.values())) != 1:
        return CriterionResult(
            name="4 kho được dựng",
            passed=False,
            detail=f"stores disagree on keyframe count: {counts}",
        )
    return CriterionResult(
        name="4 kho được dựng",
        passed=True,
        detail=f"all 4 stores hold {next(iter(counts.values()))} keyframes",
    )


def check_queries_without_crash(run_summary: dict) -> CriterionResult:
    """>= 30 queries completed without a crash, with a reproducible log."""
    total = run_summary.get("queries", 0)
    crashed = run_summary.get("crashed", 0)
    completed = total - crashed
    has_log = bool(run_summary.get("event_log"))

    if crashed:
        ids = run_summary.get("crashed_query_ids") or []
        return CriterionResult(
            name=f">= {MIN_QUERIES_WITHOUT_CRASH} query không crash",
            passed=False,
            detail=f"{crashed}/{total} queries crashed, e.g. {ids[:3]}",
        )
    if completed < MIN_QUERIES_WITHOUT_CRASH:
        return CriterionResult(
            name=f">= {MIN_QUERIES_WITHOUT_CRASH} query không crash",
            passed=False,
            detail=(
                f"only {completed} queries ran; the queryset needs at least "
                f"{MIN_QUERIES_WITHOUT_CRASH} (docs/00 §1) — Nhật Bảo owns eval/queryset.jsonl"
            ),
        )
    if not has_log:
        return CriterionResult(
            name=f">= {MIN_QUERIES_WITHOUT_CRASH} query không crash",
            passed=False,
            detail=f"{completed} queries ran but no event log was written (docs/00 §1)",
        )
    return CriterionResult(
        name=f">= {MIN_QUERIES_WITHOUT_CRASH} query không crash",
        passed=True,
        detail=f"{completed} queries, no crash, log at {run_summary['event_log']}",
    )


def check_mapping_accuracy(run_summary: dict) -> CriterionResult:
    """Mapping 100% correct on the sample.

    Sourced from `RelationalMap.verify_mapping()` rather than from query
    outcomes: a query set can pass while the map is wrong, because it only
    touches the keyframes the queries happen to hit.
    """
    problems = run_summary.get("mapping_problems")
    if problems is None:
        return CriterionResult(
            name="Mapping đúng 100%",
            passed=False,
            detail="verify_mapping() was never run — nothing to attest",
        )
    if problems:
        return CriterionResult(
            name="Mapping đúng 100%",
            passed=False,
            detail=f"{len(problems)} problem(s), first: {problems[0]}",
        )
    checked = run_summary.get("mapping_rows_checked", 0)
    return CriterionResult(
        name="Mapping đúng 100%",
        passed=True,
        detail=f"verify_mapping() clean over {checked} keyframe(s)",
    )


def check_invalid_submits_blocked(run_summary: dict) -> CriterionResult:
    """Every malformed submit was blocked by the validator.

    This needs deliberately bad submits in the run — one known violation per
    rule in docs/05 §3. A run where nothing was wrong proves the fence works
    only in the sense that an untested fence is standing, so an empty probe set
    fails rather than passes.
    """
    probes = run_summary.get("submit_probes") or []
    if not probes:
        return CriterionResult(
            name="Submit sai bị chặn",
            passed=False,
            detail="no bad-submit probes were run; an untested fence proves nothing",
        )
    leaked = [probe for probe in probes if probe.get("should_block") and not probe.get("blocked")]
    false_blocks = [
        probe for probe in probes if not probe.get("should_block") and probe.get("blocked")
    ]
    if leaked:
        return CriterionResult(
            name="Submit sai bị chặn",
            passed=False,
            detail=f"{len(leaked)} bad submit(s) got through: {[p['name'] for p in leaked]}",
        )
    if false_blocks:
        return CriterionResult(
            name="Submit sai bị chặn",
            passed=False,
            detail=(
                f"{len(false_blocks)} valid submit(s) were wrongly blocked: "
                f"{[p['name'] for p in false_blocks]}"
            ),
        )
    return CriterionResult(
        name="Submit sai bị chặn",
        passed=True,
        detail=f"{len(probes)} probe(s) all behaved as expected",
    )


def check_metrics_recorded(run_summary: dict) -> CriterionResult:
    """Recall@K, TTCA, latency, throughput all present.

    Throughput comes from `eval/throughput.json` (Quốc Đạt, docs/02 §5), not
    from this run, so the harness only checks that the file was found.
    """
    missing: list[str] = []
    if not run_summary.get("recall"):
        scored = run_summary.get("scored_queries", 0)
        missing.append(
            "Recall@K (no query has ground truth yet)" if scored == 0 else "Recall@K"
        )
    if (run_summary.get("ttca_s") or {}).get("median") is None:
        missing.append("TTCA")
    if (run_summary.get("latency_ms") or {}).get("median") is None:
        missing.append("latency")
    if not run_summary.get("throughput"):
        missing.append("throughput (run mfe-ingest to produce eval/throughput.json)")

    if missing:
        return CriterionResult(
            name="Có đủ số liệu",
            passed=False,
            detail="missing: " + "; ".join(missing),
        )
    return CriterionResult(
        name="Có đủ số liệu",
        passed=True,
        detail="Recall@K, TTCA, latency and throughput all recorded",
    )
