"""Benchmark harness entrypoint — docs/01 §2.

    python -m mfe.eval.run --queryset eval/queryset.jsonl \
        --config config/local.yaml --out eval/results/
    mfe-eval --queryset eval/queryset.jsonl --config config/local.yaml --out eval/results/

docs/01 §2 on why this exists: it is what tells the team whether there is real
progress — "vai hay bị bỏ quên nhất". Nobody's workstream feels blocked without
it, and that is exactly the problem.

The default search path is now the real cascade
------------------------------------------------
Stage A runs both arms — CLIP over the Faiss index and BM25 over the Inverted
Index — fused by RRF. The vector arm was the missing piece: BTC ships keyframe
features but not the model, so a text query had nowhere to go until the
checkpoint was pinned to `clip-ViT-B-32` (official PDF §3) and verified at
cosine 0.9992-1.0000 against the shipped vectors.

`run(...)` still takes a `search_fn`, so an experiment can swap the search path
without touching scoring, logging or the exit criteria.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from mfe.contracts.tools import Candidate
from mfe.core.cli import setup_console
from mfe.core.config import Config, load_config
from mfe.core.logging import EventType, configure, get_run_id, log_event
from mfe.core.storage import get_backend
from mfe.eval.exit_criteria import check_all
from mfe.eval.metrics import QueryRecord, compute_ttca, recall_at_k, summarize
from mfe.mapping.relational import RelationalMap
from mfe.mapping.validator import Validator
from mfe.online.cascade import Cascade
from mfe.stores.asset.manifest import AssetStore
from mfe.stores.inverted.index import InvertedIndex
from mfe.stores.vector.index import VectorIndex

logger = logging.getLogger(__name__)

SearchFn = Callable[[str, int, str], list[Candidate]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mfe-eval",
        description="Run the query set end-to-end and report metrics (docs/01 §2).",
    )
    parser.add_argument("--queryset", default=None, help="Nhật Bảo's ground truth")
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--out", default=None, help="Per-run output directory")
    return parser


def load_queryset(path: str) -> list[QueryRecord]:
    """Read `eval/queryset.jsonl`, failing loudly on a malformed line.

    A silently skipped line would lower the query count toward the >= 30 gate
    without anyone noticing which query went missing.
    """
    records: list[QueryRecord] = []
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"queryset not found: {source}")
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            records.append(QueryRecord.model_validate_json(raw))
        except Exception as err:
            raise ValueError(f"{source}:{line_number} is not a valid QueryRecord: {err}") from err
    return records


def build_default_search(
    config: Config, inverted: InvertedIndex, vector: VectorIndex | None
) -> SearchFn:
    """The full cascade: CLIP + BM25 -> fusion -> dedup -> rerank.

    Returns [] for a query that matches nothing, rather than raising: an
    unmatched query is a miss to be scored, not a crash.
    """
    cascade = Cascade.from_config(
        config, vector_index=vector, inverted_index=inverted, reranker=None
    )

    def search(query_text: str, top_k: int, run_id: str) -> list[Candidate]:
        return cascade.search(query_text=query_text, top_k=top_k, run_id=run_id)

    return search


def run_submit_probes(validator: Validator, known_good_keyframe_id: str | None) -> list[dict]:
    """Fire one deliberately bad submit per rule of docs/05 §3.

    docs/00 §1 requires "mọi submit sai schema bị validator chặn". A run in
    which nothing was wrong cannot attest to that, so the harness manufactures
    the violations itself. The known-good probe is included on purpose: a
    validator that blocks everything would pass a bad-submits-only test.
    """
    probes: list[dict] = []

    def record(name: str, keyframe_id: str, evidence_id: str | None, should_block: bool) -> None:
        result = validator.validate(
            keyframe_id, evidence_id=evidence_id, schema_format="by_frame"
        )
        probes.append(
            {
                "name": name,
                "should_block": should_block,
                "blocked": not result.valid,
                "codes": [failure.code for failure in result.failures],
            }
        )
        log_event(
            EventType.SUBMIT_BLOCKED if not result.valid else EventType.SUBMIT_ACCEPTED,
            probe=name,
            keyframe_id=keyframe_id,
            expected_block=should_block,
        )

    record("keyframe không tồn tại", "L99_V999#001", "search_text:probe:0", True)
    record("keyframe_id sai định dạng", "not-a-keyframe-id", "search_text:probe:1", True)
    if known_good_keyframe_id:
        record("thiếu bằng chứng", known_good_keyframe_id, None, True)
        record("bằng chứng sai định dạng", known_good_keyframe_id, "no-colons-here", True)
        record(
            "submit hợp lệ (không được chặn)",
            known_good_keyframe_id,
            "search_text:probe:2",
            False,
        )
    else:
        logger.warning(
            "No keyframe in the Relational Map: the evidence-rule probes were skipped. "
            "Run mfe-ingest first."
        )
    return probes


def run(
    queryset_path: str | None = None,
    config_path: str = "config/local.yaml",
    out_dir: str | None = None,
    search_fn: SearchFn | None = None,
) -> dict:
    """Run every query, score it, write the report. Owner: Vỹ Triết.

    Emits one row per query — query_id, type, Recall@K, latency_ms, TTCA_s,
    submit_valid — plus the aggregates of docs/01 §2 and a PASS/FAIL table
    against the "Done" criteria of docs/00 §1.

    A crashing query is a data point, not a reason to stop: it is caught,
    recorded and the run continues. ">= 30 query nội bộ không crash" is only
    measurable if the run reaches the end.
    """
    config = load_config(config_path)
    queryset_path = queryset_path or config.resolve_path(config.eval.queryset)
    out_dir = out_dir or config.paths.eval_results
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    configure(config.paths.event_log)
    run_id = get_run_id()
    logger.info("Run %s — queryset %s", run_id, queryset_path)

    queries = load_queryset(queryset_path)
    backend = get_backend(
        config.storage.backend, root=config.repo_root, volume=config.storage.volume
    )

    relational = RelationalMap(config.paths.relational_db)
    try:
        relational.init_schema()
        inverted = _load_optional(
            "inverted index", lambda: InvertedIndex.load(config.paths.inverted_store)
        ) or InvertedIndex(store_path=config.paths.inverted_store)
        vector = _load_optional(
            "vector index",
            lambda: VectorIndex.load(config.paths.vector_store, config.retrieval.embed_dim),
        )
        asset = _load_optional(
            "asset store",
            lambda: AssetStore.load(config.paths.asset_store, backend=backend),
        )

        search = search_fn or build_default_search(config, inverted, vector)
        rows = [_run_one_query(query, search, config, run_id) for query in queries]

        validator = Validator(relational)
        probes = run_submit_probes(validator, _any_keyframe_id(relational))
        mapping_problems = relational.verify_mapping()
        mapping_rows = relational.count_keyframes()

        summary = summarize(rows)
        summary.update(
            {
                "run_id": run_id,
                "queryset": str(queryset_path),
                "event_log": config.paths.event_log,
                "mapping_problems": mapping_problems,
                "mapping_rows_checked": mapping_rows,
                "submit_probes": probes,
                "throughput": _read_json(config.paths.throughput),
                "stores": {
                    "relational": mapping_rows,
                    "vector": len(vector) if vector is not None else 0,
                    "inverted": len(inverted),
                    "asset": len(asset) if asset is not None else 0,
                },
            }
        )
    finally:
        relational.close()

    criteria = check_all(summary)
    summary["exit_criteria"] = [criterion.model_dump() for criterion in criteria]
    summary["passed"] = all(criterion.passed for criterion in criteria)

    _write_report(Path(out_dir), run_id, rows, summary)
    _print_report(rows, summary, criteria)
    return summary


def _load_optional(label: str, loader: Callable[[], object]):
    """Load a persisted store, or None when ingest has not produced it yet.

    The harness must still run — and still report which criteria fail — on a
    repo where nobody has run `mfe-ingest`. Refusing to start would hide the
    very PASS/FAIL table that says why.
    """
    try:
        return loader()
    except Exception as err:  # noqa: BLE001 — any load failure means "not built yet"
        logger.warning("%s unavailable (%s); run mfe-ingest to build it", label, err)
        return None


def _any_keyframe_id(relational: RelationalMap) -> str | None:
    sample = relational.sample_keyframe_ids(limit=1)
    return sample[0] if sample else None


def _run_one_query(
    query: QueryRecord, search: SearchFn, config: Config, run_id: str
) -> dict:
    """Score one query, converting any crash into a recorded row."""
    log_event(EventType.QUERY_RECEIVED, query_id=query.query_id, text=query.text)
    started_at = time.perf_counter()

    row: dict = {"query_id": query.query_id, "type": query.type, "scored": False}
    try:
        candidates = search(query.text, config.retrieval.rerank_topk, run_id)
    except Exception as err:  # noqa: BLE001 — a crashed query is a data point
        row["error"] = f"{type(err).__name__}: {err}"
        row["latency_ms"] = round((time.perf_counter() - started_at) * 1000.0, 3)
        logger.warning("query %s crashed: %s", query.query_id, err)
        return row

    row["latency_ms"] = round((time.perf_counter() - started_at) * 1000.0, 3)
    row["hits"] = len(candidates)
    ranked = [candidate.keyframe_id for candidate in candidates]
    log_event(
        EventType.CANDIDATES_SHOWN,
        query_id=query.query_id,
        count=len(ranked),
        top=ranked[:5],
    )

    if query.has_ground_truth:
        row["scored"] = True
        for k in config.eval.recall_at:
            row[f"recall_at_{k}"] = recall_at_k(ranked, query.gt_keyframe_ids, k)
        if ranked:
            log_event(EventType.ANSWER_READY, query_id=query.query_id, keyframe_id=ranked[0])
    else:
        # Not a miss — nothing to compare against. See QueryRecord.has_ground_truth.
        row["note"] = "no ground truth; excluded from Recall@K"
        if ranked:
            log_event(EventType.ANSWER_READY, query_id=query.query_id, keyframe_id=ranked[0])

    events = [
        event
        for event in _events_for(query.query_id)
        if event.get("event") in (EventType.QUERY_RECEIVED, EventType.ANSWER_READY)
    ]
    row["ttca_s"] = compute_ttca(events)
    return row


def _events_for(query_id: str) -> list[dict]:
    from mfe.core.logging import get_events

    return [event for event in get_events() if event.get("query_id") == query_id]


def _read_json(path: str) -> dict | None:
    source = Path(path)
    if not source.is_file():
        return None
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_report(out_dir: Path, run_id: str, rows: list[dict], summary: dict) -> None:
    """Write the per-query rows and the aggregate, one file each.

    Named by run_id rather than overwriting a single file: comparing this run to
    last week's is the whole point of keeping the numbers (docs/01 §2).
    """
    (out_dir / f"{run_id}-queries.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows) + "\n",
        encoding="utf-8",
    )
    (out_dir / f"{run_id}-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _print_report(rows: list[dict], summary: dict, criteria: list) -> None:
    """Print the per-query table, the aggregates, then PASS/FAIL."""
    header = f"{'query_id':<10} {'type':<6} {'hits':>5} {'lat_ms':>9} {'ttca_s':>8}  recall"
    print(f"\n{header}")
    print("-" * len(header))
    for row in rows:
        recalls = " ".join(
            f"@{key.rsplit('_', 1)[1]}={row[key]:.0f}"
            for key in row
            if key.startswith("recall_at_")
        )
        note = row.get("error") or recalls or row.get("note", "")
        ttca = row.get("ttca_s")
        print(
            f"{row['query_id']:<10} {row.get('type', ''):<6} {row.get('hits', 0):>5} "
            f"{row.get('latency_ms', 0):>9.2f} "
            f"{(f'{ttca:.3f}' if ttca is not None else '-'):>8}  {note}"
        )

    print(
        f"\nqueries={summary['queries']} scored={summary['scored_queries']} "
        f"crashed={summary['crashed']} | median TTCA="
        f"{summary['ttca_s']['median']} | median latency={summary['latency_ms']['median']}ms "
        f"| recall={summary['recall'] or '-'}"
    )

    print("\nSprint 1 exit criteria (docs/00 §1)")
    for criterion in criteria:
        print(f"  [{'PASS' if criterion.passed else 'FAIL'}] {criterion.name} — {criterion.detail}")
    print(f"\n=> {'PASS' if summary['passed'] else 'FAIL'}\n")


def main() -> None:
    setup_console()
    args = build_parser().parse_args()
    run(queryset_path=args.queryset, config_path=args.config, out_dir=args.out)


if __name__ == "__main__":
    main()
