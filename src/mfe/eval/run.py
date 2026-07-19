"""Benchmark harness entrypoint — docs/01 §2.

    python -m mfe.eval.run --queryset eval/queryset.jsonl \
        --config config/local.yaml --out eval/results/
    mfe-eval --queryset eval/queryset.jsonl --config config/local.yaml --out eval/results/

docs/01 §2 on why this exists: it is what tells the team whether there is real
progress — "vai hay bị bỏ quên nhất". Nobody's workstream feels blocked without
it, and that is exactly the problem.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mfe-eval",
        description="Run the query set end-to-end and report metrics (docs/01 §2).",
    )
    parser.add_argument("--queryset", default="eval/queryset.jsonl", help="Nhật Bảo's ground truth")
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--out", default="eval/results/", help="Per-run output directory")
    return parser


def run(queryset_path: str, config_path: str, out_dir: str) -> None:
    """Run every query, score it, write the report.

    Owner: Vỹ Triết. TODO(Vỹ Triết): implement.

    Per docs/01 §2, emit one row per query:

        query_id | type | Recall@1 | Recall@5 | latency_ms | TTCA_s | submit_valid

    plus aggregates: median TTCA, mean Recall@K by `type`, share of valid
    submits. And print PASS/FAIL against the "Done" criteria of docs/00 §1 —
    >= 30 queries without a crash, mapping 100%, every bad submit blocked.

    A crashing query is a data point, not a reason to stop: catch, record, keep
    going. "Chạy được >= 30 query nội bộ không crash" is measurable only if the
    run reaches the end.
    """
    raise NotImplementedError("TODO(Vỹ Triết): run queryset -> metrics -> report + PASS/FAIL")


def main() -> None:
    args = build_parser().parse_args()
    run(queryset_path=args.queryset, config_path=args.config, out_dir=args.out)


if __name__ == "__main__":
    main()
