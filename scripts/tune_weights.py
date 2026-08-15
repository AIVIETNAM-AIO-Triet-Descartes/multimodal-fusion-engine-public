"""Tune stage-A fusion weights by measurement — ke-hoach-merge-sota.md §3.4b B3.

    python scripts/tune_weights.py --config config/pilot.yaml --queryset eval/dev/pilot.jsonl
    python scripts/tune_weights.py --config config/pilot.yaml --ablation

Retrieval runs **once** per query; every weight vector is then scored against
that cached evidence (see `mfe.eval.tuning` for why, and for what OFF means as
distinct from a weight of 0).

Two rules this script cannot enforce but the reader must:

* **Tune on `eval/dev/*`, publish on `eval/heldout/*`** (§3.4b B4). Skipping that
  turns the weights into a 115-query memo that collapses at the real contest.
* **9 pilot queries are not enough to chase 0.01.** Final Score saturates once a
  hit is in the top 5, so read the mean-rank column too, and treat a gain under
  ~0.03 on this set as noise rather than a result.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from mfe.core.cli import setup_console
from mfe.core.config import load_config
from mfe.eval.metrics.final_score import MAX_ANSWERS_PER_QUERY
from mfe.eval.official import QAQuery, TRAKEQuery, load_official_queryset
from mfe.eval.tuning import (
    OFF,
    WEIGHT_GRID,
    Assignment,
    QueryEvidence,
    ablation,
    coordinate_ascent,
)
from mfe.mapping.relational import RelationalMap
from mfe.online.cascade import Cascade

logger = logging.getLogger("tune_weights")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tune_weights", description=__doc__)
    parser.add_argument("--config", default="config/pilot.yaml")
    parser.add_argument("--queryset", default=None, help="Defaults to eval.queryset")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--objective",
        choices=("wide", "strict", "both"),
        default="wide",
        help="wide = the window BTC scores; strict = the exact-frame window",
    )
    parser.add_argument(
        "--no-off",
        action="store_true",
        help="Sweep only the {0, 0.5, 1, 2, 4} grid, without the OFF option.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Skip the ascent; just report the full set minus each arm in turn.",
    )
    parser.add_argument(
        "--en-source",
        choices=("manual", "auto", "none"),
        default="manual",
        help="Where the English query comes from (see scripts/measure_arms.py).",
    )
    parser.add_argument("--out", default=None, help="Report path (JSON)")
    return parser


def collect_evidence(cascade: Cascade, queries, coarse_topk: int, en_source: str):
    """Query every arm once per query (per event for TRAKE)."""
    evidences = []
    for query in queries:
        if en_source == "none":
            text_en = None
        elif en_source == "manual":
            text_en = query.text_en
        else:
            text_en = cascade.resolve_english(query.text, None)

        if isinstance(query, TRAKEQuery):
            searches = query.searches()
        elif isinstance(query, QAQuery):
            searches = [query.retrieval_text or query.text]
        else:
            searches = [query.text]

        per_event = [
            cascade.arms.run(text, coarse_topk, query.query_id, query_text_en=text_en)
            for text in searches
        ]
        evidences.append(QueryEvidence(query=query, per_event=per_event))
        logger.info(
            "%s: %d arm(s) answered", query.query_id, len(evidences[-1].arm_names)
        )
    return evidences


def main() -> None:
    setup_console()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()

    config = load_config(args.config)
    queryset = args.queryset or config.resolve_path(config.eval.queryset)
    queries = load_official_queryset(queryset)
    top_k = min(config.retrieval.rerank_topk, MAX_ANSWERS_PER_QUERY)

    relational = RelationalMap(config.paths.relational_db)
    relational.init_schema()
    cascade = Cascade.from_config(config)

    if cascade.reranker is not None:
        # Stage C reorders *after* fusion, so a cached-evidence sweep would tune
        # weights whose effect the reranker then partly overwrites — and paying
        # ~12 s/query × 180 evaluations is not an option either.
        raise SystemExit(
            "models.reranker is set. Tune with stage C off (it runs after fusion "
            "and would blur exactly what this measures), then re-check the winner "
            "with the reranker back on."
        )

    fusion = config.retrieval.fusion
    normalizer, rrf_k = fusion.resolved_normalizer(), fusion.rrf_k

    evidences = collect_evidence(cascade, queries, config.retrieval.coarse_topk, args.en_source)
    if cascade.translator is not None:
        cascade.translator.save_cache()

    # Only tune arms that actually returned something: an unavailable arm has no
    # weight worth choosing, and including it would waste a sixth of the sweep
    # measuring a no-op.
    answering = sorted({name for ev in evidences for name in ev.arm_names})
    arm_order = [arm.name for arm in cascade.arms if arm.name in answering]
    silent = [arm.name for arm in cascade.arms if arm.name not in answering]
    print(f"\ntuning {len(arm_order)} arm(s): {', '.join(arm_order)}")
    if silent:
        print(f"khong tham gia (khong tra ve gi tren bo nay): {', '.join(silent)}")

    weights = config.retrieval.fusion_weights.model_dump()
    start = Assignment({name: float(weights.get(name, 1.0)) for name in arm_order})
    common = dict(normalizer=normalizer, rrf_k=rrf_k, top_k=top_k)

    if args.ablation:
        rows = ablation(
            evidences, start, arm_order, relational, cascade.group, **common
        )
        print(f"\n{'cau hinh':<20} {'RONG':>7} {'HEP':>7} {'hangTB':>8} {'hit':>6}")
        print("-" * 52)
        baseline = rows[0][1]
        for label, measurement in rows:
            delta = measurement.wide - baseline.wide
            mark = "" if label == "<full>" else f"  ({delta:+.3f})"
            strict = f"{measurement.strict:.3f}" if measurement.strict is not None else "-"
            rank = (
                f"{measurement.mean_first_hit_rank:.1f}"
                if measurement.mean_first_hit_rank is not None
                else "-"
            )
            print(
                f"{label:<20} {measurement.wide:>7.3f} {strict:>7} {rank:>8} "
                f"{measurement.hits:>3}/{measurement.n_queries}{mark}"
            )
        print(
            "\nDau (+) o cot cuoi = bo arm do LAM DIEM TOT LEN, tuc dong gop bien AM.\n"
            "Hang TB doc doc lap voi Final Score: 9 cau thi Final Score bao hoa som."
        )
        report = {
            "kind": "ablation",
            "queryset": str(queryset),
            "arms": arm_order,
            "rows": [
                {
                    "config": label,
                    "wide": m.wide,
                    "strict": m.strict,
                    "mean_first_hit_rank": m.mean_first_hit_rank,
                    "hits": m.hits,
                }
                for label, m in rows
            ],
        }
    else:
        def on_step(step) -> None:
            cells = " ".join(
                f"{'OFF' if value == OFF else value}:{m.objective(args.objective):.3f}"
                for value, m in step.tried
            )
            arrow = f"{step.before} -> {step.after}" if step.changed else f"giu {step.before}"
            print(f"  [vong {step.round_index}] {step.arm:<18} {cells}   => {arrow}")

        print(
            f"\nleo doc: grid {list(WEIGHT_GRID)}"
            f"{' + OFF' if not args.no_off else ''}, "
            f"toi da {args.rounds} vong, muc tieu = {args.objective}\n"
        )
        result = coordinate_ascent(
            evidences,
            start,
            arm_order,
            relational,
            cascade.group,
            allow_off=not args.no_off,
            rounds=args.rounds,
            objective=args.objective,
            on_step=on_step,
            **common,
        )
        print(f"\nmoc nen (trong so trong config): {result.baseline.summary()}")
        print(f"sau leo doc                     : {result.best.summary()}")
        print(f"vong da chay: {result.rounds_run}\n")
        print("trong so chot:")
        for name in arm_order:
            value = result.assignment.settings[name]
            was = start.settings[name]
            mark = "  <-- doi" if value != was else ""
            print(f"  {name:<18} {'OFF' if value == OFF else value}{mark}")
        gain = result.best.objective(args.objective) - result.baseline.objective(
            args.objective
        )
        print(
            f"\nchenh lech muc tieu: {gain:+.3f} tren {len(queries)} cau. "
            "Duoi ~0.03 tren bo pilot thi coi la nhieu, khong phai ket qua.\n"
            "CHUA duoc cong bo: chinh tren dev, phai do lai tren eval/heldout (§3.4b B4)."
        )
        report = {
            "kind": "coordinate_ascent",
            "queryset": str(queryset),
            "objective": args.objective,
            "rounds_run": result.rounds_run,
            "grid": list(WEIGHT_GRID),
            "allow_off": not args.no_off,
            "baseline": {"wide": result.baseline.wide, "strict": result.baseline.strict},
            "best": {"wide": result.best.wide, "strict": result.best.strict},
            "weights": {
                name: ("off" if value == OFF else value)
                for name, value in result.assignment.settings.items()
            },
            "steps": [
                {
                    "round": step.round_index,
                    "arm": step.arm,
                    "before": step.before,
                    "after": step.after,
                    "tried": [
                        {"value": value, "wide": m.wide, "strict": m.strict}
                        for value, m in step.tried
                    ],
                }
                for step in result.steps
            ],
        }

    out = Path(args.out or Path(config.paths.eval_results) / "weight-tuning.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"bao cao: {out}")
    relational.close()


if __name__ == "__main__":
    main()
