"""Fold five members' per-video parts into the flat files ingest reads — §6.3, §6.4.

    python scripts/merge_bundle.py --transport modal --volume aic2026
    python scripts/merge_bundle.py --transport local --local-root bundle_local --emit sharded

**One person runs this.** Everything up to here is written by five people into
paths that cannot collide; this is the single step that touches the eight shared
files, and keeping it single-writer is why the collision question never has to be
answered at runtime (§6.2).

It refuses rather than warns. Loaded shard by shard instead, a duplicate key is a
silent upsert and a model mismatch is invisible — the checks below only exist
because the merge puts all five manifests in one place (`mfe.offline.merge`).

Output lands in `<workdir>/out/`; upload it with `--upload` once the report is
clean, or inspect it first and upload on a second run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mfe.core.cli import setup_console
from mfe.offline.bundle import BundleLayout, load_shard_plan
from mfe.offline.merge import merge_bundle
from mfe.offline.transport import build_transport


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="merge_bundle", description=__doc__)
    parser.add_argument("--shards", default="config/ingest_shards.yaml")
    parser.add_argument("--transport", choices=("modal", "local"), default="modal")
    parser.add_argument("--volume", default="aic2026")
    parser.add_argument("--local-root", default="bundle_local")
    parser.add_argument("--workdir", default=None, help="Default: eval/merge/<bundle>")
    parser.add_argument(
        "--emit",
        choices=("single", "sharded"),
        default="single",
        help="single = one flat file per artifact (16 video). sharded = per-video "
        "parts + an index (quy mô batch 1). Phép kiểm giống hệt nhau.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Đẩy data/vectors/ + manifest.json lên bundle sau khi báo cáo sạch.",
    )
    return parser


def main() -> None:
    setup_console()
    args = build_parser().parse_args()

    plan = load_shard_plan(args.shards)
    layout = BundleLayout(plan.bundle)
    transport = build_transport(
        args.transport, volume=args.volume, local_root=args.local_root
    )
    workdir = Path(args.workdir or Path("eval/merge") / plan.bundle)

    print(f"bundle {plan.bundle} · {len(plan.videos)} video · {len(plan.members)} nguoi")
    report = merge_bundle(transport, plan, workdir, emit=args.emit, layout=layout)

    print(f"\n{'artifact':<18} {'manh':>5} {'dong':>8} {'bo':>5}")
    print("-" * 40)
    for name, artifact in report.artifacts.items():
        print(f"{name:<18} {artifact.parts:>5} {artifact.rows:>8} {artifact.dropped:>5}")

    for warning in report.warnings:
        print(f"\nCANH BAO: {warning}")

    out = workdir / "out"
    report_path = workdir / "bundle-merge-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nbao cao: {report_path}")

    if not report.ok:
        print(f"\nTU CHOI GOP — {len(report.refusals)} loi:")
        for refusal in report.refusals:
            print(f"  {refusal}")
        print(
            "\nKhong ghi gi len bundle. Sua manh roi chay lai — day dung la cho de\n"
            "bat nhung loi ma nap tung manh se nuot mat (§6.3)."
        )
        sys.exit(1)

    print(f"\nSACH. Ket qua o {out}")
    if args.upload:
        count = 0
        for path in sorted(out.rglob("*")):
            if not path.is_file():
                continue
            remote = f"{layout.root}/{path.relative_to(out).as_posix()}"
            transport.put(path, remote)
            count += 1
        print(f"da day {count} file len {layout.root}")
    else:
        print("Chua day len bundle. Xem lai roi chay lai voi --upload.")


if __name__ == "__main__":
    main()
