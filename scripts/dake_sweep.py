"""Sweep DAKE's two knobs over cached size curves — no video, no decode.

    python scripts/dake_sweep.py --sizes scratch/sizes --fps 30 --target 700 1055

This is what caching `frame_sizes` was for (§3.4). Decoding is 92% of N1's cost;
once the curve is on disk, trying a hundred parameter pairs is arithmetic.

**Two knobs, and they are not interchangeable.**

`rho` sets the budget and is the *content-aware* part — the frames DAKE actually
picked because the compressed size moved. `max_gap_seconds` is the density floor
the paper adds on top (§5.1, page 10), and it is *content-blind*: it inserts
frames purely to bound the worst-case gap, which is uniform sampling by another
name. Measured on the shipped runs, the floor contributes more than half the
output, so cutting it removes mostly the part that is not DAKE, while cutting
`rho` removes the part that is.

Which to cut is a trade against the reason DAKE was chosen at all — a bounded
worst-case gap, against BTC's measured median 3,36s and worst case 8,4s. Raising
the floor to 3s still beats both. The table this prints is meant to make that
trade visible rather than to make it for you.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mfe.core.cli import setup_console
from mfe.offline.dake import MAX_GAP_SECONDS, RHO, select_keyframes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dake_sweep", description=__doc__)
    parser.add_argument("--sizes", required=True, help="Thu muc chua frame_sizes *.npy")
    parser.add_argument("--fps", type=float, default=30.0, help="Neu khong doc duoc tu ten file")
    parser.add_argument(
        "--rho", type=float, nargs="+", default=[0.005, 0.0075, 0.01, 0.015, 0.02],
    )
    parser.add_argument(
        "--max-gap", type=float, nargs="+", default=[2.0, 2.5, 3.0, 4.0, 5.0],
    )
    parser.add_argument(
        "--target", type=float, nargs=2, default=None,
        help="Khoang keyframe/video mong muon, vd --target 700 1055",
    )
    parser.add_argument("--rank-by", choices=("signed", "absolute"), default="signed")
    return parser


def main() -> None:
    setup_console()
    args = build_parser().parse_args()

    curves = sorted(Path(args.sizes).glob("*.npy"))
    if not curves:
        raise SystemExit(f"khong thay *.npy trong {args.sizes}")
    sizes = {path.stem: np.load(path).tolist() for path in curves}

    # fps per video when known. The density floor is a number of FRAMES, so one
    # global fps silently mis-sizes the window on every 25fps video in a 30fps
    # sweep — and batch 1 is 781 videos at 25 against 61 at 30.
    fps_path = Path(args.sizes) / "fps.json"
    known = json.loads(fps_path.read_text(encoding="utf-8")) if fps_path.is_file() else {}
    fps = {name: float(known.get(name, args.fps)) for name in sizes}
    print(f"{len(sizes)} duong cong: "
          + ", ".join(f"{k} ({len(v):,}f @ {fps[k]:g}fps)" for k, v in sizes.items()))
    if not known:
        print(f"(khong co fps.json — dung {args.fps} cho tat ca)")
    print()

    baseline: dict[str, int] = {}
    for name, curve in sizes.items():
        baseline[name] = len(
            select_keyframes(curve, fps=fps[name], rho=RHO, max_gap_seconds=MAX_GAP_SECONDS)
        )
    mean_base = sum(baseline.values()) / len(baseline)
    print(f"moc nen (rho={RHO}, gap={MAX_GAP_SECONDS}s): "
          + ", ".join(f"{k} {v}" for k, v in baseline.items())
          + f"   TB {mean_base:.0f}\n")

    columns = " ".join(f"{n[-8:]:>9}" for n in sizes)
    header = f"{'rho':>7} {'gap':>6} {columns} {'TB':>7} {'%nen':>6}"
    print(header)
    print("-" * len(header))

    rows = []
    for rho in args.rho:
        for gap in args.max_gap:
            counts = {
                name: len(select_keyframes(
                    curve, fps=fps[name], rho=rho, max_gap_seconds=gap, rank_by=args.rank_by
                ))
                for name, curve in sizes.items()
            }
            mean = sum(counts.values()) / len(counts)
            share = mean / mean_base
            mark = ""
            if args.target and args.target[0] <= mean <= args.target[1]:
                mark = "  <-- trong khoang"
            rows.append((rho, gap, counts, mean, share, mark))
            print(
                f"{rho:>7.4f} {gap:>5.1f}s "
                + " ".join(f"{counts[n]:>9}" for n in sizes)
                + f" {mean:>7.0f} {share:>5.0%}{mark}"
            )

    if args.target:
        low, high = args.target
        fits = [r for r in rows if low <= r[3] <= high]
        print(f"\n{len(fits)} cau hinh roi vao [{low:.0f}, {high:.0f}] keyframe/video.")
        if fits:
            # Prefer the one that keeps the most content-aware selection: highest
            # rho, then the tightest gap that still fits. Cutting the density floor
            # removes uniform sampling; cutting rho removes DAKE itself.
            best = max(fits, key=lambda r: (r[0], -r[1]))
            print(f"Giu duoc nhieu phan CHON NHAT: rho={best[0]}, gap={best[1]}s "
                  f"-> TB {best[3]:.0f} ({best[4]:.0%} so voi moc nen)")
            print("  (uu tien rho cao = giu phan chon theo noi dung; ha gap = bo bot lay mau deu)")


if __name__ == "__main__":
    main()
