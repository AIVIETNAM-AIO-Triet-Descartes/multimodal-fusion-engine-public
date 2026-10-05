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
from mfe.offline.dake import (
    MAX_GAP_SECONDS,
    MIN_GAP_SECONDS,
    RHO,
    select_keyframes,
)


def gap_stats(gaps: list[float]) -> dict[str, float]:
    """Trung vi va ty le khoang cach ngan, tren khoang cach da quy ve GIAY.

    Ty le khoang ngan la thu duy nhat noi len keyframe co gan trung nhau hay
    khong, va no **khong suy ra duoc tu tong so keyframe**: hai cau hinh cung cho
    400 keyframe/video co the mot ben rai deu va mot ben dinh chum.
    """
    if not gaps:
        return {"median": 0.0, "under_half": 0.0, "under_one": 0.0}
    ordered = sorted(gaps)
    return {
        "median": ordered[len(ordered) // 2],
        "under_half": sum(1 for g in gaps if g < 0.5) / len(gaps),
        "under_one": sum(1 for g in gaps if g < 1.0) / len(gaps),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dake_sweep", description=__doc__)
    parser.add_argument("--sizes", required=True, help="Thu muc chua frame_sizes *.npy")
    parser.add_argument("--fps", type=float, default=30.0, help="Neu khong doc duoc tu ten file")
    parser.add_argument(
        "--rho", type=float, nargs="+", default=[0.005, 0.0075, 0.01, 0.015, 0.02],
    )
    parser.add_argument(
        "--max-gap", type=float, nargs="+", default=[2.0, 3.0, 6.0, 8.0],
        help="Khoang cach TOI DA giua hai keyframe, giay — luat mat do",
    )
    parser.add_argument(
        "--min-gap", type=float, nargs="+", default=[MIN_GAP_SECONDS],
        help=(
            "Khoang cach TOI THIEU giua hai keyframe duoc CHON, giay. "
            "Do 19-08-2026: day la can gat quyet dinh ty le keyframe gan trung "
            "nhau — o 0,20s thi 36%% khoang cach duoi nua giay, o 1,0s con 1-3%% "
            "— va truoc do script nay khong quet duoc no."
        ),
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
            select_keyframes(
                curve, fps=fps[name], rho=RHO,
                min_gap_seconds=MIN_GAP_SECONDS, max_gap_seconds=MAX_GAP_SECONDS,
            )
        )
    mean_base = sum(baseline.values()) / len(baseline)
    print(f"moc nen (rho={RHO}, min {MIN_GAP_SECONDS}s, max {MAX_GAP_SECONDS}s): "
          + ", ".join(f"{k} {v}" for k, v in baseline.items())
          + f"   TB {mean_base:.0f}\n")

    header = (
        f"{'rho':>7} {'min':>6} {'max':>6} {'TB':>7} {'%nen':>6} "
        f"{'kc tv':>7} {'<0,5s':>6} {'<1s':>6}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for rho in args.rho:
        for min_gap in args.min_gap:
            for gap in args.max_gap:
                counts: dict[str, int] = {}
                gaps: list[float] = []
                for name, curve in sizes.items():
                    picked = select_keyframes(
                        curve, fps=fps[name], rho=rho,
                        min_gap_seconds=min_gap, max_gap_seconds=gap,
                        rank_by=args.rank_by,
                    )
                    counts[name] = len(picked)
                    # Quy ve GIAY truoc khi gop: gop bang frame la tron 25fps voi
                    # 30fps thanh mot phan bo khong doc duoc.
                    gaps += [
                        (b - a) / fps[name]
                        for a, b in zip(picked, picked[1:], strict=False)
                    ]
                mean = sum(counts.values()) / len(counts)
                share = mean / mean_base
                stats = gap_stats(gaps)
                mark = ""
                if args.target and args.target[0] <= mean <= args.target[1]:
                    mark = "  <-- trong khoang"
                rows.append((rho, gap, min_gap, counts, mean, share, stats, mark))
                print(
                    f"{rho:>7.4f} {min_gap:>5.2f}s {gap:>5.1f}s {mean:>7.0f} "
                    f"{share:>5.0%} {stats['median']:>6.2f}s "
                    f"{stats['under_half']:>5.0%} {stats['under_one']:>5.0%}{mark}"
                )

    print(
        "\n`<0,5s` va `<1s`: ty le khoang cach ngan hon nguong do — tuc ty le "
        "keyframe gan trung nhau.\nDo la thu cot `min` dieu khien, va la ly do "
        "hai cot do co mat: mot cau hinh cho IT\nkeyframe hon nhung `<0,5s` cao "
        "hon la cau hinh te hon o cung mot gia."
    )

    if args.target:
        low, high = args.target
        fits = [r for r in rows if low <= r[4] <= high]
        print(f"\n{len(fits)} cau hinh roi vao [{low:.0f}, {high:.0f}] keyframe/video.")
        if fits:
            # Prefer the one that keeps the most content-aware selection: highest
            # rho, then the tightest gap that still fits. Cutting the density floor
            # removes uniform sampling; cutting rho removes DAKE itself.
            best = max(fits, key=lambda r: (r[0], -r[1]))
            print(f"Giu duoc nhieu phan CHON NHAT: rho={best[0]}, "
                  f"min={best[2]}s, max={best[1]}s "
                  f"-> TB {best[4]:.0f} ({best[5]:.0%} so voi moc nen)")
            print("  (uu tien rho cao = giu phan chon theo noi dung; ha gap = bo bot lay mau deu)")


if __name__ == "__main__":
    main()
