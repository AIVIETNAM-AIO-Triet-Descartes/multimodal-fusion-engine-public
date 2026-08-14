"""time_to_frame.py — Đổi mốc thời gian trên video thành số frame cho ground truth.

    python scripts/time_to_frame.py L23_V001 12.04            # một mốc
    python scripts/time_to_frame.py L23_V001 12.04 25.00      # một khoảng
    python scripts/time_to_frame.py L23_V001 --keyframes 7 10 # frame của keyframe 7..10

Viết ground truth là đọc mốc thời gian trên trình phát video rồi đổi ra frame.
Phép đổi là `floor(giây × fps)` — đúng công thức BTC dùng để sinh cột
`frame_idx`, đã kiểm khớp 100% trên 4.001 dòng. Nhưng **fps khác nhau theo từng
video** (bộ mẫu có cả 25 và 29.97), nên nhẩm bằng một con số duy nhất là sai với
phần lớn video. Script này đọc fps từ chính `map-keyframes/{video_id}.csv`.

Không hiển thị được frame trên trình phát? Mọi player đều cho tua theo giây, và
giây là đủ: sai một phần mười giây ở fps 25 là 2-3 frame, nhỏ hơn nhiều so với
độ rộng cửa sổ hẹp mà bạn sẽ đặt.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

DATA = Path("data/sample_2026")


def read_map(video_id: str) -> tuple[float, list[tuple[int, float, int]]]:
    """(fps, [(n, pts_time, frame_idx), ...]) từ map-keyframes của video."""
    path = DATA / "map-keyframes" / f"{video_id}.csv"
    if not path.is_file():
        raise SystemExit(
            f"Không thấy {path}.\n"
            f"Kiểm lại video_id, hoặc video này không thuộc bộ mẫu của bạn."
        )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            (int(r["n"]), float(r["pts_time"]), int(r["frame_idx"]))
            for r in csv.DictReader(handle)
        ]
    fps_values = set()
    with path.open(encoding="utf-8", newline="") as handle:
        fps_values = {float(r["fps"]) for r in csv.DictReader(handle)}
    if len(fps_values) != 1:
        raise SystemExit(f"{video_id}: CSV khai nhiều fps khác nhau {sorted(fps_values)}")
    return fps_values.pop(), rows


def frame_of(seconds: float, fps: float) -> int:
    """Công thức BTC dùng, không phải làm tròn. Xem contracts/ids.py."""
    return math.floor(seconds * fps)


def nearest_keyframes(
    frame: int, rows: list[tuple[int, float, int]]
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """(keyframe ngay trước hoặc trùng, keyframe ngay sau) theo frame_idx."""
    before = None
    after = None
    for n, _pts, frame_idx in rows:
        if frame_idx <= frame:
            before = (n, frame_idx)
        elif after is None:
            after = (n, frame_idx)
    return before, after


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Đổi mốc thời gian trên video thành số frame cho ground truth."
    )
    parser.add_argument("video_id", help="Ví dụ L23_V001")
    parser.add_argument(
        "seconds", nargs="*", type=float, help="Một mốc, hoặc hai mốc (đầu và cuối)."
    )
    parser.add_argument(
        "--keyframes",
        nargs=2,
        type=int,
        metavar=("N_DAU", "N_CUOI"),
        help="In frame_idx của khoảng keyframe N_DAU..N_CUOI (để dựng cửa sổ cảnh).",
    )
    args = parser.parse_args()

    fps, rows = read_map(args.video_id)
    print(f"{args.video_id}: fps {fps}, {len(rows)} keyframe, "
          f"frame cuối {rows[-1][2]} (~{rows[-1][1]:.1f}s)\n")

    if args.keyframes:
        first, last = args.keyframes
        picked = [r for r in rows if first <= r[0] <= last]
        if not picked:
            raise SystemExit(f"Không có keyframe nào trong khoảng n={first}..{last}")
        print(f"{'keyframe':>9} {'pts_time':>10} {'frame_idx':>10}")
        for n, pts, frame_idx in picked:
            print(f"{n:>9} {pts:>10.2f} {frame_idx:>10}")
        print(f"\n  gt_frame_range: [{picked[0][2]}, {picked[-1][2]}]")
        return 0

    if not args.seconds:
        parser.error("cần ít nhất một mốc giây, hoặc dùng --keyframes")

    frames = [frame_of(s, fps) for s in args.seconds]
    for s, f in zip(args.seconds, frames, strict=True):
        before, after = nearest_keyframes(f, rows)
        print(f"  {s:>8.2f}s  ->  frame {f}")
        print(
            f"            keyframe gần nhất: "
            f"{f'n={before[0]} @ {before[1]}' if before else '(trước đầu video)'}"
            f"  |  {f'n={after[0]} @ {after[1]}' if after else '(sau cuối video)'}"
        )

    if len(frames) == 2:
        lo, hi = sorted(frames)
        print(f"\n  cửa sổ: [{lo}, {hi}]   ({hi - lo + 1} frame)")
        if hi - lo + 1 > 300:
            print("  (rộng hơn 300 frame — chắc là cửa sổ cảnh, không phải cửa sổ hẹp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
