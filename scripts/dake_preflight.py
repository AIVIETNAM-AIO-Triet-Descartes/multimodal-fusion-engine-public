"""Run N1 on ONE video and print every number worth checking — §3, §10.2.

    python scripts/dake_preflight.py --video data/sample_2026/videos/L21_V001.mp4
    python scripts/dake_preflight.py --video ... --rho 0.01 --rho 0.02 --rho 0.04

This exists because the decode path is the only part of the pipeline that cannot
be tested without a real video and a real OpenCV, and because committing ~175
videos of machine time to an unverified configuration is the expensive mistake
available at this milestone.

It writes nothing to the Volume and marks nothing done. Run it, read the numbers,
then start the notebook.

**What to look at, and what should worry you**

    fps raw vs used     must be one of batch 1's four values (25, 30, 29.97,
                        26.44). A large gap means this video is not one of them.
    ty le keyframe      target is rho. Far above means the density rule is doing
                        the work, i.e. the video is visually flat.
    max gap             must be <= 2.0s. That bound is the reason DAKE was chosen
                        over BTC's own keyframes (median 3.36s, worst 8.4s).
    hop dong frame_idx  must be 0 rows wrong. Any failure here and the mapping is
                        rejected by our own reader after the decode is paid for.
    thoi gian           pass 1 + pass 2, the number that turns into "how many
                        hours for my share". Multiply by your video count.

`--rho` may be repeated: the size curve is measured once and swept many times,
which is the entire point of caching it (§3.4).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from mfe.core.cli import setup_console
from mfe.offline.dake import (
    RHO,
    cuda_decode_available,
    decode_frames,
    decode_selected,
    jpeg_sizes,
    pts_for_frame,
    select_keyframes,
    video_fps,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dake_preflight", description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--rho", type=float, action="append", default=None, help="Lap lai de quet nhieu gia tri"
    )
    parser.add_argument(
        "--rank-by", choices=("signed", "absolute"), default="signed",
        help="signed = cong thuc paper. Xem dake.RankBy.",
    )
    parser.add_argument("--write-images", default=None, help="Thu muc de ghi thu anh keyframe")
    parser.add_argument("--cache", default=None, help="Duong dan frame_sizes .npy")
    parser.add_argument(
        "--decode-backend", choices=("auto", "cpu", "nvdec"), default="auto",
        help="auto do `ffmpeg -hwaccels`. Bat Accelerator=GPU MOT MINH khong du: "
             "opencv-python khong co CUDA nen no van chay toc do CPU.",
    )
    return parser


def main() -> None:
    setup_console()
    args = build_parser().parse_args()
    video = Path(args.video)
    if not video.is_file():
        raise SystemExit(f"khong thay {video}")
    rhos = args.rho or [RHO]

    backend = args.decode_backend
    if backend == "auto":
        backend = "nvdec" if cuda_decode_available() else "cpu"
    fps, fps_raw = video_fps(str(video))
    print(f"video     : {video.name}  ({video.stat().st_size / 1e6:.1f} MB)")
    print(f"giai ma   : {backend}"
          + ("" if backend == "nvdec" else "   <-- GPU KHONG tham gia giai ma"))
    print(f"fps       : dung {fps}  |  raw {fps_raw:.6f}", end="")
    print("   <-- LECH, kiem lai" if abs(fps - fps_raw) > 0.02 else "")

    cache = Path(args.cache) if args.cache else None
    if cache and cache.is_file():
        sizes = np.load(cache).tolist()
        pass1 = 0.0
        print(f"kich thuoc: doc tu cache {cache} ({len(sizes)} frame)")
    else:
        start = time.perf_counter()
        sizes = list(jpeg_sizes(decode_frames(str(video), backend=backend)))
        pass1 = time.perf_counter() - start
        if cache:
            np.save(cache, np.asarray(sizes, dtype=np.int64))
        rate = len(sizes) / max(pass1, 1e-9)
        print(f"luot 1    : {len(sizes)} frame, {pass1:.1f}s ({rate:.0f} fps giai ma)")

    duration = len(sizes) / fps
    print(f"thoi luong: {duration / 60:.1f} phut")

    print(f"\n{'rho':>6} {'keyframe':>9} {'ty le':>7} {'gap TV':>8} {'gap max':>8}  hop dong")
    print("-" * 56)
    chosen: list[int] = []
    for rho in rhos:
        picked = select_keyframes(sizes, fps=fps, rho=rho, rank_by=args.rank_by)
        gaps = np.diff(picked) if len(picked) > 1 else np.array([0])
        wrong = sum(
            1 for i in picked if math.floor(pts_for_frame(i, fps) * fps) != i
        )
        flag = "OK" if wrong == 0 else f"{wrong} DONG SAI"
        over = "" if gaps.max() / fps <= 2.001 else "  <-- VUOT 2.0s"
        print(
            f"{rho:>6.3f} {len(picked):>9} {len(picked) / len(sizes):>6.2%} "
            f"{np.median(gaps) / fps:>7.2f}s {gaps.max() / fps:>7.2f}s  {flag}{over}"
        )
        if rho == rhos[0]:
            chosen = picked

    start = time.perf_counter()
    written = 0
    out_dir = Path(args.write_images) if args.write_images else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    import cv2  # noqa: PLC0415

    for _index, frame in decode_selected(str(video), chosen):
        written += 1
        if out_dir:
            cv2.imwrite(str(out_dir / f"{written:03d}.jpg"), frame)
    pass2 = time.perf_counter() - start

    print(f"\nluot 2    : lay {written}/{len(chosen)} anh, {pass2:.1f}s")
    if written != len(chosen):
        print("  ^ THIEU ANH — bo giai ma dung som, mapping se khong khop clip-features")

    total = pass1 + pass2
    if total > 0:
        per_hour = total / (duration / 3600)
        print(f"\ntong      : {total:.1f}s cho {duration / 60:.1f} phut video")
        print(f"quy doi   : {per_hour / 60:.1f} phut may / gio video")
        print(f"            175 video (~35 gio video) -> ~{per_hour * 35 / 3600:.1f} gio may")
        if pass1 > 0:
            print(f"            luot 1 chiem {pass1 / total:.0%}, luot 2 {pass2 / total:.0%}")

    print()
    if out_dir:
        written_bytes = sum(f.stat().st_size for f in out_dir.glob("*.jpg"))
        per_image = written_bytes / max(written, 1)
        source = "do that"
    else:
        # Measured on L21_V001 (1280x720, cv2.imwrite default quality): 188 KB.
        per_image = 188 * 1024
        written_bytes = len(chosen) * per_image
        source = "uoc luong"
    print(
        f"dung luong: {per_image / 1024:.0f} KB/anh ({source}), "
        f"{written_bytes / 1e6:.0f} MB/video"
    )
    # Quy doi theo THOI LUONG, khong nhan theo so video: video nay 21 phut,
    # trung binh batch 1 chi ~12 phut (173 gio / 873 video), nen nhan thang so
    # video se thoi phong ~1,8x.
    BATCH1_HOURS = 173.0        # suy tu 177.321 keyframe BTC, xem §5.5b
    kf_per_hour = len(chosen) / (duration / 3600)
    batch1_kf = kf_per_hour * BATCH1_HOURS
    print(f"keyframe  : {kf_per_hour:,.0f}/gio video -> batch 1 ~{batch1_kf:,.0f}")
    print(f"            phan cua ban (1/5) ~{batch1_kf / 5:,.0f}")
    print(f"inode     : ca 5 nguoi ~{batch1_kf:,.0f}", end="")
    print("   <-- VUOT moc cung 500.000 cua Volume v1" if batch1_kf > 500_000 else "")
    print(f"dia Volume: batch 1 ~{batch1_kf * per_image / 1e9:,.0f} GB"
          f"  |  phan cua ban ~{batch1_kf * per_image / 5 / 1e9:,.0f} GB")


if __name__ == "__main__":
    main()
