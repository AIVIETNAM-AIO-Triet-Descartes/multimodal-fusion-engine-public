"""Write the video-id list that `assign: round_robin` deals out — §6.1.

    python scripts/make_video_list.py --from-drive gdrive:AIC2026/videos
    python scripts/make_video_list.py --from-dir data/sample_2026/videos
    python scripts/make_video_list.py --from-maps data/batch1/map-keyframes

Batch 1 is 873 videos, so the ids come from a listing rather than from anyone's
typing. Whichever source is used, the output is the same: one id per line, sorted,
which is what makes `round_robin` reproducible — deal the same list twice and
every member gets the same videos, so a re-run never silently reassigns work
somebody has already done.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from mfe.contracts.ids import video_group
from mfe.core.cli import setup_console

VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm")


def from_dir(path: Path) -> list[str]:
    return sorted(p.stem for p in path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)


def from_maps(path: Path) -> list[str]:
    """One `map-keyframes/{video_id}.csv` per video — the listing BTC ships."""
    return sorted(p.stem for p in path.glob("*.csv"))


def from_drive(remote: str) -> list[str]:
    result = subprocess.run(
        ["rclone", "lsf", remote, "--include", "*.mp4"],
        check=True, capture_output=True, text=True,
    )
    return sorted(
        line.strip().removesuffix(".mp4")
        for line in result.stdout.splitlines()
        if line.strip()
    )


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(prog="make_video_list", description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-dir", help="Thu muc chua file video")
    source.add_argument("--from-maps", help="Thu muc map-keyframes/")
    source.add_argument("--from-drive", help="Remote rclone, vd gdrive:AIC2026/videos")
    parser.add_argument("--out", default="config/videos_batch1.txt")
    parser.add_argument("--members", type=int, default=5, help="Chi de in truoc phan chia")
    args = parser.parse_args()

    if args.from_dir:
        videos, origin = from_dir(Path(args.from_dir)), args.from_dir
    elif args.from_maps:
        videos, origin = from_maps(Path(args.from_maps)), args.from_maps
    else:
        videos, origin = from_drive(args.from_drive), args.from_drive

    if not videos:
        raise SystemExit(f"khong tim thay video nao o {origin}")

    duplicates = len(videos) - len(set(videos))
    if duplicates:
        raise SystemExit(f"{duplicates} id trung trong listing — kiem tra {origin}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# {len(videos)} video, liet ke tu {origin}\n" + "\n".join(videos) + "\n",
        encoding="utf-8",
    )

    # `video_group`, KHONG phai `split("_")`: BTC dung ca gach duoi (`M01_V001`)
    # lan gach ngang (`N001-V001`), doc tu chinh zip goc 23-09.
    groups: dict[str, int] = {}
    for video_id in videos:
        groups[video_group(video_id)] = groups.get(video_group(video_id), 0) + 1

    print(f"{len(videos)} video -> {out}")
    print(f"nhom: {', '.join(f'{g} {n}' for g, n in sorted(groups.items()))}")
    print(f"\nchia {args.members} nguoi (vong tron): ~{len(videos) // args.members} video/nguoi")
    per_person_hours = len(videos) / args.members * (173 / 873)
    print(
        f"uoc luong: ~{per_person_hours:.0f} gio video/nguoi, "
        f"~{per_person_hours * 1.54:.0f} gio may -> "
        f"~{per_person_hours * 1.54 / 30:.1f} tuan quota GPU Kaggle"
    )
    print("(he so 1,54 gio may/gio video do tren bo 16 mau — xem §10)")


if __name__ == "__main__":
    main()
