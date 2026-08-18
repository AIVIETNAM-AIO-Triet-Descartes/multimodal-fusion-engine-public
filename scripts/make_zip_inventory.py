"""List what is inside BTC's zips **without downloading them** — §5.6.

    python scripts/make_zip_inventory.py
    python scripts/make_zip_inventory.py --kind Keyframes --out config/zips_keyframes.yaml

A zip keeps its central directory at the end of the file, so an HTTP `Range`
request for the last few dozen KB is enough to list every entry. Reading the
inventory this way costs a few hundred KB instead of 77 GB, and it answers the
question the shard plan depends on: **which videos live in which zip**.

That question is not optional. BTC publishes videos in 14 grouped archives, so a
member cannot fetch "their" videos — they fetch whole zips. Assigning work by
video while downloading by zip would make every member pull nearly all 77 GB,
which is precisely what splitting the work was supposed to avoid.

The download host rejects Python's default `User-Agent` with 403 but serves
`curl` fine, so `curl` is what this shells out to. Nothing here needs
credentials: BTC publishes direct links, anonymous
(`docs/triet/dataset_report.md` §A.1).
"""

from __future__ import annotations

import argparse
import io
import struct
import subprocess
import zipfile
from pathlib import PurePosixPath

from mfe.core.cli import setup_console

HOST = "https://aic-data.ledo.io.vn"

#: The archive names, from the spreadsheet the official PDF §3 links to.
VIDEO_ZIPS = (
    "Videos_L21_a", "Videos_L22_a", "Videos_L23_a", "Videos_L24_a", "Videos_L25_a",
    "Videos_L26_a", "Videos_L26_b", "Videos_L26_c", "Videos_L26_d", "Videos_L26_e",
    "Videos_L27_a", "Videos_L28_a", "Videos_L29_a", "Videos_L30_a",
)
KEYFRAME_ZIPS = (
    "Keyframes_L21", "Keyframes_L22", "Keyframes_L23", "Keyframes_L24", "Keyframes_L25",
    "Keyframes_L26_a", "Keyframes_L26_b", "Keyframes_L26_c", "Keyframes_L26_d",
    "Keyframes_L26_e", "Keyframes_L27", "Keyframes_L28", "Keyframes_L29", "Keyframes_L30",
)
OTHER_ZIPS = (
    "clip-features-32-aic25-b1", "map-keyframes-aic25-b1",
    "media-info-aic25-b1", "objects-aic25-b1",
)


def _curl(url: str, *, rng: str | None = None, head: bool = False) -> bytes:
    cmd = ["curl", "-sL", "--max-time", "180"]
    if head:
        cmd.append("-I")
    if rng:
        cmd += ["-r", rng]
    result = subprocess.run([*cmd, url], capture_output=True, check=True)
    return result.stdout


def remote_size(url: str) -> int:
    text = _curl(url, head=True).decode("utf-8", "replace")
    lengths = [
        line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if line.lower().startswith("content-length:")
    ]
    if not lengths:
        raise RuntimeError(f"khong doc duoc Content-Length tu {url}")
    return int(lengths[-1])


def list_remote_zip(url: str) -> tuple[list, int]:
    """Entry infos and the archive's byte size, via two Range requests."""
    size = remote_size(url)
    tail = _curl(url, rng=f"{max(0, size - 65_536)}-{size - 1}")
    end_marker = tail.rfind(b"PK\x05\x06")
    if end_marker < 0:
        raise RuntimeError(f"{url}: khong thay EOCD trong 64KB cuoi")
    cd_size, cd_offset = struct.unpack("<II", tail[end_marker + 12 : end_marker + 20])
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:  # ZIP64
        zip64 = tail.rfind(b"PK\x06\x06")
        cd_size, cd_offset = struct.unpack("<QQ", tail[zip64 + 40 : zip64 + 56])
    # Everything from the central directory to EOF, verbatim — not just the
    # directory. An earlier version sliced from the plain EOCD and so dropped the
    # ZIP64 end-of-central-directory record sitting in front of it; `zipfile` then
    # fell back to the 32-bit fields and added a bogus `concat` to every offset.
    # It showed up as entries pointing past the end of a 5,8 GB archive — and only
    # in archives above 4 GB, i.e. it would have looked correct on the three
    # smallest and been wrong for four of the five members.
    trailer = _curl(url, rng=f"{cd_offset}-{size - 1}")
    blob = bytes(cd_offset) + trailer
    return zipfile.ZipFile(io.BytesIO(blob)).infolist(), size


def balance_by_bytes(
    inventory: dict[str, dict], members: list[str]
) -> dict[str, list[str]]:
    """Largest-first greedy. Balances **bytes**, deliberately not video count.

    N1's cost is decode time, which tracks total duration, which tracks bytes far
    better than it tracks how many files there are — measured across these zips,
    `L26_*` holds ~100 short videos per 6.3 GB while `L28_a` holds 24 long ones in
    6.8 GB. Balancing by count would hand somebody four times the decoding.

    The residual video-count spread (119 to 227) is therefore expected and is not
    a defect; `elapsed_s` in the markers is what would show it to be wrong.
    """
    shares: dict[str, list[str]] = {m: [] for m in members}
    totals = dict.fromkeys(members, 0)
    for name, body in sorted(inventory.items(), key=lambda kv: -kv[1]["bytes"]):
        lightest = min(members, key=lambda m: totals[m])
        shares[lightest].append(name)
        totals[lightest] += body["bytes"]
    return {m: sorted(zips) for m, zips in shares.items()}


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(prog="make_zip_inventory", description=__doc__)
    parser.add_argument("--kind", choices=("Videos", "Keyframes", "other"), default="Videos")
    parser.add_argument("--out", default="config/zips_batch1.yaml")
    parser.add_argument("--members", default="triet,trong,quang,bao,dat")
    args = parser.parse_args()

    names = {"Videos": VIDEO_ZIPS, "Keyframes": KEYFRAME_ZIPS, "other": OTHER_ZIPS}[args.kind]
    members = [m.strip() for m in args.members.split(",") if m.strip()]

    inventory: dict[str, dict] = {}
    for stem in names:
        url = f"{HOST}/{stem}.zip"
        infos, size = list_remote_zip(url)
        mp4s = [i for i in infos if i.filename.lower().endswith(".mp4")]
        stored = {i.compress_type for i in mp4s}
        if stored - {0}:
            # Every mp4 measured so far is STORED (ratio 1.0000), which is what
            # makes per-video Range fetching possible at all. A compressed entry
            # would need the whole archive again, so it must not pass silently.
            raise SystemExit(
                f"{stem}: co entry bi NEN (compress_type={stored}). "
                "mfe.offline.source chi lay duoc entry STORED."
            )
        entries = {
            PurePosixPath(i.filename).stem: (i.header_offset, i.file_size)
            for i in sorted(mp4s, key=lambda i: i.filename)
        }
        inventory[f"{stem}.zip"] = {
            "bytes": size, "videos": sorted(entries), "entries": entries,
        }
        print(f"{stem:<22} {size / 2**30:6.2f} GB  {len(entries):>4} video  STORED")

    total_videos = sum(len(b["videos"]) for b in inventory.values())
    total_bytes = sum(b["bytes"] for b in inventory.values())
    print(f"\nTONG: {total_videos} video, {total_bytes / 2**30:.2f} GB, {len(inventory)} zip")

    shares = balance_by_bytes(inventory, members)
    print(f"\n{'nguoi':<8} {'GB':>6} {'video':>6}  zip")
    for member in members:
        owned = shares[member]
        gigabytes = sum(inventory[z]["bytes"] for z in owned) / 2**30
        count = sum(len(inventory[z]["videos"]) for z in owned)
        print(f"{member:<8} {gigabytes:6.2f} {count:>6}  {', '.join(z[:-4] for z in owned)}")

    lines = [
        "# SINH TU DONG boi scripts/make_zip_inventory.py — do bang HTTP Range,",
        "# khong tai file nao. Chay lai de cap nhat khi BTC phat hanh them.",
        "# `entries` = vi tri tung video trong goi -> lay rieng duoc bang HTTP Range,",
        "# khong phai tai ca goi (mfe.offline.source).",
        f"# {total_videos} video, {total_bytes / 2**30:.2f} GB, {len(inventory)} zip.",
        "",
        f"host: {HOST}",
        "",
        "zips:",
    ]
    for name, body in inventory.items():
        lines.append(f"  {name}:")
        lines.append(f"    bytes: {body['bytes']}")
        lines.append(f"    videos: [{', '.join(body['videos'])}]")
        lines.append("    entries:   # video_id: [header_offset, bytes]")
        for video_id, (offset, nbytes) in body["entries"].items():
            lines.append(f"      {video_id}: [{offset}, {nbytes}]")
    lines += ["", "# Can theo DUNG LUONG (xem balance_by_bytes).", "assignment:"]
    for member in members:
        lines.append(f"  {member}: [{', '.join(shares[member])}]")

    from pathlib import Path  # noqa: PLC0415

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nghi: {out}")


if __name__ == "__main__":
    main()
