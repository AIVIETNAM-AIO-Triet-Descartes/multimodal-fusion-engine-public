"""Fetch one video out of BTC's archives without downloading the archive — §5.6.

Every `.mp4` inside `Videos_*.zip` is **stored, not compressed** — measured on
`Videos_L25_a.zip`: `compress_type = 0` for all 88 entries, compression ratio
exactly 1.0000. Which makes sense; H.264 does not deflate.

So each video is a contiguous run of bytes at a known offset, and one HTTP `Range`
request retrieves exactly it. That removes the archive from the picture entirely:

* **Nothing large ever lands on disk.** The biggest archive is 11,97 GB against a
  ~20 GB Kaggle working directory, and unpacking it would have needed 12 GB more.
  Now the peak is one video, ~160 MB.
* **A dead session costs one video, not one archive.** Losing a partial 12 GB
  download to a session timeout was the single worst interruption available, and
  it is the one Kaggle sessions produce most often.
* **Resume granularity finally matches the markers.** Work is tracked per video,
  so fetching per video means the two agree instead of the download being coarser
  than the thing it feeds.

Offsets come from `config/zips_batch1.yaml`, read once by
`scripts/make_zip_inventory.py` from each archive's central directory. The
notebook therefore makes **zero** exploratory requests.

The host rejects Python's default `User-Agent` with 403 but serves `curl` fine,
so `curl` is what this shells out to. No credentials anywhere: BTC publishes
direct links, anonymous (`docs/triet/dataset_report.md` §A.1).
"""

from __future__ import annotations

import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: Fixed part of a zip local file header, before the variable-length name/extra.
_LOCAL_HEADER = 30


@dataclass(frozen=True)
class VideoEntry:
    """Where one video's bytes live inside an archive."""

    video_id: str
    zip_name: str
    url: str
    header_offset: int
    size: int


def _curl_range(url: str, start: int, end: int, dest: Path | None = None) -> bytes:
    command = ["curl", "-sL", "--fail", "--max-time", "1800", "-r", f"{start}-{end}"]
    if dest is not None:
        command += ["-o", str(dest)]
    result = subprocess.run([*command, url], capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"curl that bai ({result.returncode}) khi lay {start}-{end} tu {url}: "
            f"{result.stderr.decode('utf-8', 'replace')[:200]}"
        )
    return result.stdout


def fetch_video(entry: VideoEntry, dest: Path) -> Path:
    """Download exactly one video's bytes. Two requests, the second is the data.

    The archive's central directory records where each entry *starts*, but the
    payload does not begin there — a local file header sits in front, and its
    length depends on the name and extra fields it carries. Skipping that step
    and reading from `header_offset` would prepend ~60 bytes of header to the mp4,
    which most players tolerate and every decoder counts differently. So the
    header is read and parsed rather than assumed.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    head = _curl_range(entry.url, entry.header_offset, entry.header_offset + _LOCAL_HEADER - 1)
    if len(head) < _LOCAL_HEADER or head[:4] != b"PK\x03\x04":
        raise RuntimeError(
            f"{entry.video_id}: khong thay local file header o offset "
            f"{entry.header_offset} trong {entry.zip_name}"
        )
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    start = entry.header_offset + _LOCAL_HEADER + name_len + extra_len

    _curl_range(entry.url, start, start + entry.size - 1, dest=dest)
    actual = dest.stat().st_size
    if actual != entry.size:
        # A short read is not a smaller video, it is a truncated one, and the
        # decoder would simply produce fewer frames — a silently wrong keyframe
        # set rather than an error.
        #
        # This catches a transfer that ended early. It does **not** catch a wrong
        # length mid-archive: over-reading there just grabs the next entry's bytes
        # and the length comes out right. Wrong *offsets* are caught by the header
        # magic above, which is why both checks are here.
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"{entry.video_id}: lay ve {actual:,} byte, cho doi {entry.size:,}. "
            "Da xoa file cat cut — chay lai video nay."
        )
    return dest


def load_catalogue(path: str | Path = "config/zips_batch1.yaml") -> dict[str, VideoEntry]:
    """`{video_id: VideoEntry}` for every video the inventory covers."""
    import yaml  # noqa: PLC0415

    book = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    host = book["host"].rstrip("/")
    out: dict[str, VideoEntry] = {}
    for zip_name, body in book["zips"].items():
        url = f"{host}/{zip_name}"
        for video_id, (offset, size) in body["entries"].items():
            out[video_id] = VideoEntry(video_id, zip_name, url, int(offset), int(size))
    return out
