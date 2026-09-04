"""The ID model — docs/00 §3.

Every store keys off `keyframe_id`. No store may mint its own identifiers.
This module is the only place that knows how a `keyframe_id` is spelled; if you
find yourself writing an f-string with a `#` in it, call in here instead.

Anchored on the real dataset (Kaggle AI Challenge 2025). See
docs/adr/0002-id-format.md.

    keyframe_id = {video_id}#{frame_id:06d}        e.g. L21_V001#031124
    frame_id    = the real frame number in the source video (CSV `frame_idx`).

**This used to key off the ordinal `n`** (`L21_V001#001`, matching the image
filename `001.jpg`). It no longer does, and the reason is stability: `n` is a
property of *the extraction run*, not of the video. Re-run keyframe extraction
with a different threshold and every ordinal shifts, which silently invalidates
any ground truth or index built against the old numbering. `frame_id` does not
move. Two extraction runs that both pick frame 31124 produce the same id, so two
keyframe sets can be compared as sets (docs/triet/ke-hoach-merge-sota.md §3.1).

The ordinal survives as a plain column (`KeyframeRow.keyframe_index`) — it still
backs rank-based neighbour windows and the trace back to BTC's CSV — but it is
no longer an identity.

One consequence that must not be forgotten: `frame_idx` is **not unique** in the
shipped keyframe set. 614 adjacent pairs out of 177,321 in batch 1 (5 rows of
1240 in the sample) share a frame_idx, because BTC extracted two images from
timestamps less than one frame apart — they denote the same source frame. Ingest
collapses them; see `mfe.offline.keyframes.collapse_duplicate_frames`.

`frame_idx` (the real frame in the source video) is derivable, with two
conditions that are not optional. Rechecked on 2026-08-26: 4,615/4,615 rows in
the current 16-video sample and 87,305/87,305 populated rows in the 299-CSV
archive match exactly (one archive row has a blank source `frame_idx`):

    frame_idx = floor(pts_time * fps)        100% exact on every row

    - It must be **floor**, not round and not ceil. On the same two data sets,
      `round` disagreed with 1,152 and 5,272 populated rows respectively. An
      earlier revision of this module forbade derivation entirely on the
      strength of the `round` measurement; that conclusion was wrong, it was
      the rounding mode that was wrong.
    - `fps` must be the **literal value the dataset states** (29.97), not the
      exact NTSC rational it approximates (30000/1001). Substituting the
      "more correct" rational drops accuracy from 100% to 41%, because BTC
      themselves computed with the truncated number.

When a `map-keyframes` CSV is available, prefer reading `frame_idx` from it —
copying beats recomputing. `frame_idx_from_pts` exists for the case the CSV does
not cover: keyframes the team extracts itself (docs PDF §3 states the shipped
keyframes are an aid, not the competition data).
"""

from __future__ import annotations

import math
import re

from mfe.contracts.errors import ContractError, ErrorCode

#: Zero-pad width of `frame_id` inside a `keyframe_id`. Six digits cover 999,999
#: frames — 9.2 hours at 30fps, against a longest shipped video of ~35k frames.
#: The padding is not cosmetic: it makes lexicographic order match numeric order,
#: which matters for sorted listings and for range scans on a VARCHAR primary key
#: (Milvus, see src/mfe/stores/production/). Frames >= 10^6 simply use more
#: digits; parsing accepts 6+.
FRAME_ID_DIGITS = 6

#: Width of the zero-padded shot number inside a `shot_id` (`L21_V001#s07`).
SHOT_DIGITS = 2

_KEYFRAME_RE = re.compile(
    rf"^(?P<video_id>[^\s#]+)#(?P<frame_id>\d{{{FRAME_ID_DIGITS},}})$"
)
_SHOT_RE = re.compile(r"^(?P<video_id>[^\s#]+)#s(?P<shot>\d+)$")
_VIDEO_ID_RE = re.compile(r"^[^\s#]+$")


def validate_video_id(video_id: str) -> str:
    """Return `video_id` unchanged, or raise if it can't key the stores.

    A `video_id` may not contain whitespace (docs/00 §7) nor `#` (it would make
    `keyframe_id` ambiguous to parse).
    """
    if not video_id or not _VIDEO_ID_RE.match(video_id):
        raise ContractError(
            ErrorCode.INVALID_ID,
            f"video_id must be non-empty and contain no whitespace or '#': {video_id!r}",
        )
    return video_id


def build_keyframe_id(video_id: str, frame_id: int) -> str:
    """Build the global keyframe key: ``{video_id}#{frame_id:06d}``.

    `frame_id` is the source-video frame number (CSV `frame_idx`), *not* the
    keyframe's ordinal. `L21_V001#031124` is frame 31124, which happens to be
    the 250th keyframe — the two numbers are far apart and confusing them is the
    single easiest way to ship a wrong submission.

    >>> build_keyframe_id("L21_V001", 0)
    'L21_V001#000000'
    >>> build_keyframe_id("L21_V001", 31124)
    'L21_V001#031124'
    """
    validate_video_id(video_id)
    if frame_id < 0:
        raise ContractError(
            ErrorCode.INVALID_ID, f"frame_id must be >= 0, got {frame_id}"
        )
    return f"{video_id}#{frame_id:0{FRAME_ID_DIGITS}d}"


def parse_keyframe_id(keyframe_id: str) -> tuple[str, int]:
    """Inverse of :func:`build_keyframe_id`. Returns ``(video_id, frame_id)``.

    >>> parse_keyframe_id("L21_V001#031124")
    ('L21_V001', 31124)
    """
    match = _KEYFRAME_RE.match(keyframe_id)
    if match is None:
        raise ContractError(
            ErrorCode.INVALID_ID,
            f"malformed keyframe_id {keyframe_id!r}; "
            f"expected {{video_id}}#{{frame_id:0{FRAME_ID_DIGITS}d}}",
        )
    return match.group("video_id"), int(match.group("frame_id"))


def build_shot_id(video_id: str, shot_number: int) -> str:
    """Build the shot grouping key: ``{video_id}#s{shot_number:02d}``.

    Used for dedup / temporal grouping (docs/04 §3.2). Optional in Sprint 1.

    >>> build_shot_id("L21_V001", 7)
    'L21_V001#s07'
    """
    validate_video_id(video_id)
    if shot_number < 0:
        raise ContractError(
            ErrorCode.INVALID_ID, f"shot_number must be >= 0, got {shot_number}"
        )
    return f"{video_id}#s{shot_number:0{SHOT_DIGITS}d}"


def parse_shot_id(shot_id: str) -> tuple[str, int]:
    """Inverse of :func:`build_shot_id`. Returns ``(video_id, shot_number)``."""
    match = _SHOT_RE.match(shot_id)
    if match is None:
        raise ContractError(
            ErrorCode.INVALID_ID,
            f"malformed shot_id {shot_id!r}; expected {{video_id}}#s{{shot_number:02d}}",
        )
    return match.group("video_id"), int(match.group("shot"))


def frame_idx_from_pts(pts_time: float, fps: float) -> int:
    """Derive the source-video frame index: ``floor(pts_time * fps)``.

    Verified exact on all 4,615 rows of the current sample and all 87,305 rows
    with a populated source index in the local 299-CSV archive. See the module
    docstring for why it is `floor` and why `fps` must be the literal value the
    dataset states.

    Use this only when no `map-keyframes` row exists for the keyframe — for
    keyframes the team extracts itself. When the CSV has the answer, copy it.

    >>> frame_idx_from_pts(3.0, 30.0)
    90
    >>> frame_idx_from_pts(1893.66, 29.97)
    56752
    """
    if pts_time < 0:
        raise ContractError(
            ErrorCode.INVALID_ID, f"pts_time must be >= 0, got {pts_time}"
        )
    if fps <= 0:
        raise ContractError(ErrorCode.INVALID_ID, f"fps must be > 0, got {fps}")
    return math.floor(pts_time * fps)


def pts_time_to_timestamp_ms(pts_time: float) -> int:
    """Convert a keyframe's `pts_time` (seconds, from CSV) to milliseconds.

    ``timestamp_ms = round(pts_time * 1000)``

    This is the *only* time conversion the contract sanctions, and it is exact
    up to float representation — seconds to milliseconds, no fps involved.

    What is **forbidden** (docs/00 §3, docs/adr/0002-id-format.md): deriving
    `frame_idx` from `pts_time` and `fps`, or vice versa. Both come straight from
    `map-keyframes/{video_id}.csv`. `round(pts_time * fps)` disagreed with the
    dataset on 83-93% of 29.97fps rows.

    >>> pts_time_to_timestamp_ms(3.0)
    3000
    >>> pts_time_to_timestamp_ms(41.174)
    41174
    """
    if pts_time < 0:
        raise ContractError(
            ErrorCode.INVALID_ID, f"pts_time must be >= 0, got {pts_time}"
        )
    return round(pts_time * 1000)
