"""The ID model — docs/00 §3.

Every store keys off `keyframe_id`. No store may mint its own identifiers.
This module is the only place that knows how a `keyframe_id` is spelled; if you
find yourself writing an f-string with a `#` in it, call in here instead.

Anchored on the real dataset (Kaggle AI Challenge 2025). See
docs/adr/0002-id-format.md.

    keyframe_id    = {video_id}#{keyframe_index:03d}      e.g. L21_V001#001
    keyframe_index = the keyframe's sequential position `n`, matching its image
                     filename (001.jpg). It is NOT the source-video frame number.

`frame_idx` (the real frame in the source video) is derivable, with two
conditions that are not optional (both measured on the shipped sample, 1240 rows
across 5 videos including a 29.97fps one):

    frame_idx = floor(pts_time * fps)        100% exact on every row

    - It must be **floor**, not round and not ceil. `round` matched 55% of rows
      and `ceil` 28%. An earlier revision of this module forbade derivation
      entirely on the strength of the `round` measurement; that conclusion was
      wrong, it was the rounding mode that was wrong.
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

#: Zero-pad width of `keyframe_index` inside a `keyframe_id`. The dataset uses
#: 3 (`001.jpg`). Indices >= 1000 simply use more digits; parsing accepts 3+.
KEYFRAME_INDEX_DIGITS = 3

#: Width of the zero-padded shot number inside a `shot_id` (`L21_V001#s07`).
SHOT_DIGITS = 2

_KEYFRAME_RE = re.compile(
    rf"^(?P<video_id>[^\s#]+)#(?P<index>\d{{{KEYFRAME_INDEX_DIGITS},}})$"
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


def build_keyframe_id(video_id: str, keyframe_index: int) -> str:
    """Build the global keyframe key: ``{video_id}#{keyframe_index:03d}``.

    `keyframe_index` is the keyframe's ordinal `n` (matching its image
    filename), not the source-video frame number.

    >>> build_keyframe_id("L21_V001", 1)
    'L21_V001#001'
    >>> build_keyframe_id("L21_V001", 250)
    'L21_V001#250'
    """
    validate_video_id(video_id)
    if keyframe_index < 0:
        raise ContractError(
            ErrorCode.INVALID_ID, f"keyframe_index must be >= 0, got {keyframe_index}"
        )
    return f"{video_id}#{keyframe_index:0{KEYFRAME_INDEX_DIGITS}d}"


def parse_keyframe_id(keyframe_id: str) -> tuple[str, int]:
    """Inverse of :func:`build_keyframe_id`. Returns ``(video_id, keyframe_index)``.

    >>> parse_keyframe_id("L21_V001#001")
    ('L21_V001', 1)
    """
    match = _KEYFRAME_RE.match(keyframe_id)
    if match is None:
        raise ContractError(
            ErrorCode.INVALID_ID,
            f"malformed keyframe_id {keyframe_id!r}; "
            f"expected {{video_id}}#{{keyframe_index:0{KEYFRAME_INDEX_DIGITS}d}}",
        )
    return match.group("video_id"), int(match.group("index"))


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

    Verified exact on all 1240 rows of the shipped sample (30.0, 25.0 and
    29.97 fps videos). See the module docstring for why it is `floor` and why
    `fps` must be the literal value the dataset states.

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
