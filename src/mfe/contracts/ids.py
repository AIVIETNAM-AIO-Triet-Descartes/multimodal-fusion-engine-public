"""The ID model — docs/00 §3.

Every store keys off `keyframe_id`. No store may mint its own identifiers.
This module is the only place that knows how a `keyframe_id` is spelled; if you
find yourself writing an f-string with a `#` in it, call in here instead.

Two format decisions are recorded in docs/adr/0002-id-format.md:

1. `frame_number` is zero-padded to 6 digits. docs never states the width, but
   every example uses 6 (`#001234`, `#000250`, `#000000`).
2. `timestamp_ms` uses round(), per the normative formula in docs/00 §3. Two
   worked examples in the docs disagree with that formula; the formula wins and
   the examples are being corrected.
"""

from __future__ import annotations

import re

from mfe.contracts.errors import ContractError, ErrorCode

#: Width of the zero-padded frame number inside a `keyframe_id`.
#: 6 digits caps a video at 999,999 frames ~= 9.2 hours at 30fps. AIC clips are
#: far shorter; raising this is a contract change (every store re-keys).
FRAME_DIGITS = 6

#: Width of the zero-padded shot number inside a `shot_id` (`L01_V001#s07`).
SHOT_DIGITS = 2

_KEYFRAME_RE = re.compile(rf"^(?P<video_id>[^\s#]+)#(?P<frame>\d{{{FRAME_DIGITS}}})$")
_SHOT_RE = re.compile(r"^(?P<video_id>[^\s#]+)#s(?P<shot>\d+)$")
_VIDEO_ID_RE = re.compile(r"^[^\s#]+$")

MAX_FRAME_NUMBER = 10**FRAME_DIGITS - 1


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


def build_keyframe_id(video_id: str, frame_number: int) -> str:
    """Build the global keyframe key: ``{video_id}#{frame_number:06d}``.

    >>> build_keyframe_id("L01_V001", 1234)
    'L01_V001#001234'
    >>> build_keyframe_id("L01_V001", 0)
    'L01_V001#000000'
    """
    validate_video_id(video_id)
    if frame_number < 0:
        raise ContractError(
            ErrorCode.INVALID_ID, f"frame_number must be >= 0, got {frame_number}"
        )
    if frame_number > MAX_FRAME_NUMBER:
        raise ContractError(
            ErrorCode.INVALID_ID,
            f"frame_number {frame_number} exceeds {FRAME_DIGITS}-digit format "
            f"(max {MAX_FRAME_NUMBER}); widening FRAME_DIGITS is a contract change",
        )
    return f"{video_id}#{frame_number:0{FRAME_DIGITS}d}"


def parse_keyframe_id(keyframe_id: str) -> tuple[str, int]:
    """Inverse of :func:`build_keyframe_id`. Returns ``(video_id, frame_number)``.

    >>> parse_keyframe_id("L01_V001#001234")
    ('L01_V001', 1234)
    """
    match = _KEYFRAME_RE.match(keyframe_id)
    if match is None:
        raise ContractError(
            ErrorCode.INVALID_ID,
            f"malformed keyframe_id {keyframe_id!r}; "
            f"expected {{video_id}}#{{frame_number:0{FRAME_DIGITS}d}}",
        )
    return match.group("video_id"), int(match.group("frame"))


def build_shot_id(video_id: str, shot_number: int) -> str:
    """Build the shot grouping key: ``{video_id}#s{shot_number:02d}``.

    Used for dedup / temporal grouping (docs/04 §3.2). Optional in Sprint 1.

    >>> build_shot_id("L01_V001", 7)
    'L01_V001#s07'
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


def frame_to_timestamp_ms(frame_number: int, fps: float) -> int:
    """Convert a frame index to milliseconds — docs/00 §3.

    ``timestamp_ms = round(frame_number / fps * 1000)``

    This is a *derivation*, not a substitute for storage. Both `frame_number`
    and `timestamp_ms` are persisted on every keyframe row because BTC has not
    fixed which one the submission uses (TO_CONFIRM, docs/00 §3). At submit
    time, read both from the Relational Map — never re-derive one from the
    other, or a float rounding difference becomes a wrong answer.

    >>> frame_to_timestamp_ms(250, 29.97)
    8342
    >>> frame_to_timestamp_ms(1180, 25.0)
    47200
    """
    if frame_number < 0:
        raise ContractError(
            ErrorCode.INVALID_ID, f"frame_number must be >= 0, got {frame_number}"
        )
    if fps <= 0:
        raise ContractError(
            ErrorCode.INVALID_ID, f"fps must be > 0, got {fps}; read it per-video, never hardcode"
        )
    return round(frame_number / fps * 1000)


def timestamp_ms_to_frame(timestamp_ms: int, fps: float) -> int:
    """Inverse derivation, for tooling only.

    Do **not** use this to build a submission. It exists so the browser can jump
    from a user-picked timestamp back to a candidate frame. The authoritative
    pair always comes from the Relational Map (docs/05 §1.2).
    """
    if timestamp_ms < 0:
        raise ContractError(
            ErrorCode.INVALID_ID, f"timestamp_ms must be >= 0, got {timestamp_ms}"
        )
    if fps <= 0:
        raise ContractError(ErrorCode.INVALID_ID, f"fps must be > 0, got {fps}")
    return round(timestamp_ms * fps / 1000)
