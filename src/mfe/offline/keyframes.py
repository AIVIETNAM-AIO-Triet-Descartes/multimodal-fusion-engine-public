"""Keyframe extraction — step [2] of docs/02 §2.

One extraction produces rows for *two* stores at once:
  - Asset Store    : the image bytes + their dimensions (docs/03 §3)
  - Relational Map : the canonical coordinates (docs/05 §1.2)

They must agree on `keyframe_id`, because that id is what later rejoins them.
"""

from __future__ import annotations

from collections.abc import Iterator

from mfe.contracts.records import AssetRecord, KeyframeRow, VideoRow


def extract_keyframes(video: VideoRow) -> Iterator[tuple[KeyframeRow, AssetRecord]]:
    """Yield one (relational row, asset row) pair per extracted keyframe.

    Owner: TV1. TODO(TV1): implement.

    Build ids and timestamps through `mfe.contracts.ids` — never by hand:

        keyframe_id  = build_keyframe_id(video.video_id, frame_number)
        timestamp_ms = frame_to_timestamp_ms(frame_number, video.fps)

    `frame_number` must be the frame's index in the *source* video, 0-indexed
    (docs/00 §3) — not its ordinal among extracted keyframes. Those two differ
    the moment sampling is not every-frame, and confusing them puts every
    submission off by a factor of the sampling stride.

    Selection strategy (I-frame vs fixed interval vs shot boundary) is TV1's
    call and belongs in config. docs/context/buoi-1.md slide 44 notes BTC
    guarantees a query segment contains at least one provided keyframe — which
    only helps if we index the keyframes they provide.
    """
    raise NotImplementedError("TODO(TV1): decode video -> keyframe images + rows")


def assign_shot_ids(rows: list[KeyframeRow]) -> list[KeyframeRow]:
    """Group keyframes into shots — optional in Sprint 1 (docs/00 §3).

    Owner: TV1. TODO(TV1): implement, or leave shot_id None for Sprint 1.

    Worth doing early even so: TV3's dedup (docs/04 §3.2) needs it to avoid a
    grid that is 50 near-identical frames of one scene — docs/context/buoi-2.md
    §9 lists that as a place strong teams separate from weak ones.
    Use `mfe.contracts.ids.build_shot_id`.
    """
    raise NotImplementedError("TODO(TV1): shot boundary detection -> shot_id")
