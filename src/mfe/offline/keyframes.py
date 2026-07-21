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

    Owner: Quốc Đạt. TODO(Quốc Đạt): implement.

    For the current dataset the keyframes AND their coordinates are provided:
    read the images plus `map-keyframes/{video_id}.csv` (columns
    n, pts_time, fps, frame_idx). Build ids and timestamps through
    `mfe.contracts.ids` — never by hand:

        keyframe_id  = build_keyframe_id(video.video_id, n)      # n = keyframe_index
        timestamp_ms = pts_time_to_timestamp_ms(pts_time)

    `frame_idx` and `pts_time` are copied straight from the CSV and NEVER
    recomputed — round(pts_time*fps) was wrong on 83-93% of 29.97fps rows
    (docs/00 §3, adr/0002). `keyframe_index` (n) is the keyframe's ordinal /
    image filename; it is a different number from `frame_idx`.

    docs/context/buoi-1.md slide 44 notes BTC guarantees a query segment contains
    at least one provided keyframe — which only helps if we index exactly the
    keyframes they provide.
    """
    raise NotImplementedError("TODO(Quốc Đạt): read provided keyframes + CSV -> rows")


def assign_shot_ids(rows: list[KeyframeRow]) -> list[KeyframeRow]:
    """Group keyframes into shots — optional in Sprint 1 (docs/00 §3).

    Owner: Quốc Đạt. TODO(Quốc Đạt): implement, or leave shot_id None for Sprint 1.

    Worth doing early even so: Phạm Trọng's dedup (docs/04 §3.2) needs it to avoid a
    grid that is 50 near-identical frames of one scene — docs/context/buoi-2.md
    §9 lists that as a place strong teams separate from weak ones.
    Use `mfe.contracts.ids.build_shot_id`.
    """
    raise NotImplementedError("TODO(Quốc Đạt): shot boundary detection -> shot_id")
