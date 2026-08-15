"""Keyframe extraction — step [2] of docs/02 §2.

One extraction produces rows for *two* stores at once:
  - Asset Store    : the image bytes + their dimensions (docs/03 §3)
  - Relational Map : the canonical coordinates (docs/05 §1.2)

They must agree on `keyframe_id`, because that id is what later rejoins them.
Yielding them as a pair is what makes disagreement impossible: there is one
`build_keyframe_id` call per keyframe and both rows read it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.ids import build_keyframe_id, build_shot_id, pts_time_to_timestamp_ms
from mfe.contracts.records import AssetRecord, AssetType, KeyframeRow, VideoRow
from mfe.core.storage import StorageBackend
from mfe.offline.provided import MapRow, read_jpeg_size

logger = logging.getLogger(__name__)

#: Gap in seconds between consecutive keyframes that starts a new shot.
#: The provided keyframes are already shot-change driven, so consecutive ones a
#: long way apart are almost always different scenes. Deliberately coarse: this
#: only feeds Phạm Trọng's dedup (docs/04 §3.2), where over-merging costs grid
#: diversity and under-merging costs nothing but a longer grid.
SHOT_GAP_SECONDS = 5.0


def collapse_duplicate_frames(map_rows: list[MapRow]) -> tuple[list[MapRow], list[tuple[int, int]]]:
    """Drop rows whose `frame_idx` a lower-ordinal row already claims.

    Since `keyframe_id` keys off `frame_idx` (contracts/ids), two CSV rows with
    the same `frame_idx` would mint the same id. They exist: 614 adjacent pairs
    in batch 1's 177,321 rows, 5 rows of the 1240-row sample. All of them look
    like this —

        n=353  pts_time=1269.57  fps=29.97  frame_idx=38049
        n=354  pts_time=1269.60  fps=29.97  frame_idx=38049

    0.03s apart at 29.97fps is *less than one frame*, so both timestamps land on
    frame 38049. The two images are two renderings of one source frame, and
    keeping one is a de-duplication, not a loss.

    The lowest `n` wins, so the choice is deterministic across runs. Returns the
    kept rows and the (dropped_n, kept_n) pairs, for the caller to report.
    """
    kept: list[MapRow] = []
    collapsed: list[tuple[int, int]] = []
    seen: dict[int, int] = {}  # frame_idx -> the n that claimed it

    for row in sorted(map_rows, key=lambda r: r.n):
        claimed_by = seen.get(row.frame_idx)
        if claimed_by is not None:
            collapsed.append((row.n, claimed_by))
            continue
        seen[row.frame_idx] = row.n
        kept.append(row)

    return kept, collapsed


def extract_keyframes(
    video: VideoRow,
    map_rows: list[MapRow],
    keyframes_dir: str,
    backend: StorageBackend,
) -> Iterator[tuple[KeyframeRow, AssetRecord]]:
    """Yield one (relational row, asset row) pair per provided keyframe.

    Owner: Quốc Đạt.

    For this dataset the keyframes AND their coordinates are provided, so
    "extraction" is really registration: read `keyframes/{video_id}/{n:03d}.jpg`
    and pair it with its `map-keyframes/{video_id}.csv` row.

    `frame_idx` and `pts_time` are copied straight from the CSV and NEVER
    recomputed — `round(pts_time*fps)` was wrong on 83-93% of 29.97fps rows
    (docs/00 §3, adr/0002). `keyframe_index` (n) is the keyframe's ordinal and
    image filename; it is a different number from `frame_idx`.

    Note the asymmetry: the *id* keys off `frame_idx` while the *image file* is
    still named after `n`, because BTC named it. That is the only place the two
    numbers meet, and it is why the ordinal stays a column. Keyframes the team
    extracts itself should name their images after the frame instead.

    docs/context/buoi-1.md slide 44: BTC guarantees a query segment contains at
    least one provided keyframe — which only helps if we index exactly the
    keyframes they provide, so a missing image is an error, not a skip.
    """
    keyframe_root = Path(keyframes_dir) / video.video_id

    map_rows, collapsed = collapse_duplicate_frames(map_rows)
    if collapsed:
        logger.info(
            "%s: collapsed %d keyframe(s) sharing a frame_idx with an earlier one "
            "(dropped n -> kept n: %s)",
            video.video_id,
            len(collapsed),
            ", ".join(f"{dropped}->{kept}" for dropped, kept in collapsed),
        )

    for row in map_rows:
        keyframe_id = build_keyframe_id(video.video_id, row.frame_idx)
        image_path = keyframe_root / f"{row.n:03d}.jpg"

        image_path_str = str(image_path)
        if not backend.exists(image_path_str):
            raise ContractError(
                ErrorCode.STORE_UNAVAILABLE,
                f"{keyframe_id}: map-keyframes lists n={row.n} but {image_path} is missing. "
                "Indexing a coordinate with no image would produce a candidate the "
                "browser cannot display.",
            )
        image_bytes = backend.read_bytes(image_path_str)
        width, height = read_jpeg_size(image_bytes)

        keyframe_row = KeyframeRow(
            keyframe_id=keyframe_id,
            video_id=video.video_id,
            keyframe_index=row.n,
            frame_idx=row.frame_idx,
            pts_time=row.pts_time,
            timestamp_ms=pts_time_to_timestamp_ms(row.pts_time),
        )
        asset_record = AssetRecord(
            keyframe_id=keyframe_id,
            asset_type=AssetType.IMAGE,
            asset_path=image_path_str,
            width=width,
            height=height,
            byte_size=len(image_bytes),
        )
        yield keyframe_row, asset_record


def assign_shot_ids(rows: list[KeyframeRow]) -> list[KeyframeRow]:
    """Group keyframes into shots by temporal gap — optional in Sprint 1 (docs/00 §3).

    Owner: Quốc Đạt.

    Not shot-boundary detection on pixels: it groups consecutive keyframes whose
    `pts_time` are within `SHOT_GAP_SECONDS`. That is cheap and needs no model,
    which is the whole point in a sprint that runs no checkpoints — and the
    provided keyframes are already sampled at scene changes, so the gaps carry
    most of the signal a detector would find.

    Phạm Trọng's dedup (docs/04 §3.2) needs this to avoid a grid that is 50
    near-identical frames of one scene — docs/context/buoi-2.md §9 lists that as
    a place strong teams separate from weak ones.

    Rows are frozen, so this returns new rows rather than mutating in place.
    """
    if not rows:
        return []

    ordered = sorted(rows, key=lambda row: (row.video_id, row.keyframe_index))
    labelled: list[KeyframeRow] = []
    shot_number = 0
    previous: KeyframeRow | None = None

    for row in ordered:
        starts_new_video = previous is None or previous.video_id != row.video_id
        if starts_new_video:
            shot_number = 1
        elif row.pts_time - previous.pts_time > SHOT_GAP_SECONDS:
            shot_number += 1
        shot_id = build_shot_id(row.video_id, shot_number)
        labelled.append(row.model_copy(update={"shot_id": shot_id}))
        previous = row

    return labelled
