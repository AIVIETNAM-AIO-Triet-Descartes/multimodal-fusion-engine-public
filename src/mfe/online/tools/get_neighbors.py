"""`get_neighbors` — docs/04 §2.

    purpose: Lấy keyframe lân cận theo thời gian để xem ngữ cảnh trước/sau.
    input:   keyframe_id, window
    output:  [{keyframe_id, timestamp_ms}]

Why this small tool matters as much as search: docs/context/buoi-1.md §8 ranks
temporal verification third among the things that separate teams — "một frame
giống mô tả chưa chắc là đúng sự kiện; phải xem trước/sau". And §7 records that
a transition frame, or one that does not yet show the moment, may not be
accepted at all.

Backing it must be one operation in the UI (docs/04 §4). An operator who has to
manually find the video and scrub to a timestamp has already lost the round.
"""

from __future__ import annotations

from mfe.contracts.tools import Neighbor, ToolResult


def get_neighbors(keyframe_id: str, window: int = 5) -> ToolResult[Neighbor]:
    """Owner: Phạm Trọng. TODO(Phạm Trọng): implement.

    - Reads the Relational Map (`RelationalMap.get_neighbors`). This is the one
      sanctioned retrieval read of that store — it returns context, not
      verification, so it does not violate docs/00 §2.
    - `window` counts neighbouring *keyframes*, not frames or seconds. Say which
      in the tool description; an agent given "window" with no unit will guess.
    - Do not cross a video boundary. The keyframe before `L21_V002#000` is
      not the last keyframe of `L21_V001`.
    - Near the start or end, return fewer than `window` — that is correct, not
      an error.
    """
    raise NotImplementedError("TODO(Phạm Trọng): keyframe_id + window -> neighbors")
