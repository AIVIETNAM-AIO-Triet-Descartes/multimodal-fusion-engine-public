"""Submit adapter — docs/05 §2.

BTC has not fixed the submission schema (TO_CONFIRM, docs/00 §3;
docs/context/buoi-1.md §7 records the lecturer saying it could be a timestamp to
the millisecond or a frame number). So the format is a config choice, resolved
at the last possible moment, and adding the real one when it is announced means
adding one adapter file.

Every field is *read* from the Relational Map — never re-derived. The docs' own
worked examples disagreed about rounding (docs/adr/0002-id-format.md); deriving
at submit time is how that class of bug reaches a scoreboard.
"""

from __future__ import annotations

from typing import Any

from mfe.contracts.tools import SubmitFormat


def build_submission(
    keyframe_id: str,
    schema_format: SubmitFormat,
    relational_map: object,
    frame_range_window: int | None = None,
) -> dict[str, Any]:
    """Translate a keyframe_id into a submission record. docs/05 §2.

    Owner: Nhật Bảo. Dispatches to `adapters/`.

    Fails, never guesses: if `keyframe_id` is absent from the Relational Map,
    `get_keyframe` raises ContractError(KEYFRAME_NOT_FOUND). docs/05 §2: "nếu
    keyframe_id không có trong Relational Map -> lỗi, không đoán".

    `frame_range_window` (from config `submit.frame_range_window`) is forwarded
    only to the `by_frame_range` adapter; other formats ignore it. Left None,
    the adapter uses its own default.
    """
    keyframe = relational_map.get_keyframe(keyframe_id)
    video = relational_map.get_video(keyframe.video_id)

    from mfe.mapping.adapters.base import get_adapter

    # str(schema_format) in case it's an enum.
    fmt = str(schema_format)
    kwargs = {}
    if fmt == "by_frame_range" and frame_range_window is not None:
        kwargs["window"] = frame_range_window
    return get_adapter(fmt, **kwargs).build(keyframe, video)
