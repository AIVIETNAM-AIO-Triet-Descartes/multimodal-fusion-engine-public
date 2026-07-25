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
) -> dict[str, Any]:
    """Translate a keyframe_id into a submission record. docs/05 §2.

    Owner: Nhật Bảo. TODO(Nhật Bảo): implement — dispatch to `adapters/`.

    Fails, never guesses: if `keyframe_id` is absent from the Relational Map,
    raise ContractError(KEYFRAME_NOT_FOUND). docs/05 §2: "nếu keyframe_id không
    có trong Relational Map -> lỗi, không đoán".
    """
    keyframe = relational_map.get_keyframe(keyframe_id)
    video = relational_map.get_video(keyframe.video_id)
    
    from mfe.mapping.adapters.base import get_adapter
    # Call str(schema_format) in case it's an enum
    adapter = get_adapter(str(schema_format))
    return adapter.build(keyframe, video)
