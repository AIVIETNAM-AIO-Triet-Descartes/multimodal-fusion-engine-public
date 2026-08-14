"""Submit the raw key: {keyframe_id}. docs/05 §2."""

from __future__ import annotations

from typing import Any

from mfe.contracts.records import KeyframeRow, VideoRow
from mfe.mapping.adapters.base import SubmitAdapter


class RawKeyframeAdapter(SubmitAdapter):
    """Pass the id straight through. Owner: Nhật Bảo.

    The only format that assumes BTC uses *our* id convention — which they will
    not, unless they ship the keyframes and we adopt their naming. Realistically
    this is the debug/eval format: the harness (docs/01 §2) compares against
    `gt_keyframe_ids`, which are our ids.

    Still goes through the validator. Even here the keyframe must exist in the
    Relational Map — a well-formed id for a keyframe we never indexed is exactly
    the kind of thing this fence is for.
    """

    format_name = "raw_keyframe"

    def build(self, keyframe: KeyframeRow, video: VideoRow) -> dict[str, Any]:
        return {"keyframe_id": keyframe.keyframe_id}
