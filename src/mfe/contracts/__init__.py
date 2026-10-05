"""The contract — the code form of docs/00.

**Owner: Vỹ Triết. Do not change without the Vỹ Triết.** docs/01 §1: "Bất kỳ ai muốn đổi
schema một kho hoặc chữ ký một tool phải báo Vỹ Triết; Vỹ Triết cập nhật file 00 rồi
thông báo nhóm."

Import from here rather than redefining a shape locally. A `keyframe_id` built
by hand in Quốc Đạt and parsed by hand in Nhật Bảo is exactly how risk #1 in docs/00 §8
happens.
"""

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.ids import (
    FRAME_ID_DIGITS,
    build_keyframe_id,
    build_shot_id,
    parse_keyframe_id,
    parse_shot_id,
    pts_time_to_timestamp_ms,
)
from mfe.contracts.records import (
    AssetRecord,
    AssetType,
    InvertedRecord,
    KeyframeRow,
    VectorRecord,
    VideoRow,
)
from mfe.contracts.tools import (
    Candidate,
    EvidenceId,
    Neighbor,
    SubmitFormat,
    ToolError,
    ToolResult,
)

__all__ = [
    "FRAME_ID_DIGITS",
    "AssetRecord",
    "AssetType",
    "Candidate",
    "ContractError",
    "ErrorCode",
    "EvidenceId",
    "InvertedRecord",
    "KeyframeRow",
    "Neighbor",
    "SubmitFormat",
    "ToolError",
    "ToolResult",
    "VectorRecord",
    "VideoRow",
    "build_keyframe_id",
    "build_shot_id",
    "parse_keyframe_id",
    "parse_shot_id",
    "pts_time_to_timestamp_ms",
]
