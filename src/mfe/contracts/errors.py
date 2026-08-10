"""Structured failure codes.

Every tool and every validator rule fails with one of these, never with a
guessed result. docs/04 §2: "trả mã lỗi có cấu trúc, không bịa keyframe".
docs/05 §2: "nếu keyframe_id không có trong Relational Map -> lỗi, không đoán".
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """Stable machine-readable failure codes.

    The five validator rules of docs/05 §3 map onto the first five members.
    Keep the string values stable — they end up in logs the Vỹ Triết greps to
    explain a blocked submit.
    """

    # --- Validator rules, docs/05 §3 ---------------------------------------
    KEYFRAME_NOT_FOUND = "keyframe_not_found"        # rule "Tồn tại"
    FRAME_OUT_OF_BOUNDS = "frame_out_of_bounds"      # rule "Trong biên"
    TIMESTAMP_OUT_OF_BOUNDS = "timestamp_out_of_bounds"  # rule "Timestamp hợp lệ"
    SCHEMA_MISMATCH = "schema_mismatch"              # rule "Đúng schema"
    MISSING_EVIDENCE = "missing_evidence"            # rule "Có bằng chứng"
    #: rule "Nhất quán frame". Its own code, not FRAME_OUT_OF_BOUNDS: a frame
    #: inside the video that disagrees with its own pts_time is a different
    #: failure from one past the end, and sharing a code made the two
    #: indistinguishable in logs and untestable apart.
    FRAME_INCONSISTENT = "frame_inconsistent"

    # --- Contract / ID model ------------------------------------------------
    INVALID_ID = "invalid_id"
    VIDEO_NOT_FOUND = "video_not_found"

    # --- Tool runtime, docs/04 §2 ------------------------------------------
    TOOL_FAILED = "tool_failed"
    TOOL_TIMEOUT = "tool_timeout"                    # rerank over budget -> fall back to coarse
    INVALID_ARGUMENT = "invalid_argument"
    STORE_UNAVAILABLE = "store_unavailable"
    NOT_IMPLEMENTED = "not_implemented"


class ContractError(ValueError):
    """Raised when data violates the contract in docs/00.

    Carries a machine-readable `code` so callers can branch (e.g. the cascade
    falls back to coarse on TOOL_TIMEOUT) instead of string-matching a message.

    Subclasses ValueError on purpose: pydantic only converts ValueError and
    AssertionError into ValidationError. Without it, `build_keyframe_id` raising
    inside a record's validator would escape as a bare ContractError while the
    same violation caught by a Field constraint surfaced as ValidationError —
    two exception types for one kind of bug, across four workstreams. Now:

        direct call     -> ContractError, with .code
        inside a record -> ValidationError, message preserved
        either          -> caught by `except ValueError`
    """

    def __init__(self, code: ErrorCode, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")
