"""Record schemas for the 4 stores — docs/00 §4.

This is the code form of the contract tables. Quốc Đạt produces these; Công Quang and Nhật Bảo
consume them. If a field here disagrees with docs/00 §4, docs/00 wins and this
file is the bug — tell the Vỹ Triết (docs/01 §1).

Why pydantic rather than plain dataclasses: drift between the four workstreams
is risk #1 in docs/00 §8. Validation at the boundary turns a silent mismatch
into a loud one at the moment the record is built, not three stores later.
"""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mfe.contracts.ids import parse_keyframe_id

#: How far an embedding's L2 norm may drift from 1.0 before we reject it.
#: Loose enough for float32 accumulation, tight enough to catch "forgot to
#: normalize" (docs/00 §4.1: "Vector CLIP đã chuẩn hóa L2").
_L2_TOLERANCE = 1e-3


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class AssetType(StrEnum):
    """docs/00 §4.3."""

    IMAGE = "image"
    VIDEO_SEGMENT = "video_segment"


class VideoRow(_Record):
    """One row = one video, with its technical parameters. docs/05 §1.1.

    `fps` is the linchpin: each video may differ, so nothing may assume a fixed
    value (docs/00 §7 "Không hardcode FPS").
    """

    video_id: str = Field(description="Primary key")
    source_path: str = Field(description="Original file, to reopen for verification")
    fps: float = Field(gt=0, description="Frames per second; needed for frame <-> timestamp")
    duration_ms: int = Field(ge=0)
    total_frames: int = Field(ge=0, description="Used to reject out-of-bounds frames")
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class KeyframeRow(_Record):
    """One row = one keyframe with its canonical coordinates. docs/05 §1.2.

    Both `frame_number` and `timestamp_ms` are stored even though either can be
    derived from the other via `fps`, because BTC has not fixed which the
    submission uses (TO_CONFIRM, docs/00 §3).
    """

    keyframe_id: str = Field(description="Primary key, {video_id}#{frame_number:06d}")
    video_id: str = Field(description="Foreign key -> VideoRow.video_id")
    frame_number: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    shot_id: str | None = Field(default=None, description="Optional in Sprint 1")

    @model_validator(mode="after")
    def _keyframe_id_agrees_with_parts(self) -> KeyframeRow:
        """The single most expensive bug to ship: an id that disagrees with its
        own columns. docs/02 §4 calls this "điểm dễ sai nhất"."""
        parsed_video_id, parsed_frame = parse_keyframe_id(self.keyframe_id)
        if parsed_video_id != self.video_id:
            raise ValueError(
                f"keyframe_id {self.keyframe_id!r} encodes video_id "
                f"{parsed_video_id!r} but the row says {self.video_id!r}"
            )
        if parsed_frame != self.frame_number:
            raise ValueError(
                f"keyframe_id {self.keyframe_id!r} encodes frame {parsed_frame} "
                f"but the row says {self.frame_number}"
            )
        return self


class VectorRecord(_Record):
    """One row = one keyframe and its CLIP vector. docs/00 §4.1, docs/03 §1.

    Faiss itself only stores `internal_id -> embedding`; `keyframe_id` and
    `video_id` live in the sidecar that translates search hits back to global
    ids (docs/03 §1).
    """

    internal_id: int = Field(ge=0, description="Faiss position; meaningless outside Faiss")
    keyframe_id: str
    video_id: str
    embedding: list[float] = Field(description="L2-normalized CLIP vector, length D")

    @field_validator("embedding")
    @classmethod
    def _is_l2_normalized(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("embedding must not be empty")
        norm = math.sqrt(sum(component * component for component in value))
        if abs(norm - 1.0) > _L2_TOLERANCE:
            raise ValueError(
                f"embedding must be L2-normalized (docs/00 §4.1); got norm {norm:.6f}"
            )
        return value


class InvertedRecord(_Record):
    """One row = one keyframe viewed as a text document. docs/00 §4.2, docs/03 §2.1.

    This is the "document" view. How Công Quang inverts it into term -> postings
    (docs/03 §2.2) is a storage detail and not part of this contract.
    """

    keyframe_id: str
    video_id: str
    object_labels: list[str] = Field(default_factory=list)
    ocr_text: str = Field(default="", description="Empty when the OCR module is off")
    metadata_text: str = Field(default="")
    asr_text: str = Field(default="", description="Later sprint; usually empty in Sprint 1")


class AssetRecord(_Record):
    """One row = one binary resource. docs/00 §4.3, docs/03 §3.

    A manifest row *points at* bytes; it does not contain them. A keyframe may
    have more than one row (one `image`, one `video_segment`).
    """

    keyframe_id: str
    asset_type: AssetType
    asset_path: str = Field(description="Local file path, or key on a Modal volume")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    byte_size: int = Field(ge=0, description="Used to project storage at scale")
