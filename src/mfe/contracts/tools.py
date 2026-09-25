"""Shared tool contract — docs/00 §5.

docs/00 puts the tool contract in the *cross-cutting* section, not in Phạm Trọng's
file, because two consumers depend on it: the browser in Sprint 1, and the
Buổi 3 agent later. Hence the split:

    contracts/tools.py    the shape (here, Vỹ Triết-owned, stable)
    online/tools/*.py     the implementations (Phạm Trọng-owned)

An agent added in a later sprint calls the same tools. docs/context/buoi-3.md §2:
"agent không nên thay retrieval engine".
"""

from __future__ import annotations

from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from mfe.contracts.errors import ErrorCode

#: The single source of truth for the evidence_id format. CHỐT (Phạm Trọng <->
#: Nhật Bảo sync, docs/01 §1): "{tool_name}:{run_id}:{ordinal}".
#: Anyone validating an evidence_id (e.g. the validator) imports THIS — do not
#: re-inline the regex, or the two copies drift.
EVIDENCE_ID_PATTERN = r"^[a-zA-Z0-9_]+:[a-zA-Z0-9_\-]+:\d+$"

#: An opaque handle back to the tool call that produced a result.
#:
#: Every intermediate result carries one, so the validator can check that an
#: answer traces to real retrieval output (docs/05 §3, rule "Có bằng chứng") and
#: so a later agent can cite evidence per docs/context/buoi-3.md §7.
#:
#: Format: "{tool_name}:{run_id}:{ordinal}"  e.g. "search_text:req_abc123:0"
#: Resolvable against the structured log written by mfe.core.logging. Keep it
#: opaque to callers either way.
EvidenceId = Annotated[str, StringConstraints(pattern=EVIDENCE_ID_PATTERN)]


class ToolError(BaseModel):
    """A structured failure. Tools return this instead of guessing.

    docs/04 §2: "trả mã lỗi có cấu trúc, không bịa keyframe".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ErrorCode
    message: str
    tool: str


class Candidate(BaseModel):
    """One retrieval hit — the unit that flows Công Quang -> Phạm Trọng -> Nhật Bảo.

    `score` is only comparable within the tool that produced it. CLIP cosine and
    BM25 live on different scales; normalize before fusing, never add raw
    (docs/04 §3.1).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    keyframe_id: str
    video_id: str
    score: float
    evidence_id: EvidenceId
    matched_terms: list[str] = Field(
        default_factory=list, description="Populated by text_search (docs/03 §2)"
    )
    #: Toạ độ nộp, CHỈ đặt cho khung **không có trong danh mục BTC** — khung người
    #: vận hành tự tua tới trên bàn nộp, hoặc khung ta tự trích.
    #:
    #: Mặc định `None`, và `None` nghĩa là "tra Relational Map như mọi khung khác".
    #: Đường nộp (`eval.official.candidate_frame_id`) cố ý **ném** khi tra không ra,
    #: vì đoán `frame_id` từ `keyframe_id` là cách một lỗi đi thẳng tới bảng điểm.
    #: Nhưng khung tự tua thì theo định nghĩa không nằm trong bản đồ, nên nó phải
    #: **tự khai** toạ độ của mình thay vì để phép tra đoán hộ — khai tường minh
    #: thì một khung thiếu thật vẫn ném đúng như cũ.
    pts_time: float | None = Field(default=None, ge=0, allow_inf_nan=False,
                                  description="Explicit presentation time in seconds for a manually selected frame")
    frame_idx: int | None = Field(
        default=None,
        description="Chỉ cho khung ngoài danh mục BTC; None = tra Relational Map",
    )


class Neighbor(BaseModel):
    """A temporally adjacent keyframe — output of `get_neighbors` (docs/04 §2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    keyframe_id: str
    timestamp_ms: int


ItemT = TypeVar("ItemT")


class ToolResult(BaseModel, Generic[ItemT]):
    """Uniform envelope: either `items` or `error`, never both.

    Callers branch on `ok`. The cascade uses this to fall back — a rerank that
    returns TOOL_TIMEOUT means "use the coarse ranking", not "no results"
    (docs/04 §1, "Fallback bắt buộc").
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: str
    items: list[ItemT] = Field(default_factory=list)
    error: ToolError | None = None
    latency_ms: float | None = Field(
        default=None, description="Per-stage latency; the Vỹ Triết aggregates for docs/01 §2"
    )

    @property
    def ok(self) -> bool:
        return self.error is None


#: Submission formats the adapter can emit. BTC has not fixed one
#: (TO_CONFIRM, docs/05 §2), so all four stay live and the choice is config.
SubmitFormat = Literal["by_timestamp", "by_frame", "by_frame_range", "raw_keyframe"]
