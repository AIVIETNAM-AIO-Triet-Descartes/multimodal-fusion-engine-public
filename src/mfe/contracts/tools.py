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

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from mfe.contracts.errors import ErrorCode

from typing import Annotated
from pydantic import StringConstraints

#: An opaque handle back to the tool call that produced a result.
#:
#: Every intermediate result carries one, so the validator can check that an
#: answer traces to real retrieval output (docs/05 §3, rule "Có bằng chứng") and
#: so a later agent can cite evidence per docs/context/buoi-3.md §7.
#:
#: Format (Phạm Trọng <-> Nhật Bảo sync): "{tool_name}:{run_id}:{ordinal}"
#: Example: "search_text:req_abc123:0"
#: Resolvable against the structured log written by mfe.core.logging. Keep it
#: opaque to callers either way.
EvidenceId = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_]+:[a-zA-Z0-9_\-]+:\d+$")]


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
