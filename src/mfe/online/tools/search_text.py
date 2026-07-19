"""`search_text` — docs/04 §2.

    purpose:     Tìm keyframe theo mô tả văn bản (semantic + text).
    input:       query_text, top_k
    output:      [{keyframe_id, video_id, score, evidence_id}]
    constraints: top_k <= 500
    failure:     structured error code, never a fabricated keyframe

The first tool to build (docs/01 §4 step 4) and the one the whole vertical slice
hangs on.
"""

from __future__ import annotations

from mfe.contracts.tools import Candidate, ToolResult


def search_text(query_text: str, top_k: int = 100) -> ToolResult[Candidate]:
    """Owner: Phạm Trọng. TODO(Phạm Trọng): implement.

    - Encode the query with the *same* CLIP checkpoint Quốc Đạt used for images. A
      text tower from a different model against those image vectors returns
      confident nonsense.
    - Run vector_search and text_search, fuse via `fusion.py`.
    - Enforce `top_k <= 500` (docs/04 §2) — return INVALID_ARGUMENT above it,
      do not silently clamp. An agent that asked for 5000 has a bug worth
      surfacing.
    - Every Candidate carries an `evidence_id` (docs/00 §5).
    - On failure return ToolResult with an error, not an empty list. "No
      results" and "the store is down" must be distinguishable.
    """
    raise NotImplementedError("TODO(Phạm Trọng): encode query -> vector + text search -> fuse")
