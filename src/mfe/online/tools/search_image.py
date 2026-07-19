"""`search_image` — docs/04 §2.

    purpose: Tìm keyframe tương tự một ảnh tham chiếu.
    input:   image_bytes | keyframe_id, top_k
    output:  [{keyframe_id, video_id, score, evidence_id}]

Two uses (docs/context/buoi-2.md §6): exploitation — the operator found a
near-miss and wants more like it; and modality change — the query was never text
in the first place (a sketch, a generated image, a frame from a shown clip).

docs/context/buoi-2.md §3 notes a later round may show a short video clip and
forbid recording it, leaving a sketch or description as the only way to
reproduce the query. That is TO_CONFIRM, but it is why this tool takes raw bytes
and not only a keyframe_id.
"""

from __future__ import annotations

from mfe.contracts.tools import Candidate, ToolResult


def search_image(
    image_bytes: bytes | None = None,
    keyframe_id: str | None = None,
    top_k: int = 100,
) -> ToolResult[Candidate]:
    """Owner: Phạm Trọng. TODO(Phạm Trọng): implement. Sprint 1: after `search_text` works.

    - Exactly one of `image_bytes` / `keyframe_id` — both or neither is
      INVALID_ARGUMENT.
    - Given a `keyframe_id`, take the existing vector via the sidecar rather
      than re-encoding the image. Cheaper and, more importantly, guaranteed
      identical to what is indexed.
    - Given bytes, encode with the same CLIP checkpoint as ingest.
    - The query keyframe will rank #1 against itself. Decide with Phạm Trọng whether to
      drop it, and say which in the docstring — an operator seeing their own
      anchor at the top of the results learns nothing.
    """
    raise NotImplementedError("TODO(Phạm Trọng): image/keyframe -> embedding -> vector search")
