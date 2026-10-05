"""`search_image` — docs/04 §2.

    purpose: Tìm keyframe tương tự một ảnh tham chiếu.
    input:   image_bytes | keyframe_id, top_k
    output:  [{keyframe_id, video_id, score, evidence_id}]
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Sequence
from typing import Any

from mfe.contracts.errors import ErrorCode
from mfe.contracts.tools import Candidate, ToolError, ToolResult
from mfe.online.encoder import encode_image


def search_image(
    image_bytes: bytes | None = None,
    keyframe_id: str | None = None,
    top_k: int = 100,
    run_id: str = "default_run",
    vector_index: Any = None,
    mock_candidates: Sequence[Candidate] | None = None,
) -> ToolResult[Candidate]:
    """Execute image-to-image similarity search.

    Args:
        image_bytes: Raw image file bytes.
        keyframe_id: Existing keyframe ID to search by.
        top_k: Number of candidates to retrieve.
        run_id: Run identifier for evidence tracing.
        mock_candidates: Optional pre-constructed candidate list for testing.

    Returns:
        ToolResult[Candidate] envelope containing items or ToolError.
    """
    start_t = time.perf_counter()
    tool_name = "search_image"

    if (image_bytes is None) == (keyframe_id is None):
        latency = (time.perf_counter() - start_t) * 1000.0
        return ToolResult(
            tool=tool_name,
            items=[],
            error=ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message="Must provide exactly one of image_bytes or keyframe_id",
                tool=tool_name,
            ),
            latency_ms=latency,
        )

    if top_k > 500:
        latency = (time.perf_counter() - start_t) * 1000.0
        return ToolResult(
            tool=tool_name,
            items=[],
            error=ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"top_k ({top_k}) exceeds maximum constraint of 500",
                tool=tool_name,
            ),
            latency_ms=latency,
        )

    if mock_candidates is not None:
        items = list(mock_candidates[:top_k])
    elif vector_index is None:
        # Never invent hits (docs/04 §2). A fabricated keyframe here reaches the
        # scoreboard looking exactly like a real one.
        latency = (time.perf_counter() - start_t) * 1000.0
        return ToolResult(
            tool=tool_name,
            items=[],
            error=ToolError(
                code=ErrorCode.STORE_UNAVAILABLE,
                message="search_image needs a VectorIndex; pass vector_index=...",
                tool=tool_name,
            ),
            latency_ms=latency,
        )
    else:
        try:
            embedding = _embed(image_bytes, keyframe_id, vector_index)
        except Exception as err:  # noqa: BLE001 — surface as a structured error
            latency = (time.perf_counter() - start_t) * 1000.0
            return ToolResult(
                tool=tool_name,
                items=[],
                error=ToolError(
                    code=ErrorCode.STORE_UNAVAILABLE,
                    message=f"could not embed the reference image: {err}",
                    tool=tool_name,
                ),
                latency_ms=latency,
            )
        # Ask for one extra, then drop the anchor: searching by an existing
        # keyframe always returns that keyframe first at cosine 1.0, which is
        # a slot wasted on something the operator is already looking at.
        hits = vector_index.search(
            query_embedding=embedding, top_k=min(top_k + 1, 500), run_id=run_id
        )
        items = [hit for hit in hits if hit.keyframe_id != keyframe_id][:top_k]

    latency = (time.perf_counter() - start_t) * 1000.0
    return ToolResult(
        tool=tool_name,
        items=items,
        error=None,
        latency_ms=latency,
    )


def _embed(
    image_bytes: bytes | None, keyframe_id: str | None, vector_index: Any
) -> list[float]:
    """Turn the reference into a query vector.

    Searching by `keyframe_id` reuses the vector already in the index rather
    than re-encoding the image: it is free, and it is guaranteed to be the same
    vector the index was built from, so the anchor cannot drift from itself.
    """
    if keyframe_id is not None:
        internal_id = vector_index._sidecar.to_internal_id(keyframe_id)  # noqa: SLF001
        return list(vector_index._index.reconstruct(int(internal_id)))  # noqa: SLF001

    if not image_bytes:
        raise ValueError("no reference: pass either keyframe_id or image_bytes")

    # encode_image takes a path; bytes arrive from a browser upload. Written to
    # a temp file rather than held in memory so the caller's upload and the
    # encoder agree on exactly what was decoded.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
        handle.write(image_bytes)
        temp_path = handle.name
    try:
        return encode_image(temp_path).tolist()
    finally:
        os.unlink(temp_path)
