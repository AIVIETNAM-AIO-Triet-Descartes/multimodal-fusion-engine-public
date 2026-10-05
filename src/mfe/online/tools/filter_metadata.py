"""`filter_metadata` — docs/04 §2.

    purpose: Lọc tập ứng viên theo ràng buộc object/ocr/metadata.
    input:   candidate_ids, constraints
    output:  [keyframe_id]

**This is a tool shell, not a second filtering engine** (CHỐT — Vỹ Triết).
The matching logic lives in `InvertedIndex.filter`, which owns the index, the
Unicode/casefold normalisation, and the constraint vocabulary. This module only
wraps it in the ToolResult envelope of docs/00 §5 so the browser and the Buổi 3
agent can call it through the same contract as every other tool.

It used to filter on its own, against a `metadata_index` dict passed in from
outside, and it understood only `object` / `exclude` — while the index
understood `video_id`, `object_labels`, `ocr_text`, `metadata_text` and
`asr_text`. Two vocabularies for one operation is precisely the drift docs/00 §8
ranks as risk #1: a caller using this tool silently could not reach most of what
Công Quang had indexed.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from mfe.contracts.errors import ErrorCode
from mfe.contracts.tools import ToolError, ToolResult


class SupportsFilter(Protocol):
    """The one method this tool needs — `InvertedIndex` satisfies it.

    Typed as a Protocol rather than importing `InvertedIndex` so the online
    layer keeps no hard dependency on the store package (and so tests can pass
    a fake without building an index).
    """

    def filter(
        self, constraints: dict[str, Any], candidate_ids: list[str] | None = None
    ) -> list[str]: ...


def filter_metadata(
    constraints: dict[str, Any],
    candidate_ids: list[str] | None = None,
    inverted_index: SupportsFilter | None = None,
) -> ToolResult[str]:
    """Filter candidate keyframe IDs against the Inverted Index.

    Args:
        constraints: Constraint mapping. The vocabulary is the index's:
            `video_id`, `object_labels` (alias `object`), `ocr_text` (`ocr`),
            `metadata_text` (`metadata`), `asr_text` (`asr`). Values may be a
            string, a list, or `{"contains": [...]}`.
        candidate_ids: Restrict to these ids. `None` filters the whole store.
        inverted_index: The store to filter against. Required — see below.

    Returns:
        ToolResult[str] of keyframe_ids. This is a set operation, not a ranking,
        so it returns ids rather than Candidates.

    Without an index this returns INVALID_ARGUMENT rather than an empty list.
    An empty result and "nobody gave me anything to search" are different
    answers, and the caller that cannot tell them apart is the one that ships a
    query silently matching nothing (docs/04 §2: structured error, never a
    fabricated result).
    """
    start_t = time.perf_counter()
    tool_name = "filter_metadata"

    def fail(code: ErrorCode, message: str) -> ToolResult[str]:
        return ToolResult(
            tool=tool_name,
            items=[],
            error=ToolError(code=code, message=message, tool=tool_name),
            latency_ms=(time.perf_counter() - start_t) * 1000.0,
        )

    if inverted_index is None:
        return fail(
            ErrorCode.INVALID_ARGUMENT,
            "filter_metadata needs an inverted_index to filter against; "
            "pass the InvertedIndex the cascade was built with",
        )
    if not constraints:
        return fail(ErrorCode.INVALID_ARGUMENT, "constraints must not be empty")

    try:
        filtered = inverted_index.filter(constraints, candidate_ids=candidate_ids)
    except ValueError as err:
        # InvertedIndex rejects an unknown field or a malformed value. Surfacing
        # it as a structured error keeps the operator's typo visible instead of
        # returning zero hits that look like "nothing matched".
        return fail(ErrorCode.INVALID_ARGUMENT, str(err))

    return ToolResult(
        tool=tool_name,
        items=filtered,
        error=None,
        latency_ms=(time.perf_counter() - start_t) * 1000.0,
    )
