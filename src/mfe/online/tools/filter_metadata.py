"""`filter_metadata` — docs/04 §2.

    purpose: Lọc tập ứng viên theo ràng buộc object/ocr/metadata.
    input:   candidate_ids, constraints
    output:  [keyframe_id]
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from mfe.contracts.tools import ToolResult


def filter_metadata(
    constraints: dict[str, Any],
    candidate_ids: list[str] | None = None,
    metadata_index: Mapping[str, dict[str, Any]] | None = None,
) -> ToolResult[str]:
    """Filter candidate keyframe IDs according to object/OCR/metadata constraints.

    Args:
        constraints: Dictionary of constraint criteria (e.g. {'object': 'bag', 'exclude': 'car'}).
        candidate_ids: Target candidate keyframe IDs to filter. If None, operates on all.
        metadata_index: Optional pre-indexed keyframe metadata dictionary.

    Returns:
        ToolResult[str] containing filtered keyframe IDs.
    """
    start_t = time.perf_counter()
    tool_name = "filter_metadata"

    if candidate_ids is not None:
        target_ids = list(candidate_ids)
    elif metadata_index:
        target_ids = list(metadata_index.keys())
    else:
        target_ids = []

    filtered: list[str] = []
    required_object = constraints.get("object")
    excluded_object = constraints.get("exclude")

    for kf_id in target_ids:
        meta = metadata_index.get(kf_id, {}) if metadata_index else {}
        objs = meta.get("objects", [])

        if required_object and required_object.lower() not in [o.lower() for o in objs]:
            continue
        if excluded_object and excluded_object.lower() in [o.lower() for o in objs]:
            continue

        filtered.append(kf_id)

    latency = (time.perf_counter() - start_t) * 1000.0
    return ToolResult(
        tool=tool_name,
        items=filtered,
        error=None,
        latency_ms=latency,
    )
