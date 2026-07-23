"""Tool registry — docs/00 §5.

Every online capability is wrapped as a tool with a contract and is not tied
to the UI (docs/04 §2).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mfe.online.tools.build_segment import build_segment
from mfe.online.tools.filter_metadata import filter_metadata
from mfe.online.tools.get_neighbors import get_neighbors
from mfe.online.tools.get_video_segment import get_video_segment
from mfe.online.tools.rerank import rerank
from mfe.online.tools.search_image import search_image
from mfe.online.tools.search_text import search_text


class ToolSpec:
    """A tool's contract, in the shape of docs/00 §5."""

    def __init__(
        self,
        name: str,
        purpose: str,
        constraints: str,
        fn: Callable[..., Any],
    ) -> None:
        self.name = name
        self.purpose = purpose
        self.constraints = constraints
        self.fn = fn


_REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> None:
    """Register a tool spec by name."""
    _REGISTRY[spec.name] = spec


def get(name: str) -> ToolSpec:
    """Retrieve a registered tool spec by name. Raise KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown tool '{name}'. Registered: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_specs() -> list[ToolSpec]:
    """List all registered tool specs."""
    return list(_REGISTRY.values())


# Register core online tools
register(
    ToolSpec(
        name="search_text",
        purpose="Tìm keyframe theo mô tả văn bản (semantic + text).",
        constraints="top_k <= 500",
        fn=search_text,
    )
)
register(
    ToolSpec(
        name="search_image",
        purpose="Tìm keyframe tương tự một ảnh tham chiếu.",
        constraints="top_k <= 500, exactly one of image_bytes or keyframe_id",
        fn=search_image,
    )
)
register(
    ToolSpec(
        name="filter_metadata",
        purpose="Lọc tập ứng viên theo ràng buộc object/ocr/metadata.",
        constraints="Negative constraints supported",
        fn=filter_metadata,
    )
)
register(
    ToolSpec(
        name="get_neighbors",
        purpose="Lấy keyframe lân cận theo thời gian để xem ngữ cảnh trước/sau.",
        constraints="window > 0, bounded within video",
        fn=get_neighbors,
    )
)
register(
    ToolSpec(
        name="build_segment",
        purpose="Ghép keyframe thành khung thời gian start_ms/end_ms.",
        constraints=">=1 keyframe_id",
        fn=build_segment,
    )
)
register(
    ToolSpec(
        name="get_video_segment",
        purpose="Trả về video gốc + khung thời gian segment.",
        constraints="Valid video_id",
        fn=get_video_segment,
    )
)
register(
    ToolSpec(
        name="rerank",
        purpose="Xếp lại tập nhỏ bằng cross-encoder.",
        constraints="Enforces latency budget",
        fn=rerank,
    )
)
