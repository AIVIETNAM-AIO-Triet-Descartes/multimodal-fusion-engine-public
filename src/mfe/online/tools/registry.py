"""Tool registry — docs/00 §5.

Every online capability is wrapped as a tool with a contract and is **not tied
to the UI** (docs/04 §2). That constraint is what lets the Buổi 3 agent call the
same tools instead of a reimplementation: docs/context/buoi-3.md §2 is explicit
that "agent không nên thay retrieval engine", and §10 makes "chuẩn hóa
retrieval/browser Buổi 2 thành API/tool có contract" the first step of that
sprint.

So the test for whether a tool belongs here: could an agent call it with no
browser running? If it needs UI state, it is in the wrong place.

docs/context/buoi-3.md §9 also warns the other way — "Action space gọn". Too many
overlapping tools and a planner picks wrong. Five tools in Sprint 1, no more.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class ToolSpec:
    """A tool's contract, in the shape of docs/00 §5.

    Kept as data, not just a docstring, because the agent phase needs to *read*
    these to decide what to call. docs/context/buoi-3.md §7: the description
    must say what the tool does, its input, output, limits, and when **not** to
    use it.
    """

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
    """Owner: TV3. TODO(TV3): implement."""
    raise NotImplementedError("TODO(TV3): register tool spec by name")


def get(name: str) -> ToolSpec:
    """Owner: TV3. TODO(TV3): implement. Raise on unknown — never no-op."""
    raise NotImplementedError("TODO(TV3): name -> ToolSpec")


def list_specs() -> list[ToolSpec]:
    """All registered tools. The agent phase reads this to build its action space."""
    raise NotImplementedError("TODO(TV3): list registered specs")
