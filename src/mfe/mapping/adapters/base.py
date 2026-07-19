"""Submit adapter interface — docs/05 §2.

One adapter per candidate BTC format. All four stay live until BTC announces the
real one; config picks (`submit.schema_format`).

When the real schema lands: add a file here, add its name to `SubmitFormat` in
contracts/tools.py, set it in config. No other code changes — that is the point
of the indirection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mfe.contracts.records import KeyframeRow, VideoRow


class SubmitAdapter(ABC):
    """Turns the authoritative rows into one submission record."""

    #: Value of `submit.schema_format` that selects this adapter.
    format_name: str

    @abstractmethod
    def build(self, keyframe: KeyframeRow, video: VideoRow) -> dict[str, Any]:
        """Build the record.

        Both rows come from the Relational Map and are already authoritative.
        Read fields off them; do not recompute anything from `fps`.
        """


_REGISTRY: dict[str, type[SubmitAdapter]] = {}


def register(adapter_cls: type[SubmitAdapter]) -> None:
    """Owner: Nhật Bảo. TODO(Nhật Bảo): implement — key on `adapter_cls.format_name`."""
    raise NotImplementedError("TODO(Nhật Bảo): register adapter under format_name")


def get_adapter(format_name: str) -> SubmitAdapter:
    """Owner: Nhật Bảo. TODO(Nhật Bảo): implement.

    Raise on an unknown name. A default-to-something here would submit in the
    wrong format on a config typo, and nothing upstream would notice.
    """
    raise NotImplementedError("TODO(Nhật Bảo): format_name -> adapter instance")
