"""Modal volume backend — STUB. Not active in Sprint 1.

docs/00 §1 scope table puts "Full dataset trên Modal" and "Scale-out, GPU
orchestration thật" outside this sprint. docs/01 §3.2: "để sẵn khung nhưng
không kích hoạt".

This file exists so the seam is real rather than hypothetical. When the switch
happens, the diff is confined to this file, `infra/modal/`, and config.

TODO(scale-out sprint), noted in docs/01 §3.2 as things to size before starting:
  - extracting keyframes across hundreds of hours of video
  - Faiss index size vs available RAM
  - VRAM ceiling for the CLIP/detector batch size
"""

from __future__ import annotations

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.core.storage import StorageBackend

_STUB_MESSAGE = (
    "Modal backend is a stub (docs/01 §3.2). Sprint 1 runs local only — "
    "set storage.backend: local in your config."
)


class ModalBackend(StorageBackend):
    """Placeholder. Fails loudly rather than silently falling back to local.

    A silent fallback would let someone believe they were testing the Modal path
    when they were not.
    """

    def __init__(self, volume: str) -> None:
        self.volume = volume

    def read_bytes(self, path: str) -> bytes:
        raise ContractError(ErrorCode.NOT_IMPLEMENTED, _STUB_MESSAGE)

    def write_bytes(self, path: str, data: bytes) -> None:
        raise ContractError(ErrorCode.NOT_IMPLEMENTED, _STUB_MESSAGE)

    def exists(self, path: str) -> bool:
        raise ContractError(ErrorCode.NOT_IMPLEMENTED, _STUB_MESSAGE)
