"""Storage backend abstraction — docs/01 §3.2.

The whole point: switching local -> Modal is a config change, not a logic
change. Nothing outside this module may call `open()` on a store path. If TV1
writes keyframes with `Path.write_bytes`, the Modal switch means editing TV1's
code, which is exactly what this indirection exists to prevent.

Sprint 1 ships the local backend only. `backends/modal.py` is a deliberate stub.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Minimal byte-level interface — docs/01 §3.2.

    Deliberately small. It is a byte store, not a filesystem: no listdir, no
    rename, no seek. Anything richer would not port to a bucket.
    """

    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        """Return the bytes at `path`, or raise ContractError(STORE_UNAVAILABLE)."""

    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None:
        """Write `data` to `path`, creating parents as needed."""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Whether `path` holds anything."""


def get_backend(name: str, **options: object) -> StorageBackend:
    """Resolve `storage.backend` from config to an implementation.

    Owner: Lead. TODO(Lead): implement — dispatch "local" -> LocalBackend,
    "modal" -> ModalBackend. Keep the registry here so adding a backend never
    touches a caller.
    """
    raise NotImplementedError("TODO(Lead): dispatch config storage.backend -> StorageBackend")
