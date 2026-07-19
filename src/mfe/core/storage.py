"""Storage backend abstraction — docs/01 §3.2.

The whole point: switching local -> Modal is a config change, not a logic
change. Nothing outside this module may call `open()` on a store path. If Quốc Đạt
writes keyframes with `Path.write_bytes`, the Modal switch means editing Quốc Đạt's
code, which is exactly what this indirection exists to prevent.

Sprint 1 ships the local backend only. `backends/modal.py` is a deliberate stub.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from mfe.contracts.errors import ContractError, ErrorCode


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

    Owner: Vỹ Triết.

        local: root=<repo_root>            (from Config.repo_root)
        modal: volume=<config.storage.volume>

    Imports are local to each branch so that callers using the local backend do
    not pull in Modal's dependencies, and vice versa. Adding a backend touches
    only this function.
    """
    if name == "local":
        from mfe.core.backends.local import LocalBackend

        return LocalBackend(root=str(options.get("root", ".")))
    if name == "modal":
        from mfe.core.backends.modal import ModalBackend

        volume = options.get("volume")
        if not volume:
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT, "modal backend requires a 'volume' option"
            )
        return ModalBackend(volume=str(volume))
    raise ContractError(
        ErrorCode.INVALID_ARGUMENT,
        f"unknown storage backend {name!r}; expected 'local' or 'modal'",
    )
