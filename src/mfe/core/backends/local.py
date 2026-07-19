"""Local filesystem backend — the only implementation in Sprint 1 (docs/01 §3.2)."""

from __future__ import annotations

from pathlib import Path

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.core.storage import StorageBackend


class LocalBackend(StorageBackend):
    """Reads and writes plain files under a root directory.

    Owner: Lead.

    Every access is confined to `root`. A path read from a manifest — an
    `asset_path` written by TV1, or later something on a Modal volume — is
    untrusted input; a `..` in it must not read outside the data tree. Absolute
    paths are joined the pathlib way (`root / "/x"` == `/x`) and then rejected
    by the same containment check, so an absolute escape fails too.
    """

    def __init__(self, root: str = ".") -> None:
        self.root = root
        self._root_resolved = Path(root).resolve()

    def _full(self, path: str) -> Path:
        """Resolve `path` under root, refusing anything that escapes it."""
        full = (self._root_resolved / path).resolve()
        if full != self._root_resolved and self._root_resolved not in full.parents:
            raise ContractError(
                ErrorCode.STORE_UNAVAILABLE,
                f"path {path!r} escapes storage root {self._root_resolved}",
            )
        return full

    def read_bytes(self, path: str) -> bytes:
        full = self._full(path)
        try:
            return full.read_bytes()
        except FileNotFoundError as exc:
            raise ContractError(
                ErrorCode.STORE_UNAVAILABLE, f"no bytes at {path!r}"
            ) from exc
        except OSError as exc:
            raise ContractError(
                ErrorCode.STORE_UNAVAILABLE, f"cannot read {path!r}: {exc}"
            ) from exc

    def write_bytes(self, path: str, data: bytes) -> None:
        full = self._full(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)

    def exists(self, path: str) -> bool:
        return self._full(path).exists()
