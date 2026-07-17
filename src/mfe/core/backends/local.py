"""Local filesystem backend — the only implementation in Sprint 1 (docs/01 §3.2)."""

from __future__ import annotations

from mfe.core.storage import StorageBackend


class LocalBackend(StorageBackend):
    """Reads and writes plain files under a root directory.

    Owner: Lead. TODO(Lead): implement all three methods. Resolve `path`
    against the configured root and reject escapes above it — a `..` in an
    asset_path read from a manifest must not read outside the data dir.
    """

    def __init__(self, root: str = ".") -> None:
        self.root = root

    def read_bytes(self, path: str) -> bytes:
        raise NotImplementedError("TODO(Lead): read file under self.root")

    def write_bytes(self, path: str, data: bytes) -> None:
        raise NotImplementedError("TODO(Lead): mkdir parents + write file under self.root")

    def exists(self, path: str) -> bool:
        raise NotImplementedError("TODO(Lead): stat file under self.root")
