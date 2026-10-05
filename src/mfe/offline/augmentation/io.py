"""Small, deterministic I/O helpers used by every augmentation gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hash one file or a directory tree with stable relative-path identity.

    TensorFlow SavedModel checkpoints are directories (``saved_model.pb`` plus
    ``variables/*``), so hashing only the protobuf would leave most learned
    weights outside the experiment fingerprint. Directory entries are ordered
    by POSIX relative path and both their names, sizes and bytes are committed.
    """

    source = Path(path)
    if source.is_file():
        return sha256_file(source, chunk_size)
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = sorted(
        (item for item in source.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(source).as_posix(),
    )
    if not files:
        raise ValueError(f"cannot hash empty directory: {source}")
    digest = hashlib.sha256()
    digest.update(b"mfe-directory-sha256-v1\0")
    for item in files:
        relative = item.relative_to(source).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash a JSON-compatible object independent of mapping insertion order."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def write_jsonl_atomic(path: str | Path, rows: Iterable[BaseModel | dict]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    lines = []
    for row in rows:
        value = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
        lines.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    temporary.replace(target)
    return target


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        rows.append(value)
    return rows
