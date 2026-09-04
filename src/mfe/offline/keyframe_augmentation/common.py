"""Small deterministic I/O helpers shared by Gate 0-2."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


def pts_time_for_frame_idx(frame_idx: int, literal_fps: float) -> float:
    """Return a synthetic PTS safely inside one literal-FPS frame interval.

    New frames are identified by the decoder's integer source-frame counter.
    A synthetic PTS is only a compatibility field for the existing relational
    contract; it must never be used to recover or replace that identity.  The
    interval midpoint avoids the floating-point lower-bound problem of plain
    ``frame_idx / fps`` and guarantees the BTC audit invariant.
    """
    if frame_idx < 0:
        raise ValueError(f"frame_idx must be >= 0, got {frame_idx}")
    if not math.isfinite(literal_fps) or literal_fps <= 0:
        raise ValueError(f"literal_fps must be finite and > 0, got {literal_fps}")
    pts_time = (frame_idx + 0.5) / literal_fps
    if math.floor(pts_time * literal_fps) != frame_idx:
        raise AssertionError("synthetic PTS failed literal-FPS floor round-trip")
    return pts_time


def canonical_json_bytes(value: Any) -> bytes:
    """Return a stable UTF-8 JSON representation suitable for fingerprints."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            stream.write("\n")
    os.replace(temporary, target)
