"""Adapters that expose a loaded :class:`FlatBundle` as QA text evidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import SimpleNamespace


@dataclass(frozen=True)
class FlatTextStore:
    """Small read-only text-store adapter for ``answering.gather_evidence``.

    ``FlatBundle`` stores one mapping per source while the evidence API consumes
    rows with ``video_id``, ``frame_id``, ``source`` and ``text`` attributes.
    Keeping the adapter here avoids teaching either the retrieval bundle or the
    generic evidence collector about the other's storage layout.
    """

    bundle: object

    def fetch(
        self,
        keys: Iterable[tuple[str, int]],
        *,
        source: str | None = None,
        limit: int = 500,
    ) -> list[object]:
        wanted = {(str(video), int(frame)) for video, frame in keys}
        if not wanted:
            return []
        texts = getattr(self.bundle, "texts", {})
        sources = [source] if source is not None else list(texts)
        rows: list[object] = []
        for video, frame in wanted:
            key = f"{video}#{frame:06d}"
            for source_name in sources:
                value = texts.get(source_name, {}).get(key, "")
                if not str(value or "").strip():
                    continue
                rows.append(
                    SimpleNamespace(
                        video_id=video,
                        frame_id=frame,
                        source=source_name,
                        text=str(value),
                        score=0.0,
                    )
                )
                if len(rows) >= limit:
                    return rows
        return rows
