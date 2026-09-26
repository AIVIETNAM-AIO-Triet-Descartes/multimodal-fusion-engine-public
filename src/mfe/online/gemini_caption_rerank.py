"""Optional caption based reranking through the Trammay Gemini gateway.

The coarse multimodal cascade remains the default.  This module is deliberately
opt-in because a remote reranker adds latency and its ranking must be observable
before an operator uses it for a submission.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mfe.contracts.tools import Candidate
from mfe.online.trammay import TrammayClient

logger = logging.getLogger(__name__)

DEFAULT_DEPTH = 30
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+#[0-9]+$")
_RERANK_CACHE_MAX = 128
_RERANK_CACHE: OrderedDict[str, tuple[str, ...]] = OrderedDict()
_RERANK_CACHE_LOCK = threading.RLock()


@dataclass(frozen=True)
class CaptionRerankOutcome:
    enabled: bool
    ran: bool
    cached: bool = False
    moved: int = 0
    duration_ms: float = 0.0
    reason: str | None = None


def _cache_key(
    candidates: Sequence[Candidate],
    query: str,
    query_en: str | None,
    model: str,
    depth: int,
) -> str:
    payload = {
        "model": model,
        "depth": depth,
        "query": query.strip(),
        "query_en": (query_en or "").strip(),
        "ids": [item.keyframe_id for item in candidates[:depth]],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _cache_get(key: str) -> tuple[str, ...] | None:
    with _RERANK_CACHE_LOCK:
        value = _RERANK_CACHE.get(key)
        if value is not None:
            _RERANK_CACHE.move_to_end(key)
        return value


def _cache_put(key: str, value: Sequence[str]) -> None:
    with _RERANK_CACHE_LOCK:
        _RERANK_CACHE[key] = tuple(value)
        _RERANK_CACHE.move_to_end(key)
        while len(_RERANK_CACHE) > _RERANK_CACHE_MAX:
            _RERANK_CACHE.popitem(last=False)


def _parse_ids(raw: str) -> list[str]:
    """Parse either a JSON array or ``{"ranking": [...]}`` response."""
    match = re.search(r"\[[\s\S]*?\]", raw or "")
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except (TypeError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]


def _text_for(bundle: object, keyframe_id: str, names: Sequence[str]) -> str:
    texts = getattr(bundle, "texts", {})
    parts: list[str] = []
    for name in names:
        value = str(texts.get(name, {}).get(keyframe_id, "") or "").strip()
        if value:
            parts.append(f"{name}: {value[:500]}")
    return " | ".join(parts) or "(khong co van ban)"


def rerank_candidates(
    candidates: Sequence[Candidate],
    query: str,
    bundle: object,
    *,
    query_en: str | None = None,
    client: Any | None = None,
    depth: int = DEFAULT_DEPTH,
) -> tuple[list[Candidate], CaptionRerankOutcome]:
    """Return a safe reordered copy and a diagnostic outcome.

    Unknown, duplicated or malformed IDs from the model are ignored.  If the
    gateway is unavailable or returns an invalid ranking, the original order is
    returned so enabling the switch can never erase a retrieval result.
    """
    started = time.perf_counter()
    head = list(candidates[: max(1, int(depth))])
    if not head:
        return [], CaptionRerankOutcome(True, False, duration_ms=0.0, reason="khong co ung vien")
    gateway = client if client is not None else TrammayClient.from_env()
    if gateway is None:
        return list(candidates), CaptionRerankOutcome(
            True, False, duration_ms=_elapsed(started), reason="thieu TRAMMAY_API_KEY"
        )

    # Only cache the real gateway path. Tests and callers that inject a client
    # may intentionally change the answer between calls.
    use_cache = client is None
    model = str(getattr(gateway, "model", ""))
    cache_key = _cache_key(head, query, query_en, model, len(head)) if use_cache else ""
    if use_cache:
        cached_order = _cache_get(cache_key)
        if cached_order is not None:
            by_id = {item.keyframe_id: item for item in head}
            reordered = [by_id[keyframe_id] for keyframe_id in cached_order if keyframe_id in by_id]
            if len(reordered) == len(head):
                moved = sum(
                    a.keyframe_id != b.keyframe_id
                    for a, b in zip(head, reordered, strict=True)
                )
                return (
                    reordered + list(candidates[len(head):]),
                    CaptionRerankOutcome(
                        True,
                        True,
                        cached=True,
                        moved=moved,
                        duration_ms=_elapsed(started),
                        reason="cache",
                    ),
                )

    rows = [
        {"id": item.keyframe_id, "text": _text_for(
            bundle,
            item.keyframe_id,
            ("caption_gemini", "caption_qwen", "caption", "ocr_gemini", "ocr_qwen25", "subtitle"),
        )}
        for item in head
    ]
    prompt = (
        "Rank the candidate keyframe IDs for this video retrieval query using only "
        "the supplied caption, OCR and subtitle evidence. Return ONLY a JSON array "
        "of the supplied IDs, from most relevant to least relevant. Include every "
        "ID exactly once; do not invent IDs.\n"
        f"QUERY VI: {query.strip()}\n"
        f"QUERY EN: {(query_en or '').strip()}\n"
        f"CANDIDATES: {json.dumps(rows, ensure_ascii=False)}"
    )
    try:
        raw = gateway.chat(prompt, max_tokens=max(256, len(head) * 18), temperature=0.0)
        order = _parse_ids(raw)
    except Exception as exc:  # noqa: BLE001 - retrieval must survive an optional remote tier
        logger.warning("caption rerank skipped: %s: %s", type(exc).__name__, exc)
        return list(candidates), CaptionRerankOutcome(
            True, False, duration_ms=_elapsed(started), reason=f"{type(exc).__name__}: {exc}"
        )

    by_id = {item.keyframe_id: item for item in head}
    seen: set[str] = set()
    reordered: list[Candidate] = []
    for keyframe_id in order:
        if keyframe_id in by_id and keyframe_id not in seen and _ID_RE.fullmatch(keyframe_id):
            reordered.append(by_id[keyframe_id])
            seen.add(keyframe_id)
    if len(reordered) != len(head):
        return list(candidates), CaptionRerankOutcome(
            True, False, duration_ms=_elapsed(started), reason="Gemini tra ve thu tu khong day du"
        )
    if use_cache:
        _cache_put(cache_key, [item.keyframe_id for item in reordered])
    output = reordered + list(candidates[len(head):])
    moved = sum(a.keyframe_id != b.keyframe_id for a, b in zip(head, reordered, strict=True))
    return output, CaptionRerankOutcome(True, True, moved=moved, duration_ms=_elapsed(started))


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
