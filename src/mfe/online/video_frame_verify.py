"""Fast, explicit VLM verification inside a video that retrieval already found.

The expensive and inaccurate shape of this feature would be to upload a whole
video.  Instead we score every catalogue keyframe locally with the already
loaded BGE-M3 caption vectors, retain a small and temporally useful shortlist,
and make one multimodal Trammay/Gemini request.  The remote model therefore
sees enough evidence to choose the exact moment without paying video upload,
decode, or thousands of image tokens on every click.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from mfe.contracts.records import KeyframeRow
from mfe.online.dense_arms import encode_query
from mfe.online.trammay import TrammayClient

_BGE_CHECKPOINT = "BAAI/bge-m3"
_BGE_DIMENSION = 1024
_TEXT_SOURCES = (
    "caption_gemini",
    "caption_qwen",
    "caption",
    "ocr_gemini",
    "ocr_qwen25",
    "subtitle",
)
_CACHE_MAX = 128
_CACHE: OrderedDict[str, Any] = OrderedDict()
_CACHE_LOCK = threading.RLock()
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class VideoFrameVerification:
    best_keyframe_id: str
    alternatives: tuple[str, ...]
    confidence: float
    confidence_label: str
    reason: str
    auto_selected: bool
    total_keyframes_scanned: int
    shortlisted: int
    images_sent: int
    image_parts_sent: int
    evidence_sources: tuple[str, ...]
    duration_ms: float
    cached: bool = False


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 1)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).strip()


def _tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return {token for token in _WORD_RE.findall(normalized) if len(token) >= 2}


def _metadata(bundle: object, keyframe_id: str) -> dict[str, str]:
    stores = getattr(bundle, "texts", {})
    return {
        source: _as_text(stores.get(source, {}).get(keyframe_id, ""))
        for source in _TEXT_SOURCES
    }


def _lexical_score(query_tokens: set[str], metadata: Mapping[str, str]) -> float:
    if not query_tokens:
        return 0.0
    weights = {
        "caption_gemini": 1.0,
        "caption_qwen": 0.7,
        "caption": 0.6,
        "ocr_gemini": 1.25,
        "ocr_qwen25": 1.0,
        "subtitle": 1.1,
    }
    score = 0.0
    for source, text in metadata.items():
        overlap = query_tokens.intersection(_tokens(text))
        score += weights.get(source, 1.0) * len(overlap) / max(1, len(query_tokens))
    return score


def shortlist_video_keyframes(
    query: str,
    rows: Sequence[KeyframeRow],
    bundle: object,
    *,
    anchor_keyframe_id: str,
    dense_key_index: Mapping[str, int] | None = None,
    related_keyframe_ids: Sequence[str] = (),
    max_images: int = 12,
) -> list[KeyframeRow]:
    """Scan one complete video locally and return a diverse VLM shortlist."""
    if not rows:
        return []
    limit = max(1, int(max_images))
    by_id = {row.keyframe_id: row for row in rows}
    query_tokens = _tokens(query)
    metadata = {row.keyframe_id: _metadata(bundle, row.keyframe_id) for row in rows}
    lexical = sorted(
        rows,
        key=lambda row: _lexical_score(query_tokens, metadata[row.keyframe_id]),
        reverse=True,
    )

    dense_ranked: list[KeyframeRow] = []
    vectors = getattr(bundle, "vectors", {}).get("bge_m3")
    if dense_key_index and vectors is not None:
        valid = [row for row in rows if row.keyframe_id in dense_key_index]
        if valid:
            indices = np.asarray(
                [dense_key_index[row.keyframe_id] for row in valid], dtype=np.int64
            )
            query_vector = encode_query(query, _BGE_CHECKPOINT, _BGE_DIMENSION)
            scores = np.asarray(vectors[indices] @ query_vector, dtype=np.float32)
            order = np.argsort(scores)[::-1]
            dense_ranked = [valid[int(index)] for index in order]

    anchor_index = next(
        (index for index, row in enumerate(rows) if row.keyframe_id == anchor_keyframe_id),
        0,
    )
    nearby = [
        rows[index]
        for offset in (0, -1, 1, -2, 2)
        for index in (anchor_index + offset,)
        if 0 <= index < len(rows)
    ]
    related = [by_id[key] for key in related_keyframe_ids if key in by_id]

    # Dense caption semantics sees the whole video.  Lexical evidence catches
    # exact OCR/subtitle words, while nearby frames protect fine moment timing.
    pools = (
        dense_ranked[:6],
        lexical[:2],
        nearby,
        related[:2],
        dense_ranked[6:],
        lexical[2:],
    )
    selected: list[KeyframeRow] = []
    seen: set[str] = set()
    for pool in pools:
        for row in pool:
            if row.keyframe_id in seen:
                continue
            selected.append(row)
            seen.add(row.keyframe_id)
            if len(selected) >= limit:
                return selected
    return selected


def _encode_contact_sheet(items: Sequence[tuple[Path, str]]) -> str:
    """Pack four labeled frames into one image part to reduce gateway latency."""
    columns = 2
    tile_width, tile_height, header = 512, 288, 28
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * (tile_height + header)), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(items):
        left = (index % columns) * tile_width
        top = (index // columns) * (tile_height + header)
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        image_left = left + (tile_width - image.width) // 2
        image_top = top + header + (tile_height - image.height) // 2
        canvas.paste(image, (image_left, image_top))
        draw.text((left + 8, top + 8), label, fill="white")
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=82, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _parse_object(raw: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", raw or "")
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _confidence(value: Any) -> float:
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    text = str(value or "").strip().lower()
    try:
        return min(1.0, max(0.0, float(text.rstrip("%")) / (100.0 if "%" in text else 1.0)))
    except ValueError:
        labels = {
            "high": 0.9,
            "cao": 0.9,
            "medium": 0.65,
            "vua": 0.65,
            "low": 0.35,
            "thap": 0.35,
        }
        return labels.get(text, 0.0)


def _cache_key(
    query: str,
    video_id: str,
    anchor_keyframe_id: str,
    model: str,
    ids: Sequence[str],
    auto_threshold: float,
) -> str:
    value = json.dumps(
        {
            "query": query.strip(),
            "video": video_id,
            "anchor": anchor_keyframe_id,
            "model": model,
            "ids": list(ids),
            "threshold": auto_threshold,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode()).hexdigest()


def verify_video_keyframe(
    query: str,
    video_id: str,
    anchor_keyframe_id: str,
    rows: Sequence[KeyframeRow],
    bundle: object,
    image_resolver: Callable[[str], Path | None],
    *,
    dense_key_index: Mapping[str, int] | None = None,
    related_keyframe_ids: Sequence[str] = (),
    model: str,
    client: Any | None = None,
    max_images: int = 12,
    auto_threshold: float = 0.8,
) -> VideoFrameVerification:
    """Choose the exact catalogue keyframe with one bounded Gemini request."""
    started = time.perf_counter()
    shortlisted = shortlist_video_keyframes(
        query,
        rows,
        bundle,
        anchor_keyframe_id=anchor_keyframe_id,
        dense_key_index=dense_key_index,
        related_keyframe_ids=related_keyframe_ids,
        max_images=max_images,
    )
    image_paths: dict[str, Path] = {}
    available: list[KeyframeRow] = []
    for row in shortlisted:
        path = image_resolver(row.keyframe_id)
        if path is not None:
            available.append(row)
            image_paths[row.keyframe_id] = path
    if not available:
        raise RuntimeError("video khong co anh keyframe local de Gemini xac nhan")

    key = _cache_key(
        query,
        video_id,
        anchor_keyframe_id,
        model,
        [row.keyframe_id for row in available],
        auto_threshold,
    )
    if client is None:
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
            if cached is not None:
                _CACHE.move_to_end(key)
                return replace(cached, cached=True, duration_ms=_elapsed(started))

    rows_for_prompt: list[dict[str, Any]] = []
    feature_scores: dict[str, float] = {}
    feature_ranks: dict[str, int] = {}
    vectors = getattr(bundle, "vectors", {}).get("bge_m3")
    if dense_key_index and vectors is not None:
        feature_rows = [row for row in rows if row.keyframe_id in dense_key_index]
        if feature_rows:
            indices = np.asarray(
                [dense_key_index[row.keyframe_id] for row in feature_rows], dtype=np.int64
            )
            query_vector = encode_query(query, _BGE_CHECKPOINT, _BGE_DIMENSION)
            scores = np.asarray(vectors[indices] @ query_vector, dtype=np.float32)
            order = np.argsort(scores)[::-1]
            feature_scores = {
                row.keyframe_id: float(score)
                for row, score in zip(feature_rows, scores, strict=True)
            }
            feature_ranks = {
                feature_rows[int(position)].keyframe_id: rank
                for rank, position in enumerate(order, 1)
            }
    anchor_position = next(
        (index for index, row in enumerate(rows) if row.keyframe_id == anchor_keyframe_id),
        0,
    )
    positions = {row.keyframe_id: index for index, row in enumerate(rows)}
    for row in available:
        metadata = _metadata(bundle, row.keyframe_id)
        rows_for_prompt.append(
            {
                "id": row.keyframe_id,
                "time_s": round(float(row.pts_time), 3),
                "caption": (
                    metadata["caption_gemini"]
                    or metadata["caption_qwen"]
                    or metadata["caption"]
                )[:500],
                "ocr": (metadata["ocr_gemini"] or metadata["ocr_qwen25"])[:300],
                "subtitle": metadata["subtitle"][:300],
                "bge_m3_cosine": round(feature_scores.get(row.keyframe_id, 0.0), 4),
                "bge_m3_rank_in_video": feature_ranks.get(row.keyframe_id),
                "keyframes_from_current": positions[row.keyframe_id] - anchor_position,
            }
        )
    encoded_images = [
        _encode_contact_sheet(
            [
                (image_paths[row.keyframe_id], row.keyframe_id)
                for row in available[start : start + 4]
            ]
        )
        for start in range(0, len(available), 4)
    ]

    gateway = client if client is not None else TrammayClient.from_env()
    if gateway is None:
        raise RuntimeError("thieu TRAMMAY_API_KEY")
    prompt = (
        "You verify the exact keyframe inside a video whose video_id is already known "
        "to be correct. "
        "The backend scanned ALL catalogue keyframes in this video with BGE-M3 and text evidence; "
        "the attached contact sheets contain the strongest shortlist, with each tile labeled by "
        "keyframe id. Inspect pixels, action timing, "
        "counts, OCR and subtitles. Pick the frame that best matches the requested exact moment. "
        "Lower confidence when the exact action is between sampled frames, repeated, obscured, or "
        "not visibly proven. confidence must be 0..1 and should be >=0.80 only when "
        "the image clearly "
        "confirms the requested moment. Return ONLY JSON with keys best_keyframe_id, alternatives "
        "(up to 3 supplied ids), confidence, is_match, reason. Write reason briefly in Vietnamese. "
        "Never invent an id.\n"
        f"QUERY: {query.strip()}\n"
        f"VIDEO_ID: {video_id}\n"
        f"CURRENT_FRAME: {anchor_keyframe_id}\n"
        f"CANDIDATES_IN_IMAGE_ORDER: {json.dumps(rows_for_prompt, ensure_ascii=False)}"
    )
    raw = gateway.chat(
        prompt,
        images_b64=encoded_images,
        model=model,
        max_tokens=512,
        temperature=0.0,
    )
    parsed = _parse_object(raw)
    allowed = {row.keyframe_id for row in available}
    best = str(parsed.get("best_keyframe_id") or "").strip()
    if best not in allowed:
        raise RuntimeError("Gemini khong tra ve keyframe hop le trong shortlist")
    confidence = _confidence(parsed.get("confidence"))
    is_match = parsed.get("is_match") is not False
    auto_selected = bool(is_match and confidence >= auto_threshold)
    if confidence >= 0.8:
        label = "cao"
    elif confidence >= 0.55:
        label = "vừa"
    else:
        label = "thấp"
    raw_alternatives = parsed.get("alternatives")
    alternatives = tuple(
        candidate
        for candidate in (
            str(value).strip() for value in raw_alternatives if isinstance(value, str)
        )
        if candidate in allowed and candidate != best
    )[:3] if isinstance(raw_alternatives, list) else ()
    outcome = VideoFrameVerification(
        best_keyframe_id=best,
        alternatives=alternatives,
        confidence=round(confidence, 4),
        confidence_label=label,
        reason=str(parsed.get("reason") or "Gemini da doi chieu anh va metadata.").strip()[:1000],
        auto_selected=auto_selected,
        total_keyframes_scanned=len(rows),
        shortlisted=len(shortlisted),
        images_sent=len(available),
        image_parts_sent=len(encoded_images),
        evidence_sources=("BGE-M3 toàn video", "caption Gemini", "OCR", "subtitle"),
        duration_ms=_elapsed(started),
    )
    if client is None:
        with _CACHE_LOCK:
            _CACHE[key] = outcome
            _CACHE.move_to_end(key)
            while len(_CACHE) > _CACHE_MAX:
                _CACHE.popitem(last=False)
    return outcome
