"""Protected BTC-first pHash and embedding deduplication."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from mfe.offline.augmentation.models import CandidateRow
from mfe.offline.dedup_keyframes import hamming


@dataclass(frozen=True)
class ProtectedReference:
    keyframe_id: str
    shot_id: str
    phash: int
    embedding: np.ndarray


@dataclass(frozen=True)
class DedupDrop:
    keyframe_id: str
    duplicate_of: str
    reason: str
    value: float


@dataclass(frozen=True)
class DedupResult:
    kept: tuple[CandidateRow, ...]
    dropped: tuple[DedupDrop, ...]


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"embedding shape mismatch: {left.shape} != {right.shape}")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right) / denominator if denominator else 0.0


def protected_dedup(
    candidates: Iterable[CandidateRow],
    protected_btc: Iterable[ProtectedReference],
    candidate_phashes: Mapping[str, int],
    candidate_embeddings: Mapping[str, np.ndarray],
    *,
    phash_max_distance: int,
    embedding_max_cosine: float,
) -> DedupResult:
    """Drop only supplemental candidates, comparing within the same shot."""

    references: dict[str, list[ProtectedReference]] = {}
    protected_ids: set[str] = set()
    for reference in protected_btc:
        references.setdefault(reference.shot_id, []).append(reference)
        protected_ids.add(reference.keyframe_id)

    kept: list[CandidateRow] = []
    dropped: list[DedupDrop] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.keyframe_id
        if key in protected_ids or key in seen:
            dropped.append(DedupDrop(key, key, "exact_identity", 0.0))
            continue
        if key not in candidate_phashes or key not in candidate_embeddings:
            raise KeyError(f"missing pHash/embedding for candidate {key}")

        duplicate: DedupDrop | None = None
        same_shot = references.get(candidate.shot_id, [])
        for reference in same_shot:
            distance = hamming(candidate_phashes[key], reference.phash)
            if distance <= phash_max_distance:
                duplicate = DedupDrop(key, reference.keyframe_id, "phash", float(distance))
                break
        if duplicate is None:
            for reference in same_shot:
                similarity = _cosine(candidate_embeddings[key], reference.embedding)
                if similarity >= embedding_max_cosine:
                    duplicate = DedupDrop(
                        key, reference.keyframe_id, "embedding", similarity
                    )
                    break
        if duplicate is not None:
            dropped.append(duplicate)
            continue

        candidate_reference = ProtectedReference(
            keyframe_id=key,
            shot_id=candidate.shot_id,
            phash=candidate_phashes[key],
            embedding=np.asarray(candidate_embeddings[key], dtype=np.float32),
        )
        references.setdefault(candidate.shot_id, []).append(candidate_reference)
        kept.append(candidate)
        seen.add(key)
    return DedupResult(tuple(kept), tuple(dropped))
