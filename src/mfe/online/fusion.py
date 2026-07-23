"""Late fusion — stage A of docs/04 §1. Combine CLIP and BM25 scores.

Uses Reciprocal Rank Fusion (RRF) and weighted score normalization to merge
disparate score scales without risking unbounded BM25 dominate CLIP.
"""

from __future__ import annotations

from typing import Literal

from mfe.contracts.tools import Candidate
from mfe.online.tools.evidence import create_evidence_id


def normalize_scores(
    candidates: list[Candidate],
    method: Literal["rrf", "min_max"] = "rrf",
    rrf_k: int = 60,
) -> list[Candidate]:
    """Normalize candidate scores within a single tool result list.

    Args:
        candidates: Input list of candidates.
        method: Normalization method ('rrf' or 'min_max'). Default 'rrf'.
        rrf_k: RRF constant factor (default 60).

    Returns:
        New list of Candidates with normalized scores in [0, 1].
    """
    if not candidates:
        return []

    # Sort candidates by score descending to assign ranks
    sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
    norm_cands: list[Candidate] = []

    if method == "rrf":
        for rank, c in enumerate(sorted_cands, start=1):
            rrf_score = 1.0 / (rrf_k + rank)
            norm_cands.append(
                Candidate(
                    keyframe_id=c.keyframe_id,
                    video_id=c.video_id,
                    score=rrf_score,
                    evidence_id=c.evidence_id,
                    matched_terms=c.matched_terms,
                )
            )
    else:  # min_max
        scores = [c.score for c in sorted_cands]
        min_s = min(scores)
        max_s = max(scores)
        denom = (max_s - min_s) if max_s > min_s else 1.0

        for c in sorted_cands:
            norm_s = (c.score - min_s) / denom if max_s > min_s else 1.0
            norm_cands.append(
                Candidate(
                    keyframe_id=c.keyframe_id,
                    video_id=c.video_id,
                    score=norm_s,
                    evidence_id=c.evidence_id,
                    matched_terms=c.matched_terms,
                )
            )

    return norm_cands


def weighted_fusion(
    result_lists: dict[str, list[Candidate]],
    weights: dict[str, float],
    run_id: str = "fusion_run",
) -> list[Candidate]:
    """Merge multiple per-tool result lists into a unified ranking.

    Args:
        result_lists: Dictionary mapping tool_name -> candidate_list.
        weights: Dictionary mapping tool_name -> fusion_weight.
        run_id: Tracing identifier for fused evidence.

    Returns:
        List of fused Candidate objects sorted by score descending.
    """
    if not result_lists:
        return []

    # Normalize each list independently
    normalized_map: dict[str, list[Candidate]] = {}
    for tool_name, cands in result_lists.items():
        normalized_map[tool_name] = normalize_scores(cands, method="rrf")

    # Aggregate scores by keyframe_id
    combined_scores: dict[str, float] = {}
    candidate_metadata: dict[str, dict] = {}

    for tool_name, cands in normalized_map.items():
        w = weights.get(tool_name, 1.0)
        for c in cands:
            kf_id = c.keyframe_id
            if kf_id not in combined_scores:
                combined_scores[kf_id] = 0.0
                candidate_metadata[kf_id] = {
                    "video_id": c.video_id,
                    "matched_terms": set(c.matched_terms),
                    "ev_sources": [c.evidence_id],
                }
            combined_scores[kf_id] += w * c.score
            candidate_metadata[kf_id]["matched_terms"].update(c.matched_terms)
            if c.evidence_id not in candidate_metadata[kf_id]["ev_sources"]:
                candidate_metadata[kf_id]["ev_sources"].append(c.evidence_id)

    # Sort keyframes by combined score descending
    sorted_kf_ids = sorted(combined_scores.keys(), key=lambda kf: combined_scores[kf], reverse=True)

    fused_cands: list[Candidate] = []
    for idx, kf_id in enumerate(sorted_kf_ids, start=1):
        meta = candidate_metadata[kf_id]
        ev_id = create_evidence_id("late_fusion", run_id, idx)
        fused_cands.append(
            Candidate(
                keyframe_id=kf_id,
                video_id=meta["video_id"],
                score=combined_scores[kf_id],
                evidence_id=ev_id,
                matched_terms=sorted(list(meta["matched_terms"])),
            )
        )

    return fused_cands
