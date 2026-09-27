"""Late fusion — stage A of docs/04 §1. Combine CLIP and BM25 scores.

Uses Reciprocal Rank Fusion (RRF) and weighted score normalization to merge
disparate score scales without risking unbounded BM25 dominate CLIP.
"""

from __future__ import annotations

from typing import Literal

from mfe.contracts.tools import Candidate
from mfe.online.tools.evidence import create_evidence_id

Normalizer = Literal["rrf", "linear_rank", "min_max"]


def normalize_scores(
    candidates: list[Candidate],
    method: Normalizer = "rrf",
    rrf_k: int = 60,
) -> list[Candidate]:
    """Normalize candidate scores within a single tool result list.

    Three methods, and the choice is not cosmetic:

    ``rrf``          ``1/(k + rank)`` — reads rank only, so an arm cannot
                     dominate through the size of its numbers. Steep at the top.
    ``linear_rank``  ``1 - (rank-1)/n`` — also rank-only, but decays evenly. The
                     honest partner for `weighted_sum`: with both modes reading
                     rank, the only difference left between them is the decay
                     shape, which is the thing worth measuring.
    ``min_max``      ``(s-min)/(max-min)`` — reads the raw values, so use it only
                     where they are calibrated. **Fragile to outliers**: measured
                     on SOTA_AIC's subtitle arm, one exact-match score of 100.0
                     against a [0.05, 1.0] body collapses the entire rest of the
                     arm into [0, 0.005] — see ke-hoach-merge-sota.md §3.4b.

    Args:
        candidates: Input list of candidates.
        method: Normalization method. Default 'rrf'.
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
    elif method == "linear_rank":
        total = len(sorted_cands)
        for rank, c in enumerate(sorted_cands, start=1):
            norm_cands.append(
                Candidate(
                    keyframe_id=c.keyframe_id,
                    video_id=c.video_id,
                    score=1.0 - (rank - 1) / total,
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
    normalizer: Normalizer = "rrf",
    rrf_k: int = 60,
) -> list[Candidate]:
    """Merge multiple per-tool result lists into a unified ranking.

    Every arm is normalized independently, then weighted, then summed. Which
    normalizer runs decides what "mode" means: `rrf` reproduces Reciprocal Rank
    Fusion, `linear_rank` gives the evenly-decaying weighted sum.

    **A weight of 0 does not disable an arm.** The arm still contributes its
    candidates to the pool, it just adds nothing to their score — which is why a
    zero-weighted BM25 arm was still visibly changing results. To remove an arm,
    take it out of `result_lists` (or disable it in `ArmRegistry`).

    Args:
        result_lists: Dictionary mapping arm name -> candidate_list.
        weights: Dictionary mapping arm name -> fusion_weight.
        run_id: Tracing identifier for fused evidence.
        normalizer: How each arm's scores are made comparable.
        rrf_k: RRF constant, used only by the `rrf` normalizer.

    Returns:
        List of fused Candidate objects sorted by score descending.
    """
    if not result_lists:
        return []

    # Normalize each list independently
    normalized_map: dict[str, list[Candidate]] = {}
    for tool_name, cands in result_lists.items():
        normalized_map[tool_name] = normalize_scores(cands, method=normalizer, rrf_k=rrf_k)

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
                frame_idx=int(kf_id.split("#")[1]) if "#" in kf_id else None,
            )
        )

    return fused_cands
