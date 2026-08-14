"""Dedup + temporal grouping — stage B of docs/04 §1.

Collapses near-duplicate keyframes (same shot or close keyframe_index) to preserve
grid diversity without losing retrieved items.
"""

from __future__ import annotations

from mfe.contracts.ids import parse_keyframe_id
from mfe.contracts.tools import Candidate


def group_by_shot(candidates: list[Candidate], temporal_window: int = 5) -> list[list[Candidate]]:
    """Cluster candidates into shot groups or temporal proximity clusters.

    A group spans at most `temporal_window` keyframes, measured from the first
    keyframe in the group — not from the previous one.

    Measuring from the previous keyframe chains transitively: keyframes 1, 6,
    11, ... 262 each sit within 5 of their predecessor, so they collapse into a
    single "shot" covering the entire video. That is not hypothetical — on the
    2026 sample it produced a 216-candidate group spanning keyframes 1 to 262 of
    L27_V004, leaving one representative for a 22-minute video. Because
    `Cascade._group` places every representative ahead of every non-
    representative, the effect was to bury correct answers: the ground-truth
    keyframe of the KIS baseline query fell from fused rank 60 to rank 180 and
    out of the submitted top 100, scoring 0.0 on a query the vector arm had
    ranked 30th.

    Args:
        candidates: Candidate list sorted by rank/score.
        temporal_window: Keyframe index span of one group (default 5).

    Returns:
        List of candidate groups (each group is a list of Candidate objects).
    """
    if not candidates:
        return []

    groups: list[list[Candidate]] = []
    video_groups: dict[str, list[Candidate]] = {}

    # Separate candidates by video_id
    for c in candidates:
        video_groups.setdefault(c.video_id, []).append(c)

    for cands in video_groups.values():
        # Sort candidates in video by keyframe_index
        indexed_cands = []
        for c in cands:
            try:
                _, kf_idx = parse_keyframe_id(c.keyframe_id)
            except ValueError:
                kf_idx = 0
            indexed_cands.append((kf_idx, c))

        indexed_cands.sort(key=lambda item: item[0])

        current_group: list[Candidate] = []
        anchor_idx: int | None = None

        for kf_idx, c in indexed_cands:
            if anchor_idx is None or (kf_idx - anchor_idx) <= temporal_window:
                current_group.append(c)
                if anchor_idx is None:
                    anchor_idx = kf_idx
            else:
                groups.append(current_group)
                current_group = [c]
                anchor_idx = kf_idx

        if current_group:
            groups.append(current_group)

    # Sort groups by the highest candidate score in each group
    groups.sort(key=lambda grp: max(c.score for c in grp), reverse=True)
    return groups


def select_representatives(groups: list[list[Candidate]]) -> list[Candidate]:
    """Select one representative candidate per group.

    Args:
        groups: Clustered candidate groups.

    Returns:
        List of representative Candidate objects.
    """
    representatives: list[Candidate] = []
    for grp in groups:
        if not grp:
            continue
        # Pick the candidate with the highest score as group representative
        best_cand = max(grp, key=lambda c: c.score)
        representatives.append(best_cand)

    return representatives


def dedup(candidates: list[Candidate], temporal_window: int = 5) -> list[Candidate]:
    """Stage B entrypoint: group_by_shot + select_representatives.

    Args:
        candidates: Candidate list.
        temporal_window: Proximity window threshold.

    Returns:
        Deduplicated candidate list.
    """
    groups = group_by_shot(candidates, temporal_window=temporal_window)
    return select_representatives(groups)
