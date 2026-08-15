"""Dedup + temporal grouping — stage B of docs/04 §1.

Collapses near-duplicate keyframes (same shot, or close in time) to preserve grid
diversity without losing retrieved items.

**The window is measured in source-video frames, not in keyframes.** It used to
be keyframe ordinals, back when `keyframe_id` encoded one. When the id switched
to `frame_id` the same comparison would have kept compiling, kept passing its
tests, and quietly stopped grouping anything — 5 frames is 0.17s, so every
candidate would have become its own group. The unit is in the parameter name for
exactly that reason.

The default (640) reproduces the old behaviour on the shipped keyframes: their
median spacing is 128 frames, so the old span of 5 keyframes was ~640 frames
(~21s at 30fps). It is a translation of the old setting, not a new tuning — and
worth revisiting, since 21s is a generous idea of "near-duplicate".
"""

from __future__ import annotations

from mfe.contracts.ids import parse_keyframe_id
from mfe.contracts.tools import Candidate

#: Frame span of one group. See the module docstring for where 640 comes from.
DEFAULT_TEMPORAL_WINDOW_FRAMES = 640


def group_by_shot(
    candidates: list[Candidate],
    temporal_window_frames: int = DEFAULT_TEMPORAL_WINDOW_FRAMES,
) -> list[list[Candidate]]:
    """Cluster candidates into shot groups or temporal proximity clusters.

    A group spans at most `temporal_window_frames` frames, measured from the
    first keyframe in the group — not from the previous one.

    Measuring from the previous keyframe chains transitively: a run of keyframes
    each within the window of its predecessor collapses into a single "shot"
    covering the entire video. That is not hypothetical — on the 2026 sample it
    produced a 216-candidate group spanning keyframes 1 to 262 of L27_V004,
    leaving one representative for a 22-minute video. Because `Cascade._group`
    places every representative ahead of every non-representative, the effect was
    to bury correct answers: the ground-truth keyframe of the KIS baseline query
    fell from fused rank 60 to rank 180 and out of the submitted top 100, scoring
    0.0 on a query the vector arm had ranked 30th.

    Args:
        candidates: Candidate list sorted by rank/score.
        temporal_window_frames: Source-video frame span of one group.

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
        # Sort candidates in video by frame
        indexed_cands = []
        for c in cands:
            try:
                _, frame_id = parse_keyframe_id(c.keyframe_id)
            except ValueError:
                frame_id = 0
            indexed_cands.append((frame_id, c))

        indexed_cands.sort(key=lambda item: item[0])

        current_group: list[Candidate] = []
        anchor_frame: int | None = None

        for frame_id, c in indexed_cands:
            if anchor_frame is None or (frame_id - anchor_frame) <= temporal_window_frames:
                current_group.append(c)
                if anchor_frame is None:
                    anchor_frame = frame_id
            else:
                groups.append(current_group)
                current_group = [c]
                anchor_frame = frame_id

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


def dedup(
    candidates: list[Candidate],
    temporal_window_frames: int = DEFAULT_TEMPORAL_WINDOW_FRAMES,
) -> list[Candidate]:
    """Stage B entrypoint: group_by_shot + select_representatives.

    Args:
        candidates: Candidate list.
        temporal_window_frames: Proximity threshold, in source-video frames.

    Returns:
        Deduplicated candidate list.
    """
    groups = group_by_shot(candidates, temporal_window_frames=temporal_window_frames)
    return select_representatives(groups)
