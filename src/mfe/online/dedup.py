"""Dedup + temporal grouping — stage B of docs/04 §1.

The problem this solves (docs/04 §3.2): without it the grid is 50 near-identical
frames of one scene, and the operator scrolls past the answer. docs/context/
buoi-2.md §9 puts "Kết quả đa dạng" first among the places strong teams win:
"không để cả grid là 50 frame cùng một shot".

Note this is a *display* concern with a *retrieval* implementation. Collapsing a
group raises the diversity the human sees without changing what was retrieved.
"""

from __future__ import annotations

from mfe.contracts.tools import Candidate


def group_by_shot(candidates: list[Candidate]) -> list[list[Candidate]]:
    """Cluster candidates into shots.

    Owner: Phạm Trọng. TODO(Phạm Trọng): implement.

    Prefers `shot_id` when Quốc Đạt assigned one (optional in Sprint 1 — docs/00 §3).
    Falls back to temporal proximity within a video. `keyframe_index` (the
    keyframe's order) comes from `mfe.contracts.ids.parse_keyframe_id` — good
    enough for adjacency without a store lookup; use `pts_time` from the
    Relational Map when true elapsed time matters.
    """
    raise NotImplementedError("TODO(Phạm Trọng): cluster candidates by shot / temporal proximity")


def select_representatives(groups: list[list[Candidate]]) -> list[Candidate]:
    """Keep one candidate per group. docs/04 §3.2.

    Owner: Phạm Trọng. TODO(Phạm Trọng): implement.

    Keep the group's members reachable — docs/04 §3.2 says "Giữ đại diện mỗi
    nhóm, mở rộng khi người dùng cần". A collapsed group the operator cannot
    expand has hidden the answer rather than decluttered the view. The
    representative is usually the highest-scoring member, but for a shot that is
    often a transition frame; docs/context/buoi-1.md §7 records that a
    transition frame may not be accepted as an answer at all.
    """
    raise NotImplementedError("TODO(Phạm Trọng): one representative per group, kept expandable")


def dedup(candidates: list[Candidate]) -> list[Candidate]:
    """group_by_shot + select_representatives. Stage B entrypoint."""
    raise NotImplementedError("TODO(Phạm Trọng): group -> select representatives")
