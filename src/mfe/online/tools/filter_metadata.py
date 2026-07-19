"""`filter_metadata` — docs/04 §2.

    purpose: Lọc tập ứng viên theo ràng buộc object/ocr/metadata.
    input:   candidate_ids, constraints
    output:  [keyframe_id]

A set operation, not a ranking — hence it returns ids, not Candidates. This is
the tool that narrows "person with a bag" to "person with a bag AND a car in
frame", using signals CLIP is weak at (docs/context/buoi-1.md §8: small text,
numbers, proper nouns, relations, negation).
"""

from __future__ import annotations

from mfe.contracts.tools import ToolResult


def filter_metadata(
    constraints: dict,
    candidate_ids: list[str] | None = None,
) -> ToolResult[str]:
    """Owner: Phạm Trọng. TODO(Phạm Trọng): implement.

    - `candidate_ids=None` filters the whole store; otherwise filter within the
      given set (docs/03 §2).
    - Support negative constraints. docs/context/buoi-2.md §6 lists marking a
      negative to exclude a wrong kind of scene as core exploitation behaviour,
      and §9 calls out negative constraints as a place teams pull ahead.
    - Term normalization must match how Công Quang indexed (docs/03 §2.2 stores "18h",
      Quốc Đạt emitted "BẢN TIN 18H"). Normalizing here differently from there means
      the filter silently matches nothing — which looks like "no such
      keyframes", not like a bug.
    - An empty result is a valid answer, not an error.
    """
    raise NotImplementedError("TODO(Phạm Trọng): apply constraints -> filtered keyframe_ids")
