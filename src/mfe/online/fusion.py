"""Late fusion — stage A of docs/04 §1. Combine CLIP and BM25 scores.

**The warning in docs/04 §3.1, in full:** the two scores are on different
scales — do not add them directly. CLIP cosine sits in [-1, 1]; BM25 is
unbounded and depends on the corpus. Adding them means BM25 silently decides
every ranking. docs/context/buoi-2.md §9 lists "Fusion có calibration" as a
place strong teams separate from weak ones.

Weights live in config (`retrieval.fusion_weights`). docs/00 §8 risk 4: do not
calibrate them deep into last year's data — 2026's data is different, and a
tuned constant is a liability that looks like an improvement.
"""

from __future__ import annotations

from mfe.contracts.tools import Candidate


def normalize_scores(candidates: list[Candidate]) -> list[Candidate]:
    """Bring one tool's scores onto a comparable scale.

    Owner: TV3. TODO(TV3): implement.

    Pick the method deliberately and write down why:
      - min-max      : bounded, but one outlier compresses everything else
      - z-score      : outlier-tolerant, unbounded output
      - rank-based (RRF): discards magnitude, very robust, no calibration to
        overfit — a strong default given docs/00 §8 risk 4

    Whatever you pick, normalize *within* a single tool's result list. Across
    lists is meaningless.
    """
    raise NotImplementedError("TODO(TV3): normalize scores within one result list")


def weighted_fusion(
    result_lists: dict[str, list[Candidate]],
    weights: dict[str, float],
) -> list[Candidate]:
    """Merge per-tool result lists into one ranking. docs/04 §3.1.

    Owner: TV3. TODO(TV3): implement.

    - Normalize each list first, then weight.
    - A keyframe found by one tool and not the other still competes: treat a
      missing score as the bottom of that tool's list, not as zero. Zero is a
      real value on some scales and would reward absence.
    - Carry `evidence_id` through. A fused candidate needs to trace back to what
      produced it (docs/00 §5), and after a merge that provenance is the only
      way the validator can check it.
    """
    raise NotImplementedError("TODO(TV3): normalize -> weight -> merge")
