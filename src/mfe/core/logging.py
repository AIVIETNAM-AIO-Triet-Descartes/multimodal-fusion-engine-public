"""Structured logging — the raw material for TTCA and for reproducible runs.

Two hard requirements ride on this module:

  - docs/00 §1: ">= 30 query nội bộ không crash, có log tái hiện".
  - docs/04 §5: TTCA is computed from per-query logs (query text, candidates
    seen, clicks, timestamps). docs/context/buoi-2.md §2: TTCA, not vector
    search latency, is what the team optimizes.

So logs here are events the Vỹ Triết's harness reads, not prose for humans. Write
one JSON object per line.
"""

from __future__ import annotations

from enum import StrEnum


class EventType(StrEnum):
    """Event vocabulary. The harness (docs/01 §2) aggregates on these."""

    QUERY_RECEIVED = "query_received"      # TTCA clock starts
    TOOL_CALLED = "tool_called"
    TOOL_RETURNED = "tool_returned"        # carries latency_ms per stage (docs/04 §5)
    CANDIDATES_SHOWN = "candidates_shown"
    CANDIDATE_CLICKED = "candidate_clicked"
    ANSWER_READY = "answer_ready"          # TTCA clock stops
    SUBMIT_ATTEMPTED = "submit_attempted"
    SUBMIT_BLOCKED = "submit_blocked"      # which of the 5 rules fired (docs/05 §3)
    SUBMIT_ACCEPTED = "submit_accepted"


def get_run_id() -> str:
    """Stable id for one harness run or one operator session.

    Owner: Vỹ Triết. TODO(Vỹ Triết): implement. `evidence_id` is expected to embed this
    (see contracts/tools.py) so a candidate can be traced back to the exact call
    that produced it.
    """
    raise NotImplementedError("TODO(Vỹ Triết): mint a run id")


def log_event(event: EventType, **fields: object) -> None:
    """Append one structured event.

    Owner: Vỹ Triết. TODO(Vỹ Triết): implement — JSON per line, always stamped with
    run_id and a monotonic timestamp. Use a monotonic clock for durations: a
    wall-clock adjustment mid-session would corrupt TTCA.
    """
    raise NotImplementedError("TODO(Vỹ Triết): write one JSON event per line")
