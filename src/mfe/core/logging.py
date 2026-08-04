"""Structured logging — the raw material for TTCA and for reproducible runs.

Two hard requirements ride on this module:

  - docs/00 §1: ">= 30 query nội bộ không crash, có log tái hiện".
  - docs/04 §5: TTCA is computed from per-query logs (query text, candidates
    seen, clicks, timestamps). docs/context/buoi-2.md §2: TTCA, not vector
    search latency, is what the team optimizes.

So logs here are events the Vỹ Triết's harness reads, not prose for humans. Write
one JSON object per line.

Every event carries two clocks:

    ts_monotonic  perf_counter seconds — what durations are computed from
    ts_wall       unix seconds — what a human correlates with a screen recording

Durations must use the monotonic one. A wall-clock adjustment (NTP, DST, a
laptop waking up) mid-session would otherwise produce a negative TTCA, and the
median of a set containing one negative is quietly wrong rather than loudly.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from enum import StrEnum
from pathlib import Path


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


_lock = threading.Lock()
_run_id: str | None = None
_sink_path: Path | None = None
_buffer: list[dict] = []


def get_run_id() -> str:
    """Stable id for one harness run or one operator session. Owner: Vỹ Triết.

    Minted once per process and reused, because `evidence_id` embeds it
    (contracts/tools.py): if this changed between two calls in the same query,
    two candidates from the same run would look like they came from different
    ones and the validator's provenance check would lose its meaning.

    Shape is `{YYYYmmdd-HHMMSS}-{pid}-{4 hex}` — sortable by time, unique across
    concurrent processes, and matching the `[a-zA-Z0-9_-]+` the evidence_id
    contract allows in the middle field.
    """
    global _run_id
    with _lock:
        if _run_id is None:
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            _run_id = f"{stamp}-{os.getpid()}-{uuid.uuid4().hex[:4]}"
        return _run_id


def configure(path: str | None, run_id: str | None = None) -> None:
    """Point the event log at a file and reset in-memory state. Owner: Vỹ Triết.

    Called once at the start of a harness run or a browser session. `path=None`
    keeps events in memory only — which is what unit tests want, and what keeps
    an import of this module from creating files as a side effect.
    """
    global _run_id, _sink_path, _buffer
    with _lock:
        _buffer = []
        _run_id = run_id
        if path is None:
            _sink_path = None
        else:
            _sink_path = Path(path)
            _sink_path.parent.mkdir(parents=True, exist_ok=True)
            _sink_path.write_text("", encoding="utf-8")
    if run_id is None:
        get_run_id()


def log_event(event: EventType, **fields: object) -> dict:
    """Append one structured event. Owner: Vỹ Triết.

    One JSON object per line, always stamped with run_id and both clocks.
    Returns the event so a caller can keep it without re-reading the file.

    Appends per call rather than buffering to the end: docs/00 §1 asks for a log
    that reproduces a run, and a run that crashed is exactly the one whose log
    must survive.
    """
    record = {
        "run_id": get_run_id(),
        "event": str(event),
        "ts_monotonic": time.perf_counter(),
        "ts_wall": time.time(),
        **fields,
    }
    with _lock:
        _buffer.append(record)
        if _sink_path is not None:
            with _sink_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return record


def get_events() -> list[dict]:
    """Every event logged since `configure`. The harness scores off this."""
    with _lock:
        return list(_buffer)


def read_events(path: str) -> list[dict]:
    """Read an event log back from disk.

    Malformed lines are skipped rather than fatal: a run killed mid-write can
    leave a truncated last line, and losing the whole log of a crashed run is
    exactly the wrong response to the one run worth reading.
    """
    events: list[dict] = []
    source = Path(path)
    if not source.is_file():
        return events
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
