"""SQLite backend for the Relational Map — Sprint 1. docs/05 §1.

"Đủ cho sprint, đổi sang Postgres khi scale."
"""

from __future__ import annotations

import sqlite3


class SqliteBackend:
    """Owner: TV4. TODO(TV4): implement.

    Settings worth turning on deliberately rather than discovering later:

    - `PRAGMA foreign_keys = ON`. SQLite ignores FK constraints unless asked,
      per connection. Without it, `schema.sql`'s REFERENCES is decoration and a
      keyframe can point at a video that does not exist.
    - Bulk-insert keyframes with `executemany` in one transaction. Row-at-a-time
      commits make a full-dataset ingest hours slower for no reason.
    - Row factory returning dicts, so `relational.py` can hand straight to
      `KeyframeRow.model_validate`.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def connect(self) -> sqlite3.Connection:
        raise NotImplementedError("TODO(TV4): connect with foreign_keys ON")

    def executescript(self, sql: str) -> None:
        raise NotImplementedError("TODO(TV4): run schema.sql")
