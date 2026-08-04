"""Shared entrypoint setup for `mfe-ingest` and `mfe-eval`.

Exists for one unglamorous reason: the team runs Windows, where a console
defaults to cp1252, and every user-facing string in this repo is Vietnamese.
Without the reconfigure below, `mfe-eval --help` dies with a UnicodeEncodeError
before printing anything — argparse writes the help text straight to stdout, so
the crash happens inside the standard library and reads like a bug in the tool.

Called from `main()`, never at import time: reconfiguring a stream is a global
side effect, and importing a module should not have one.
"""

from __future__ import annotations

import logging
import sys


def setup_console(level: int = logging.INFO) -> None:
    """Make stdout/stderr UTF-8 capable, then configure logging."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # errors="replace" so an exotic character degrades to '?' rather
            # than killing a run that is otherwise finished and correct.
            reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
