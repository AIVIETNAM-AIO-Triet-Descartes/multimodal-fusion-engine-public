"""Minimal video browser — docs/04 §4.

    streamlit run src/mfe/browser/app.py

**Keep it MVP.** docs/00 §8 names gold-plating this the sprint's second risk:
building all 45 slides of Buổi 2 instead of reaching a working slice. Four core
capabilities, then stop.

    Grid ổn định                required  top-k, diverse, no layout overflow
    Timeline/neighbor           required  prev/next frame in ONE action
    Query refine                required  edit query, add/remove concepts
    Positive/negative feedback  nice      mark right/wrong -> ranking changes
    Sketch/pose/scene-graph     LATER     do not start

docs/context/buoi-1.md §7 on whether a pretty UI scores points: it does not —
but a *usable* one lets the operator find the answer faster, and that is TTCA,
which does score. Optimize for the operator's hands, not for looks.
"""

from __future__ import annotations


def render_grid(candidates: list) -> None:
    """Top-k results as a grid. Owner: TV3. TODO(TV3): implement.

    Diversity comes from `dedup.py` before this point (docs/04 §3.2) — do not
    re-solve it here. A group's members must stay expandable: collapsing a shot
    with no way to open it hides the answer instead of decluttering.
    """
    raise NotImplementedError("TODO(TV3): render candidate grid")


def render_timeline(keyframe_id: str) -> None:
    """Neighbours around a keyframe. Owner: TV3. TODO(TV3): implement.

    **One action.** docs/04 §4 makes this a hard requirement, and it is the
    single highest-leverage UI feature: temporal verification is the third
    hardest problem in docs/context/buoi-1.md §8, and an operator who has to
    manually reopen the video and scrub has lost the round.

    Backed by the `get_neighbors` tool — not by a direct store read.
    """
    raise NotImplementedError("TODO(TV3): render neighbors via get_neighbors tool")


def render_submit_panel(keyframe_id: str) -> None:
    """Submit, gated. Owner: TV3. TODO(TV3): implement.

    **"Xem" and "submit" are separate buttons** (docs/04 §4, docs/context/
    buoi-2.md §8). Submit goes through TV4's validator with a confirmation step.

    Never call `build_submission` without `Validator.validate` first. docs/05 §3:
    submit sai còn tệ hơn không submit. A one-click submit next to a one-click
    preview is how a wrong answer leaves the room at speed.
    """
    raise NotImplementedError("TODO(TV3): confirm -> validate -> build_submission")


def main() -> None:
    """Owner: TV3. TODO(TV3): wire the Streamlit page.

    Log every interaction through `mfe.core.logging` — query text, candidates
    shown, clicks, timestamps. Without it the Lead cannot compute TTCA
    (docs/04 §5), and TTCA is the metric the whole browser exists to move.
    """
    raise NotImplementedError("TODO(TV3): streamlit app entrypoint")


if __name__ == "__main__":
    main()
