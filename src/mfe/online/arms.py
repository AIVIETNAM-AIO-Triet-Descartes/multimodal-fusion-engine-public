"""Stage-A arm registry — what the cascade fuses, and what the UI can toggle.

Stage A used to be two hardcoded `if` branches (`vector_search`, `search_text`)
plus a `_WEIGHT_KEY` dict translating tool names into config's signal names. That
shape stopped fitting the moment the merge plan called for eight arms
(docs/triet/ke-hoach-merge-sota.md §6.0 step 9), and the indirection was pure
overhead: an arm's name *is* its weight key.

One registry serves three needs that would otherwise be three mechanisms:

  * **toggling** — the browser's on/off switches set `enabled`
  * **per-arm scoring** — §3.4b B1 runs each arm alone by enabling exactly one
  * **plugging in** — a new arm is a registration, not an edit to `_coarse`

**`enabled` and `available` are different things, deliberately.** `enabled=False`
means a human turned it off. `available=False` means it cannot run — no store, no
model, no data. Collapsing them is exactly the bug in Bảo's UI: the ColPali switch
defaults to on while `search_engine.py` pins `has_colpali = False`, so the button
looks live, does nothing, and says nothing. An arm that cannot run must be able to
explain why.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace

from mfe.contracts.tools import Candidate

logger = logging.getLogger(__name__)

#: `(query_text, top_k, run_id) -> candidates`
SearchFn = Callable[[str, int, str], list[Candidate]]


@dataclass(frozen=True)
class Arm:
    """One named retrieval arm in stage A."""

    name: str
    search: SearchFn | None = None
    weight: float = 1.0
    enabled: bool = True
    #: Why this arm cannot run. Set whenever `search` is None; shown in the UI.
    unavailable_reason: str | None = None
    #: One line for the operator: what this arm actually matches on.
    description: str = ""
    #: Which language this arm's query must be in.
    #:
    #:   "vi"  the arm matches Vietnamese text (OCR, subtitle, Vietnamese labels)
    #:   "en"  the arm's model only understands English (SigLIP SO400M's text
    #:         tower, CLAP, PaliGemma/ColPali) — feeding it Vietnamese does not
    #:         error, it just retrieves badly, which is worse
    #:   "any" the model is cross-lingual (BAAI/bge-m3), so either works
    #:
    #: An "en" arm with no English query available goes unavailable with a reason
    #: rather than silently searching with the wrong language.
    query_language: str = "vi"

    @property
    def available(self) -> bool:
        return self.search is not None

    @property
    def active(self) -> bool:
        """Runs on the next query: switched on *and* able to run."""
        return self.enabled and self.available

    def status(self) -> str:
        if not self.available:
            return f"unavailable — {self.unavailable_reason or 'no reason recorded'}"
        return "on" if self.enabled else "off"


class ArmRegistry:
    """Ordered, name-addressed collection of arms.

    Order is preserved for display only; fusion is order-independent.
    """

    def __init__(self, arms: list[Arm] | None = None) -> None:
        self._arms: dict[str, Arm] = {}
        for arm in arms or []:
            self.register(arm)

    def register(self, arm: Arm) -> None:
        if arm.name in self._arms:
            raise ValueError(f"arm {arm.name!r} is already registered")
        if arm.search is None and arm.unavailable_reason is None:
            raise ValueError(
                f"arm {arm.name!r} has no search function and no unavailable_reason; "
                "an arm that cannot run must say why (see module docstring)"
            )
        self._arms[arm.name] = arm

    def __len__(self) -> int:
        return len(self._arms)

    def __iter__(self):
        return iter(self._arms.values())

    def __contains__(self, name: object) -> bool:
        return name in self._arms

    def get(self, name: str) -> Arm:
        if name not in self._arms:
            raise KeyError(f"no arm named {name!r}; have {sorted(self._arms)}")
        return self._arms[name]

    def active(self) -> list[Arm]:
        return [arm for arm in self._arms.values() if arm.active]

    def needs_english(self) -> bool:
        """Whether any arm about to run would consume an English query.

        Lets a caller skip translating — and skip loading a translation model
        that weighs more than a gigabyte — when nothing would read the result.
        Asked of the *active* arms, not all of them: an English arm whose
        collection never opened is not a reason to translate.
        """
        return any(arm.query_language == "en" for arm in self.active())

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Flip one switch. Enabling an unavailable arm is allowed and still
        will not run it — the UI shows the reason rather than silently lying."""
        self._arms[name] = replace(self.get(name), enabled=enabled)

    def set_weight(self, name: str, weight: float) -> None:
        self._arms[name] = replace(self.get(name), weight=float(weight))

    def only(self, name: str) -> None:
        """Enable exactly one arm. The per-arm measurement of §3.4b B1."""
        self.get(name)
        for arm_name in self._arms:
            self.set_enabled(arm_name, arm_name == name)

    def weights(self) -> dict[str, float]:
        return {arm.name: arm.weight for arm in self._arms.values()}

    def report(self) -> list[dict[str, object]]:
        """Rows for the UI and for `tiers_enabled` in an eval run (§3.6)."""
        return [
            {
                "name": arm.name,
                "status": arm.status(),
                "weight": arm.weight,
                "description": arm.description,
            }
            for arm in self._arms.values()
        ]

    def run_named(
        self,
        names: list[str],
        query_text: str,
        top_k: int,
        run_id: str,
        query_text_en: str | None = None,
    ) -> dict[str, list[Candidate]]:
        """`run()` restricted to specific arms, ignoring their `enabled` flags.

        For measuring one arm in isolation without mutating registry state.
        """
        saved = {arm.name: arm.enabled for arm in self}
        try:
            for name in saved:
                self.set_enabled(name, name in names)
            return self.run(query_text, top_k, run_id, query_text_en=query_text_en)
        finally:
            for name, was in saved.items():
                self.set_enabled(name, was)

    def run(
        self,
        query_text: str,
        top_k: int,
        run_id: str,
        query_text_en: str | None = None,
    ) -> dict[str, list[Candidate]]:
        """Run every active arm. A failing arm is dropped with a warning, never
        raised: one broken arm must not take the whole query down.

        `query_text` is the Vietnamese query the user typed. `query_text_en` is
        the optional English form; arms declaring `query_language="en"` need it
        and are skipped (loudly) without it.
        """
        results: dict[str, list[Candidate]] = {}
        for arm in self.active():
            text = query_text
            if arm.query_language == "en":
                if not query_text_en:
                    logger.warning(
                        "arm %s needs an English query and none was supplied; skipping. "
                        "Add `text_en` to the query, or disable the arm.",
                        arm.name,
                    )
                    continue
                text = query_text_en
            try:
                hits = arm.search(text, top_k, run_id)  # type: ignore[misc]
            except ValueError as err:
                # e.g. a query with no indexable term — a miss, not a crash.
                logger.info("arm %s returned nothing for %r (%s)", arm.name, query_text, err)
                continue
            except Exception as err:  # noqa: BLE001 — degrade, do not abort the query
                logger.warning("arm %s failed (%s); continuing without it", arm.name, err)
                continue
            if hits:
                results[arm.name] = hits
        return results
