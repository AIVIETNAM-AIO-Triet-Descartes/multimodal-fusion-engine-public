"""Locks the stage-A arm registry — mfe.online.arms.

The behaviour worth protecting is not "arms can be listed". It is the distinction
between *switched off* and *cannot run*, because collapsing the two is what makes
a UI lie to its operator.
"""

from __future__ import annotations

import pytest

from mfe.contracts.tools import Candidate
from mfe.online.arms import Arm, ArmRegistry


def _cand(frame: int, score: float, video: str = "L21_V001") -> Candidate:
    return Candidate(
        keyframe_id=f"{video}#{frame:06d}",
        video_id=video,
        score=score,
        evidence_id=f"bm25:test:{frame}",
    )


def _fixed(hits: list[Candidate]):
    return lambda query, top_k, run_id: hits[:top_k]


def _boom(query, top_k, run_id):
    raise RuntimeError("store went away mid-query")


class TestArmState:
    def test_available_arm_that_is_on_is_active(self):
        arm = Arm(name="bm25", search=_fixed([]))
        assert arm.available and arm.enabled and arm.active
        assert arm.status() == "on"

    def test_switched_off_is_not_active(self):
        arm = Arm(name="bm25", search=_fixed([]), enabled=False)
        assert arm.available and not arm.active
        assert arm.status() == "off"

    def test_unavailable_reports_why(self):
        arm = Arm(name="siglip", unavailable_reason="collection not ingested")
        assert not arm.available and not arm.active
        assert "collection not ingested" in arm.status()

    def test_enabling_an_unavailable_arm_still_does_not_run_it(self):
        """The ColPali bug, as a test.

        SOTA_AIC ships a ColPali switch that defaults to on while the engine
        pins has_colpali = False: the switch looks live and does nothing. Here
        the switch may be flipped, but `active` stays False and the reason is
        still readable — so the UI can render it disabled instead of lying.
        """
        registry = ArmRegistry(
            [Arm(name="colpali", unavailable_reason="colpali-engine not installed")]
        )
        registry.set_enabled("colpali", True)
        arm = registry.get("colpali")
        assert arm.enabled is True
        assert arm.active is False
        assert registry.active() == []
        assert "colpali-engine not installed" in arm.status()


class TestRegistry:
    def test_rejects_an_arm_that_cannot_run_and_will_not_say_why(self):
        with pytest.raises(ValueError, match="must say why"):
            ArmRegistry([Arm(name="mystery")])

    def test_rejects_duplicate_names(self):
        registry = ArmRegistry([Arm(name="bm25", search=_fixed([]))])
        with pytest.raises(ValueError, match="already registered"):
            registry.register(Arm(name="bm25", search=_fixed([])))

    def test_only_enables_exactly_one(self):
        """Backs the per-arm measurement of ke-hoach-merge-sota.md §3.4b B1."""
        registry = ArmRegistry(
            [
                Arm(name="bm25", search=_fixed([])),
                Arm(name="clip", search=_fixed([])),
                Arm(name="ocr", search=_fixed([])),
            ]
        )
        registry.only("clip")
        assert [a.name for a in registry.active()] == ["clip"]

    def test_only_rejects_an_unknown_name_before_disabling_everything(self):
        registry = ArmRegistry([Arm(name="bm25", search=_fixed([]))])
        with pytest.raises(KeyError):
            registry.only("nope")
        assert [a.name for a in registry.active()] == ["bm25"], "state must be untouched"

    def test_weights_key_by_arm_name(self):
        registry = ArmRegistry(
            [Arm(name="bm25", search=_fixed([]), weight=2.0), Arm(name="clip", search=_fixed([]))]
        )
        assert registry.weights() == {"bm25": 2.0, "clip": 1.0}


class TestRun:
    def test_runs_only_active_arms(self):
        registry = ArmRegistry(
            [
                Arm(name="bm25", search=_fixed([_cand(90, 0.9)])),
                Arm(name="clip", search=_fixed([_cand(261, 0.8)]), enabled=False),
                Arm(name="siglip", unavailable_reason="not ingested"),
            ]
        )
        assert sorted(registry.run("q", 10, "run1")) == ["bm25"]  # clip off, siglip unavailable

    def test_a_failing_arm_does_not_take_the_query_down(self):
        """One broken store must cost its own arm, not the whole result."""
        registry = ArmRegistry(
            [Arm(name="bm25", search=_boom), Arm(name="clip", search=_fixed([_cand(90, 0.9)]))]
        )
        results = registry.run("q", 10, "run1")
        assert list(results) == ["clip"]

    def test_an_arm_with_no_hits_is_omitted_not_empty(self):
        registry = ArmRegistry([Arm(name="bm25", search=_fixed([]))])
        assert registry.run("q", 10, "run1") == {}

    def test_respects_top_k(self):
        hits = [_cand(f, 1.0 - f / 1000) for f in (90, 261, 351, 420)]
        registry = ArmRegistry([Arm(name="bm25", search=_fixed(hits))])
        assert len(registry.run("q", 2, "run1")["bm25"]) == 2


class TestReport:
    def test_report_covers_every_arm_including_unavailable(self):
        """`tiers_enabled` in an eval result (§3.6) has to record the ones that
        did NOT run too, or two runs look comparable when they are not."""
        registry = ArmRegistry(
            [
                Arm(name="bm25", search=_fixed([])),
                Arm(name="clip", search=_fixed([]), enabled=False),
                Arm(name="siglip", unavailable_reason="not ingested"),
            ]
        )
        rows = registry.report()
        assert [r["name"] for r in rows] == ["bm25", "clip", "siglip"]
        assert [r["status"] for r in rows] == ["on", "off", "unavailable — not ingested"]
