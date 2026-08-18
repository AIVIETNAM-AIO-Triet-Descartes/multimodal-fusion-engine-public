"""Locks DAKE — mfe.offline.dake.

Every sequence here is built by hand, so nothing in this file needs a video, a
codec or a GPU (ke-hoach-offline-ingest.md §3, step B2). That is the point of
keeping the algorithm pure: the parts that can be wrong in a subtle way are the
parts that can be tested cheaply.
"""

from __future__ import annotations

import math

import pytest

from mfe.contracts.errors import ContractError
from mfe.offline.dake import pts_for_frame, select_keyframes, steepness


class TestSteepness:
    def test_returns_one_value_per_frame_except_the_last(self):
        """The last frame has no lookahead, so it gets no value — not a zero.

        Zero is mid-range on a signed scale, so defaulting the final frame to 0.0
        would rank it above every genuinely-darkening frame in the video.
        """
        assert len(steepness([1.0, 2.0, 3.0, 4.0])) == 3
        assert steepness([1.0]) == []
        assert steepness([]) == []

    def test_flat_sequence_has_no_steepness(self):
        assert steepness([500.0] * 8) == [0.0] * 7

    def test_bounded_in_minus_one_to_one(self):
        """It is the sine of the slope angle, so the bound is structural. A raw
        slope would not be bounded, and one frame pair could then dominate."""
        spiky = [1.0, 10_000.0, 1.0, 10_000.0, 1.0, 10_000.0]
        assert all(-1.0 <= value <= 1.0 for value in steepness(spiky))

    def test_scale_invariant(self):
        """Normalising by `s_max` is what makes values comparable across videos
        of different resolution and encoder quality. Doubling every size — the
        same content at a different bitrate — must not move the curve."""
        sizes = [100.0, 140.0, 130.0, 700.0, 690.0, 200.0]
        doubled = [value * 2 for value in sizes]
        for left, right in zip(steepness(sizes), steepness(doubled), strict=True):
            assert left == pytest.approx(right)

    def test_peaks_at_a_step_up_and_troughs_at_a_step_down(self):
        """The paper keeps the sign, and this is what that means in practice.

        One step up at index 4 and one step down at index 9. Under the paper's
        formula the rise scores highest and the fall scores *lowest* — the drop is
        not "a strong signal with a minus sign", it is ranked last. That is a real
        consequence of following the paper and the reason `rank_by` exists.
        """
        sizes = [100.0] * 5 + [900.0] * 5 + [100.0] * 5
        scores = steepness(sizes)
        assert scores.index(max(scores)) == 4
        assert scores.index(min(scores)) == 9
        assert max(scores) > 0 > min(scores)

    def test_zero_sizes_do_not_divide_by_zero(self):
        assert steepness([0.0, 0.0, 0.0]) == [0.0, 0.0]

    def test_rejects_a_zero_lookahead(self):
        with pytest.raises(ContractError, match="lookahead"):
            steepness([1.0, 2.0], lookahead=0)


class TestSelectKeyframes:
    def test_rho_sets_the_budget_before_the_density_rule_adds_to_it(self):
        """200 frames, rho 0.02 -> 4 frames of budget. fps is set high enough
        that the 2-second window spans the clip, so nothing is added and the
        budget is visible on its own."""
        sizes = [100.0 + (i % 7) * 50 for i in range(200)]
        picked = select_keyframes(sizes, fps=200.0, rho=0.02, min_gap=1)
        assert len(picked) == 4

    def test_density_rule_can_push_the_count_above_the_budget(self):
        """A guarantee that yields to a budget is not a guarantee.

        600 flat frames at 25fps: the budget is 12 frames, but a 2-second window
        is 50 frames, so covering 600 frames needs at least 12 — and the flat
        curve makes the greedy pass clump them at the front, so the density rule
        has to add more.
        """
        picked = select_keyframes([500.0] * 600, fps=25.0, rho=0.02, min_gap=5)
        assert len(picked) > 12

    @pytest.mark.parametrize("fps", [25.0, 30.0, 29.97, 26.44])
    def test_the_two_second_bound_holds_in_seconds_not_frames(self, fps):
        """Regression: the window was `round(2.0 * fps)`, which overshoots.

        60 frames at 29.97fps is 2.002s and 53 at 26.44 is 2.005s — and both of
        those rates are in batch 1. Small, but this bound is the entire reason
        DAKE was chosen over BTC's own keyframes (median 3.36s, worst 8.4s), so a
        bound the code itself exceeds is the wrong kind of small.
        """
        picked = select_keyframes([100.0 + (i % 5) * 20 for i in range(2000)], fps=fps)
        gaps = [(b - a) / fps for a, b in zip(picked, picked[1:], strict=False)]
        assert max(gaps) <= 2.0

    def test_honours_the_max_gap_guarantee(self):
        """The promise the whole plan rests on: worst-case gap 2.0s, against
        BTC's measured median 3.36s and worst case 8.4s (§3.1)."""
        sizes = [100.0 + (i % 3) * 10 for i in range(1000)]
        fps, window = 25.0, 50
        picked = select_keyframes(sizes, fps=fps, rho=0.02)
        assert picked[0] <= window - 1
        assert picked[-1] >= len(sizes) - window
        assert all(b - a <= window for a, b in zip(picked, picked[1:], strict=False))

    def test_density_rule_spaces_frames_instead_of_keeping_every_one(self):
        """Regression: the density fill used to keep an entire static stretch.

        On a flat curve every candidate scores the same, and `max` returns the
        leftmost of equals — so each pass advanced the frontier by one frame and
        the rule ended up selecting frames 4,5,6,...,30 consecutively instead of
        every 10th. It terminated and it satisfied the gap guarantee, so no other
        test noticed; the cost was silent and expensive — a static video would
        have blown the keyframe budget by ~10x and dragged every downstream model
        along with it.

        40 flat frames at 5fps: budget 4, window 10.
        """
        picked = select_keyframes([100.0] * 40, fps=5.0, rho=0.1, min_gap=1)
        assert picked == [0, 1, 2, 3, 13, 23, 33]

    def test_honours_min_gap_within_the_budget(self):
        """Guards the anti-clumping rule: one cut yields several adjacent
        high-steepness frames, and keeping them all spends the budget on one
        moment. Checked on the greedy pass alone — the density rule is allowed to
        place frames closer, because covering a gap outranks spacing."""
        sizes = [100.0] * 20 + [900.0] * 20 + [100.0] * 20
        picked = select_keyframes(sizes, fps=1000.0, rho=0.2, min_gap=8)
        assert all(b - a >= 8 for a, b in zip(picked, picked[1:], strict=False))

    def test_finds_the_step_it_is_supposed_to_find(self):
        sizes = [100.0] * 50 + [900.0] * 50
        picked = select_keyframes(sizes, fps=1000.0, rho=0.02, min_gap=1)
        assert 49 in picked

    def test_signed_and_absolute_disagree_on_a_downward_cut(self):
        """The one place the two rankings visibly differ, which is why the choice
        is a parameter and not a hardcoded `abs()`. A cut from a dense scene to a
        plain one is still a cut; under the paper's signed ranking it is ranked
        below a static frame.

        The fall must be *larger* than the rise for the two to separate at all:
        with a symmetric step the magnitudes tie and the earlier index wins, so
        both rankings return the rise. Measured here — medium scene (500) to busy
        (900) to plain (100): steepness +0.9988 at 29, -0.9997 at 59.
        """
        sizes = [500.0] * 30 + [900.0] * 30 + [100.0] * 30
        common = {"fps": 1000.0, "rho": 0.02, "min_gap": 1}
        assert select_keyframes(sizes, rank_by="signed", **common) == [29]
        assert select_keyframes(sizes, rank_by="absolute", **common) == [59]

    def test_is_deterministic(self):
        """Two runs on one video must give one keyframe set, or the bundle is not
        reproducible and two members' shards cannot be compared."""
        sizes = [100.0 + (i * 37 % 400) for i in range(300)]
        first = select_keyframes(sizes, fps=25.0)
        assert first == select_keyframes(sizes, fps=25.0)

    def test_min_k_saves_short_clips_from_an_empty_result(self):
        """`floor(0.02 * n)` is 0 for every n < 50, so without MIN_K a short clip
        or a short shot yields nothing at all."""
        assert select_keyframes([100.0] * 10, fps=1000.0) == [0]

    def test_edge_cases(self):
        assert select_keyframes([], fps=25.0) == []
        assert select_keyframes([100.0], fps=25.0) == [0]

    def test_returns_ascending_indices(self):
        picked = select_keyframes([100.0 + (i * 91 % 500) for i in range(400)], fps=25.0)
        assert picked == sorted(picked)
        assert len(set(picked)) == len(picked)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"fps": 0}, "fps"),
            ({"fps": 25.0, "rho": 0}, "rho"),
            ({"fps": 25.0, "rho": 1.5}, "rho"),
            ({"fps": 25.0, "min_gap": 0}, "min_gap"),
        ],
    )
    def test_rejects_impossible_parameters(self, kwargs, match):
        with pytest.raises(ContractError, match=match):
            select_keyframes([100.0] * 10, **kwargs)


class TestPtsForFrame:
    """The mapping N1 writes has to satisfy the contract its own reader enforces."""

    @pytest.mark.parametrize("fps", [25.0, 30.0, 29.97, 26.44, 23.976, 50.0, 59.94])
    def test_round_trips_through_the_contract(self, fps):
        """frame_idx == floor(pts_time * fps), on every frame of a long video.

        All four fps values batch 1 actually contains are covered, including the
        26.44 that appears in exactly one video (L24_V044).
        """
        for frame_idx in range(0, 50_000, 7):
            pts = pts_for_frame(frame_idx, fps)
            assert math.floor(pts * fps) == frame_idx

    def test_the_naive_formula_is_what_this_replaces(self):
        """Regression, and a correction to something the code used to claim.

        `round(frame_idx / fps, 4)` was called a tautology - true in real
        arithmetic, false in floating point. It breaks 34% of rows at 30fps and
        ~50% at 29.97, so the check catches a real defect rather than nothing.
        """
        naive_bad = sum(
            1 for i in range(20_000) if math.floor(round(i / 29.97, 4) * 29.97) != i
        )
        assert naive_bad > 9_000
        assert all(
            math.floor(pts_for_frame(i, 29.97) * 29.97) == i for i in range(20_000)
        )

    def test_stays_inside_the_frames_own_interval(self):
        """Smallest valid value, not just any: pts_time must not drift forward
        into the next frame's window."""
        fps = 30.0
        for frame_idx in (0, 1, 999, 31_499):
            pts = pts_for_frame(frame_idx, fps)
            assert frame_idx / fps <= pts < (frame_idx + 1) / fps
