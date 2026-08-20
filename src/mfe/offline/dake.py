"""DAKE — keyframe extraction with no model and no training.

    docs/triet/ke-hoach-offline-ingest.md §3
    U-CESE (đội Nomial, AIC HCMC 2025), arXiv:2605.23274 §4.1, Algorithm 1

The signal is **JPEG file size per frame**. A frame with more motion, texture or
light change compresses worse, so it is bigger; the *rate of change* of that size
along the timeline marks where the content turns over. That is the whole idea, and
it is why the method needs no GPU: the encoder has already done the measuring.

Everything in this module is a pure function over a sequence of numbers, so the
tests build their own size sequences and never open a video
(`tests/unit/offline/test_dake.py`). Decoding lives at the bottom, behind a lazy
import, because it is the only part that needs OpenCV — and it is the part the
notebook runs on Kaggle, not the part we verify on this machine.

**What the paper does not settle.** Its whole DAKE-vs-AutoShot comparison runs on
*one* video (`K01_V001`), and it measures agreement with shot boundaries rather
than usefulness for retrieval. So `RHO = 0.02` is a starting point carried over
from a dataset that happens to be ours (§3.1), not a tuned value. `frame_sizes`
exists as a separate, cacheable step precisely so sweeping `RHO` later costs
minutes instead of a day of re-decoding (§3.4).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Iterator, Sequence
from typing import Literal

from mfe.contracts.errors import ContractError, ErrorCode

logger = logging.getLogger(__name__)

#: Fraction of source frames to keep. **Back to the paper's 0.02**, decided by
#: the Lead on 19-08-2026 (plan P1, `notebooks/dake/hieu-chinh-tham-so-dake.md` §7).
#:
#: The 18-08 value was 0.015, cut to bring the keyframe count down. Measured on
#: 20 videos / 424.818 frames, that was cutting the wrong knob: `rho` is the
#: *content-aware* half of DAKE, while the density floor below is the
#: content-blind half, and the floor was producing **52%** of all output. Lowering
#: `rho` therefore removed the part that is DAKE and kept the part that is not.
#:
#: Raising `rho` back to the paper value while widening the floor gives **fewer**
#: keyframes overall and **more** chosen ones: 8.692 vs 13.205 total, of which
#: 8.486 (98%) come from selection rather than 6.361 (48%). See §4 of that doc.
RHO_PAPER = 0.02
RHO = 0.02

#: How many frames ahead `S(i, j)` averages over. Paper value.
LOOKAHEAD = 3

#: Minimum spacing between two selected keyframes, in **seconds**. **Not in the
#: paper** — added because the top-k of a noisy steepness curve clumps: a single
#: cut produces several adjacent high-steepness frames that all describe the same
#: transition, and keeping them spends the keyframe budget on one moment.
#:
#: Seconds, not frames, since 19-08-2026 (decision 5 in that doc's §7). It used
#: to be `MIN_GAP = 5` *frames*, which is 0,20s at 25fps and 0,17s at 30fps — the
#: same setting meaning two different things, and batch 1 runs 781 videos at
#: 25fps against 92 at three other rates. `max_gap_seconds` was already in
#: seconds, so the pair also disagreed with each other.
#:
#: The value matters more than the unit did. At 5 frames, **36% of keyframe gaps
#: were under half a second** — near-duplicates that no content decision asked
#: for, produced because the algorithm was *allowed* to take two frames 5 apart.
#: At 1,0s that falls to 1–3%, which is what let the Lead defer the whole
#: dedup-filter question (decision 4) rather than build a filter to delete frames
#: a parameter could have stopped from existing.
MIN_GAP_SECONDS = 1.0

#: Floor on the keyframe count. **Not in the paper** — `floor(RHO * n)` is 0 for
#: any n < 50, so a short video or a short shot would otherwise yield nothing.
MIN_K = 1

#: The density guarantee: at least one keyframe in every window this many
#: **seconds** wide. **Paper says 2.0s; we use 3.0s.**
#:
#: The paper's wording, §5.1 page 10: it adopts rho=0.02 and additionally
#: enforces at least one keyframe within every `2 x fps`-frame window, "for a
#: near-perfect recall" — where recall means agreement with AutoShot, measured on
#: **one** video. See §3.3.
#:
#: This is the *content-blind* half of the algorithm: it inserts frames purely to
#: bound the gap, which is uniform sampling wearing DAKE's name. Measured on 20
#: videos it was producing **52%** of all keyframes — so the tighter it is, the
#: less of the output is actually DAKE.
#:
#: **8.0s, widened from 3.0s on 19-08-2026.** The 3.0s value came from an
#: argument that does not hold: "tighter than BTC's median gap of 3.36s". That
#: compares an *upper bound* against a *median* — BTC having a median of 3.36s
#: means half their gaps are longer, so a hard 3.0s cap is **denser than BTC
#: across their entire upper half**, cementing keyframes every 3 seconds exactly
#: where BTC leaves the footage alone. To be comparable to BTC the bound belongs
#: near their **maximum** (210 frames, 7.0-8.4s depending on fps), not their median.
#:
#: At 8.0s the floor contributes 2% of output instead of 52%, while the *median*
#: realised gap (1.44s) stays well under BTC's — the bound now fires only on
#: static stretches, which is what a floor is for.
#:
#: Turning it off entirely is **not** an option, and that is measured, not
#: assumed: without it the worst gap reaches **324 seconds** (§1.3 of that doc).
MAX_GAP_SECONDS_PAPER = 2.0
MAX_GAP_SECONDS = 8.0

#: How the top-k is ranked. `"signed"` is the paper: `S(i, j)` keeps the sign of
#: the size change, so frames whose size *rises* sharply rank high and frames
#: whose size *falls* sharply rank last. `"absolute"` treats a sharp drop as
#: equally interesting.
#:
#: The paper is followed by default, but the choice is exposed because it is a
#: real question its n=1 evidence cannot answer: a cut from a dense scene to a
#: plain one is just as much a cut, and under `"signed"` it is ranked *below* a
#: static frame. The existing DAKE notebook silently used `abs()` (see
#: notebooks/README.md), so the two are already in play; one A/B on the sample
#: settles it, and until then the paper wins by default rather than by argument.
RankBy = Literal["signed", "absolute"]


def steepness(
    sizes: Sequence[float], lookahead: int = LOOKAHEAD
) -> list[float]:
    """Per-frame steepness for frames ``0 .. n-2``. Returns ``n - 1`` values.

    For each frame `i`, the mean of `S(i, j)` over the next `lookahead` frames:

        ŝ      = 100 * (s_j - s_i) / s_max
        S(i,j) = ŝ / sqrt((j - i)^2 + ŝ^2)

    which is the **sine of the slope angle** in (frame index, normalised size)
    space, hence always in `[-1, 1]`. Two consequences worth naming: dividing by
    `s_max` makes the value independent of resolution and encoder quality, so it
    is comparable across videos; and the bound means no single frame pair can
    dominate an average the way a raw slope could.

    `s_max` is the largest size **within this sequence** — per video, never per
    dataset (§3.4). Passing a whole dataset's sizes would let one busy video
    flatten every other video's curve.

    The last frame gets no value at all rather than a zero: it has no lookahead,
    and zero is mid-range on a signed scale, so a default of 0.0 would rank it
    above every genuinely-darkening frame. It can still be picked by the density
    rule in :func:`select_keyframes`, which is the correct way for it to be
    chosen.
    """
    if lookahead < 1:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT, f"lookahead must be >= 1, got {lookahead}"
        )
    n = len(sizes)
    if n < 2:
        return []

    s_max = max(sizes)
    if s_max <= 0:
        # Degenerate but reachable: a fully black clip encodes to a near-constant
        # tiny size, and a caller could hand us zeros. Flat curve, no opinion.
        return [0.0] * (n - 1)

    scale = 100.0 / s_max
    out: list[float] = []
    for i in range(n - 1):
        last = min(n - 1, i + lookahead)
        total = 0.0
        count = 0
        for j in range(i + 1, last + 1):
            delta = (sizes[j] - sizes[i]) * scale
            total += delta / math.sqrt((j - i) ** 2 + delta**2)
            count += 1
        out.append(total / count)
    return out


def _greedy_top_k(scores: Sequence[float], k: int, min_gap: int, rank_by: RankBy) -> list[int]:
    """Highest-ranked `k` indices, no two closer than `min_gap`.

    Ties break toward the earlier frame (`idx` in the sort key), so the selection
    is deterministic — two runs on the same video must produce the same keyframe
    set or the bundle is not reproducible.
    """
    if rank_by == "signed":
        order = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
    else:
        order = sorted(range(len(scores)), key=lambda idx: (-abs(scores[idx]), idx))

    chosen: list[int] = []
    for idx in order:
        if len(chosen) >= k:
            break
        if all(abs(idx - taken) >= min_gap for taken in chosen):
            chosen.append(idx)
    return sorted(chosen)


def _fill_density(
    chosen: list[int], scores: Sequence[float], n: int, window: int, rank_by: RankBy
) -> list[int]:
    """Add frames until every `window`-wide window of frames holds a keyframe.

    The guarantee, stated exactly: with keyframes sorted, `chosen[0] <= window-1`,
    `chosen[-1] >= n-window`, and `chosen[i+1] - chosen[i] <= window`.

    Candidates for a hole are restricted to the first `window` frames *after* its
    left edge, and the best-scoring one there wins. The restriction is what makes
    this terminate: every insertion lands strictly inside the hole and strictly
    closes it from the left, so the number of violating windows falls each pass.
    Picking the globally best frame in the hole would not — it could land right
    next to the left edge and leave the same hole open forever.
    """
    if n == 0:
        return chosen
    if window < 1:
        window = 1

    def best_in(lo: int, hi: int) -> int | None:
        """Highest-ranked index in the inclusive range, or None if empty."""
        lo, hi = max(lo, 0), min(hi, len(scores) - 1)
        if lo > hi:
            # Past the end of `scores` (only the final frame lives there). Take
            # the frame itself: it has no steepness but it does close the hole.
            return min(max(lo, 0), n - 1) if lo <= n - 1 else None
        score_of = (
            (lambda idx: scores[idx]) if rank_by == "signed" else (lambda idx: abs(scores[idx]))
        )
        # Ties break toward the RIGHT, unlike the greedy pass which breaks left.
        # The asymmetry is deliberate and it is load-bearing: on a static stretch
        # every candidate scores the same, and a left-breaking tie advances the
        # frontier by one frame per pass, so the rule ends up keeping *every*
        # frame of the stretch instead of every `window`-th one. Rightmost is also
        # simply the better answer when nothing distinguishes the candidates — it
        # closes the most gap for one keyframe.
        return max(range(lo, hi + 1), key=lambda idx: (score_of(idx), idx))

    picked = set(chosen)
    guard = 0
    while True:
        guard += 1
        if guard > n + 2:  # pragma: no cover - the bound above makes this dead
            raise AssertionError("_fill_density did not converge")
        current = sorted(picked)
        hole: tuple[int, int] | None = None  # (left edge, exclusive right bound)
        if not current:
            hole = (-1, n)
        elif current[0] > window - 1:
            hole = (-1, current[0])
        else:
            for left, right in zip(current, current[1:], strict=False):
                if right - left > window:
                    hole = (left, right)
                    break
            if hole is None and current[-1] < n - window:
                hole = (current[-1], n)
        if hole is None:
            return current

        left, right = hole
        candidate = best_in(left + 1, min(right - 1, left + window))
        if candidate is None or candidate in picked:  # pragma: no cover - defensive
            return current
        picked.add(candidate)


def select_keyframes(
    sizes: Sequence[float],
    *,
    fps: float,
    rho: float = RHO,
    lookahead: int = LOOKAHEAD,
    min_gap_seconds: float = MIN_GAP_SECONDS,
    min_k: int = MIN_K,
    max_gap_seconds: float = MAX_GAP_SECONDS,
    rank_by: RankBy = "signed",
) -> list[int]:
    """Algorithm 1: pick keyframe indices from a video's per-frame JPEG sizes.

    Returns **source-video frame indices**, ascending — 0-based, the same numbers
    the decode loop counts. They are the `frame_idx` of the contract, not
    ordinals; `mfe.contracts.ids` explains at length why those must not be
    confused, and this function returning the former is where that starts.

    Order of operations, and it matters: top-k first, density second. The density
    rule *adds* frames the budget missed, so the final count is
    `>= max(min_k, floor(rho * n))` and can exceed it on a long static stretch.
    That is the intended behaviour — a bounded gap is a guarantee, and a
    guarantee that yields to a budget is not one.

    **Both gap parameters are in seconds and both convert with `floor`.** They are
    the two ends of one spacing rule — no keyframes closer than `min_gap_seconds`,
    none further apart than `max_gap_seconds` — so having them in different units
    made the pair unreadable, and made one setting mean different things on the
    four frame rates present in batch 1.

    A flat 40-frame clip at 5fps: the budget picks 4 frames, then the density
    rule spaces the rest at the 10-frame (2-second) limit. Both gaps are passed
    explicitly — a doctest that reads the module defaults tests the constants
    rather than the function, and rewrites itself every time the Lead retunes.

    >>> select_keyframes([100.0] * 40, fps=5.0, rho=0.1,
    ...                  min_gap_seconds=0.2, max_gap_seconds=2.0)
    [0, 1, 2, 3, 13, 23, 33]
    """
    if fps <= 0:
        raise ContractError(ErrorCode.INVALID_ARGUMENT, f"fps must be > 0, got {fps}")
    if not 0 < rho <= 1:
        raise ContractError(ErrorCode.INVALID_ARGUMENT, f"rho must be in (0, 1], got {rho}")
    if min_gap_seconds <= 0:
        raise ContractError(
            ErrorCode.INVALID_ARGUMENT,
            f"min_gap_seconds must be > 0, got {min_gap_seconds}",
        )

    n = len(sizes)
    if n == 0:
        return []
    if n == 1:
        return [0]

    scores = steepness(sizes, lookahead)
    k = max(min_k, math.floor(rho * n))
    k = min(k, len(scores))
    # `max(1, ...)`: a min_gap that floors to 0 frames would let the greedy pass
    # take the same index twice over. One frame is the smallest spacing that
    # still means "not the same frame".
    min_gap_frames = max(1, math.floor(min_gap_seconds * fps))
    chosen = _greedy_top_k(scores, k, min_gap_frames, rank_by)
    # floor, not round: the guarantee is stated in SECONDS, so the window has to
    # be the largest whole number of frames that still fits inside it. Rounding
    # overshoots on any fps that is not a whole number — 60 frames at 29.97 is
    # 2.002s, 53 at 26.44 is 2.005s — and both of those rates are in batch 1. A
    # bound that the bound-enforcing code itself exceeds is not a bound, and this
    # one is the reason DAKE was picked over BTC's keyframes at all (§3.1).
    window = max(1, math.floor(max_gap_seconds * fps))
    return _fill_density(chosen, scores, n, window, rank_by)


# ---------------------------------------------------------------------------
# Decoding. The only part that touches a video, and the only part with a heavy
# dependency — hence the lazy import and the separation from everything above.
# ---------------------------------------------------------------------------


def jpeg_sizes(frames: Iterable, quality: int = 90) -> Iterator[int]:
    """Encoded JPEG byte length of each BGR frame. Lazily imports OpenCV.

    `quality` must be held constant across a whole run: the size *series* is the
    measurement, so re-encoding half a video at a different quality shifts the
    curve without shifting the content, and `s_max` normalisation does not undo
    that (it rescales, it does not re-shape).
    """
    import cv2  # noqa: PLC0415 - deliberately lazy; see module docstring

    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    for frame in frames:
        ok, buffer = cv2.imencode(".jpg", frame, params)
        if not ok:  # pragma: no cover - encoder failure is not reproducible here
            raise RuntimeError("cv2.imencode failed on a decoded frame")
        yield int(buffer.size)


#: Decode backend for pass 1. `"auto"` probes for NVDEC and silently falls back.
DecodeBackend = Literal["auto", "cpu", "nvdec"]


def cuda_decode_available() -> bool:
    """Can this machine's ffmpeg decode on the GPU?

    Probed rather than assumed, and the distinction matters: **selecting a GPU
    accelerator on Kaggle does not by itself make N1 faster.** PyPI's
    `opencv-python` is built without CUDA, so the OpenCV path runs at CPU speed on
    a GPU session while spending GPU quota. Only a decoder that actually uses
    NVDEC changes anything, and whether one is present is a property of the image,
    not of the accelerator setting.
    """
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return False
    return "cuda" in out.stdout.lower()


def decode_frames_nvdec(video_path: str, max_side: int | None = 320) -> Iterator:
    """Decode on the GPU via ffmpeg/NVDEC, yielding BGR frames on the host.

    Scaling happens on the GPU too (`scale_cuda`), so what crosses the PCIe bus is
    the small frame, not the full-resolution one — at 320px that is ~170 KB per
    frame instead of ~2.7 MB.

    Still strictly sequential, and still every frame in order, so the loop counter
    remains `frame_idx` exactly as in the CPU path. Nothing about the contract
    changes; only who does the decoding.

    **Not verified on real hardware.** There is no GPU on the machine this was
    written on, so this path has never decoded a real frame. `decode_frames`
    treats it as an optimisation and falls back, which is the only responsible
    way to ship something untested on the expensive path.
    """
    import subprocess  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    width, height = probe_size(video_path)
    if max_side is not None and max(width, height) > max_side:
        scale = max_side / max(width, height)
        width = (int(width * scale) // 2) * 2
        height = (int(height * scale) // 2) * 2
    filters = f"scale_cuda={width}:{height},hwdownload,format=nv12,format=bgr24"

    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", video_path,
        "-vf", filters, "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    frame_bytes = width * height * 3
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            buffer = process.stdout.read(frame_bytes)
            if len(buffer) < frame_bytes:
                break
            yield np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
    finally:
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", "replace")
        process.stderr.close()
        code = process.wait()
        if code not in (0, None) and stderr.strip():
            raise RuntimeError(f"ffmpeg NVDEC that bai ({code}): {stderr.strip()[:300]}")


def probe_size(video_path: str) -> tuple[int, int]:
    """`(width, height)`, read with OpenCV — cheap, and no decode."""
    import cv2  # noqa: PLC0415

    capture = cv2.VideoCapture(video_path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"khong doc duoc kich thuoc tu {video_path}")
    return width, height


def decode_frames(
    video_path: str, max_side: int | None = 320, backend: DecodeBackend = "auto"
) -> Iterator:
    """Yield every frame of a video **sequentially**, in decode order.

    Sequential and never seeking, which is a correctness requirement and not a
    performance choice: `cap.set(CAP_PROP_POS_FRAMES)` on long-GOP H.264 lands on
    the preceding keyframe, so a seek-per-frame loop silently returns the same
    picture several times and both the size curve and the frame numbering come
    out wrong. The loop counter *is* `frame_idx`, exactly (§3.4).

    `max_side` downscales before measuring. That is safe for this signal — what
    DAKE reads is the *shape* of the size curve, and `s_max` normalisation removes
    the scale — and it is what makes 20k frames per video affordable. Set `None`
    to measure at full resolution.

    `backend="auto"` uses NVDEC when ffmpeg reports it and OpenCV otherwise. The
    fallback is deliberate: an untested GPU path must not be able to fail a run
    that a tested CPU path would have completed.
    """
    if backend == "auto":
        backend = "nvdec" if cuda_decode_available() else "cpu"
    if backend == "nvdec":
        yield from decode_frames_nvdec(video_path, max_side)
        return

    import cv2  # noqa: PLC0415

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"khong mo duoc video: {video_path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            if max_side is not None:
                height, width = frame.shape[:2]
                longest = max(height, width)
                if longest > max_side:
                    scale = max_side / longest
                    frame = cv2.resize(
                        frame,
                        (max(1, int(width * scale)), max(1, int(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
            yield frame
    finally:
        capture.release()


#: Decimal places `map-keyframes` writes `pts_time` with, matching BTC's CSVs.
PTS_DECIMALS = 4


def pts_for_frame(frame_idx: int, fps: float, decimals: int = PTS_DECIMALS) -> float:
    """A `pts_time` that actually satisfies ``floor(pts_time * fps) == frame_idx``.

    The obvious `round(frame_idx / fps, 4)` **does not**, and not rarely:
    measured over 20,000 frames it breaks 5.7% of rows at 25fps, **34% at 30fps
    and 50% at 29.97fps**. Rounding to four places can land a hair below the
    frame boundary, and `floor` then returns `frame_idx - 1`.

    That matters more here than a rounding bug usually would. The contract in
    `mfe.contracts.ids` is `frame_idx = floor(pts_time * fps)`; `read_keyframes`
    enforces it on every row, so a DAKE mapping written the naive way is rejected
    by our own reader — after the decode, which is the expensive part.

    It also settles something stated wrongly elsewhere: writing `pts_time` from
    `frame_idx / fps` was called a tautology that made the check meaningless. In
    real arithmetic it would be. In floating point it is not, and the check earns
    its keep by catching exactly this.

    The fix is to pick the **smallest** value at this precision that lands inside
    the frame's own interval `[frame_idx/fps, (frame_idx+1)/fps)`.
    """
    if fps <= 0:
        raise ContractError(ErrorCode.INVALID_ARGUMENT, f"fps must be > 0, got {fps}")
    scale = 10**decimals
    step = 1.0 / scale
    candidate = math.ceil(frame_idx / fps * scale) / scale
    upper = (frame_idx + 1) / fps
    # At most a couple of nudges: the interval is 1/fps wide, far wider than 1e-4
    # for any real frame rate.
    for _ in range(4):
        if candidate >= upper:
            break
        if math.floor(candidate * fps) == frame_idx:
            return round(candidate, decimals)
        candidate = round(candidate + step, decimals)
    raise ContractError(
        ErrorCode.INVALID_ARGUMENT,
        f"khong tim duoc pts_time {decimals} chu so thap phan cho frame {frame_idx} "
        f"o fps {fps}. Khoang cua frame hep hon do phan giai cua pts_time.",
    )


#: Decimal places batch 1 states fps with. All four values measured in
#: `docs/triet/dataset_report.md` §A.3 — 25.0, 30.0, 29.97, 26.44 — fit exactly.
FPS_DECIMALS = 2


def video_fps(video_path: str, decimals: int | None = FPS_DECIMALS) -> tuple[float, float]:
    """Return ``(fps_used, fps_raw)``. Read per video, never assumed.

    Batch 1 carries **four** different frame rates: 25.0 (781 videos), 30.0 (61),
    29.97 (30) and 26.44 in exactly one video (`L24_V044`), measured in
    `docs/triet/dataset_report.md` §A.3. Hardcoding 25 or 30 would put at least
    92 videos' `frame_idx` off.

    **Why it is rounded, and why that is not a liberty.** The contract wants the
    *literal value the dataset states* — `frame_idx = floor(pts_time * fps)`
    matches BTC on 100% of rows at 29.97 and on 41% at 30000/1001, because BTC
    themselves computed with the truncated number (`mfe.contracts.ids`). OpenCV
    reports what the container holds, which for a 30000/1001 stream is
    29.969999…, not 29.97. Rounding to two places lands back on the number the
    dataset states, for all four values it actually contains.

    Both numbers come back so the caller can record them. If they disagree by
    more than a rounding hair, that is a video whose rate is not one of the four
    and it belongs in the marker, not swept under a round().
    """
    import cv2  # noqa: PLC0415 - deliberately lazy

    capture = cv2.VideoCapture(video_path)
    try:
        raw = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if raw <= 0:
        raise RuntimeError(f"khong doc duoc fps tu {video_path}")
    if decimals is None:
        return raw, raw
    used = round(raw, decimals)
    if abs(used - raw) > 0.02:
        logger.warning(
            "%s: fps %.6f lam tron thanh %.2f — lech %.4f. Video nay khong thuoc "
            "4 gia tri fps cua batch 1; kiem lai truoc khi tin mapping.",
            video_path, raw, used, abs(used - raw),
        )
    return used, raw


def decode_selected(video_path: str, indices: Iterable[int]) -> Iterator[tuple[int, object]]:
    """Yield ``(frame_index, frame)`` for just the wanted frames, in one pass.

    Still strictly sequential — seeking a long-GOP stream lands on the previous
    keyframe and would hand back the wrong picture. But `grab()` advances without
    reconstructing the output image, and only the ~2% of frames DAKE selected get
    `retrieve()`d. N1 has to walk the video twice (sizes must all be known before
    any frame can be chosen), and this is what keeps the second walk from costing
    as much as the first.
    """
    import cv2  # noqa: PLC0415 - deliberately lazy

    wanted = set(indices)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"khong mo duoc video: {video_path}")
    try:
        index = 0
        remaining = len(wanted)
        while remaining:
            if not capture.grab():
                return
            if index in wanted:
                ok, frame = capture.retrieve()
                if not ok:  # pragma: no cover - decoder failure, not reproducible
                    raise RuntimeError(f"retrieve() that bai o frame {index}")
                remaining -= 1
                yield index, frame
            index += 1
    finally:
        capture.release()
