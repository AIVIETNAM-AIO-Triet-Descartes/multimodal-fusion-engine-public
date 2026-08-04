"""Throughput measurement — docs/02 §5. Writes eval/throughput.json for the Vỹ Triết.

Why this exists in Sprint 1, on a tiny sample: "Với data lớn (hàng trăm giờ),
thứ hay vỡ khi scale không phải logic mà là thông lượng và lưu trữ" (docs/02 §5).
The numbers are useless after the fact — they have to be recorded while the
small run happens, so the full-dataset estimate exists before anyone commits to
it.

The four metrics docs/02 §5 asks for:

    seconds / hour of video     -> how long a full extraction takes
    seconds / 1000 keyframes    -> GPU demand for CLIP
    MB / hour of video (vector) -> RAM/disk for the Faiss index
    MB / hour of video (asset)  -> Modal volume size
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from mfe.contracts.errors import ContractError, ErrorCode

_MS_PER_HOUR = 3_600_000.0
_BYTES_PER_MB = 1024.0 * 1024.0


class ThroughputRecorder:
    """Accumulates timings and sizes across an ingest run.

    Owner: Quốc Đạt.

    Reports per hour of *video*, not per video — videos differ in length, so a
    per-video mean does not extrapolate. Stage timings stay separate (read vs
    features vs write): a single total cannot tell the Vỹ Triết whether the
    answer at scale is more GPU or more CPU.

    Uses `time.perf_counter`, which is monotonic — a wall-clock adjustment
    mid-run would otherwise produce a negative stage duration.
    """

    def __init__(self) -> None:
        self._stage_totals_s: dict[str, float] = {}
        self._stage_started_at: dict[str, float] = {}
        self._video_count = 0
        self._keyframe_count = 0
        self._video_ms = 0
        self._vector_bytes = 0
        self._asset_bytes = 0
        self._run_started_at = time.perf_counter()

    def start_stage(self, stage: str) -> None:
        if stage in self._stage_started_at:
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT, f"stage {stage!r} is already running"
            )
        self._stage_started_at[stage] = time.perf_counter()

    def end_stage(self, stage: str) -> None:
        try:
            started_at = self._stage_started_at.pop(stage)
        except KeyError:
            raise ContractError(
                ErrorCode.INVALID_ARGUMENT, f"stage {stage!r} was never started"
            ) from None
        self._stage_totals_s[stage] = self._stage_totals_s.get(stage, 0.0) + (
            time.perf_counter() - started_at
        )

    def record_video(self, duration_ms: int, keyframe_count: int) -> None:
        self._video_count += 1
        self._video_ms += duration_ms
        self._keyframe_count += keyframe_count

    def record_bytes(self, vector_bytes: int = 0, asset_bytes: int = 0) -> None:
        """Record store sizes. Vector bytes are `n * embed_dim * 4` (float32);
        asset bytes come from `AssetStore.total_bytes()`."""
        self._vector_bytes += vector_bytes
        self._asset_bytes += asset_bytes

    def summary(self) -> dict[str, Any]:
        """The 4 metrics of docs/02 §5, plus the raw counts behind them.

        The raw counts are included on purpose: a ratio computed on 5 videos is
        an estimate, and whoever reads it later needs to see how thin the
        sample was before extrapolating to hundreds of hours.
        """
        elapsed_s = time.perf_counter() - self._run_started_at
        video_hours = self._video_ms / _MS_PER_HOUR
        keyframe_thousands = self._keyframe_count / 1000.0

        def per_hour(value: float) -> float | None:
            return round(value / video_hours, 3) if video_hours > 0 else None

        return {
            "sample": {
                "videos": self._video_count,
                "keyframes": self._keyframe_count,
                "video_hours": round(video_hours, 4),
                "wall_clock_s": round(elapsed_s, 3),
            },
            "stages_s": {
                stage: round(total, 3)
                for stage, total in sorted(self._stage_totals_s.items())
            },
            "seconds_per_video_hour": per_hour(elapsed_s),
            "seconds_per_1000_keyframes": (
                round(elapsed_s / keyframe_thousands, 3) if keyframe_thousands > 0 else None
            ),
            "vector_mb_per_video_hour": per_hour(self._vector_bytes / _BYTES_PER_MB),
            "asset_mb_per_video_hour": per_hour(self._asset_bytes / _BYTES_PER_MB),
        }

    def write(self, path: str) -> None:
        """Write the summary to `paths.throughput` (default eval/throughput.json)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
