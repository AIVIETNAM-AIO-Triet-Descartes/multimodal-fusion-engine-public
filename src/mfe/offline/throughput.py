"""Throughput measurement — docs/02 §5. Writes eval/throughput.json for the Lead.

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

from typing import Any


class ThroughputRecorder:
    """Accumulates timings and sizes across an ingest run.

    Owner: TV1. TODO(TV1): implement.

    Report per hour of *video*, not per video — videos differ in length, so a
    per-video mean does not extrapolate. Keep stage timings separate
    (extraction vs CLIP vs detector): a single total cannot tell the Lead
    whether the answer at scale is more GPU or more CPU.
    """

    def start_stage(self, stage: str) -> None:
        raise NotImplementedError("TODO(TV1)")

    def end_stage(self, stage: str) -> None:
        raise NotImplementedError("TODO(TV1)")

    def record_video(self, duration_ms: int, keyframe_count: int) -> None:
        raise NotImplementedError("TODO(TV1)")

    def summary(self) -> dict[str, Any]:
        raise NotImplementedError("TODO(TV1): the 4 metrics of docs/02 §5")

    def write(self, path: str) -> None:
        """Write the summary to `paths.throughput` (default eval/throughput.json)."""
        raise NotImplementedError("TODO(TV1): dump summary to JSON")
