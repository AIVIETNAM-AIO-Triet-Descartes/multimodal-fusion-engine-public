"""Read technical metadata from a video file — step [1] of docs/02 §2.

This module is where `fps` enters the system. Everything downstream — every
timestamp, every submission — is derived from what it returns. docs/00 §7:
"Không hardcode FPS. Đọc từ metadata từng video." docs/context/buoi-1.md §7
records that BTC may ship videos with differing FPS and resolution.
"""

from __future__ import annotations

from mfe.contracts.records import VideoRow


def probe_video(source_path: str, video_id: str) -> VideoRow:
    """Read fps / duration / resolution / frame count from a video file.

    Owner: Quốc Đạt. TODO(Quốc Đạt): implement (PyAV or ffprobe).

    Notes that will bite otherwise:

    - Report fps as the exact rational the container declares (30000/1001 =
      29.97), not a rounded 30.0. A 0.1% fps error is ~36 frames of drift over
      10 minutes — far wider than a correct submission window.
    - Some containers lie about, or omit, total_frames. If it must be estimated,
      say so in the log; `total_frames` is what the validator uses to bound-check
      a frame (docs/05 §1.1), so a wrong value blocks good submits or lets bad
      ones through.
    """
    raise NotImplementedError("TODO(Quốc Đạt): probe container metadata -> VideoRow")


def derive_video_id(source_path: str) -> str:
    """Derive a `video_id` from a filename, e.g. data/raw/L21_V001.mp4 -> L21_V001.

    Owner: Quốc Đạt. TODO(Quốc Đạt): implement.

    Must round-trip: `VideoRow.source_path` is what reopens the video for
    verification (docs/05 §1.1). Validate the result with
    `mfe.contracts.ids.validate_video_id` — a filename with a space or a '#' in
    it would silently break every keyframe_id built from it.
    """
    raise NotImplementedError("TODO(Quốc Đạt): filename -> video_id")
