"""Pinned adapter for the official TransNetV2 PyTorch inference code.

The model source and weights stay outside this repository.  A run must point at
a checked-out ``soCzech/TransNetV2`` tree and records both the git commit and
weight hash in every shot proposal.
"""

from __future__ import annotations

import importlib.util
import subprocess
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from mfe.offline.keyframe_augmentation.common import sha256_file, sha256_json
from mfe.offline.keyframe_augmentation.models import ShotProposal


def _git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def transnet_fingerprint(repo: str | Path, weights: str | Path) -> tuple[str, dict[str, str]]:
    root = Path(repo).resolve()
    weight_path = Path(weights).resolve()
    source = root / "inference-pytorch" / "transnetv2_pytorch.py"
    if not source.is_file() or not weight_path.is_file():
        raise ValueError("TransNetV2 source or PyTorch weights are missing")
    detail = {
        "repository": "https://github.com/soCzech/TransNetV2",
        "commit": _git_commit(root),
        "source_sha256": sha256_file(source),
        "weights_sha256": sha256_file(weight_path),
    }
    return f"transnetv2:{sha256_json(detail)}", detail


def _load_model(source: Path, weights: Path, device: str) -> Any:
    import torch  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("mfe_transnetv2_pytorch", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import TransNetV2 model from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TransNetV2()
    state = torch.load(str(weights), map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.eval().to(device)


@lru_cache(maxsize=4)
def _load_model_cached(source: str, weights: str, device: str) -> Any:
    """Load one model per source/weight/device tuple for the process lifetime.

    Kaggle workers process many videos in one interpreter.  Reloading the same
    checkpoint for every video wastes GPU time and can fragment CUDA memory.
    The cache key contains both provenance paths and the device, so changing any
    of them creates a distinct model rather than silently reusing stale state.
    """

    return _load_model(Path(source), Path(weights), device)


def _decode_video(path: Path) -> tuple[Any, float]:
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(cv2.resize(frame, (48, 27)), cv2.COLOR_BGR2RGB)
            frames.append(rgb)
    finally:
        capture.release()
    if not frames or fps <= 0:
        raise ValueError(f"no decodable frames or invalid FPS: {path}")
    return np.asarray(frames, dtype=np.uint8), fps


def _input_batches(frames: Any):
    import numpy as np  # noqa: PLC0415

    remainder = len(frames) % 50
    padding_end = 25 + 50 - (remainder if remainder else 50)
    padded = np.concatenate(
        [[frames[0]] * 25, frames, [frames[-1]] * padding_end],
        axis=0,
    )
    for start in range(0, len(padded) - 99, 50):
        yield padded[start : start + 100]


def _predict_once(
    model: Any,
    frames: Any,
    device: str,
    *,
    window_batch_size: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> Any:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if window_batch_size < 1:
        raise ValueError("window_batch_size must be >= 1")
    windows = list(_input_batches(frames))
    total = len(windows)
    predictions: list[Any] = []
    with torch.no_grad():
        for start in range(0, total, window_batch_size):
            packed = np.stack(windows[start : start + window_batch_size], axis=0)
            tensor = torch.from_numpy(packed).to(device)
            single, _ = model(tensor)
            predictions.append(torch.sigmoid(single)[:, 25:75, 0].cpu().numpy())
            if progress is not None:
                progress(min(start + window_batch_size, total), total)
    return np.concatenate(predictions, axis=0).reshape(-1)[: len(frames)]


def _batch_fallback_sequence(window_batch_size: int) -> tuple[int, ...]:
    """Return the only permitted CUDA OOM retry ladder for this run."""

    ladder = (8, 4, 2, 1)
    if window_batch_size not in ladder:
        return (window_batch_size,)
    return ladder[ladder.index(window_batch_size) :]


def _is_cuda_oom(error: RuntimeError, device: str, torch: Any) -> bool:
    if not str(device).startswith("cuda"):
        return False
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _predict_with_batch_fallback(
    model: Any,
    frames: Any,
    device: str,
    *,
    window_batch_size: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Any, int]:
    """Retry a CUDA OOM only by walking 8 -> 4 -> 2 -> 1.

    A retry restarts the complete window sequence so a failed partial inference
    cannot change the ordering or endpoints.  Other exceptions, and OOM on
    batch 1, remain fatal rather than silently changing the detector contract.
    """

    import torch  # noqa: PLC0415

    if window_batch_size < 1:
        raise ValueError("window_batch_size must be >= 1")
    for batch_size in _batch_fallback_sequence(window_batch_size):
        progress_events: list[tuple[int, int]] = []

        def attempt_progress(done: int, total: int, events=progress_events) -> None:
            events.append((done, total))

        try:
            predictions = _predict_once(
                model,
                frames,
                device,
                window_batch_size=batch_size,
                progress=attempt_progress if progress is not None else None,
            )
        except RuntimeError as error:
            if not _is_cuda_oom(error, device, torch) or batch_size == 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        if progress is not None:
            for done, total in progress_events:
                progress(done, total)
        return predictions, batch_size
    raise AssertionError("unreachable CUDA batch fallback state")


def _predict(
    model: Any,
    frames: Any,
    device: str,
    *,
    window_batch_size: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> Any:
    """Predict with bounded batching while preserving the reference ordering."""

    predictions, _effective_batch_size = _predict_with_batch_fallback(
        model,
        frames,
        device,
        window_batch_size=window_batch_size,
        progress=progress,
    )
    return predictions


def warmup_transnet(
    *,
    repo: str | Path,
    weights: str | Path,
    device: str,
    window_batch_size: int = 8,
) -> dict[str, Any]:
    """Load the pinned model and run one tiny inference before downloading video.

    This is a runtime preflight, not a detector-quality measurement.  It catches
    incompatible weights, a CPU fallback, and CUDA/model errors early while also
    populating the process-wide model cache used by subsequent videos.
    """

    import numpy as np  # noqa: PLC0415

    root = Path(repo).resolve()
    weight_path = Path(weights).resolve()
    fingerprint, detail = transnet_fingerprint(root, weight_path)
    source = (root / "inference-pytorch" / "transnetv2_pytorch.py").resolve()
    before = _load_model_cached.cache_info()
    started = time.perf_counter()
    model = _load_model_cached(str(source), str(weight_path), device)
    model_load_seconds = time.perf_counter() - started
    after = _load_model_cached.cache_info()
    dummy = np.zeros((50, 27, 48, 3), dtype=np.uint8)
    infer_started = time.perf_counter()
    predictions, effective_batch_size = _predict_with_batch_fallback(
        model,
        dummy,
        device,
        window_batch_size=window_batch_size,
    )
    inference_seconds = time.perf_counter() - infer_started
    if len(predictions) != len(dummy):
        raise RuntimeError("TransNet warmup returned the wrong prediction length")
    return {
        "status": "PASS_RUNTIME_ONLY",
        "detector_fingerprint": fingerprint,
        "detector": detail,
        "device": device,
        "window_batch_size": effective_batch_size,
        "model_cache_hit": after.hits > before.hits,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
    }


def predictions_to_scenes(predictions: Any, threshold: float = 0.5) -> list[tuple[int, int]]:
    """Official endpoint convention: zero-based and inclusive on both sides."""
    binary = (predictions > threshold).astype("uint8")
    scenes: list[tuple[int, int]] = []
    previous = 0
    start = 0
    current = 0
    for index, current in enumerate(binary):
        if previous == 1 and current == 0:
            start = index
        if previous == 0 and current == 1 and index != 0:
            scenes.append((start, index))
        previous = int(current)
    if current == 0:
        scenes.append((start, len(binary) - 1))
    return scenes or [(0, len(binary) - 1)]


def detect_video(
    video_path: str | Path,
    *,
    repo: str | Path,
    weights: str | Path,
    literal_fps: float,
    threshold: float = 0.5,
    device: str = "cpu",
    window_batch_size: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[ShotProposal], dict[str, Any]]:
    total_started = time.perf_counter()
    video = Path(video_path).resolve()
    root = Path(repo).resolve()
    weight_path = Path(weights).resolve()
    fingerprint, detail = transnet_fingerprint(root, weight_path)
    source = root / "inference-pytorch" / "transnetv2_pytorch.py"
    decode_started = time.perf_counter()
    frames, decoder_fps = _decode_video(video)
    decode_seconds = time.perf_counter() - decode_started
    if abs(decoder_fps - literal_fps) > 0.05:
        raise ValueError(
            f"decoder FPS {decoder_fps} disagrees with locked literal FPS {literal_fps}"
        )
    before = _load_model_cached.cache_info()
    load_started = time.perf_counter()
    model = _load_model_cached(str(source.resolve()), str(weight_path), device)
    model_load_seconds = time.perf_counter() - load_started
    after = _load_model_cached.cache_info()
    inference_started = time.perf_counter()
    predictions, effective_batch_size = _predict_with_batch_fallback(
        model,
        frames,
        device,
        window_batch_size=window_batch_size,
        progress=progress,
    )
    inference_seconds = time.perf_counter() - inference_started
    scenes = predictions_to_scenes(predictions, threshold=threshold)
    proposals: list[ShotProposal] = []
    for index, (start, end) in enumerate(scenes):
        left = 1.0 if start == 0 else float(max(predictions[max(0, start - 2) : start + 2]))
        right = (
            1.0
            if end == len(frames) - 1
            else float(max(predictions[max(0, end - 2) : min(len(frames), end + 3)]))
        )
        proposals.append(
            ShotProposal(
                video_id=video.stem,
                shot_id=f"{video.stem}#s{index:04d}",
                start_frame=start,
                end_frame=end,
                fps=literal_fps,
                decoder_fps=decoder_fps,
                detector_fingerprint=fingerprint,
                confidence=min(left, right),
            )
        )
    report = {
        "video": str(video),
        "frames": len(frames),
        "literal_fps": literal_fps,
        "decoder_fps": decoder_fps,
        "frame_identity": "zero-based integer decoder counter",
        "threshold": threshold,
        "scenes": len(scenes),
        "device": device,
        "window_batch_size": effective_batch_size,
        "model_cache_hit": after.hits > before.hits,
        "timings_seconds": {
            "decode": decode_seconds,
            "model_load": model_load_seconds,
            "inference": inference_seconds,
            "total": time.perf_counter() - total_started,
        },
        "detector": detail,
        "endpoint_contract": "zero-based inclusive",
    }
    report["run_hash"] = sha256_json(report)
    return proposals, report
