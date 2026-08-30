"""Pinned adapter for the official TransNetV2 PyTorch inference code.

The model source and weights stay outside this repository.  A run must point at
a checked-out ``soCzech/TransNetV2`` tree and records both the git commit and
weight hash in every shot proposal.
"""

from __future__ import annotations

import importlib.util
import subprocess
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


def _predict(model: Any, frames: Any, device: str) -> Any:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    predictions = []
    with torch.no_grad():
        for batch in _input_batches(frames):
            tensor = torch.from_numpy(batch[None, ...]).to(device)
            single, _ = model(tensor)
            predictions.append(torch.sigmoid(single)[0, 25:75, 0].cpu().numpy())
    return np.concatenate(predictions)[: len(frames)]


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
) -> tuple[list[ShotProposal], dict[str, Any]]:
    video = Path(video_path).resolve()
    root = Path(repo).resolve()
    weight_path = Path(weights).resolve()
    fingerprint, detail = transnet_fingerprint(root, weight_path)
    source = root / "inference-pytorch" / "transnetv2_pytorch.py"
    frames, decoder_fps = _decode_video(video)
    if abs(decoder_fps - literal_fps) > 0.05:
        raise ValueError(
            f"decoder FPS {decoder_fps} disagrees with locked literal FPS {literal_fps}"
        )
    model = _load_model(source, weight_path, device)
    predictions = _predict(model, frames, device)
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
        "detector": detail,
        "endpoint_contract": "zero-based inclusive",
    }
    report["run_hash"] = sha256_json(report)
    return proposals, report
