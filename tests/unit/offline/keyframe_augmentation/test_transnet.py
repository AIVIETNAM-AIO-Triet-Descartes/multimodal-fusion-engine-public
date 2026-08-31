from __future__ import annotations

import numpy as np
import pytest
import torch

from mfe.offline.keyframe_augmentation import transnet as transnet_module
from mfe.offline.keyframe_augmentation.transnet import (
    _predict,
    detect_video,
    predictions_to_scenes,
)


def test_transnet_scene_endpoints_are_inclusive() -> None:
    predictions = np.array([0.0, 0.0, 0.9, 0.9, 0.0, 0.0], dtype=np.float32)

    assert predictions_to_scenes(predictions) == [(0, 2), (4, 5)]


def test_transnet_no_transition_covers_last_frame() -> None:
    predictions = np.zeros(4, dtype=np.float32)

    assert predictions_to_scenes(predictions) == [(0, 3)]


class _DeterministicWindowModel:
    def __call__(self, tensor):
        single = tensor[:, :, 0, 0, 0].float().unsqueeze(-1) / 255.0
        return single, single


@pytest.mark.parametrize("frame_count", [1, 49, 50, 51, 99, 100, 101, 173])
def test_batched_windows_match_the_original_one_window_execution(
    frame_count: int,
) -> None:
    frames = np.random.default_rng(2026 + frame_count).integers(
        0,
        256,
        size=(frame_count, 27, 48, 3),
        dtype=np.uint8,
    )
    model = _DeterministicWindowModel()

    original = _predict(model, frames, "cpu", window_batch_size=1)
    batched_two = _predict(model, frames, "cpu", window_batch_size=2)
    batched = _predict(model, frames, "cpu", window_batch_size=8)

    np.testing.assert_array_equal(batched_two, original)
    np.testing.assert_array_equal(batched, original)
    assert len(batched) == len(frames)
    assert predictions_to_scenes(batched) == predictions_to_scenes(original)


def test_window_batch_size_fails_closed() -> None:
    frames = np.zeros((50, 27, 48, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="window_batch_size"):
        _predict(_DeterministicWindowModel(), frames, "cpu", window_batch_size=0)


def test_model_cache_reuses_only_the_same_source_weight_and_device(monkeypatch) -> None:
    calls = []

    def fake_load(source, weights, device):
        value = (str(source), str(weights), device, len(calls))
        calls.append(value)
        return value

    monkeypatch.setattr(transnet_module, "_load_model", fake_load)
    transnet_module._load_model_cached.cache_clear()
    try:
        first = transnet_module._load_model_cached("source.py", "weights.pth", "cpu")
        repeated = transnet_module._load_model_cached("source.py", "weights.pth", "cpu")
        other_weight = transnet_module._load_model_cached(
            "source.py", "weights-v2.pth", "cpu"
        )
        other_device = transnet_module._load_model_cached(
            "source.py", "weights.pth", "cuda"
        )
    finally:
        transnet_module._load_model_cached.cache_clear()

    assert repeated is first
    assert other_weight != first
    assert other_device != first
    assert len(calls) == 3


def test_detect_video_reuses_model_for_two_videos_with_the_same_cache_key(
    monkeypatch, tmp_path
) -> None:
    frames = np.random.default_rng(2048).integers(
        0,
        256,
        size=(101, 27, 48, 3),
        dtype=np.uint8,
    )
    calls = []

    monkeypatch.setattr(
        transnet_module,
        "transnet_fingerprint",
        lambda *_args, **_kwargs: ("fingerprint", {"commit": "pinned"}),
    )
    monkeypatch.setattr(
        transnet_module,
        "_decode_video",
        lambda _path: (frames, 30.0),
    )

    def fake_load(_source, _weights, _device):
        calls.append(1)
        return _DeterministicWindowModel()

    monkeypatch.setattr(transnet_module, "_load_model", fake_load)
    transnet_module._load_model_cached.cache_clear()
    try:
        first_shots, first_report = detect_video(
            tmp_path / "first.mp4",
            repo=tmp_path / "repo",
            weights=tmp_path / "weights.pth",
            literal_fps=30.0,
            device="cpu",
        )
        second_shots, second_report = detect_video(
            tmp_path / "second.mp4",
            repo=tmp_path / "repo",
            weights=tmp_path / "weights.pth",
            literal_fps=30.0,
            device="cpu",
        )
    finally:
        transnet_module._load_model_cached.cache_clear()

    assert len(calls) == 1
    assert first_report["model_cache_hit"] is False
    assert second_report["model_cache_hit"] is True
    assert [shot.start_frame for shot in first_shots] == [shot.start_frame for shot in second_shots]


def test_cuda_oom_retries_only_the_permitted_batch_ladder(monkeypatch) -> None:
    frames = np.zeros((401, 27, 48, 3), dtype=np.uint8)
    seen_batch_sizes = []

    def passthrough_cuda(self, device, *args, **kwargs):
        if device == "cuda":
            return self
        return torch.Tensor.to(self, device, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", passthrough_cuda)

    class _OomUntilBatchOne:
        def __call__(self, tensor):
            seen_batch_sizes.append(tensor.shape[0])
            if tensor.shape[0] > 1:
                raise RuntimeError("CUDA out of memory in adversarial test")
            single = torch.zeros((tensor.shape[0], 100, 1))
            return single, single

    predictions = _predict(
        _OomUntilBatchOne(),
        frames,
        "cuda",
        window_batch_size=8,
    )

    assert len(predictions) == len(frames)
    assert seen_batch_sizes[:3] == [8, 4, 2]
    assert set(seen_batch_sizes[3:]) == {1}
