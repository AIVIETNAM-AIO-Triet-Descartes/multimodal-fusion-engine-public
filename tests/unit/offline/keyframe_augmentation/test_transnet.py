from __future__ import annotations

import numpy as np

from mfe.offline.keyframe_augmentation.transnet import predictions_to_scenes


def test_transnet_scene_endpoints_are_inclusive() -> None:
    predictions = np.array([0.0, 0.0, 0.9, 0.9, 0.0, 0.0], dtype=np.float32)

    assert predictions_to_scenes(predictions) == [(0, 2), (4, 5)]


def test_transnet_no_transition_covers_last_frame() -> None:
    predictions = np.zeros(4, dtype=np.float32)

    assert predictions_to_scenes(predictions) == [(0, 3)]
