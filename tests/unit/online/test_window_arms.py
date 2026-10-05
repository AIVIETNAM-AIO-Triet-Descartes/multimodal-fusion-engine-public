"""Bảo vệ hợp đồng của hai arm bản ghi cửa sổ.

Window artifacts trả `video#w0000`, còn mọi tầng online phía sau chỉ nhận
`video#frame_idx`. Một phép kiểm trực tiếp ở đây chặn lỗi hỏng âm thầm đó.
"""

from pathlib import Path

import numpy as np

from mfe.online.dense_arms import DENSE_ARM_SPECS
from mfe.online.flat_arms import (
    DENSE_FILE,
    SOURCE_FILE,
    FlatBundle,
    _window_candidates,
    build_flat_dense_arms,
    build_flat_text_arms,
)
from mfe.online.text_arms import TEXT_ARM_SPECS


def test_window_arm_registries_declare_the_right_source_and_language():
    assert SOURCE_FILE["window_bm25"] == "window_text.json"
    assert TEXT_ARM_SPECS["window_bm25"] == ("window_bm25", False, "en")
    assert DENSE_FILE["window_bge_m3"] == "window_dense.npz"
    assert DENSE_ARM_SPECS["window_bge_m3"][1:4] == (
        1024,
        "BAAI/bge-m3",
        "any",
    )


def test_window_map_uses_frame_idx_and_real_fps(tmp_path: Path):
    maps = tmp_path / "maps"
    maps.mkdir()
    (maps / "L21_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,15.0,30.0,450\n",
        encoding="utf-8",
    )
    bundle = FlatBundle()
    bundle.vector_keys["siglip"] = [
        "L21_V001#000000",
        "L21_V001#000449",
        "L21_V001#000450",
    ]

    mapping = bundle.build_window_keyframes(maps)

    assert mapping == {
        "L21_V001#w0000": ["L21_V001#000000", "L21_V001#000449"],
        "L21_V001#w0001": ["L21_V001#000450"],
    }


def test_window_candidates_always_return_keyframe_ids():
    bundle = FlatBundle()
    bundle.window_keyframes = {
        "L21_V001#w0000": ["L21_V001#000001", "L21_V001#000002"],
        "L21_V001#w0001": ["L21_V001#000003", "L21_V001#000004"],
    }

    candidates = _window_candidates(
        [("L21_V001#w0000", 0.9), ("L21_V001#w0001", 0.8)],
        bundle,
        "window_bm25",
        "run",
        3,
    )

    assert [candidate.keyframe_id for candidate in candidates] == [
        "L21_V001#000001",
        "L21_V001#000002",
        "L21_V001#000003",
    ]
    assert all("#w" not in candidate.keyframe_id for candidate in candidates)


def test_window_arms_are_unavailable_without_the_bridge():
    bundle = FlatBundle()
    bundle.texts["window_bm25"] = {"L21_V001#w0000": "a lion"}
    bundle.text_keys["window_bm25"] = ["L21_V001#w0000"]
    bundle.vectors["window_bge_m3"] = np.ones((1, 1024), dtype=np.float32)
    bundle.vector_keys["window_bge_m3"] = ["L21_V001#w0000"]
    bundle.missing["window_bm25"] = "missing bridge"
    bundle.missing["window_bge_m3"] = "missing bridge"

    text_arm = next(arm for arm in build_flat_text_arms(bundle) if arm.name == "window_bm25")
    dense_arm = next(arm for arm in build_flat_dense_arms(bundle) if arm.name == "window_bge_m3")

    assert not text_arm.available
    assert not dense_arm.available
