from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mfe.contracts.ids import frame_idx_from_pts, pts_time_to_timestamp_ms
from mfe.offline.keyframe_augmentation.common import pts_time_for_frame_idx
from mfe.offline.keyframe_augmentation.gate0 import (
    audit_mapping_math,
    read_and_validate_mapping,
    verify_split_isolation,
)


def test_mapping_contract_uses_literal_fps_floor_and_collapses_btc_aliases(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "V.csv"
    mapping.write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,0.0333333,30.0,0\n3,1.0,30.0,30\n",
        encoding="utf-8",
    )

    frames, stats = read_and_validate_mapping(mapping, video_id="V", total_frames=31)

    assert [frame.frame_idx for frame in frames] == [0, 30]
    assert stats["csv_rows"] == 3
    assert stats["unique_frames"] == 2
    assert stats["duplicate_aliases"] == [{"frame_idx": 0, "kept_n": 1, "dropped_n": 2}]
    assert frames[1].keyframe_id == "V#000030"


def test_mapping_rejects_round_instead_of_floor(tmp_path: Path) -> None:
    mapping = tmp_path / "V.csv"
    mapping.write_text(
        "n,pts_time,fps,frame_idx\n1,0.019,30.0,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="floor"):
        read_and_validate_mapping(mapping, video_id="V", total_frames=10)


def test_mapping_accepts_first_and_last_frame_but_rejects_exclusive_upper_bound(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.csv"
    valid.write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,10.0,0\n2,0.9,10.0,9\n",
        encoding="utf-8",
    )
    frames, _ = read_and_validate_mapping(valid, video_id="V", total_frames=10)
    assert [row.frame_idx for row in frames] == [0, 9]

    invalid = tmp_path / "invalid.csv"
    invalid.write_text(
        "n,pts_time,fps,frame_idx\n1,1.0,10.0,10\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"outside \[0,10\)"):
        read_and_validate_mapping(invalid, video_id="V", total_frames=10)


@pytest.mark.parametrize("fps", [25.0, 30.0, 29.97, 26.44])
@pytest.mark.parametrize("frame_idx", [0, 1, 2, 999, 99999])
def test_synthetic_pts_floor_roundtrip_for_literal_fps(fps: float, frame_idx: int) -> None:
    pts_time = pts_time_for_frame_idx(frame_idx, fps)

    assert math.floor(pts_time * fps) == frame_idx
    assert frame_idx_from_pts(pts_time, fps) == frame_idx


@pytest.mark.parametrize("fps", [25.0, 30.0, 29.97, 26.44])
def test_pts_to_frame_uses_floor_at_integer_boundary_with_epsilon(fps: float) -> None:
    boundary_frame = 100
    below = (boundary_frame - 1e-7) / fps
    above = (boundary_frame + 1e-7) / fps

    assert frame_idx_from_pts(below, fps) == boundary_frame - 1
    assert frame_idx_from_pts(above, fps) == boundary_frame


@pytest.mark.parametrize(
    ("pts_time", "fps"),
    [
        (-1.0, 25.0),
        (1.0, 0.0),
        (1.0, -25.0),
        (float("nan"), 25.0),
        (1.0, float("nan")),
        (float("inf"), 25.0),
    ],
)
def test_pts_to_frame_fails_closed_for_invalid_numeric_inputs(
    pts_time: float, fps: float
) -> None:
    with pytest.raises((ValueError, OverflowError)):
        frame_idx_from_pts(pts_time, fps)


def test_timestamp_ms_round_is_a_separate_contract() -> None:
    assert pts_time_to_timestamp_ms(1.2346) == 1235
    assert pts_time_to_timestamp_ms(1.2344) == 1234


def test_mapping_math_audit_preserves_csv_identity_and_measures_counterfactual_round(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "V.csv"
    mapping.write_text(
        "n,pts_time,fps,frame_idx\n"
        "1,0.0,30.0,0\n"
        "2,0.019,30.0,0\n"
        "3,0.04,30.0,1\n",
        encoding="utf-8",
    )

    report = audit_mapping_math([mapping])

    assert report["floor_mismatches"] == 0
    assert report["round_mismatches"] == 1
    assert report["coordinate_source"].startswith("frame_idx copied")


def _write_queries(path: Path, split: str, rows: list[tuple[str, str]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "query_id": query_id,
                    "split": split,
                    "type": "KIS",
                    "gt_video_id": video_id,
                }
            )
            + "\n"
            for query_id, video_id in rows
        ),
        encoding="utf-8",
    )


def test_dev_and_heldout_must_be_video_disjoint(tmp_path: Path) -> None:
    dev = tmp_path / "dev.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    _write_queries(dev, "dev", [("d1", "V1")])
    _write_queries(heldout, "heldout", [("h1", "V1")])

    with pytest.raises(ValueError, match="leakage"):
        verify_split_isolation(dev, heldout)


def test_dev_and_heldout_lock_records_hashes(tmp_path: Path) -> None:
    dev = tmp_path / "dev.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    _write_queries(dev, "dev", [("d1", "V1")])
    _write_queries(heldout, "heldout", [("h1", "V2")])

    result = verify_split_isolation(dev, heldout)

    assert result["query_overlap"] == []
    assert result["video_overlap"] == []
    assert len(result["dev"]["sha256"]) == 64
    assert len(result["heldout"]["sha256"]) == 64
