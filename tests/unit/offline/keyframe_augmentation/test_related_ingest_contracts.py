from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[4] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_btc_relational_rounds_timestamp_ms_by_contract(tmp_path: Path) -> None:
    module = _load_script("build_btc_relational")
    mapping = tmp_path / "V.csv"
    mapping.write_text(
        "n,pts_time,fps,frame_idx\n1,0.0015,1000.0,1\n",
        encoding="utf-8",
    )
    info = tmp_path / "V.json"
    info.write_text(json.dumps({"width": 10, "height": 10}), encoding="utf-8")

    video, keyframes, _ = module.rows_for("V", mapping, info)

    assert video.duration_ms == 2
    assert keyframes[0].timestamp_ms == 2
    assert keyframes[0].frame_idx == 1


def test_vector_ingest_requires_canonical_frame_idx_column(tmp_path: Path) -> None:
    module = _load_script("ingest_btc_vectors")
    mapping_root = tmp_path / "map-keyframes"
    feature_root = tmp_path / "clip-features"
    mapping_root.mkdir()
    feature_root.mkdir()
    with (mapping_root / "V.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerow({"n": 1, "pts_time": 0.0, "fps": 30.0, "frame_idx": ""})
    np.save(feature_root / "V.npy", np.zeros((1, 512), dtype=np.float32))

    with pytest.raises(ValueError, match="canonical frame_idx is blank"):
        list(module.iter_records(tmp_path, quarantine=[]))
