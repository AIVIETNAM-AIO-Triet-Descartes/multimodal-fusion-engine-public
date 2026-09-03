from __future__ import annotations

import json
from pathlib import Path

import pytest

GENERATOR = Path("scratchpad/gen_nb30_btc_b0_b1_dev_ab.py")
NOTEBOOK = Path("notebooks/btc/30-btc-b0-b1-dev-ab.ipynb")


def _payload() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source() -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in _payload()["cells"])


def _helpers() -> dict:
    cell = next(
        cell
        for cell in _payload()["cells"]
        if "dev-ab-helpers" in cell.get("metadata", {}).get("tags", [])
    )
    namespace: dict = {}
    exec(compile("".join(cell["source"]), "dev-ab-helpers", "exec"), namespace)
    return namespace


def test_notebook_30_regenerates_and_compiles() -> None:
    exec(compile(GENERATOR.read_text(encoding="utf-8"), str(GENERATOR), "exec"))
    payload = _payload()
    assert payload["nbformat"] == 4
    assert all(not cell.get("outputs") for cell in payload["cells"])
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-30-cell-{index}", "exec")


def test_notebook_30_locks_inputs_and_stays_dev_only() -> None:
    source = _source()
    assert "EXPECTED_B1_COUNT = 5301" in source
    assert "d2c38a6a5f7fe810b09d87b6a384b850bcbf942564bf83d910c353eaeafc5fa6" in source
    assert 'RUN_SCOPE = "PAIRED_DEV_AB"' in source
    assert "gate1-b0-b1-dev-ab-v1" in source
    assert "eval/queryset_dev.jsonl" in source
    assert "eval/queryset_heldout" not in source
    assert '"heldout_accessed": False' in source
    assert '"btc_mutated": False' in source
    assert "load_overlay_flat_bundle" in source
    assert "build_relational_snapshot" in source
    assert "paired_bootstrap_interval" in source
    assert "hãy chọn GPU T4, không chọn P100" in source
    assert "research.put(complete_path, complete_remote)" in source
    assert source.index("for path in (report_path, per_query_path)") < source.index(
        "research.put(complete_path, complete_remote)"
    )


@pytest.mark.parametrize(
    ("mean_delta", "lower", "expected"),
    [
        (0.02, 0.001, "DEV_UPLIFT_CONFIRMED"),
        (0.02, -0.001, "DEV_POSITIVE_BUT_UNCERTAIN"),
        (0.0, 0.0, "NO_MEASURED_CHANGE"),
        (-0.01, -0.02, "DEV_REGRESSION"),
    ],
)
def test_verdict_is_explicit(mean_delta: float, lower: float, expected: str) -> None:
    assert _helpers()["classify_dev_ab"](mean_delta, lower) == expected


def test_snapshot_records_fail_closed() -> None:
    helper = _helpers()["snapshot_records"]
    rows = [{
        "keyframe_id": "L21_V001#000025",
        "video_id": "L21_V001",
        "frame_idx": 25,
        "fps": 25.0,
        "pts_time": 1.0,
        "shot_ids": ["L21_V001#shot-1"],
    }]
    records = helper(rows)
    assert records[0].frame_idx == 25
    with pytest.raises(RuntimeError, match="round-trip"):
        helper([{**rows[0], "pts_time": 0.9}])
    with pytest.raises(RuntimeError, match="no shot id"):
        helper([{**rows[0], "shot_ids": []}])
