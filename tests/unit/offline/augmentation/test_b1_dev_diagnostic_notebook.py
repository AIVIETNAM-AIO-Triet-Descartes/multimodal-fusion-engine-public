from __future__ import annotations

import json
from pathlib import Path

GENERATOR = Path("scratchpad/gen_nb31_btc_b1_dev_diagnostic.py")
NOTEBOOK = Path("notebooks/btc/31-btc-b1-dev-diagnostic.ipynb")


def _payload() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source() -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in _payload()["cells"])


def _helpers() -> dict:
    cell = next(
        cell
        for cell in _payload()["cells"]
        if "diagnostic-helpers" in cell.get("metadata", {}).get("tags", [])
    )
    namespace: dict = {}
    exec(compile("".join(cell["source"]), "diagnostic-helpers", "exec"), namespace)
    return namespace


def test_notebook_31_regenerates_and_compiles() -> None:
    exec(compile(GENERATOR.read_text(encoding="utf-8"), str(GENERATOR), "exec"))
    payload = _payload()
    assert payload["nbformat"] == 4
    assert all(not cell.get("outputs") for cell in payload["cells"])
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-31-cell-{index}", "exec")


def test_notebook_31_is_fast_dev_only_handoff() -> None:
    source = _source()
    assert "EXPECTED_B1_COUNT = 5301" in source
    assert "EXPECTED_CHANGED = 3" in source
    assert "PROMISING_BUT_INCONCLUSIVE" in source
    assert "floor(176707 * 0.03)" in source
    assert "gate1-b1-dev-diagnostic-v1" in source
    assert "leader-handoff.md" in source
    assert "eval/queryset_dev.jsonl" in source
    assert "eval/queryset_heldout" not in source
    assert "torch" not in source
    assert '"heldout_accessed": False' in source
    assert '"btc_mutated": False' in source
    data_publish = source.index(
        "for path in (decision_path, diagnostic_path, handoff_path)"
    )
    complete_publish = source.index("research.put(complete_path_out, complete_remote)")
    assert data_publish < complete_publish


def test_candidate_memberships_support_kis_and_trake() -> None:
    helper = _helpers()["candidate_memberships"]
    candidates = [
        {"keyframe_id": "V#000015", "video_id": "V", "frame_idx": 15},
        {"keyframe_id": "V#000035", "video_id": "V", "frame_idx": 35},
        {"keyframe_id": "X#000015", "video_id": "X", "frame_idx": 15},
    ]
    kis = {
        "type": "KIS",
        "gt_video_id": "V",
        "gt_frame_range": [10, 40],
        "gt_frame_range_strict": [12, 20],
    }
    kis_result = helper(kis, candidates)
    assert len(kis_result["b1_in_wide_gt"]) == 2
    assert len(kis_result["b1_in_strict_gt"]) == 1

    trake = {
        "type": "TRAKE",
        "gt_video_id": "V",
        "gt_events": [[10, 20], [30, 40]],
        "gt_events_strict": [[14, 16], [34, 36]],
    }
    trake_result = helper(trake, candidates)
    assert [row["matching_events"] for row in trake_result["b1_in_strict_gt"]] == [[0], [1]]
