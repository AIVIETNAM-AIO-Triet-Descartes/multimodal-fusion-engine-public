from __future__ import annotations

import json
import runpy
from pathlib import Path

GENERATOR = Path("scratchpad/gen_nb33_btc_b1_subtitle_producer.py")
NOTEBOOK = Path("notebooks/btc/33-btc-b1-subtitle-producer.ipynb")


def _regenerate() -> None:
    runpy.run_path(str(GENERATOR), run_name="__main__")


def _source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebook_regenerates_idempotently_and_compiles() -> None:
    _regenerate()
    first = NOTEBOOK.read_bytes()
    _regenerate()
    assert NOTEBOOK.read_bytes() == first

    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert all(not cell.get("outputs") for cell in payload["cells"])
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"nb33-cell-{index}", "exec")


def test_notebook_is_research_only_and_never_uses_synthetic_pts() -> None:
    source = _source()
    assert 'RUN_SCOPE = "PREFLIGHT"' in source
    assert 'REPO_REF = "feature/quang/storage"' in source
    assert "EXPECTED_FRAMES = 5301" in source
    assert "inherit_btc_next_keyframe_window_by_frame_idx" in source
    assert '"candidate_pts_time_used": False' in source
    assert "subtitles_dense.npz" in source
    assert "subtitle-alignment-audit.jsonl" in source
    assert "CONFIRM_AGGREGATE" in source
    assert "RESEARCH.put" in source
    assert "BTC.put" not in source
    assert "WhisperModel" not in source
    assert "pts_time_for_frame_idx" not in source
    assert "eval/queryset_heldout" not in source


def test_complete_marker_is_written_after_all_aggregate_artifacts() -> None:
    source = _source()
    artifact_upload = source.index("for relative,path in uploads.items():")
    complete_upload = source.index(
        'RESEARCH.put(complete_path, f"{OUTPUT_ROOT}/COMPLETE.json")'
    )
    assert artifact_upload < complete_upload
