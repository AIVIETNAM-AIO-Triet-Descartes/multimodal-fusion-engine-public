from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("notebooks/btc/21-btc-o1s-exact-interval-scan.ipynb")


def notebook_source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def test_exact_scan_notebook_is_generated_and_fail_closed() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = notebook_source()
    assert payload["nbformat"] == 4
    assert "# 21 — O1-S exact interval scan" in source
    assert 'RUN_SCOPE = "PREFLIGHT"' in source
    assert 'RUN_ID = "o1s-exact-v1"' in source
    assert 'INTERVAL_POLICY = "inclusive_full_interval_exact"' in source
    assert "DEV_TARGETS = sorted(" in source
    assert "EXPECTED_DEV_TARGETS = 71" in source
    assert "EXPECTED_FULL_TARGETS = 61" in source
    assert "for frame_idx in range(target.frame_range_start, target.frame_range_end + 1)" in source
    assert '"heldout_accessed": False' in source
    assert '"btc_mutated": False' in source
    assert "if RUN_SCOPE == \"PREFLIGHT\":" in source
    assert "O1S_EXACT_SIGLIP_VIDEO_COMPLETE" in source
    assert "reference_lb64_canary_report_hash" in source


def test_exact_scan_uses_video_disjoint_resume_markers() -> None:
    source = notebook_source()
    assert '"marker": f"{OUTPUT_ROOT}/markers/{RUNNER}/{video_id}.json"' in source
    assert "valid_remote_completion(video_id)" in source
    assert "quang_keys.intersection(trong_keys)" in source
    assert "Hai runner COMPLETE -> notebook 22 exact O1-S aggregate" in source
