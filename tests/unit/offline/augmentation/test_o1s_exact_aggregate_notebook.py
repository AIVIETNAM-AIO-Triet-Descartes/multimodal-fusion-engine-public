from __future__ import annotations

import json
from pathlib import Path
from pathlib import Path as _Path

import pytest as _pytest

NOTEBOOK = Path("notebooks/btc/22-btc-o1s-exact-aggregate.ipynb")
GENERATOR = Path("scratchpad/gen_nb22_btc_o1s_exact_aggregate.py")


def _payload_and_source() -> tuple[dict, str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    return payload, source


def test_exact_aggregate_is_fail_closed_dev_only_and_read_only() -> None:
    payload, source = _payload_and_source()
    assert all(not cell.get("outputs") for cell in payload["cells"])
    assert "accelerator" not in payload["metadata"]
    assert "# 22 — O1-S exact aggregate" in source
    assert 'RUN_SCOPE = "AGGREGATE"' in source
    assert 'CONFIRM_AGGREGATE = "O1S-EXACT-DEV"' in source
    assert 'RUNNERS = ("quang", "trong")' in source
    assert "EXPECTED_EXACT_TARGETS = 61" in source
    assert "EXPECTED_EXACT_VIDEOS = 35" in source
    assert "EXPECTED_EXACT_FRAMES = 32867" in source
    assert "EXPECTED_EXACT_PLAN_HASH" in source
    assert "validate_exact_worker_manifest" in source
    assert "exact_remote_paths" in source
    assert "O1S_EXACT_SIGLIP_VIDEO_COMPLETE" in source
    assert "exact_minus_lb64_source_score" in source
    assert "O1S_EXACT_AGGREGATE_COMPLETE" in source
    assert '"notebook23_status": "BLOCKED_PENDING_FULL_FEATURE_PARITY"' in source
    assert "eval/queryset_heldout" not in source
    assert '"heldout_accessed": True' not in source
    assert "btc.put(" not in source
    assert "btc.put_many(" not in source
    assert "research.put_many(" in source
    assert 'REPO_REF = "feature/quang/storage"' in source


@_pytest.mark.skipif(
    not _Path("scratchpad").is_dir(),
    reason="notebook generators (scratchpad/) are not included in the public replica",
)
def test_exact_aggregate_is_regenerated_exactly_and_cells_compile() -> None:
    payload, _ = _payload_and_source()
    before = NOTEBOOK.read_bytes()
    namespace = {"__name__": "__main__"}
    exec(compile(GENERATOR.read_text(encoding="utf-8"), str(GENERATOR), "exec"), namespace)
    assert NOTEBOOK.read_bytes() == before
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"o1s-exact-aggregate-cell-{index}", "exec")
