from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("notebooks/btc/20-btc-o1s-lb64-aggregate.ipynb")
GENERATOR = Path("scratchpad/gen_nb20_btc_o1s_aggregate.py")


def _source() -> tuple[dict, str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    return payload, source


def test_o1s_aggregate_is_dev_only_read_only_and_fail_closed() -> None:
    payload, source = _source()
    assert all(not cell.get("outputs") for cell in payload["cells"])
    assert "accelerator" not in payload["metadata"]
    assert 'RUN_SCOPE = "AGGREGATE"' in source
    assert 'RUNNERS = ("quang", "trong")' in source
    assert "EXPECTED_SCAN_VIDEOS = 90" in source
    assert "EXPECTED_SCAN_FRAMES = 12982" in source
    assert "EXPECTED_DEV_TARGETS = 71" in source
    assert "EXPECTED_BTC_SIGLIP_ROWS = 176707" in source
    assert "EXPECTED_SIGLIP_CANARY_REPORT_HASH" in source
    assert "validate_o1s_worker_manifest" in source
    assert "validate_scan_completion_marker" in source
    assert 'f"{scan_root}/samples/{video_id}.jsonl"' in source
    assert 'f"{scan_root}/markers/{runner}/{video_id}.json"' in source
    assert 'f"{scan_root}/samples/{runner}/{video_id}.jsonl"' not in source
    assert "best_source_rank" in source
    assert "best_btc_rank" in source
    assert "top_btc_outside_interval_rank" in source
    assert "o1s_math_call_site_ledger" in source
    assert "adversarial_counterexample_summary" in source
    assert "hypothesis-backlog.json" in source
    assert "workspace-safety-manifest.json" in source
    assert '"decision_authority": "diagnostic_only"' in source
    assert '"full_o1s_status": "BLOCKED_' in source
    assert '"official_score_status": "UNVERIFIED_' in source
    assert "eval/queryset_heldout" not in source
    assert '"heldout_accessed": True' not in source
    assert "btc.put(" not in source
    assert "btc.put_many(" not in source
    assert "research.put_many(" in source
    assert 'REPO_REF = "feature/quang/storage"' in source


def test_o1s_aggregate_is_regenerated_exactly_and_cells_compile() -> None:
    payload, _ = _source()
    before = NOTEBOOK.read_bytes()
    namespace = {"__name__": "__main__"}
    exec(compile(GENERATOR.read_text(encoding="utf-8"), str(GENERATOR), "exec"), namespace)
    assert NOTEBOOK.read_bytes() == before
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"o1s-aggregate-cell-{index}", "exec")
