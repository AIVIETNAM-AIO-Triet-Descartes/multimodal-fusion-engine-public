from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("notebooks/btc/19-btc-o1s-representative-scan-producer.ipynb")


def _source() -> tuple[dict, str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    return payload, source


def test_o1s_scan_notebook_is_dev_only_research_and_fail_closed() -> None:
    payload, source = _source()
    assert all(not cell.get("outputs") for cell in payload["cells"])
    assert "accelerator" not in payload["metadata"]
    assert 'RUNNER = "quang"' in source
    assert 'RUNNERS = ("quang", "trong")' in source
    assert 'RUN_SCOPE = "PREFLIGHT"' in source
    assert 'RUN_ID = "o1s-lb64-scan-v1"' in source
    assert 'SAMPLING_POLICY = "inclusive_uniform_integer_lb64"' in source
    assert "EXPECTED_TARGETS = 219" in source
    assert "EXPECTED_VIDEOS = 90" in source
    assert "EXPECTED_SCAN_FRAMES = 12982" in source
    assert "3d524cbfa77ceae0026a51ff48bdab0685fb9f66abca0975520112ae353e997c" in source
    assert "build_representative_scan_plan" in source
    assert "partition_representative_scan" in source
    assert "validate_scan_completion_marker" in source
    assert "RepresentativeResolvedSample" in source
    assert '"frame_idx_source": "decoder_counter"' in source
    assert '"pts_time_source": "decoder_pts"' in source
    assert '"decision_authority": "diagnostic_only"' in source
    assert '"full_o1s_status": "BLOCKED_SAMPLED_INTERVALS_ARE_LOWER_BOUND"' in source
    assert "CANARY_MAX_ABS_DIFF" in source
    assert "checked.kept_ids" in source
    assert "checked.ids" not in source
    assert "use_fast=True" in source
    assert 'REPO_REF = "feature/quang/storage"' in source
    assert "eval/queryset_heldout" not in source
    assert "heldout_accessed\": True" not in source
    assert "BTC.put(" not in source
    assert "BTC.remove" not in source
    assert "RESEARCH.put" in source


def test_o1s_scan_notebook_is_regenerated_exactly_and_cells_compile() -> None:
    payload, _ = _source()
    before = NOTEBOOK.read_bytes()
    generator = Path("scratchpad/gen_nb19_btc_o1s_scan_producer.py")
    namespace = {"__name__": "__main__"}
    exec(compile(generator.read_text(encoding="utf-8"), str(generator), "exec"), namespace)
    assert NOTEBOOK.read_bytes() == before
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"o1s-scan-cell-{index}", "exec")
