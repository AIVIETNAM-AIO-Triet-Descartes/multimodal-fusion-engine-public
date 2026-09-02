from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("notebooks/btc/23-btc-feature-parity-worklist-freeze.ipynb")
GENERATOR = Path("scratchpad/gen_nb23_btc_feature_parity_worklist.py")


def test_feature_parity_freeze_contract_and_regeneration() -> None:
    namespace = {"__name__": "__main__"}
    exec(compile(GENERATOR.read_text(encoding="utf-8"), str(GENERATOR), "exec"), namespace)
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    before = NOTEBOOK.read_bytes()
    exec(compile(GENERATOR.read_text(encoding="utf-8"), str(GENERATOR), "exec"), namespace)
    assert NOTEBOOK.read_bytes() == before
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert all(not cell.get("outputs") for cell in payload["cells"])
    assert "accelerator" not in payload["metadata"]
    assert 'RUN_SCOPE = "FREEZE"' in source
    assert 'RUNNERS = ("quang", "trong")' in source
    assert "EXPECTED_FEATURE_FRAMES = 17399" in source
    assert "EXPECTED_NB22_COMPLETE_SHA256" in source
    assert "partition_feature_worklist_by_runner" in source
    assert 'for artifact in ("images", "image_manifest", "siglip", "report")' in source
    assert '"dependency": "qwen25.COMPLETE"' in source
    assert "FEATURE_PARITY_WORKLIST_FROZEN" in source
    assert "eval/queryset_heldout" not in source
    assert "btc.put(" not in source and "btc.put_many(" not in source
    assert 'REPO_REF = "feature/quang/storage"' in source
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"feature-parity-freeze-cell-{index}", "exec")
