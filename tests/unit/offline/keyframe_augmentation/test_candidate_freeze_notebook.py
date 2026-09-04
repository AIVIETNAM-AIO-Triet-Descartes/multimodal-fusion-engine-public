from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
NOTEBOOK = ROOT / "notebooks" / "btc" / "12-btc-b0-b4-candidate-freeze.ipynb"


def test_notebook_locks_measured_audit_and_never_writes_btc() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert payload["metadata"]["kaggle"]["accelerator"] == "none"
    assert "370de9c53cdd84ee3c2bb5eb12e4a40c7d7d36676ba27f7dec0650d618d3586b" in text
    assert "2e0887652464e0cc7942bedb94177cfe2dff5f7baf77081902461cf3c885d75f" in text
    assert "equalize_budget" in text
    assert "validate_frozen_selection" in text
    assert "BTC.put(" not in text
    assert "BTC.remove_file(" not in text
    assert '"heldout_accessed": False' in text
    assert "SELECTION_FROZEN" in text


def test_every_candidate_freeze_code_cell_compiles() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"candidate-freeze-cell-{index}", "exec")
