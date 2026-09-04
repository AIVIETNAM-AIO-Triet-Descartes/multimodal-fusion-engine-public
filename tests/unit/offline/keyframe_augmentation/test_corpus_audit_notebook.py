from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
NOTEBOOK = ROOT / "notebooks" / "btc" / "11-btc-shot-coverage-audit.ipynb"


def test_notebook_is_cpu_only_and_keeps_btc_read_only() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert payload["metadata"]["kaggle"]["accelerator"] == "none"
    assert "EXPECTED_VIDEO_COUNT = 873" in text
    assert 'create_if_missing=False' in text
    assert "BTC.put(" not in text
    assert "BTC.remove_file(" not in text
    assert '"heldout_accessed": False' in text
    assert '"decision_authority": "diagnostic_only"' in text
    assert "temporal diagnostic" in text
    assert "AUDIT_COMPLETE" in text


def test_every_code_cell_compiles() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
