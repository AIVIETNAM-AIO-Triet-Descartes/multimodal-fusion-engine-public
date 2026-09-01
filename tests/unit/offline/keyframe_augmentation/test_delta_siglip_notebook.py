from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
NOTEBOOK = ROOT / "notebooks" / "btc" / "13-btc-delta-siglip-producer.ipynb"


def test_notebook_is_preflight_first_and_research_only() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert payload["metadata"]["kaggle"]["accelerator"] == "gpu"
    assert 'RUN_SCOPE = "PREFLIGHT"' in text
    assert "75296b93ab4428ae721776d90167cdf3bca6b262137fd93610d04082def5a630" in text
    assert "CANARY_MAX_ABS_DIFF" in text
    assert "ENV_MISMATCHES" in text
    assert "use_fast=True" in text
    assert "UNVERIFIED_LEGACY_MANIFEST" in text
    assert "BLOCKED_MISSING_OCR_CAPTION_BGE_AND_DOWNSTREAM" in text
    assert "BTC.put(" not in text
    assert "BTC.remove_file(" not in text
    assert "selection_freeze_sha256" in text


def test_every_delta_siglip_code_cell_compiles() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"delta-siglip-cell-{index}", "exec")
