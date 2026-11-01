from __future__ import annotations

import json
from pathlib import Path
from pathlib import Path as _Path

import pytest as _pytest

GENERATOR = Path("scratchpad/gen_nb32_btc_b0_b1_dev163_ab.py")
NOTEBOOK = Path("notebooks/btc/32-btc-b0-b1-dev163-ab.ipynb")


def _payload() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source() -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in _payload()["cells"])


@_pytest.mark.skipif(
    not _Path("scratchpad").is_dir(),
    reason="notebook generators (scratchpad/) are not included in the public replica",
)
def test_notebook_32_regenerates_and_compiles() -> None:
    exec(compile(GENERATOR.read_text(encoding="utf-8"), str(GENERATOR), "exec"))
    payload = _payload()
    assert payload["nbformat"] == 4
    assert all(not cell.get("outputs") for cell in payload["cells"])
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-32-cell-{index}", "exec")


def test_notebook_32_locks_exact_expanded_dev_contract() -> None:
    source = _source()
    assert "EXPECTED_DEV_QUERIES = 163" in source
    assert "EXPECTED_KIS_QUERIES = 115" in source
    assert "EXPECTED_QA_QUERIES = 48" in source
    assert "EXPECTED_LEGACY_KIS_QA = 54" in source
    assert "EXPECTED_EXPANDED_ONLY = 109" in source
    assert 'RUN_SCOPE = "PAIRED_DEV_KIS_QA_163"' in source
    assert "gate1-b0-b1-dev-kis-qa-163-v1" in source
    assert "query_source_ledger" in source
    assert "legacy_kis_qa_54" in source
    assert "expanded_only_109" in source
    assert '"NOTEBOOK32": "COMPLETE"' in source
    assert 'query.type in {"KIS", "QA"}' in source
    assert 'query.split != "dev"' in source
    assert "eval/heldout" not in source
    assert '"heldout_accessed": False' in source
    assert '"btc_mutated": False' in source


@_pytest.mark.skipif(
    not _Path("eval/dev").is_dir(),
    reason="raw per-annotator dev files (eval/dev/) are not included in the public replica",
)
def test_repository_sources_form_exact_163_query_set() -> None:
    from mfe.eval.official import load_official_queryset

    queries = []
    for member in ("quang", "triet", "trong"):
        queries.extend(
            query
            for query in load_official_queryset(Path("eval/dev") / f"{member}.jsonl")
            if query.type in {"KIS", "QA"}
        )
    assert len(queries) == 163
    assert len({query.query_id for query in queries}) == 163
    assert sum(query.type == "KIS" for query in queries) == 115
    assert sum(query.type == "QA" for query in queries) == 48
    assert all(query.split == "dev" for query in queries)
