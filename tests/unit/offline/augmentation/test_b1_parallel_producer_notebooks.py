from __future__ import annotations

import json
from pathlib import Path

GENERATOR = Path("scratchpad/gen_nb25_26_b1_parallel_producers.py")
NOTEBOOKS = [
    Path("notebooks/btc/25-btc-b1-easyocr-producer.ipynb"),
    Path("notebooks/btc/26-btc-b1-qwen25-producer.ipynb"),
    Path("notebooks/btc/27-btc-b1-paddle-vietocr-producer.ipynb"),
]


def _run_generator(namespace: dict) -> None:
    exec(  # noqa: S102 - notebook sinh bang script, chay lai la phep kiem
        compile(GENERATOR.read_text(encoding="utf-8"), str(GENERATOR), "exec"),
        namespace,
    )


def test_parallel_producers_regenerate_compile_and_are_resumable() -> None:
    ns = {"__name__": "__main__"}
    _run_generator(ns)
    before = {path: path.read_bytes() for path in NOTEBOOKS}
    _run_generator(ns)
    assert before == {path: path.read_bytes() for path in NOTEBOOKS}
    for path in NOTEBOOKS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in payload["cells"]
        )
        assert all(not cell.get("outputs") for cell in payload["cells"])
        assert "EXPECTED_B1_COUNT = 5301" in source
        assert 'RUNNERS = ("quang", "trong")' in source
        assert "marker_payload_hash" in source and "failures_this_session" in source
        assert "eval/queryset_heldout" not in source and "keyframes-btc" not in source
        assert 'REPO_REF = "feature/quang/storage"' in source
        for index, cell in enumerate(payload["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path.name}-cell-{index}", "exec")
