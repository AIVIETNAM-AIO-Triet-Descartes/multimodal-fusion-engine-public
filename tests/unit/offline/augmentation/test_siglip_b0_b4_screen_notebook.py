from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("notebooks/btc/24-btc-siglip-b0-b4-screen.ipynb")
GENERATOR = Path("scratchpad/gen_nb24_btc_siglip_b0_b4_screen.py")


def test_siglip_screen_contract_regeneration_and_compile() -> None:
    namespace = {"__name__": "__main__"}
    exec(compile(GENERATOR.read_text(encoding="utf-8"), str(GENERATOR), "exec"), namespace)
    before = NOTEBOOK.read_bytes()
    exec(compile(GENERATOR.read_text(encoding="utf-8"), str(GENERATOR), "exec"), namespace)
    assert NOTEBOOK.read_bytes() == before
    payload = json.loads(before)
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert all(not cell.get("outputs") for cell in payload["cells"])
    assert "accelerator" not in payload["metadata"]
    assert 'RUN_SCOPE = "SCREEN"' in source
    assert "EXPECTED_CANDIDATES_PER_ARM = 5301" in source
    assert "EXPECTED_FEATURE_FRAMES = 17399" in source
    assert "build_keyframe_id(str(row[\"video_id\"]), int(row[\"frame_idx\"]))" in source
    assert "ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)" in source
    assert "paired_bootstrap_95ci" in source
    assert "SIGLIP_B0_B4_SCREEN_COMPLETE" in source
    assert "full_feature_score_status" in source
    assert "eval/queryset_heldout" not in source
    assert "btc.put(" not in source and "btc.put_many(" not in source
    assert 'REPO_REF = "feature/quang/storage"' in source
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"siglip-screen-cell-{index}", "exec")
