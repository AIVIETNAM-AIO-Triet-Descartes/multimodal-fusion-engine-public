from __future__ import annotations

import json
from pathlib import Path


def test_oracle_eval_notebook_is_dev_only_fail_closed_and_compilable() -> None:
    """Notebook 18 must evaluate an isolated oracle overlay, never BTC live."""

    path = Path("notebooks/btc/18-btc-keyframe-augmentation-oracle-eval.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert 'RUN_ID = "oracle-eval-quang-v1"' in source
    assert 'MATERIALIZE_RUN_ID = "oracle-materialize-v1"' in source
    assert 'REPO_REF = "feature/quang/storage"' in source
    assert "EXPECTED_CANDIDATES = 202" in source
    assert "EXPECTED_TARGETS = 219" in source
    assert "validate_oracle_completion_marker" in source
    assert "OracleArtifactCandidate" in source
    assert "OracleArtifactTarget" in source
    assert "oracle_candidate_for_snapshot" in source
    assert "build_relational_snapshot" in source
    assert "allow_preexisting_mapping_problems=True" in source
    assert '"oracle_artifact": "decoder_pts"' in source
    assert '"relational_snapshot": "synthetic_literal_fps_interval"' in source
    assert '"frame_from_pts": "floor"' in source
    assert 'name="B0"' in source
    assert 'name="P1-N-SIGLIP"' in source
    assert 'name="O1-R-FINAL"' in source
    assert '"decision_authority": "diagnostic_only"' in source
    assert '"O1-S": "BLOCKED_' in source
    assert '"feature_parity_status": "BLOCKED_' in source
    assert (
        '"o1r_force_stage": '
        '"post_cascade_final_candidate_list_before_submission_mapping"'
    ) in source
    assert 'checkpoint_revision = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"' in source
    assert "use_fast=True" in source
    assert 'name="O1-T"' not in source
    assert 'heldout_accessed": False' in source
    assert "eval/queryset_heldout" not in source
    assert 'ModalVolumeTransport("keyframes-btc"' not in source
    assert "BTC_VOLUME" not in source
    assert "btc.put(" not in source
    assert "hf_" not in source

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"oracle-eval-cell-{index}", "exec")
