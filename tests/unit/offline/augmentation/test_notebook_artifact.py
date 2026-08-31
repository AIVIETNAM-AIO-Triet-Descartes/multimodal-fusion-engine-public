from __future__ import annotations

import ast
import json
from pathlib import Path

NOTEBOOK = Path("notebooks/btc/10-btc-keyframe-augmentation-integrated.ipynb")


def test_augmentation_notebook_is_valid_and_defaults_to_safe_smoke_mode():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'RUN_MODE = "SMOKE"' in source
    assert 'MEMBER = "quang"' in source
    assert 'WORKER = "quang1"' in source
    assert '"quang": ("quang1", "quang2", "quang3", "quang4")' in source
    assert 'OUTPUT_TRANSPORT = "local"' in source
    assert "ALLOW_MODAL_WRITES = False" in source
    assert "READ_SMOKE_BTC_FROM_MODAL = True" in source
    assert "SMOKE bắt buộc OUTPUT_TRANSPORT='local'" in source
    assert "isinstance(transport, LocalDirTransport)" in source
    assert "MODAL_TOKEN_SECRET" in source
    assert "create_if_missing=False" in source
    assert "modal_read.put" not in source
    assert "modal_read.put_many" not in source
    assert "https://{github_token}@" not in source
    assert 'env["GIT_ASKPASS"] = str(askpass)' in source
    assert 'env["MFE_GITHUB_TOKEN"] = github_token' in source
    assert 'git_env.pop("MFE_GITHUB_TOKEN", None)' in source
    assert "askpass_path.unlink(missing_ok=True)" in source
    assert "raise RuntimeError(" in source
    assert ") from None" in source
    assert "--overwrite" not in source
    assert "FULL_CORPUS" not in source
    assert "heldout" in source
    assert "write_jsonl_atomic(runtime_heldout, [])" in source
    assert 'raw_config["dataset"]["extra_pilot_videos"] = 0' in source
    assert "videos = {SMOKE_VIDEO_ID: videos[SMOKE_VIDEO_ID]}" in source
    assert '"--videos-root", str(ACTIVE_VIDEOS_ROOT)' in source
    assert '"--maps-root", str(ACTIVE_MAPS_ROOT)' in source
    assert "Experiment root trên Modal" not in source
    assert 'SAVED_MODEL_GLOB = "AUTO"' in source
    assert "https://github.com/soCzech/TransNetV2.git" in source
    assert 'transnet_env["GIT_LFS_SKIP_SMUDGE"] = "1"' in source
    assert 'saved_model_pb = _configured_file(SAVED_MODEL_GLOB, ("saved_model.pb",))' in source
    assert 'checkpoint_target / "variables" / source.name' in source
    assert "sha256_path(checkpoint_target)" in source
    assert "BLOCKED_MISSING_O1_S_FEATURE_RANK_ARTIFACT" in source
    assert "O1-T_DIAGNOSTIC_ONLY" in source
    assert "MEASURED_O1_S_O1_R_PASS" in source
    assert '"--b1", str(b1_kept), "--b2", str(b2_kept)' in source
    assert "validate_primary_equal_budget" in source

    requirements = Path("notebooks/requirements-kaggle-btc-augmentation.txt").read_text(
        encoding="utf-8"
    )
    assert "tensorflow>=2.16,<3" in requirements


def test_smoke_report_exposes_effectiveness_without_claiming_dev_gate():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert '"effectiveness_signal": effectiveness_signal' in source
    assert '"b3_minus_b0": b3_delta_b0' in source
    assert '"b3_minus_matched_b4": b3_delta_b4' in source
    assert "One-video smoke signal only" in source


def test_augmentation_notebook_code_cells_compile_without_outputs():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        ast.parse("".join(cell["source"]))
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
