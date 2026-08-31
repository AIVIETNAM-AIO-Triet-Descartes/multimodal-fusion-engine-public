from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
NOTEBOOK = ROOT / "notebooks/btc/tmp-kaggle-keyframe-producer-v2.ipynb"
CONFIG = ROOT / "config/keyframe_augmentation_kaggle.yaml"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_is_clean_and_code_cells_compile() -> None:
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        assert cell.get("outputs", []) == []
        assert cell.get("execution_count") is None
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_notebook_uses_pinned_upstream_and_separate_research_volume() -> None:
    notebook = _load_notebook()
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert 'MEMBER = "quang"' in source
    assert 'WORKER = "quang1"' in source
    assert config["transnetv2"]["repository"] == "https://github.com/soCzech/TransNetV2.git"
    assert config["transnetv2"]["commit"] == "85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed"
    assert config["transnetv2"]["converter_relpath"] == "inference-pytorch/convert_weights.py"
    assert config["transnetv2"]["tensorflow_weights_relpath"] == "inference/transnetv2-weights"
    assert config["workers"] == {
        "triet": ["triet1", "triet2"],
        "trong": ["trong1", "trong2"],
        "quang": ["quang1", "quang2", "quang3", "quang4"],
        "bao": ["bao1", "bao2"],
        "dat": ["dat1", "dat2"],
    }
    assert config["btc"]["volume"] != config["output"]["volume"]
    assert config["output"]["root"].startswith("/research/")
    assert "create_if_missing=False" in source
    assert "run_gate1" not in source
    assert 'resolve_video_source(video_id, local["video"])' in source
    assert "kaggle-input-read-only" in source
    assert 'RUN_SCOPE = "PREFLIGHT"' in source
    assert 'CONFIRM_SHARD = ""' in source
    assert 'CONFIRM_SHARD != WORKER' in source
    assert "warmup_transnet(" in source
    assert "INFERENCE_WINDOW_BATCH_SIZE = 8" in source
    assert "OUTPUT_ROOT_OVERRIDE = None" in source
    assert "OUTPUT_ROOT_OVERRIDE phải là root con /sandbox/" in source
    assert "OUTPUT_ROOT_OVERRIDE or CFG[\"output\"][\"root\"]" in source
    assert "requirements-kaggle-keyframe-augmentation.txt" in source
    assert "REPO_URL.replace" not in source
    assert "https://{token}@" not in source
    assert "GIT_ASKPASS" in source
    assert "secret_values" in source
    assert "secret_values=(token,) if token else ()" in source
    assert "askpass.unlink(missing_ok=True)" in source
    assert 'transnet_env["GIT_LFS_SKIP_SMUDGE"] = "1"' in source
    assert '"clone", "--filter=blob:none", "--no-checkout"' in source
    assert '"fetch", "--depth", "1", "origin", tn["commit"]' in source
    assert 'tn["tensorflow_weights_relpath"].rstrip("/") + "/**"' in source
    assert '"lfs", "pull", "--include", tf_weights_glob, "--exclude", ""' in source
    assert 'str(TRANSNET_CONVERTER), "--tf_weights", str(TRANSNET_TF_WEIGHTS)' in source
    assert "TRANSNET_TF_WEIGHTS_SHA256" in source
    assert '"transnet_converter_sha256": TRANSNET_CONVERTER_SHA256' in source
    assert '"transnet_tf_weights_sha256": TRANSNET_TF_WEIGHTS_SHA256' in source
    assert 'pending_by_size[:1] if RUN_SCOPE == "PREFLIGHT"' in source
    assert "SESSION_MAX_MINUTES" in source
    assert "ordinal > 1 and session_elapsed >= SESSION_MAX_MINUTES * 60" in source
    assert '"session_checkpoint": "TIME_BUDGET_REACHED"' in source
    assert '"model_cache_hit": marker["model_cache_hit"]' in source
    assert '"inference_window_batch_size": detector_report["window_batch_size"]' in source
    assert "eta_hours_estimate_this_todo" in source


def test_notebook_is_a_self_contained_team_handoff() -> None:
    notebook = _load_notebook()
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )

    for heading in ("Step 0a", "Step 0b", "Step 0c", "Step 0d", "Step 1", "Step 2"):
        assert heading in markdown
    for instruction in (
        "Bấm Run All",
        "Bạn chỉ cần sửa hai dòng",
        "Kaggle Secrets",
        "Khi nào phần của bạn xong?",
        "Lỗi thường gặp",
    ):
        assert instruction in markdown


def test_completion_marker_is_uploaded_last_after_hash_verification() -> None:
    notebook = _load_notebook()
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    start = source.index("def process_video(")
    end = source.index("def build_member_manifest(")
    function = source[start:end]

    shots_upload = function.index('output_transport.put(local["shots"]')
    report_upload = function.index('output_transport.put(local["report"]')
    remote_hash_check = function.index("remote_shots_sha = output_transport.sha256")
    marker_upload = function.index('output_transport.put(local["marker"]')
    assert shots_upload < remote_hash_check < marker_upload
    assert report_upload < remote_hash_check < marker_upload
    assert "Remote hash mismatch" in function
    assert "unsigned_marker.pop(\"marker_payload_hash\", None)" in source
    assert "sha256_json(unsigned_marker) != marker_payload_hash" in source
    assert "transnet_source_sha256" in source
    assert "not marker.get(\"shots_sha256\")" in source
    assert "not marker.get(\"report_sha256\")" in source
    assert 'markers/{MEMBER}/{WORKER}/' in source
    assert 'worker-manifests/{WORKER}.json' in source


def test_notebook_git_runner_redacts_secret_from_called_process_error() -> None:
    notebook = _load_notebook()
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    start = source.index("def _redact_text(")
    end = source.index("def clone_repository(")
    namespace = {"subprocess": subprocess}
    exec(source[start:end], namespace)

    secret = "ghp_adversarial_test_secret"

    class _FailingSubprocess:
        CalledProcessError = subprocess.CalledProcessError

        @staticmethod
        def run(*_args, **_kwargs):
            raise subprocess.CalledProcessError(
                128,
                ["git", "clone", secret],
                output=f"stdout contains {secret}",
                stderr=f"stderr contains {secret}",
            )

    namespace["subprocess"] = _FailingSubprocess
    with pytest.raises(RuntimeError) as raised:
        namespace["run"](["git", "clone", secret], secret_values=(secret,))

    message = str(raised.value)
    assert secret not in message
    assert message.count("[REDACTED]") >= 3
