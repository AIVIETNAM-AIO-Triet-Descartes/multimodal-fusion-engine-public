from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
NOTEBOOK = ROOT / "notebooks/btc/10-btc-keyframes-argumentation-final.ipynb"
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
    assert 'markers/{MEMBER}/{WORKER}/' in source
    assert 'worker-manifests/{WORKER}.json' in source
