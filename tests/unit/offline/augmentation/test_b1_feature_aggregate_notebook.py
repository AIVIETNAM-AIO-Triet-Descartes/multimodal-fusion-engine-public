from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mfe.offline.keyframe_augmentation.common import sha256_json

GENERATOR = Path("scratchpad/gen_nb29_btc_b1_feature_aggregate.py")
NOTEBOOK = Path("notebooks/btc/29-btc-b1-feature-aggregate.ipynb")


def _source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def _helper_namespace() -> dict:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    helper = next(
        "".join(cell["source"])
        for cell in payload["cells"]
        if "aggregate-contract-helpers" in cell.get("metadata", {}).get("tags", [])
    )
    namespace: dict = {}
    exec(compile(helper, "aggregate-contract-helpers", "exec"), namespace)
    namespace.update(
        {
            "RUNNERS": ("quang", "trong"),
            "EXPECTED_NB24_COMPLETE_SHA256": "n" * 64,
            "EXPECTED_PARITY_FREEZE_SHA256": "p" * 64,
        }
    )
    return namespace


def test_notebook_29_regenerates_and_all_cells_compile() -> None:
    exec(compile(GENERATOR.read_text(encoding="utf-8"), str(GENERATOR), "exec"))
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert all(not cell.get("outputs") for cell in payload["cells"])
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-29-cell-{index}", "exec")

    source = _source()
    assert "EXPECTED_B1_COUNT = 5301" in source
    assert 'RUN_SCOPE = "AGGREGATE"' in source
    assert 'CONFIRM_AGGREGATE = "B1"' in source
    assert "gate1-b1-feature-aggregate-v1" in source
    assert "gate1-b1-easyocr-v1" in source
    assert "gate1-b1-qwen25-v1" in source
    assert "gate1-b1-paddle-vietocr-v1" in source
    assert "gate1-b1-bge-m3-v1" in source
    assert "siglip_vectors.npz" in source and "dense_vectors.npz" in source
    assert '"rank-bm25>=0.2.2,<1"' in source
    assert "B1_FEATURE_AGGREGATE" in source
    assert "heldout_accessed" in source and '"btc_mutated": False' in source
    assert "path.is_file() and path != complete_path" in source
    assert source.index("for ordinal, path in enumerate(publish_files, 1)") < source.index(
        "research.put(complete_path, complete_remote)"
    )
    assert "torch.cuda" not in source
    assert "eval/queryset_heldout" not in source


def test_split_and_manifest_contract_are_fail_closed() -> None:
    ns = _helper_namespace()
    rows = [
        {"video_id": "V1", "frame_idx": index} for index in range(4)
    ] + [
        {"video_id": "V2", "frame_idx": index} for index in range(3)
    ] + [
        {"video_id": "V3", "frame_idx": index} for index in range(2)
    ]
    partitions, loads = ns["split_b1_rows"](rows)
    assert loads == {"quang": 4, "trong": 5}
    assert set(partitions["quang"]) | set(partitions["trong"]) == {"V1", "V2", "V3"}
    assert set(partitions["quang"]).isdisjoint(partitions["trong"])

    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "pipeline": "easyocr",
        "runner": "quang",
        "assigned_videos": 1,
        "verified_videos": 1,
        "assigned_frames": 4,
        "verified_frames": 4,
        "nb24_complete_sha256": "n" * 64,
    }
    manifest["manifest_payload_hash"] = sha256_json(manifest)
    ns["validate_runner_manifest"](
        manifest,
        pipeline="easyocr",
        runner="quang",
        expected_videos=1,
        expected_frames=4,
        hash_json=sha256_json,
    )
    manifest["verified_frames"] = 3
    with pytest.raises(RuntimeError, match="Invalid signed payload"):
        ns["validate_runner_manifest"](
            manifest,
            pipeline="easyocr",
            runner="quang",
            expected_videos=1,
            expected_frames=4,
            hash_json=sha256_json,
        )


def test_vector_and_text_contract_reject_drift(tmp_path: Path) -> None:
    ns = _helper_namespace()
    path = tmp_path / "part.npz"
    ids = ["L21_V001#000001", "L21_V001#000002"]
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.savez_compressed(path, ids=np.asarray(ids, dtype=np.str_), vectors=vectors)
    observed = ns["inspect_vector_payload"](
        path, expected_ids=ids, dim=2, norm_atol=1e-6
    )
    assert observed.shape == (2, 2)
    with pytest.raises(RuntimeError, match="identity/order drift"):
        ns["inspect_vector_payload"](
            path, expected_ids=list(reversed(ids)), dim=2, norm_atol=1e-6
        )

    assert ns["validate_text_payload"](
        {ids[0]: " hello ", ids[1]: ""},
        allowed_ids=set(ids),
        artifact="qwen_captions",
    ) == {ids[0]: "hello"}
    with pytest.raises(RuntimeError, match="unexpected ids"):
        ns["validate_text_payload"](
            {"other": "text"}, allowed_ids=set(ids), artifact="qwen_captions"
        )
