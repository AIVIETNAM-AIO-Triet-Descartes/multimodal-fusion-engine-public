from __future__ import annotations

import ast
import json
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mfe.offline.keyframe_augmentation.common import sha256_file, sha256_json
from mfe.offline.transport import LocalDirTransport

GENERATOR = Path("scratchpad/gen_nb28_btc_b1_bge_m3_producer.py")
NOTEBOOK = Path("notebooks/btc/28-btc-b1-bge-m3-producer.ipynb")


def _regenerate() -> None:
    runpy.run_path(str(GENERATOR), run_name="__main__")


def _payload() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(*, code_only: bool = False) -> str:
    cells = _payload()["cells"]
    if code_only:
        cells = [cell for cell in cells if cell["cell_type"] == "code"]
    return "\n".join("".join(cell.get("source", [])) for cell in cells)


def _helpers() -> dict[str, Any]:
    cells = _payload()["cells"]
    cell = next(
        value
        for value in cells
        if "bge-contract-helpers" in value.get("metadata", {}).get("tags", [])
    )
    namespace: dict[str, Any] = {"__name__": "nb28_contract_test", "np": np}
    exec(compile("".join(cell["source"]), "nb28-contract-helpers", "exec"), namespace)
    return namespace


def _signed(payload: dict[str, Any], field: str) -> dict[str, Any]:
    out = dict(payload)
    out[field] = sha256_json(out)
    return out


def _write_remote_json(
    transport: LocalDirTransport,
    tmp_path: Path,
    remote: str,
    payload: dict[str, Any],
) -> None:
    local = tmp_path / "upload" / Path(remote).name
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    transport.put(local, remote)


def _unit_vectors(rows: int, dim: int = 1024) -> np.ndarray:
    vectors = np.zeros((rows, dim), dtype=np.float32)
    for index in range(rows):
        vectors[index, index % dim] = 1.0
    return vectors


def test_generator_is_idempotent_and_all_code_cells_compile() -> None:
    _regenerate()
    first = NOTEBOOK.read_bytes()
    _regenerate()
    assert NOTEBOOK.read_bytes() == first

    payload = _payload()
    assert payload["nbformat"] == 4
    assert all(not cell.get("outputs") for cell in payload["cells"])
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"nb28-cell-{index}", "exec")


def test_first_cell_only_contains_the_six_runner_parameters() -> None:
    first = _payload()["cells"][0]
    assert first["cell_type"] == "code"
    tree = ast.parse("".join(first["source"]))
    names = []
    for statement in tree.body:
        assert isinstance(statement, ast.Assign)
        assert len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name)
        names.append(statement.targets[0].id)
    assert names == [
        "RUNNER",
        "RUNNERS",
        "RUN_SCOPE",
        "CONFIRM_SHARD",
        "SESSION_MAX_MINUTES",
        "BGE_BATCH_SIZE",
    ]


def test_notebook_has_no_forbidden_data_or_model_paths() -> None:
    source = _source(code_only=True)
    forbidden = (
        "eval/queryset_heldout",
        "keyframes-btc",
        "requirements-kaggle-btc-caption",
        "from vllm",
        "import vllm",
        "stage_n6_caption_qwen",
        "stage_n6_ocr_qwen25",
        "PIL",
        "cv2",
        "tarfile",
        ".jpg",
        ".mp4",
    )
    assert all(value not in source for value in forbidden)
    assert "requirements-kaggle-btc-bge.txt" in source
    assert "stage_n6_bge_m3" in source
    assert "qwen_captions" in source


def test_runner_split_is_deterministic_disjoint_and_exactly_5301_frames() -> None:
    split = _helpers()["split_b1_rows"]
    rows = []
    for frame_idx in range(5301):
        video_id = f"L21_V{frame_idx % 688 + 1:03d}"
        rows.append(
            {
                "video_id": video_id,
                "frame_idx": frame_idx,
                "keyframe_id": f"{video_id}#{frame_idx:06d}",
                "arms": ["B1"],
            }
        )
    first, first_loads = split(rows, ("quang", "trong"))
    second, second_loads = split(list(reversed(rows)), ("quang", "trong"))
    assert first == second
    assert first_loads == second_loads
    quang_ids = {row["keyframe_id"] for values in first["quang"].values() for row in values}
    trong_ids = {row["keyframe_id"] for values in first["trong"].values() for row in values}
    assert not quang_ids & trong_ids
    assert len(quang_ids | trong_ids) == 5301
    assert abs(first_loads["quang"] - first_loads["trong"]) <= 1


def test_shard_lock_requires_exact_runner_confirmation() -> None:
    validate = _helpers()["validate_run_lock"]
    validate("quang", ("quang", "trong"), "PREFLIGHT", "")
    validate("trong", ("quang", "trong"), "SHARD", "trong")
    with pytest.raises(RuntimeError, match="SHARD is locked"):
        validate("quang", ("quang", "trong"), "SHARD", "")


def test_qwen_partial_manifest_is_rejected() -> None:
    validate = _helpers()["validate_qwen_runner_manifest"]
    manifest = _signed(
        {
            "status": "PARTIAL",
            "pipeline": "qwen25",
            "runner": "quang",
            "assigned_videos": 2,
            "verified_videos": 1,
            "assigned_frames": 3,
            "verified_frames": 2,
            "nb24_complete_sha256": "n24",
        },
        "manifest_payload_hash",
    )
    with pytest.raises(RuntimeError, match="not COMPLETE"):
        validate(
            manifest,
            runner="quang",
            assigned_videos=2,
            assigned_frames=3,
            expected_nb24_sha="n24",
            hash_json=sha256_json,
        )


def test_qwen_marker_and_caption_artifact_hash_are_verified(tmp_path: Path) -> None:
    helpers = _helpers()
    load_qwen = helpers["load_and_validate_qwen_video"]
    qwen_root = helpers["QWEN_ROOT"]
    transport = LocalDirTransport(tmp_path / "remote")
    video_id = "L21_V001"
    expected_ids = [f"{video_id}#000010", f"{video_id}#000020"]
    caption_remote = f"{qwen_root}/parts/qwen_captions/{video_id}.json"
    ocr_remote = f"{qwen_root}/parts/qwen_ocr/{video_id}.json"
    captions = {expected_ids[0]: "A person walks", expected_ids[1]: "A red sign"}
    _write_remote_json(transport, tmp_path, caption_remote, captions)
    _write_remote_json(transport, tmp_path, ocr_remote, {})
    caption_sha = transport.sha256(caption_remote)
    ocr_sha = transport.sha256(ocr_remote)
    marker = _signed(
        {
            "status": "COMPLETE",
            "pipeline": "qwen25",
            "runner": "quang",
            "video_id": video_id,
            "frame_count": 2,
            "keyframe_ids_sha256": sha256_json(expected_ids),
            "nb24_complete_sha256": "n24",
            "artifacts": {
                "qwen_captions": {
                    "remote": caption_remote,
                    "sha256": caption_sha,
                    "rows": 2,
                },
                "qwen_ocr": {"remote": ocr_remote, "sha256": ocr_sha, "rows": 0},
            },
        },
        "marker_payload_hash",
    )
    marker_remote = f"{qwen_root}/markers/quang/{video_id}.json"
    _write_remote_json(transport, tmp_path, marker_remote, marker)
    loaded, provenance = load_qwen(
        transport,
        tmp_path / "work",
        runner="quang",
        video_id=video_id,
        expected_ids=expected_ids,
        expected_nb24_sha="n24",
        hash_json=sha256_json,
        hash_file=sha256_file,
    )
    assert loaded == captions
    assert provenance["qwen_caption_sha256"] == caption_sha

    _write_remote_json(transport, tmp_path, caption_remote, {expected_ids[0]: "changed"})
    with pytest.raises(RuntimeError, match="remote SHA mismatch"):
        load_qwen(
            transport,
            tmp_path / "work-hash",
            runner="quang",
            video_id=video_id,
            expected_ids=expected_ids,
            expected_nb24_sha="n24",
            hash_json=sha256_json,
            hash_file=sha256_file,
        )

    _write_remote_json(transport, tmp_path, caption_remote, captions)
    marker["frame_count"] = 1  # payload hash is now stale
    _write_remote_json(transport, tmp_path, marker_remote, marker)
    with pytest.raises(RuntimeError, match="payload hash mismatch"):
        load_qwen(
            transport,
            tmp_path / "work2",
            runner="quang",
            video_id=video_id,
            expected_ids=expected_ids,
            expected_nb24_sha="n24",
            hash_json=sha256_json,
            hash_file=sha256_file,
        )


def _install_bge_artifact(
    tmp_path: Path,
    transport: LocalDirTransport,
    *,
    ids: list[str],
    vectors: np.ndarray,
    vector_remote: str,
    marker_remote: str,
    fields: dict[str, Any],
) -> None:
    local_npz = tmp_path / "artifact.npz"
    np.savez_compressed(local_npz, ids=np.asarray(ids, dtype=np.str_), vectors=vectors)
    transport.put(local_npz, vector_remote)
    marker = _signed(
        {
            **fields,
            "artifact": {"remote": vector_remote, "sha256": sha256_file(local_npz)},
        },
        "marker_payload_hash",
    )
    _write_remote_json(transport, tmp_path, marker_remote, marker)


def test_valid_bge_marker_resumes_and_invalid_npz_is_pending(tmp_path: Path) -> None:
    completed = _helpers()["completed"]
    transport = LocalDirTransport(tmp_path / "remote")
    video_id = "L21_V002"
    ids = [f"{video_id}#000001", f"{video_id}#000002"]
    vector_remote = f"/research/bge/parts/dense_vectors/{video_id}.npz"
    marker_remote = f"/research/bge/markers/quang/{video_id}.json"
    fields = {"status": "COMPLETE", "pipeline": "bge_m3", "runner": "quang", "video_id": video_id}
    fields["repo_commit"] = "a" * 40
    _install_bge_artifact(
        tmp_path,
        transport,
        ids=ids,
        vectors=_unit_vectors(2),
        vector_remote=vector_remote,
        marker_remote=marker_remote,
        fields=fields,
    )
    assert completed(
        transport,
        tmp_path / "work",
        marker_remote=marker_remote,
        expected_marker_fields=fields,
        vector_remote=vector_remote,
        expected_ids=ids,
        video_id=video_id,
        hash_json=sha256_json,
        hash_file=sha256_file,
    ) is not None

    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not-an-npz")
    transport.put(corrupt, vector_remote)
    assert completed(
        transport,
        tmp_path / "work-hash-bad",
        marker_remote=marker_remote,
        expected_marker_fields=fields,
        vector_remote=vector_remote,
        expected_ids=ids,
        video_id=video_id,
        hash_json=sha256_json,
        hash_file=sha256_file,
    ) is None

    bad = _unit_vectors(2)
    bad[0] *= 2.0
    _install_bge_artifact(
        tmp_path,
        transport,
        ids=ids,
        vectors=bad,
        vector_remote=vector_remote,
        marker_remote=marker_remote,
        fields=fields,
    )
    assert completed(
        transport,
        tmp_path / "work-bad",
        marker_remote=marker_remote,
        expected_marker_fields=fields,
        vector_remote=vector_remote,
        expected_ids=ids,
        video_id=video_id,
        hash_json=sha256_json,
        hash_file=sha256_file,
    ) is None


@pytest.mark.parametrize(
    ("ids", "vectors"),
    [
        (["L21_V003#000002", "L21_V003#000001"], _unit_vectors(2)),
        (["L21_V003#000001", "L21_V003#000002"], _unit_vectors(2, dim=8)),
        (["L21_V003#000001", "L21_V003#000002"], _unit_vectors(2).astype(np.float64)),
        (["L21_V003#000001", "L21_V003#000002"], np.full((2, 1024), np.nan, dtype=np.float32)),
    ],
)
def test_npz_id_dimension_dtype_or_finite_drift_is_rejected(
    tmp_path: Path,
    ids: list[str],
    vectors: np.ndarray,
) -> None:
    inspect = _helpers()["inspect_bge_npz"]
    expected = ["L21_V003#000001", "L21_V003#000002"]
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, ids=np.asarray(ids, dtype=np.str_), vectors=vectors)
    with pytest.raises(RuntimeError):
        inspect(path, expected_ids=expected, video_id="L21_V003")


def test_runner_manifest_complete_only_when_the_whole_shard_is_verified() -> None:
    build = _helpers()["build_runner_manifest"]
    assigned = {
        "L21_V001": [{"keyframe_id": "L21_V001#000001"}],
        "L21_V002": [{"keyframe_id": "L21_V002#000001"}],
    }
    marker = {"caption_row_count": 1, "vector_row_count": 1}
    common = {
        "runner": "quang",
        "assigned": assigned,
        "failures": [],
        "environment": {"torch": "x"},
        "repo_commit": "a" * 40,
        "model_revision": "r",
        "model_fingerprint": "f",
    }
    partial = build(verified={"L21_V001": marker}, **common)
    complete = build(verified={"L21_V001": marker, "L21_V002": marker}, **common)
    assert partial["status"] == "PARTIAL"
    assert complete["status"] == "COMPLETE"
    assert complete["verified_frames"] == complete["assigned_frames"] == 2


def test_model_factory_is_cached_across_videos() -> None:
    load_model = _helpers()["load_bge_model"]
    cache: dict[str, object] = {}
    calls = 0

    def factory() -> object:
        nonlocal calls
        calls += 1
        return object()

    def fake_cached(key: str, build: Any) -> object:
        if key not in cache:
            cache[key] = build()
        return cache[key]

    first = load_model(factory, fake_cached)
    second = load_model(factory, fake_cached)
    assert first is second
    assert calls == 1
