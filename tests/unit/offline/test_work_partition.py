from pathlib import Path

import pytest
import yaml

from mfe.offline.bundle import load_shard_plan
from mfe.offline.work_partition import load_worker_slice

ROOT = Path(__file__).resolve().parents[3]
SHARDS = "config/ingest_shards_batch1_btc_ocr2.yaml"
PARTITION = ROOT / "config" / "ingest_workers_batch1_btc_ocr2.yaml"


def _plan():
    return load_shard_plan(ROOT / SHARDS)


def test_real_partition_is_disjoint_complete_and_never_touches_quang():
    plan = _plan()
    kaggle = load_worker_slice(
        PARTITION, plan, member="trong", worker="kaggle", expected_shards=SHARDS
    )
    modal = load_worker_slice(
        PARTITION, plan, member="trong", worker="modal", expected_shards=SHARDS
    )

    assert set(kaggle.zip_names).isdisjoint(modal.zip_names)
    assert set(kaggle.videos).isdisjoint(modal.videos)
    assert set(kaggle.videos) | set(modal.videos) == set(plan.videos_for("trong"))
    assert (len(kaggle.videos), len(modal.videos)) == (216, 188)
    assert (set(kaggle.videos) | set(modal.videos)).isdisjoint(plan.videos_for("quang"))

    overlay = yaml.safe_load(PARTITION.read_text(encoding="utf-8"))
    inventory = yaml.safe_load(
        (ROOT / "config" / "zips_keyframes_btc.yaml").read_text(encoding="utf-8")
    )["zips"]
    measured = {}
    for worker_slice in (kaggle, modal):
        measured[worker_slice.worker] = {
            "videos": len(worker_slice.videos),
            "keyframes": sum(
                sum(inventory[name]["counts"].values()) for name in worker_slice.zip_names
            ),
            "zip_bytes": sum(inventory[name]["bytes"] for name in worker_slice.zip_names),
        }
    assert measured == overlay["expected"]
    ratio = measured["modal"]["keyframes"] / measured["kaggle"]["keyframes"]
    assert ratio == pytest.approx(1.491, abs=0.001)


def test_partition_rejects_a_zip_owned_by_quang(tmp_path):
    raw = yaml.safe_load(PARTITION.read_text(encoding="utf-8"))
    raw["workers"]["modal"]["zips"].append("Keyframes_L22.zip")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Khong duoc cham vao phan cua Quang"):
        load_worker_slice(
            path, _plan(), member="trong", worker="modal", expected_shards=SHARDS
        )


def test_partition_rejects_overlap_between_workers(tmp_path):
    raw = yaml.safe_load(PARTITION.read_text(encoding="utf-8"))
    raw["workers"]["modal"]["zips"].append("Keyframes_L24.zip")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="bi giao cho ca"):
        load_worker_slice(
            path, _plan(), member="trong", worker="modal", expected_shards=SHARDS
        )
