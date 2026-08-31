from __future__ import annotations

import json

import numpy as np
import pytest

from mfe.offline.augmentation.evaluation import VariantRun, dev_gate, heldout_gate
from mfe.offline.augmentation.overlay import load_overlay_flat_bundle


def write_flat(root, ids, vectors, *, text=None):
    data = root / "data" / "vectors"
    data.mkdir(parents=True)
    np.savez_compressed(data / "siglip_vectors.npz", ids=np.array(ids), vectors=vectors)
    if text is not None:
        (data / "dense_captions.json").write_text(json.dumps(text), encoding="utf-8")


def run(name, scores, latency=100.0):
    return VariantRun(
        name=name,
        query_ids=("q1", "q2", "q3"),
        query_types=("KIS", "QA", "TRAKE"),
        scores=tuple(scores),
        latency_ms=(latency, latency, latency),
    )


def test_overlay_reuses_base_and_filters_delta_by_allowlist(tmp_path):
    base = tmp_path / "base"
    delta = tmp_path / "delta"
    write_flat(base, ["L21_V001#000001"], np.array([[3.0, 4.0]]))
    write_flat(
        delta,
        ["L21_V001#000002", "L21_V001#000003"],
        np.array([[0.0, 2.0], [1.0, 1.0]]),
        text={"L21_V001#000002": "target", "L21_V001#000003": "excluded"},
    )

    bundle = load_overlay_flat_bundle(base, delta, ["L21_V001#000002"])

    assert bundle.vector_keys["siglip"] == [
        "L21_V001#000001",
        "L21_V001#000002",
    ]
    assert np.linalg.norm(bundle.vectors["siglip"], axis=1) == pytest.approx([1.0, 1.0])
    assert bundle.texts["caption"] == {"L21_V001#000002": "target"}


def test_overlay_fails_when_selected_frame_has_no_required_siglip(tmp_path):
    base = tmp_path / "base"
    delta = tmp_path / "delta"
    write_flat(base, ["L21_V001#000001"], np.array([[1.0, 0.0]]))
    write_flat(delta, ["L21_V001#000002"], np.array([[0.0, 1.0]]))

    with pytest.raises(ValueError, match="missing required siglip"):
        load_overlay_flat_bundle(base, delta, ["L21_V001#000099"])


def test_balanced_dev_gate_checks_score_random_control_cap_and_latency():
    baseline = run("B0", (0.4, 0.5, 0.6))
    b3 = run("B3", (0.42, 0.52, 0.62), latency=105.0)
    b4 = run("B4", (0.401, 0.501, 0.601), latency=104.0)

    gate = dev_gate(
        baseline,
        b3,
        b4,
        target_query_ids={"q1"},
        delta_frames=3,
        base_frames=100,
        cap_ratio=0.03,
        noninferiority_margin=-0.005,
        max_latency_ratio=1.10,
        bootstrap_samples=500,
        bootstrap_seed=2026,
    )

    assert gate["go"] is True
    assert gate["delta_ratio"] == pytest.approx(0.03)
    assert gate["bootstrap"]["lower_95"] > 0


def test_heldout_gate_exposes_only_aggregates_and_catches_type_regression():
    baseline = run("B0", (0.4, 0.5, 0.6))
    treatment = run("B3", (0.45, 0.60, 0.58))

    gate = heldout_gate(
        baseline,
        treatment,
        by_type_margin=-0.01,
        max_latency_ratio=1.10,
    )

    assert gate["go"] is False
    assert gate["split"] == "heldout"
    assert "query_ids" not in gate and "scores" not in gate
    assert any("TRAKE" in reason for reason in gate["reasons"])
