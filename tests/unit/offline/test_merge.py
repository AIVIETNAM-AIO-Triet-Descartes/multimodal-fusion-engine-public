"""Locks the merge — mfe.offline.merge.

Every refusal in §6.4 gets a test that proves it actually stops the merge, because
a refusal that does not fire is worse than no check: it reads like coverage. The
whole file runs against `LocalDirTransport`, so it exercises the real code path a
Modal run takes, minus the network.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mfe.offline.bundle import BundleLayout, BundleWriter, ShardPlan
from mfe.offline.merge import merge_bundle
from mfe.offline.transport import LocalDirTransport

# Two videos, two members, two frames each. Small enough to reason about by hand,
# which is the only way a duplicate-key test means anything.
FRAMES = {"L21_V001": [0, 90], "L23_V001": [0, 125]}
PLAN = ShardPlan(
    bundle="aic_btc_v1",
    keyframe_set="btc",
    calibration_video="L21_V001",
    members={"triet": ["L21_V001"], "trong": ["L23_V001"]},
    stages=["n2_subtitles", "n4_siglip"],
)
LAYOUT = BundleLayout("aic_btc_v1")


def key(video_id: str, frame: int) -> str:
    return f"{video_id}#{frame:06d}"


@pytest.fixture
def bundle(tmp_path):
    """A complete, valid bundle. Each test breaks exactly one thing."""
    transport = LocalDirTransport(tmp_path / "volume")
    for video_id, frames in FRAMES.items():
        member = PLAN.owner_of(video_id)
        writer = BundleWriter(transport, LAYOUT, member, tmp_path / f"work_{member}")

        csv_path = tmp_path / f"{video_id}.csv"
        rows = ["n,pts_time,fps,frame_idx"]
        rows += [f"{i + 1},{f / 30:.4f},30.0,{f}" for i, f in enumerate(frames)]
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        writer.write_file(csv_path, LAYOUT.map_keyframes(video_id))

        subs = writer.write_text_part(
            "subtitles",
            video_id,
            {key(video_id, f): f"loi thoai {f}" for f in frames},
            model="faster-whisper large-v3",
            params={"beam_size": 5},
        )
        vecs = writer.write_vector_part(
            "siglip_vectors",
            video_id,
            [key(video_id, f) for f in frames],
            np.ones((len(frames), 1152), dtype=np.float32),
            model="google/siglip-so400m-patch14-384",
        )
        writer.write_part_manifest(video_id, [subs, vecs], keyframes=len(frames))
    return transport


def run(bundle, tmp_path, plan=PLAN, **kwargs):
    return merge_bundle(bundle, plan, tmp_path / "merge", layout=LAYOUT, **kwargs)


class TestHappyPath:
    def test_clean_bundle_merges(self, bundle, tmp_path):
        report = run(bundle, tmp_path)
        assert report.ok, [str(r) for r in report.refusals]
        assert report.artifacts["subtitles"].rows == 4
        assert report.artifacts["siglip_vectors"].rows == 4
        assert report.authors == ["triet", "trong"]

    def test_flat_files_are_what_the_ingest_scripts_read(self, bundle, tmp_path):
        run(bundle, tmp_path)
        out = tmp_path / "merge" / "out" / "data" / "vectors"
        payload = json.loads((out / "subtitles.json").read_text(encoding="utf-8"))
        assert set(payload) == {key(v, f) for v, fs in FRAMES.items() for f in fs}

        vectors = np.load(out / "siglip_vectors.npz", allow_pickle=False)
        assert set(vectors.files) == {"ids", "vectors"}
        assert vectors["vectors"].shape == (4, 1152)
        assert vectors["vectors"].dtype == np.float32

    def test_manifest_records_the_agreed_fingerprint(self, bundle, tmp_path):
        run(bundle, tmp_path)
        manifest = json.loads(
            (tmp_path / "merge" / "out" / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["key_format"] == "keyframe_id"
        assert manifest["counts"] == {"videos": 2, "keyframes": 4}
        assert manifest["fingerprint_check"]["subtitles"] == "faster-whisper large-v3"

    def test_sharded_emit_carries_the_same_rows(self, bundle, tmp_path):
        """Scale changes the output shape, never the checks — otherwise the checks
        get dropped exactly when the data is big enough to need them."""
        report = run(bundle, tmp_path, emit="sharded")
        assert report.ok
        shard_dir = tmp_path / "merge" / "out" / "data" / "vectors" / "siglip_vectors"
        parts = sorted(shard_dir.glob("part-*.npz"))
        assert len(parts) == 2
        assert sum(len(np.load(p, allow_pickle=False)["ids"]) for p in parts) == 4


class TestRefusals:
    def test_duplicate_key_between_two_members(self, bundle, tmp_path):
        """The decisive argument for merging at all (§6.3).

        Loaded shard by shard this is an upsert: the later one wins and nothing
        is logged. Here it stops the merge before anything reaches a store.
        """
        writer = BundleWriter(bundle, LAYOUT, "trong", tmp_path / "w")
        writer.transport.put(
            _json_file(tmp_path, {key("L23_V001", 0): "trung khoa"}),
            LAYOUT.part("subtitles", "L23_V001"),
        )
        # Re-point trong's part at a key L21_V001 already owns.
        writer.transport.put(
            _json_file(tmp_path, {key("L21_V001", 0): "trung khoa"}),
            LAYOUT.part("subtitles", "L23_V001"),
        )
        report = run(bundle, tmp_path)
        assert any(r.kind == "khoa_trung" for r in report.refusals)

    def test_model_mismatch_between_shards(self, bundle, tmp_path):
        """Sharding by video buys parallelism and pays with five machines running
        one model. This is the only moment that mixture is visible."""
        _patch_manifest(
            bundle, tmp_path, "L23_V001", lambda m: m["artifacts"]["subtitles"].update(
                {"model": "faster-whisper medium"}
            )
        )
        report = run(bundle, tmp_path)
        assert any(r.kind == "model_lech" for r in report.refusals)

    def test_numeric_package_mismatch(self, bundle, tmp_path):
        _patch_manifest(
            bundle, tmp_path, "L23_V001", lambda m: m["env"].update({"transformers": "0.0.1"})
        )
        report = run(bundle, tmp_path)
        assert any(r.kind == "env_lech" for r in report.refusals)

    def test_non_numeric_package_mismatch_only_warns(self, bundle, tmp_path):
        """`modal` does not change a vector, so it is provenance, not a gate.
        Refusing on it would train people to pass --force."""
        _patch_manifest(bundle, tmp_path, "L23_V001", lambda m: m["env"].update({"modal": "9.9"}))
        report = run(bundle, tmp_path)
        assert report.ok
        assert any("modal" in w for w in report.warnings)

    def test_video_written_by_the_wrong_member(self, bundle, tmp_path):
        """Detection, not prevention — one Modal workspace gives no write
        boundary (§6.5). It turns the yaml convention into a machine check."""
        _patch_manifest(bundle, tmp_path, "L23_V001", lambda m: m.update({"author": "quang"}))
        report = run(bundle, tmp_path)
        assert any(r.kind == "sai_tac_gia" for r in report.refusals)

    def test_missing_part(self, bundle, tmp_path):
        """A bundle short three videos looks exactly like a complete one unless
        somebody counts — the same reason LoadReport counts empties."""
        plan = ShardPlan(
            bundle=PLAN.bundle,
            keyframe_set="btc",
            calibration_video="L21_V001",
            members={**PLAN.members, "quang": ["L25_V001"]},
            stages=PLAN.stages,
        )
        with pytest.raises(FileNotFoundError):
            run(bundle, tmp_path, plan=plan)

    def test_key_outside_the_bundles_own_keyframe_set(self, bundle, tmp_path):
        """Frame 91 is in the video but is not a keyframe. Letting it through puts
        a key in Milvus that the Relational Map cannot resolve — which surfaces as
        a raise in the middle of scoring, not here."""
        bundle.put(
            _json_file(tmp_path, {key("L21_V001", 91): "khong phai keyframe"}),
            LAYOUT.part("subtitles", "L21_V001"),
        )
        report = run(bundle, tmp_path)
        assert any(r.kind == "khoa_ngoai_bo_keyframe" for r in report.refusals)

    def test_wrong_vector_width(self, bundle, tmp_path):
        path = tmp_path / "bad.npz"
        np.savez_compressed(
            path,
            ids=np.asarray([key("L21_V001", 0)], dtype=np.str_),
            vectors=np.ones((1, 512), dtype=np.float32),
        )
        bundle.put(path, LAYOUT.part("siglip_vectors", "L21_V001"))
        report = run(bundle, tmp_path)
        assert any(r.kind == "so_chieu_lech" for r in report.refusals)

    def test_missing_part_manifest(self, bundle, tmp_path):
        report = run(bundle, tmp_path)
        assert report.ok
        # Remove one and it must be noticed, not shrugged off.
        (bundle.root / LAYOUT.part_manifest("L23_V001").lstrip("/")).unlink()
        report = run(bundle, tmp_path)
        assert any(r.kind == "thieu_manifest" for r in report.refusals)


def _json_file(tmp_path, payload):
    path = tmp_path / "tmp_part.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _patch_manifest(transport, tmp_path, video_id, mutate):
    local = tmp_path / "patch.json"
    transport.get(LAYOUT.part_manifest(video_id), local)
    manifest = json.loads(local.read_text(encoding="utf-8"))
    mutate(manifest)
    local.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    transport.put(local, LAYOUT.part_manifest(video_id))
