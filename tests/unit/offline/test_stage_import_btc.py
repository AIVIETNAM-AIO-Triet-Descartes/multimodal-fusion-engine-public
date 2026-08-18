"""Locks the BTC import stage — mfe.offline.stages.stage_n0_import_btc.

This one needs no stub: it moves files, so it can run for real on a laptop. What
it must get right is not the moving but the refusing — an import that finishes
with fewer images than the mapping claims, or with a different number of object
files than images, produces a bundle whose own files disagree with each other.
Both are silent downstream: `clip-features` is an (N, D) array indexed by row
order, and `read_object_labels` addresses by the same ordinal as the image.
"""

from __future__ import annotations

import json

import pytest

from mfe.offline.bundle import BundleLayout, BundleWriter, StageStatus
from mfe.offline.stages import StageContext, read_keyframes, stage_n0_import_btc
from mfe.offline.transport import LocalDirTransport

VIDEO = "L21_V001"
FRAMES = [(1, 0.0, 0), (2, 3.0, 90), (3, 8.7, 261)]


@pytest.fixture
def source(tmp_path):
    """A miniature copy of what BTC ships for one video."""
    keyframes = tmp_path / "keyframes" / VIDEO
    objects = tmp_path / "objects" / VIDEO
    keyframes.mkdir(parents=True)
    objects.mkdir(parents=True)
    for n, _, _ in FRAMES:
        (keyframes / f"{n:03d}.jpg").write_bytes(b"\xff\xd8" + bytes(200))
        (objects / f"{n:03d}.json").write_text(
            json.dumps({"detection_class_entities": ["Person"], "detection_scores": ["0.9"]}),
            encoding="utf-8",
        )
    csv_path = tmp_path / f"{VIDEO}.csv"
    rows = ["n,pts_time,fps,frame_idx"]
    rows += [f"{n},{pts},30.0,{frame}" for n, pts, frame in FRAMES]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return keyframes, objects, csv_path


@pytest.fixture
def ctx(tmp_path, source):
    keyframes, objects, csv_path = source
    transport = LocalDirTransport(tmp_path / "volume")
    layout = BundleLayout("aic_btc_b1_v1")
    writer = BundleWriter(transport, layout, "triet", tmp_path / "work")
    return StageContext(
        VIDEO, tmp_path / "unused.mp4", [], writer, layout, tmp_path / "work",
        params={
            "keyframes_dir": str(keyframes),
            "objects_dir": str(objects),
            "map_csv": str(csv_path),
        },
    )


class TestImport:
    def test_uploads_images_objects_and_mapping(self, ctx, tmp_path):
        marker = stage_n0_import_btc(ctx)
        assert (marker["keyframes"], marker["images"], marker["objects"]) == (3, 3, 3)
        assert marker["fps"] == 30.0

        transport, layout = ctx.writer.transport, ctx.layout
        assert len(transport.listdir(layout.keyframes_dir(VIDEO))) == 3
        assert len(transport.listdir(layout.objects_dir(VIDEO))) == 3
        assert transport.exists(layout.map_keyframes(VIDEO))

    def test_the_mapping_round_trips_through_the_volume(self, ctx, tmp_path):
        """BTC's own numbers, checked against the contract rather than assumed.

        Here `floor(pts_time * fps)` is a genuine check — unlike on the DAKE path,
        where we generate `pts_time` ourselves — so it is the thing that would
        catch an fps read off the wrong video.
        """
        stage_n0_import_btc(ctx)
        local = tmp_path / "back.csv"
        ctx.writer.transport.get(ctx.layout.map_keyframes(VIDEO), local)
        keyframes = read_keyframes(local, tmp_path / "work")
        assert [kf.frame_idx for kf in keyframes] == [f for _, _, f in FRAMES]

    def test_refuses_when_an_image_is_missing(self, ctx, source):
        """The mapping claims N rows and `clip-features` is an (N, D) array indexed
        by row order, so a short import is not a partial result — it is a bundle
        whose files disagree."""
        keyframes, _, _ = source
        (keyframes / "002.jpg").unlink()
        with pytest.raises(RuntimeError, match="thieu"):
            stage_n0_import_btc(ctx)

    def test_refuses_when_objects_and_images_disagree(self, ctx, source):
        """`read_object_labels` addresses by the same ordinal as the image, so a
        count mismatch means some keyframe silently has no labels — and BM25 is
        sensitive to term counts, which would skew the very A/B this bundle exists
        to enable."""
        _, objects, _ = source
        (objects / "003.json").unlink()
        with pytest.raises(RuntimeError, match="objects"):
            stage_n0_import_btc(ctx)

    def test_no_marker_survives_a_refusal(self, ctx, source):
        keyframes, _, _ = source
        (keyframes / "001.jpg").unlink()
        with pytest.raises(RuntimeError):
            stage_n0_import_btc(ctx)
        status = StageStatus.load(ctx.writer.transport, ctx.layout, "triet")
        assert not status.is_done("n0_import_btc", VIDEO)

    def test_objects_are_optional(self, ctx):
        """A run can legitimately import keyframes alone; the count check only
        fires when objects are actually supplied."""
        ctx.params["objects_dir"] = ""
        marker = stage_n0_import_btc(ctx)
        assert marker["images"] == 3
        assert marker["objects"] == 0


class TestShippedPlan:
    def test_the_btc_plan_covers_batch_1_disjointly(self):
        from mfe.offline.bundle import load_shard_plan

        plan = load_shard_plan("config/ingest_shards_batch1_btc.yaml")
        assert plan.keyframe_set == "btc"
        assert plan.stages == ["n0_import_btc"]
        assert len(plan.videos) == len(set(plan.videos)) == 873

    def test_the_btc_bundle_is_not_the_dake_bundle(self):
        """Two keyframe sets must never share a bundle: images and `objects/` are
        both addressed by the extraction ordinal `n`, so a mixed bundle stops
        matching its own `map-keyframes` (§1.2)."""
        from mfe.offline.bundle import load_shard_plan

        btc = load_shard_plan("config/ingest_shards_batch1_btc.yaml")
        dake = load_shard_plan("config/ingest_shards_batch1_dake.yaml")
        assert btc.bundle != dake.bundle
        assert btc.keyframe_set != dake.keyframe_set
