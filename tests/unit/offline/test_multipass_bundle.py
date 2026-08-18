"""Locks the two things that change when one bundle is filled by several passes.

The BTC bundle is written by three notebooks — `02` imports the keyframes, `03`
adds SigLIP and the two OCR sources, `04` adds subtitles and CLAP — over the same
videos, into the same `aic_btc_b1_v1`. That is a shape the earlier single-pass
code did not have, and it breaks two assumptions that were safe before:

1. a stage may load its model once and be called for hundreds of videos, and
2. `part-manifest/{video}.json` is written more than once for the same video.

Both failures are silent. The first only costs time (measured at 5-16 hours per
member of pure model loading); the second loses data while every marker still
reports success.
"""

from __future__ import annotations

import json

import pytest

from mfe.offline.bundle import BundleLayout, BundleWriter, load_shard_plan
from mfe.offline.stages import cached_model, unload_models
from mfe.offline.transport import LocalDirTransport

VIDEO = "L21_V001"


@pytest.fixture
def writer(tmp_path):
    transport = LocalDirTransport(tmp_path / "volume")
    layout = BundleLayout("aic_btc_b1_v1")
    return BundleWriter(transport, layout, "triet", tmp_path / "work")


def read_manifest(writer, tmp_path, video_id=VIDEO) -> dict:
    local = tmp_path / "read_back.json"
    writer.transport.get(writer.layout.part_manifest(video_id), local)
    return json.loads(local.read_text(encoding="utf-8"))


def marker(artifact: str, rows: int) -> dict:
    return {"artifact": artifact, "video_id": VIDEO, "stage": "x", "rows": rows}


class TestModelCache:
    def setup_method(self):
        unload_models()

    def teardown_method(self):
        unload_models()

    def test_builds_once_and_reuses(self):
        calls = []

        def build():
            calls.append(1)
            return object()

        first = cached_model("siglip", build)
        second = cached_model("siglip", build)
        assert first is second
        assert len(calls) == 1

    def test_different_checkpoints_do_not_collide(self):
        a = cached_model("siglip", lambda: "A")
        b = cached_model("qwen", lambda: "B")
        assert (a, b) == ("A", "B")

    def test_unload_forces_a_rebuild(self):
        """`unload_models` is what a video-major notebook calls between stages —
        it must genuinely let go, or §4.2's VRAM argument stops holding."""
        calls = []
        cached_model("siglip", lambda: calls.append(1))
        unload_models()
        cached_model("siglip", lambda: calls.append(1))
        assert len(calls) == 2

    def test_a_miss_is_the_only_thing_that_calls_build(self):
        """The lambda in each stage closes over `from_pretrained`, so calling it
        on a hit would download and initialise the model anyway — the cache would
        look present and buy nothing."""
        cached_model("siglip", lambda: "first")
        assert cached_model("siglip", lambda: pytest.fail("build chay tren cache hit")) == "first"


class TestManifestMerge:
    def test_a_second_pass_erases_the_first_without_merge(self, writer, tmp_path):
        """The behaviour being guarded against, asserted so it stays deliberate."""
        writer.write_part_manifest(VIDEO, [marker("map_keyframes", 307)], keyframes=307)
        writer.write_part_manifest(VIDEO, [marker("siglip_vectors", 307)], keyframes=307)
        assert set(read_manifest(writer, tmp_path)["artifacts"]) == {"siglip_vectors"}

    def test_merge_keeps_what_earlier_passes_recorded(self, writer, tmp_path):
        writer.write_part_manifest(VIDEO, [marker("map_keyframes", 307)], keyframes=307)
        writer.write_part_manifest(
            VIDEO, [marker("siglip_vectors", 307), marker("ocr_text", 41)],
            keyframes=307, merge=True,
        )
        writer.write_part_manifest(
            VIDEO, [marker("subtitles", 260)], keyframes=307, merge=True
        )
        assert set(read_manifest(writer, tmp_path)["artifacts"]) == {
            "map_keyframes", "siglip_vectors", "ocr_text", "subtitles",
        }

    def test_a_rerun_of_one_stage_replaces_only_that_artifact(self, writer, tmp_path):
        writer.write_part_manifest(
            VIDEO, [marker("map_keyframes", 307), marker("ocr_text", 41)], keyframes=307
        )
        writer.write_part_manifest(VIDEO, [marker("ocr_text", 44)], keyframes=307, merge=True)
        artifacts = read_manifest(writer, tmp_path)["artifacts"]
        assert artifacts["ocr_text"]["rows"] == 44
        assert artifacts["map_keyframes"]["rows"] == 307

    def test_merge_works_when_there_is_nothing_to_merge_with(self, writer, tmp_path):
        """First pass of the first video: no manifest exists yet, and a missing
        one is the normal case rather than an error."""
        writer.write_part_manifest(VIDEO, [marker("map_keyframes", 307)],
                                   keyframes=307, merge=True)
        assert set(read_manifest(writer, tmp_path)["artifacts"]) == {"map_keyframes"}

    def test_no_scratch_file_is_left_behind(self, writer, tmp_path):
        writer.write_part_manifest(VIDEO, [marker("a", 1)], keyframes=1, merge=True)
        writer.write_part_manifest(VIDEO, [marker("b", 1)], keyframes=1, merge=True)
        assert list((tmp_path / "work").glob("prev_manifest_*.json")) == []


class TestThreePassesAgreeOnOwnership:
    """The invariant that makes the three-notebook split safe.

    `part-manifest/{video}.json` records exactly one `author`, and
    `merge.check_ownership` refuses a shard whose author is not the member the
    plan assigns. So if two passes over the same bundle split the videos
    differently, each video gets its author rewritten by whichever pass ran last
    and the merge refuses the whole bundle.

    This is not hypothetical: BTC groups keyframes and videos into different
    archives, and splitting by the video archives gives `bao` 148 videos that
    overlap his 295 keyframe-archive videos in **zero** places. Pass `04` reads
    videos, so splitting it that way is the obvious move — and it is wrong.
    It gets away with the keyframe split because it fetches each video by HTTP
    Range, which makes archive grouping irrelevant to bandwidth.
    """

    PLANS = [
        "config/ingest_shards_batch1_btc.yaml",
        "config/ingest_shards_batch1_btc_images.yaml",
        "config/ingest_shards_batch1_btc_video.yaml",
    ]

    def test_every_pass_assigns_the_same_videos_to_the_same_member(self):
        plans = [load_shard_plan(p) for p in self.PLANS]
        first = plans[0]
        for plan in plans[1:]:
            assert plan.bundle == first.bundle
            assert plan.members == first.members

    def test_the_passes_together_cover_every_stage_except_recap(self):
        stages = [s for p in self.PLANS for s in load_shard_plan(p).stages]
        assert stages == [
            "n0_import_btc",
            "n4_siglip", "n5_ocr_easyocr", "n5_ocr_qwen",
            "n2_subtitles", "n3_audio",
        ]
        # n4_objects: BTC ships them, imported by pass 1.
        # n6_recap / n6_bge_m3: deferred by the Lead.
        assert "n4_objects" not in stages

    def test_every_assigned_video_can_actually_be_fetched(self):
        """Pass `04` needs a Range entry per video, and it reads the assignment
        from a *different* file than the offsets. Nothing else checks that the
        two files cover the same 873 videos."""
        from mfe.offline.source import load_catalogue

        plan = load_shard_plan("config/ingest_shards_batch1_btc_video.yaml")
        catalogue = load_catalogue("config/zips_batch1.yaml")
        assert [v for v in plan.videos if v not in catalogue] == []


class TestZipSizesAreKnown:
    """`zip_bytes` is what turns "wget said OK" into a checked claim.

    `wget -c` resumes with an HTTP `Range` request. If the host answers `200`
    with the whole body instead of `206` with the tail — which a throttling or
    misconfigured server does — wget appends that body after the bytes already
    on disk. The archive is then larger than the real one and corrupt from the
    join, and nothing in the download reports a problem. The only signal is the
    size, and it only exists because `make_zip_inventory.py` measured it.
    """

    @pytest.mark.parametrize(
        "config",
        [
            "config/ingest_shards_batch1_btc.yaml",
            "config/ingest_shards_batch1_btc_images.yaml",
        ],
    )
    def test_every_archive_a_member_downloads_has_a_size(self, config):
        plan = load_shard_plan(config)
        for member in plan.members:
            for name, _ in plan.zips_for(member):
                assert plan.zip_bytes.get(name, 0) > 0, f"{name} khong co kich thuoc"

    def test_the_sizes_match_the_inventory_they_came_from(self):
        import yaml

        book = yaml.safe_load(open("config/zips_keyframes_btc.yaml", encoding="utf-8"))
        plan = load_shard_plan("config/ingest_shards_batch1_btc.yaml")
        assert plan.zip_bytes == {n: b["bytes"] for n, b in book["zips"].items()}
