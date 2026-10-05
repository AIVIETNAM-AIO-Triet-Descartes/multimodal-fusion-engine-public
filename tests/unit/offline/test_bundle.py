"""Locks the bundle protocol — mfe.offline.bundle + mfe.offline.transport.

Runs entirely against `LocalDirTransport`, which does real filesystem writes with
the same path strings Modal sees. So layout, validation and the resume logic are
all exercised here; what is *not* covered is the network itself — partial uploads
and last-write-wins — and §6.2 removes those by construction rather than by test.
"""

from __future__ import annotations

import json
import unicodedata
from unittest import mock

import numpy as np
import pytest

from mfe.contracts.errors import ContractError
from mfe.offline.bundle import (
    ARTIFACTS,
    DIRECTORY_STAGES,
    NUMERIC_PACKAGES,
    OTHER_PACKAGES,
    STAGE_PACKAGES,
    BundleLayout,
    BundleWriter,
    StageStatus,
    check_text_payload,
    check_vectors,
    load_shard_plan,
    round_robin,
)
from mfe.offline.transport import LocalDirTransport

NEWLINE = chr(10)
VIDEO = "L21_V001"


@pytest.fixture
def writer(tmp_path):
    transport = LocalDirTransport(tmp_path / "volume")
    layout = BundleLayout("aic_btc_v1")
    return BundleWriter(transport, layout, "triet", tmp_path / "work")


class TestRegistry:
    def test_text_artifacts_match_what_the_ingest_script_reads(self):
        """Guards the one drift that would cost a whole tier in silence.

        `ingest_sota_text.py` treats a missing file as a valid state, so a bundle
        writing `subtitle.json` while the reader looks for `subtitles.json`
        produces an index short one arm and no error anywhere.
        """
        from mfe.offline.adapters.sota_payloads import SOURCE_FILES

        for artifact in ARTIFACTS.values():
            if artifact.kind == "text" and artifact.source is not None:
                assert SOURCE_FILES[artifact.source] == artifact.flat_name

    def test_a_text_artifact_may_carry_no_source_at_all(self):
        """`source=None` means "written, but nothing ingests it" — a legal state.

        It is how a payload gets generated for measurement before anyone decides
        whether an arm should read it. `paddle_ocr` and `viet_ocr` lived here for
        one day before lượt 4 wired them in, and the next probe will too, so the
        registry must keep accepting it.
        """
        import dataclasses

        from mfe.offline.bundle import _assert_registries_agree

        probe = dataclasses.replace(
            ARTIFACTS["ocr_text"], name="probe_ocr", flat_name="probe_ocr.json", source=None
        )
        with mock.patch.dict(ARTIFACTS, {"probe_ocr": probe}):
            _assert_registries_agree()   # khong duoc ném gi

    def test_every_shipped_text_artifact_is_now_wired_to_an_arm(self):
        """Ngược lại: không artifact nào trong registry hiện đang treo lơ lửng.

        Một artifact có producer nhưng không có `source` là dữ liệu được sinh ra
        rồi không ai đọc — im lặng, và chỉ phát hiện được bằng cách đếm.
        """
        orphans = [
            a.name for a in ARTIFACTS.values() if a.kind == "text" and a.source is None
        ]
        assert orphans == [], f"artifact text khong co arm nao doc: {orphans}"

    def test_a_source_that_disagrees_with_its_filename_still_raises(self):
        """Skipping source-less artifacts must not have opened a hole."""
        import dataclasses

        from mfe.offline.bundle import _assert_registries_agree

        broken = dataclasses.replace(ARTIFACTS["ocr_text"], flat_name="ocr.json")
        with mock.patch.dict(ARTIFACTS, {"ocr_text": broken}):
            with pytest.raises(AssertionError, match="ocr_text"):
                _assert_registries_agree()

    def test_vector_dims_are_the_three_the_stores_accept(self):
        from mfe.stores.production.contracts import SUPPORTED_EMBEDDING_DIMS

        for artifact in ARTIFACTS.values():
            if artifact.kind == "vector":
                assert artifact.dim in SUPPORTED_EMBEDDING_DIMS


class TestShardPlan:
    def test_the_shipped_plan_is_disjoint_and_complete(self):
        plan = load_shard_plan()
        assert len(plan.videos) == 16
        assert len(set(plan.videos)) == 16
        assert plan.owner_of(plan.calibration_video) is not None

    def test_round_robin_deals_873_videos_disjointly(self, tmp_path):
        """Batch 1 is 873 videos, so the assignment has to come from a rule.

        Hand-listing 873 ids across five members guarantees a typo, and a typo
        here means a video belongs to nobody: the merge reports a missing part
        and nothing says why.
        """
        ids = [f"L{g}_V{v:03d}" for g in range(21, 31) for v in range(1, 88)]
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "videos.txt").write_text(
            NEWLINE.join(ids), encoding="utf-8"
        )
        (tmp_path / "config" / "shards.yaml").write_text(
            NEWLINE.join([
                "bundle: b",
                "keyframe_set: btc",
                "calibration_video: L21_V001",
                "assign: round_robin",
                "members: [triet, trong, quang, bao, dat]",
                "video_list: config/videos.txt",
                "stages: [n2_subtitles]",
            ]),
            encoding="utf-8",
        )
        plan = load_shard_plan(tmp_path / "config" / "shards.yaml")
        assert len(plan.videos) == len(set(plan.videos)) == len(ids)
        sizes = [len(v) for v in plan.members.values()]
        assert max(sizes) - min(sizes) <= 1

    def test_round_robin_spreads_programmes_across_members(self):
        """Dealing, not slicing. Ids are grouped by programme and the groups
        differ in typical length, so contiguous blocks would hand one member
        every lecture video."""
        ids = [f"L{g}_V{v:03d}" for g in range(21, 31) for v in range(1, 21)]
        shares = round_robin(ids, ["a", "b", "c", "d", "e"])
        assert len({v.split("_")[0] for v in shares["a"]}) == 10

    def test_round_robin_is_reproducible(self):
        """A re-run must not reassign work somebody already finished."""
        ids = [f"L21_V{v:03d}" for v in range(1, 50)]
        names = ["a", "b", "c"]
        assert round_robin(ids, names) == round_robin(list(reversed(ids)), names)

    def test_rejects_a_video_given_to_two_people(self, tmp_path):
        """Disjointness is the assumption every later simplification rests on:
        it is what makes per-video paths collision-free and what makes a
        duplicate key at merge time a real error rather than a race."""
        path = tmp_path / "shards.yaml"
        path.write_text(
            "bundle: b\nkeyframe_set: btc\ncalibration_video: L21_V001\n"
            "members:\n  a:\n    videos: [L21_V001]\n  b:\n    videos: [L21_V001]\n"
            "stages: [n2_subtitles]\n",
            encoding="utf-8",
        )
        with pytest.raises(ContractError, match="ROI NHAU"):
            load_shard_plan(path)

    def test_rejects_an_unknown_stage(self, tmp_path):
        path = tmp_path / "shards.yaml"
        path.write_text(
            "bundle: b\nkeyframe_set: btc\ncalibration_video: L21_V001\n"
            "members:\n  a:\n    videos: [L21_V001]\nstages: [n9_khong_co]\n",
            encoding="utf-8",
        )
        with pytest.raises(ContractError, match="stages"):
            load_shard_plan(path)


class TestTextPayloadChecks:
    def test_normalises_to_nfc(self):
        """NFD text scores zero on every text arm and raises nothing at all, so
        it is normalised where the file is written, not only where it is read.

        `StorageRecord.validate_text` normalises too, but that only cleans the
        index - the file five people copy around keeps whatever the model emitted.

        The decomposed input is built with NFD rather than typed as escapes, so
        the test carries no hand-counted magic length to get wrong.
        """
        composed = "Điện Biên Phủ"   # Dien Bien Phu, NFC
        decomposed = unicodedata.normalize("NFD", composed)
        assert decomposed != composed

        out = check_text_payload({f"{VIDEO}#000090": decomposed}, VIDEO)
        assert out[f"{VIDEO}#000090"] == composed

    def test_drops_empty_text(self):
        out = check_text_payload({f"{VIDEO}#000090": "   ", f"{VIDEO}#000261": "x"}, VIDEO)
        assert list(out) == [f"{VIDEO}#000261"]

    def test_rejects_a_key_from_another_video(self):
        """A part holds one video. That is what makes the merge a plain union
        with nothing to reconcile."""
        with pytest.raises(ContractError, match="khoa cua"):
            check_text_payload({"L23_V001#000090": "x"}, VIDEO)

    def test_rejects_an_ordinal_key(self):
        """Our notebooks key by frame. An ordinal here would resolve to a real
        but wrong frame downstream — the exact trap §1.2 exists to remove."""
        with pytest.raises(ContractError):
            check_text_payload({f"{VIDEO}_002": "x"}, VIDEO)


class TestVectorChecks:
    def _ids(self, count):
        return [f"{VIDEO}#{i:06d}" for i in range(count)]

    def test_normalises_and_casts_to_float32(self):
        vectors = np.full((3, 4), 2.0, dtype=np.float64)
        out = check_vectors(self._ids(3), vectors, VIDEO, 4)
        assert out.kept.dtype == np.float32
        assert np.allclose(np.linalg.norm(out.kept, axis=1), 1.0)
        assert out.renormalised == 3

    def test_drops_zero_and_nan_rows_and_counts_them(self):
        """`StorageRecord.from_values` would reject these at ingest — days later,
        on someone else's machine. Dropping here turns that into a number in the
        manifest."""
        vectors = np.array(
            [[1.0, 0.0], [0.0, 0.0], [np.nan, 1.0], [0.0, 3.0]], dtype=np.float32
        )
        out = check_vectors(self._ids(4), vectors, VIDEO, 2)
        assert out.dropped == 2
        assert len(out.kept_ids) == 2

    def test_rejects_the_wrong_width(self):
        with pytest.raises(ContractError, match="so chieu"):
            check_vectors(self._ids(2), np.ones((2, 7), dtype=np.float32), VIDEO, 1152)

    def test_rejects_a_length_mismatch(self):
        with pytest.raises(ContractError, match="khong khop"):
            check_vectors(self._ids(3), np.ones((2, 4), dtype=np.float32), VIDEO, 4)


class TestWriteThenMark:
    def test_text_part_lands_at_the_documented_path(self, writer):
        writer.write_text_part(
            "subtitles", VIDEO, {f"{VIDEO}#000090": "Chào mừng"}, model="faster-whisper large-v3"
        )
        expected = f"/bundle/aic_btc_v1/parts/subtitles/{VIDEO}.json"
        assert writer.transport.sha256(expected) is not None

    def test_marker_carries_the_counts_and_the_hash(self, writer):
        marker = writer.write_text_part(
            "subtitles",
            VIDEO,
            {f"{VIDEO}#000090": "co tieng noi", f"{VIDEO}#000261": "  "},
            model="faster-whisper large-v3",
            params={"beam_size": 5},
        )
        assert marker["rows"] == 1
        assert marker["empty"] == 1
        assert marker["author"] == "triet"
        assert len(marker["sha256"]) == 64

    def test_marker_hash_matches_the_stored_bytes(self, writer):
        """The read-back check. `modal volume put` is not verified to validate a
        checksum, so the protocol does it rather than assuming."""
        marker = writer.write_vector_part(
            "siglip_vectors",
            VIDEO,
            [f"{VIDEO}#000090"],
            np.ones((1, 1152), dtype=np.float32),
            model="google/siglip-so400m-patch14-384",
        )
        stored = writer.transport.sha256(f"/bundle/aic_btc_v1/parts/siglip_vectors/{VIDEO}.npz")
        assert stored == marker["sha256"]

    def test_vector_part_reloads_as_ids_plus_vectors(self, writer, tmp_path):
        """The shape `ingest_sota_vectors.py` reads — and no pickle, so loading a
        teammate's artifact is not code execution."""
        writer.write_vector_part(
            "dense_vectors",
            VIDEO,
            [f"{VIDEO}#000090", f"{VIDEO}#000261"],
            np.ones((2, 1024), dtype=np.float32),
            model="BAAI/bge-m3",
        )
        local = tmp_path / "back.npz"
        writer.transport.get(f"/bundle/aic_btc_v1/parts/dense_vectors/{VIDEO}.npz", local)
        payload = np.load(local, allow_pickle=False)
        assert set(payload.files) == {"ids", "vectors"}
        assert payload["vectors"].shape == (2, 1024)
        # Regression: `ids` was written with dtype=object, which makes np.load
        # need allow_pickle=True — i.e. loading a teammate's artifact would
        # execute pickle, the exact hazard §1.5 rejects Bảo's .npy dict for. The
        # earlier version of this test never touched `ids`, and np.load is lazy,
        # so it passed. Reading the array is the whole check.
        assert payload["ids"].dtype.kind == "U"
        assert payload["ids"].tolist() == [f"{VIDEO}#000090", f"{VIDEO}#000261"]

    def test_no_marker_is_written_when_validation_fails(self, writer):
        """The ordering that makes resume trustworthy: nothing is claimed done
        until the bytes are up and checked. A marker written first would let a
        restart skip work that never happened."""
        with pytest.raises(ContractError):
            writer.write_text_part("subtitles", VIDEO, {"L23_V001#000090": "sai video"}, model="m")
        status = StageStatus.load(writer.transport, writer.layout, "triet")
        assert not status.is_done("n2_subtitles", VIDEO)


class TestResume:
    def test_finished_work_is_skipped_on_the_next_run(self, writer):
        writer.write_text_part("subtitles", VIDEO, {f"{VIDEO}#000090": "x"}, model="m")
        status = StageStatus.load(writer.transport, writer.layout, "triet")
        assert status.is_done("n2_subtitles", VIDEO)
        assert not status.is_done("n5_ocr_easyocr", VIDEO)

    def test_pending_lists_exactly_what_is_left(self, writer):
        writer.write_text_part("subtitles", VIDEO, {f"{VIDEO}#000090": "x"}, model="m")
        status = StageStatus.load(writer.transport, writer.layout, "triet")
        pending = status.pending(["n2_subtitles", "n4_siglip"], [VIDEO, "L21_V002"])
        assert ("n2_subtitles", VIDEO) not in pending
        assert ("n2_subtitles", "L21_V002") in pending
        assert ("n4_siglip", VIDEO) in pending

    def test_status_is_read_per_member(self, writer, tmp_path):
        """One member's progress must not let another skip work."""
        writer.write_text_part("subtitles", VIDEO, {f"{VIDEO}#000090": "x"}, model="m")
        other = StageStatus.load(writer.transport, writer.layout, "quang")
        assert other.done == set()

    def test_a_half_written_part_without_a_marker_is_not_done(self, writer, tmp_path):
        """The reason the marker is the commit point and the data file is not:
        an interrupted upload leaves a file that exists and is wrong."""
        stray = tmp_path / "stray.json"
        stray.write_text("{}", encoding="utf-8")
        writer.transport.put(stray, writer.layout.part("subtitles", "L21_V002"))
        status = StageStatus.load(writer.transport, writer.layout, "triet")
        assert not status.is_done("n2_subtitles", "L21_V002")


class TestPartManifest:
    def test_records_model_and_environment(self, writer, tmp_path):
        """Five machines running one model is the risk chia-theo-video buys, so
        every shard states what it actually loaded — `merge_bundle.py` compares."""
        marker = writer.write_text_part(
            "subtitles", VIDEO, {f"{VIDEO}#000090": "x"}, model="faster-whisper large-v3"
        )
        writer.write_part_manifest(VIDEO, [marker], keyframes=307)
        local = tmp_path / "m.json"
        writer.transport.get(writer.layout.part_manifest(VIDEO), local)
        manifest = json.loads(local.read_text(encoding="utf-8"))
        assert manifest["key_format"] == "keyframe_id"
        assert manifest["artifacts"]["subtitles"]["model"] == "faster-whisper large-v3"
        assert "numpy" in manifest["env"]
        assert "numpy" in manifest["stages"]["n2_subtitles"]["env"]

    def test_a_second_pass_keeps_the_first_pass_environment(self, writer, tmp_path):
        """The bug per-stage `env` exists to remove.

        Four notebooks fill one BTC manifest. Top-level `env` is the last writer's,
        so if it were the only record, `04` would overwrite what `03` measured
        under and `merge_bundle.py` would compare two shards on a difference that
        belongs to two notebooks rather than to two measurements.
        """
        subs = writer.write_text_part(
            "subtitles", VIDEO, {f"{VIDEO}#000090": "x"}, model="faster-whisper large-v3"
        )
        with mock.patch(
            "mfe.offline.bundle.freeze_environment",
            return_value={"faster-whisper": "1.0.0", "torch": "2.4.0"},
        ):
            writer.write_part_manifest(VIDEO, [subs], keyframes=307)

        vecs = writer.write_vector_part(
            "siglip_vectors", VIDEO, [f"{VIDEO}#000090"], np.ones((1, 1152), np.float32),
            model="google/siglip-so400m-patch14-384",
        )
        with mock.patch(
            "mfe.offline.bundle.freeze_environment",
            return_value={"transformers": "4.44.0", "torch": "2.6.0"},
        ):
            writer.write_part_manifest(VIDEO, [vecs], keyframes=307, merge=True)

        local = tmp_path / "m.json"
        writer.transport.get(writer.layout.part_manifest(VIDEO), local)
        manifest = json.loads(local.read_text(encoding="utf-8"))
        assert manifest["stages"]["n2_subtitles"]["env"]["torch"] == "2.4.0"
        assert manifest["stages"]["n4_siglip"]["env"]["torch"] == "2.6.0"
        # Top-level env still describes only the pass that wrote last — left
        # in place, and no longer the thing the merge compares.
        assert manifest["env"]["torch"] == "2.6.0"

    def test_an_old_manifest_is_not_backfilled(self, writer, tmp_path):
        """A manifest from before this change has one `env` belonging to its last
        pass. Copying it onto the stages it does not describe would invent
        provenance; the merge reports the gap as unknown instead."""
        subs = writer.write_text_part(
            "subtitles", VIDEO, {f"{VIDEO}#000090": "x"}, model="faster-whisper large-v3"
        )
        writer.write_part_manifest(VIDEO, [subs], keyframes=307)

        local = tmp_path / "old.json"
        writer.transport.get(writer.layout.part_manifest(VIDEO), local)
        old = json.loads(local.read_text(encoding="utf-8"))
        old.pop("stages")
        local.write_text(json.dumps(old), encoding="utf-8")
        writer.transport.put(local, writer.layout.part_manifest(VIDEO))

        vecs = writer.write_vector_part(
            "siglip_vectors", VIDEO, [f"{VIDEO}#000090"], np.ones((1, 1152), np.float32),
            model="google/siglip-so400m-patch14-384",
        )
        writer.write_part_manifest(VIDEO, [vecs], keyframes=307, merge=True)

        writer.transport.get(writer.layout.part_manifest(VIDEO), local)
        manifest = json.loads(local.read_text(encoding="utf-8"))
        assert set(manifest["artifacts"]) == {"subtitles", "siglip_vectors"}
        assert set(manifest["stages"]) == {"n4_siglip"}


class TestStagePackages:
    def test_every_known_stage_is_declared(self):
        """An undeclared stage falls back to comparing every NUMERIC package,
        which refuses on versions it never loaded. That fallback is a safety net,
        not a place to leave a stage — so adding one to `ARTIFACTS` without a row
        here fails right here rather than as a mystery refusal at merge time."""
        known = set(DIRECTORY_STAGES) | {a.stage for a in ARTIFACTS.values()}
        assert known - set(STAGE_PACKAGES) == set()

    def test_declared_packages_are_actually_recorded(self):
        """A package `freeze_environment` never looks up reads as "(khong cai)"
        in every shard — perfect agreement, measured on nothing."""
        recorded = set(NUMERIC_PACKAGES) | set(OTHER_PACKAGES)
        for stage, packages in STAGE_PACKAGES.items():
            assert set(packages) - recorded == set(), stage


class TestTransportExists:
    """`exists` for files, `listdir` for directories - they are not interchangeable."""

    def test_exists_answers_for_a_file_path(self, tmp_path):
        """`listdir` on a file path returns nothing, which reads as 'not there'
        and is wrong in the one way that matters: the notebook used it to decide
        whether the shared video list had been published yet, so every member
        would have re-listed Drive and dealt a possibly different assignment.
        """
        transport = LocalDirTransport(tmp_path / "vol")
        source = tmp_path / "list.txt"
        source.write_text("L21_V001", encoding="utf-8")

        assert transport.exists("/shared/videos.txt") is False
        transport.put(source, "/shared/videos.txt")
        assert transport.exists("/shared/videos.txt") is True
        assert transport.listdir("/shared/videos.txt") == []
        assert transport.listdir("/shared") == ["/shared/videos.txt"]
