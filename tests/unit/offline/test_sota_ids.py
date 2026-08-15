"""Locks the SOTA_AIC id translation against the real map-keyframes CSVs.

The point of these tests is that they compare against the CSV, never against
another string. A test that asserts ``"L21_V001_001" -> "L21_V001#000001"`` would
pass while encoding exactly the bug the adapter exists to prevent.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.offline.adapters.sota_ids import SotaIdMap, build_sota_id, parse_sota_id
from mfe.offline.provided import ProvidedDataset

PILOT_VIDEOS = ["L21_V001", "L21_V002", "L21_V003"]
_DATASET_ROOT = Path("data/sample_2026")
_MAP_DIR = _DATASET_ROOT / "map-keyframes"

pytestmark = pytest.mark.skipif(
    not _MAP_DIR.is_dir(),
    reason="needs data/sample_2026/ — see docs/task/huongdan-upload-download-hf.md",
)


@pytest.fixture(scope="module")
def id_map() -> SotaIdMap:
    dataset = ProvidedDataset(
        keyframes_dir=str(_DATASET_ROOT / "keyframes"),
        map_keyframes_dir=str(_MAP_DIR),
        clip_features_dir=str(_DATASET_ROOT / "clip-features"),
        objects_dir=str(_DATASET_ROOT / "objects"),
        media_info_dir=str(_DATASET_ROOT / "media-info"),
        videos_dir=str(_DATASET_ROOT / "videos"),
    )
    return SotaIdMap(dataset, PILOT_VIDEOS)


def _csv_rows(video_id: str) -> list[dict[str, str]]:
    with (_MAP_DIR / f"{video_id}.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestParseSotaId:
    def test_splits_from_the_right(self):
        """`video_id` contains underscores; a left split reads the group as the ordinal."""
        assert parse_sota_id("L21_V001_001") == ("L21_V001", 1)
        assert parse_sota_id("L21_V001_250") == ("L21_V001", 250)

    @pytest.mark.parametrize("malformed", ["L21_V001", "L21_V001_abc", "_001", ""])
    def test_rejects_malformed(self, malformed):
        with pytest.raises(ContractError) as exc:
            parse_sota_id(malformed)
        assert exc.value.code is ErrorCode.INVALID_ID

    def test_round_trips(self):
        assert parse_sota_id(build_sota_id("L21_V001", 7)) == ("L21_V001", 7)


class TestAgainstTheRealCsv:
    def test_every_pilot_keyframe_translates(self, id_map: SotaIdMap):
        """All 855 rows, not a sample of three."""
        checked = 0
        for video_id in PILOT_VIDEOS:
            for row in _csv_rows(video_id):
                sota_id = build_sota_id(video_id, int(row["n"]))
                expected = f"{video_id}#{int(row['frame_idx']):06d}"
                assert id_map.to_keyframe_id(sota_id) == expected
                checked += 1
        assert checked == 855, f"pilot set should be 855 keyframes, counted {checked}"

    def test_ordinal_and_frame_really_do_differ(self, id_map: SotaIdMap):
        """Guards the premise. If some future dataset numbered keyframes by frame,
        the adapter would look correct while doing nothing, and this test says so."""
        differing = 0
        for video_id in PILOT_VIDEOS:
            for row in _csv_rows(video_id):
                if int(row["n"]) != int(row["frame_idx"]):
                    differing += 1
        assert differing > 800, (
            f"only {differing}/855 rows have n != frame_idx; the two numbering "
            "schemes may have converged, which would make this adapter a no-op"
        )

    def test_round_trips_through_our_id(self, id_map: SotaIdMap):
        for video_id in PILOT_VIDEOS:
            for row in _csv_rows(video_id):
                sota_id = build_sota_id(video_id, int(row["n"]))
                assert id_map.to_sota_id(id_map.to_keyframe_id(sota_id)) == sota_id

    def test_pilot_set_has_no_duplicate_frames(self, id_map: SotaIdMap):
        """Documents why the pilot is the easy case: the 20 collapsing rows of the
        16-video set all sit in L25/L27, none in L21_V001..V003."""
        for video_id in PILOT_VIDEOS:
            frames = [int(row["frame_idx"]) for row in _csv_rows(video_id)]
            assert len(frames) == len(set(frames)), f"{video_id} has duplicate frame_idx"


class TestRefusesToGuess:
    def test_unknown_ordinal_raises(self, id_map: SotaIdMap):
        """Bảo's p0_adapter returns "0" here — submittable, plausible, wrong."""
        with pytest.raises(ContractError) as exc:
            id_map.to_keyframe_id("L21_V001_9999")
        assert exc.value.code is ErrorCode.KEYFRAME_NOT_FOUND

    def test_unknown_video_raises(self, id_map: SotaIdMap):
        with pytest.raises(ContractError):
            id_map.to_keyframe_id("L99_V001_001")

    def test_translate_mapping_reports_unknown_instead_of_dropping_silently(
        self, id_map: SotaIdMap
    ):
        payload = {"L21_V001_001": "co that", "L99_V001_001": "khong thuoc bo pilot"}
        translated, unknown = id_map.translate_mapping(payload)
        assert len(translated) == 1
        assert unknown == ["L99_V001_001"]
