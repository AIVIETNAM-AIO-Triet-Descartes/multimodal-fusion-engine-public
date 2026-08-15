"""Locks the SOTA_AIC payload loader — mfe.offline.adapters.sota_payloads."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mfe.offline.adapters.sota_ids import SotaIdMap
from mfe.offline.adapters.sota_payloads import SOURCE_FILES, load_all, load_payload
from mfe.offline.provided import ProvidedDataset

PILOT_VIDEOS = ["L21_V001", "L21_V002", "L21_V003"]
_DATASET_ROOT = Path("data/sample_2026")
_MAP_DIR = _DATASET_ROOT / "map-keyframes"
_SOTA_VECTORS = Path("D:/Projects/SOTA_AIC/data/vectors")

pytestmark = pytest.mark.skipif(
    not _MAP_DIR.is_dir(), reason="needs data/sample_2026/"
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


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class TestLoadPayload:
    def test_translates_ids_and_tags_the_source(self, tmp_path, id_map):
        # n=2 of L21_V001 is frame 90 — the id is NOT L21_V001#000002.
        path = _write(tmp_path, "subtitles.json", {"L21_V001_002": "Chào mừng quý vị"})
        records, report = load_payload(path, "subtitle", id_map)
        assert len(records) == 1
        assert records[0].video_id == "L21_V001"
        assert records[0].frame_id == 90
        assert records[0].source == "subtitle"
        assert report.records == 1

    def test_missing_file_is_reported_not_raised(self, tmp_path, id_map):
        """Four of five files have no producer in Bảo's repo (§1b), so absent is
        an expected state — but it must be visible, not silent."""
        records, report = load_payload(tmp_path / "nope.json", "subtitle", id_map)
        assert records == []
        assert report.present is False
        assert "not found" in report.summary()

    def test_empty_text_is_dropped_and_counted(self, tmp_path, id_map):
        path = _write(
            tmp_path, "ocr.json", {"L21_V001_002": "  ", "L21_V001_003": "BẢN TIN 60 GIÂY"}
        )
        records, report = load_payload(path, "ocr", id_map)
        assert report.records == 1
        assert report.empty_texts == 1

    def test_unknown_id_is_collected_not_dropped_silently(self, tmp_path, id_map):
        path = _write(tmp_path, "ocr.json", {"L99_V001_001": "video ngoài bộ pilot"})
        records, report = load_payload(path, "ocr", id_map)
        assert records == []
        assert report.unknown_ids == ["L99_V001_001"]
        assert "1 unknown id" in report.summary()

    def test_rejects_an_unregistered_source_tag(self, tmp_path, id_map):
        """`source` must satisfy Quang's SOURCE_PATTERN and be one we planned for;
        a typo would create a fourth silent bucket in Elasticsearch."""
        path = _write(tmp_path, "x.json", {"L21_V001_002": "x"})
        with pytest.raises(ValueError, match="unknown source"):
            load_payload(path, "OCR_TYPO", id_map)


class TestAgainstBaosRealExport:
    pytestmark = pytest.mark.skipif(
        not _SOTA_VECTORS.is_dir(), reason="needs D:/Projects/SOTA_AIC/data/vectors"
    )

    def test_every_source_loads_and_ids_resolve(self, id_map):
        records, reports = load_all(_SOTA_VECTORS, id_map)
        by_source = {r.source: r for r in reports}
        assert set(by_source) == set(SOURCE_FILES)
        for report in reports:
            assert report.present, report.summary()
            assert report.records > 0, report.summary()
            # Bảo's export covers exactly the pilot videos, so nothing should
            # fail to translate. A non-empty list here means the export drifted.
            assert report.unknown_ids == [], report.summary()

        assert {r.source for r in records} == set(SOURCE_FILES)
        # Every record accounted for: sum of per-source counts, no double count.
        assert len(records) == sum(r.records for r in reports)

    def test_the_two_ocr_sources_stay_separate(self, id_map):
        """The reason they are two sources and not one concatenated string.

        Measured on the pilot export: easyocr emits text for all 855 keyframes
        (much of it noise off the clock overlay), while SVIPTR emits for 513 and
        stays silent on the other 342. Concatenating them would give every frame
        a non-empty OCR field and hide exactly that difference — including from
        BM25, which would then see 855 documents where 513 carry signal.
        """
        _, reports = load_all(_SOTA_VECTORS, id_map, sources=["ocr", "ocr_sviptr"])
        by_source = {r.source: r for r in reports}
        assert by_source["ocr"].empty_texts == 0
        assert by_source["ocr_sviptr"].empty_texts > 0
        assert by_source["ocr_sviptr"].records < by_source["ocr"].records

    def test_subtitle_covers_fewer_keyframes_than_ocr(self, id_map):
        """Not every keyframe has speech; every keyframe has pixels. If these
        ever match exactly, one of the files is not what it claims to be."""
        _, reports = load_all(_SOTA_VECTORS, id_map, sources=["subtitle", "ocr"])
        by_source = {r.source: r for r in reports}
        assert by_source["subtitle"].keys_read < by_source["ocr"].keys_read

    def test_frame_ids_land_inside_the_ingested_pilot_set(self, id_map):
        """Guards the whole translation chain end to end: every frame_id the
        loader emits must be a real row of map-keyframes, not an ordinal that
        happens to look like a frame."""
        valid: set[tuple[str, int]] = set()
        for video_id in PILOT_VIDEOS:
            with (_MAP_DIR / f"{video_id}.csv").open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    valid.add((video_id, int(row["frame_idx"])))

        records, _ = load_all(_SOTA_VECTORS, id_map)
        assert all((r.video_id, r.frame_id) in valid for r in records)
