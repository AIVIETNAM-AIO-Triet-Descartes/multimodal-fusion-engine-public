"""Locks the SOTA_AIC payload loader — mfe.offline.adapters.sota_payloads."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from mfe.offline.adapters.sota_ids import SotaIdMap
from mfe.offline.adapters.sota_payloads import (
    SOTA_SOURCES,
    SOURCE_FILES,
    load_all,
    load_payload,
)
from mfe.offline.provided import ProvidedDataset

PILOT_VIDEOS = ["L21_V001", "L21_V002", "L21_V003"]
_DATASET_ROOT = Path("data/sample_2026")
_MAP_DIR = _DATASET_ROOT / "map-keyframes"
_SOTA_VECTORS = Path(os.getenv("MFE_SOTA_VECTORS", "SOTA_AIC/data/vectors"))
_HAS_PILOT_MAPS = all((_MAP_DIR / f"{video_id}.csv").is_file() for video_id in PILOT_VIDEOS)

pytestmark = pytest.mark.skipif(
    not _HAS_PILOT_MAPS, reason="needs the L21_V001..V003 pilot CSVs"
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

    def test_reports_which_grammar_the_file_used(self, tmp_path, id_map):
        ordinal = _write(tmp_path, "a.json", {"L21_V001_002": "x"})
        assert load_payload(ordinal, "ocr", id_map)[1].key_format == "ordinal"
        ours = _write(tmp_path, "b.json", {"L21_V001#000090": "x"})
        assert load_payload(ours, "ocr", id_map)[1].key_format == "keyframe_id"


class TestOurOwnKeyGrammar:
    """The `keyframe_id`-keyed payloads our Kaggle notebooks produce — §1.2.

    Frame 90 is ordinal n=2 of L21_V001. Every case below leans on that gap: an
    id that merely *looks* resolvable is the failure this grammar exists to make
    impossible, so the tests must not use a video where the two numbers coincide.
    """

    def test_keyframe_id_key_needs_no_lookup(self, tmp_path, id_map):
        path = _write(tmp_path, "subtitles.json", {"L21_V001#000090": "Chào mừng quý vị"})
        records, report = load_payload(path, "subtitle", id_map)
        assert len(records) == 1
        assert (records[0].video_id, records[0].frame_id) == ("L21_V001", 90)
        assert report.records == 1
        assert report.unknown_ids == []
        assert report.malformed_ids == []

    def test_both_grammars_agree_on_the_same_frame(self, tmp_path, id_map):
        """The whole point of accepting two grammars: they must be two spellings
        of one thing. `L21_V001_002` and `L21_V001#000090` are the same frame."""
        by_ordinal = _write(tmp_path, "a.json", {"L21_V001_002": "x"})
        by_frame = _write(tmp_path, "b.json", {"L21_V001#000090": "x"})
        left, _ = load_payload(by_ordinal, "ocr", id_map)
        right, _ = load_payload(by_frame, "ocr", id_map)
        assert (left[0].video_id, left[0].frame_id) == (right[0].video_id, right[0].frame_id)

    def test_frame_outside_the_bundle_goes_to_unknown_ids(self, tmp_path, id_map):
        """This is the filter that dropping the CSV lookup would otherwise lose.

        Frame 91 exists in the video but is **not a keyframe**, so no row of
        map-keyframes names it. Letting it through would put a key in
        Elasticsearch/Milvus that the Relational Map cannot resolve — and that
        surfaces as a raise in the middle of scoring, not here.
        """
        path = _write(tmp_path, "ocr.json", {"L21_V001#000091": "khong phai keyframe"})
        records, report = load_payload(path, "ocr", id_map)
        assert records == []
        assert report.unknown_ids == ["L21_V001#000091"]
        assert report.malformed_ids == []

    def test_video_outside_the_bundle_goes_to_unknown_ids(self, tmp_path, id_map):
        path = _write(tmp_path, "ocr.json", {"L99_V001#000090": "video ngoai bo"})
        _, report = load_payload(path, "ocr", id_map)
        assert report.unknown_ids == ["L99_V001#000090"]

    def test_malformed_id_is_counted_apart_from_unknown(self, tmp_path, id_map):
        """A broken producer must not hide behind the state we learned to tolerate.

        `#90` fails the 6-digit contract, so no reader can resolve it. That is a
        notebook writing bad keys — a different problem from an export covering
        videos we did not ingest, and with five members writing payloads the two
        need separate counters.
        """
        path = _write(tmp_path, "ocr.json", {"L21_V001#90": "thieu chu so"})
        records, report = load_payload(path, "ocr", id_map)
        assert records == []
        assert report.malformed_ids == ["L21_V001#90"]
        assert report.unknown_ids == []
        assert "MALFORMED" in report.summary()

    def test_a_file_mixing_both_grammars_is_flagged(self, tmp_path, id_map):
        """No correct producer emits both. Seeing both means a run changed
        convention half way, and the caller should refuse rather than pick."""
        # n=2 -> frame 90, n=3 -> frame 261 (read off map-keyframes/L21_V001.csv;
        # picking a round number like 180 gives a frame that is not a keyframe).
        path = _write(
            tmp_path, "ocr.json", {"L21_V001_002": "theo thu tu", "L21_V001#000261": "theo frame"}
        )
        records, report = load_payload(path, "ocr", id_map)
        assert report.key_format == "mixed"
        assert len(records) == 2  # still readable — flagged, not silently dropped


class TestAgainstBaosRealExport:
    pytestmark = pytest.mark.skipif(
        not _SOTA_VECTORS.is_dir(), reason="needs MFE_SOTA_VECTORS (SOTA_AIC/data/vectors)"
    )

    def test_every_source_loads_and_ids_resolve(self, id_map):
        """Every source **Bảo ships** must load. Scoped to `SOTA_SOURCES`, not to
        `SOURCE_FILES`, since lượt 4 added two sources only we produce.

        Asserting over `SOURCE_FILES` conflated two different claims — "Bảo's
        export is intact" and "no source may exist unless Bảo ships it" — and the
        second one would block adding any producer of our own.
        """
        records, reports = load_all(_SOTA_VECTORS, id_map, sources=sorted(SOTA_SOURCES))
        by_source = {r.source: r for r in reports}
        assert set(by_source) == SOTA_SOURCES
        for report in reports:
            assert report.present, report.summary()
            assert report.records > 0, report.summary()
            # Bảo's export covers exactly the pilot videos, so nothing should
            # fail to translate. A non-empty list here means the export drifted.
            assert report.unknown_ids == [], report.summary()

        assert {r.source for r in records} == SOTA_SOURCES
        # Every record accounted for: sum of per-source counts, no double count.
        assert len(records) == sum(r.records for r in reports)

    def test_a_source_bao_never_shipped_is_reported_absent_not_raised(self, id_map):
        """Sources from our own passes. Pointed at Bảo's directory they must be
        *reported* missing, because "absent" is a valid state for a payload whose
        producer is a different pass entirely.

        The list is asserted rather than merely iterated: a new source that
        forgets to declare itself in `SOURCE_FILES` is invisible, and one that
        lands in `SOTA_SOURCES` by mistake would claim Bảo ships it.

        `caption_qwen` / `ocr_qwen25` joined on 25-08-2026 — the two halves of one
        Qwen2.5-VL generation (`n6_caption_qwen` / `n6_ocr_qwen25`). Bảo's
        *method* is where they come from, but his **export** never carried them,
        which is exactly the distinction this test keeps.
        """
        ours = sorted(set(SOURCE_FILES) - SOTA_SOURCES)
        assert ours == ["caption_qwen", "ocr_paddle", "ocr_qwen25", "ocr_vietocr"]
        records, reports = load_all(_SOTA_VECTORS, id_map, sources=ours)
        assert records == []
        assert all(not r.present for r in reports)
        assert all("not found" in r.summary() for r in reports)

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
