import pytest

from mfe.contracts.records import InvertedRecord
from mfe.stores.inverted.index import InvertedIndex


def _records() -> list[InvertedRecord]:
    return [
        InvertedRecord(
            keyframe_id="L21_V001#001",
            video_id="L21_V001",
            object_labels=["person", "car"],
            ocr_text="BẢN TIN 18H",
            metadata_text="Thời sự tối",
        ),
        InvertedRecord(
            keyframe_id="L21_V002#001",
            video_id="L21_V002",
            object_labels=["person", "bicycle"],
            metadata_text="Camera giao lộ Nguyễn Huệ",
        ),
        InvertedRecord(
            keyframe_id="L22_V003#001",
            video_id="L22_V003",
            object_labels=["tree"],
            metadata_text="Công viên buổi sáng",
        ),
    ]


def test_add_and_search_return_matching_candidate_with_terms(tmp_path):
    index = InvertedIndex(store_path=str(tmp_path))
    index.add(_records())

    candidates = index.search("CAR 18h", top_k=2)

    assert [candidate.keyframe_id for candidate in candidates] == ["L21_V001#001"]
    assert candidates[0].matched_terms == ["car", "18h"]
    assert candidates[0].video_id == "L21_V001"
    assert candidates[0].evidence_id == "text_search:store:0"


def test_search_fields_restrict_where_terms_can_match(tmp_path):
    index = InvertedIndex(store_path=str(tmp_path))
    index.add(_records())

    assert index.search("18h", fields=["ocr"], top_k=10)
    assert index.search("18h", fields=["metadata"], top_k=10) == []


def test_add_updates_existing_document_and_invalidates_search_cache(tmp_path):
    index = InvertedIndex(store_path=str(tmp_path))
    original = _records()[0]
    index.add([original])
    assert index.search("car", top_k=1)

    updated = original.model_copy(
        update={
            "object_labels": ["person", "bus"],
            "ocr_text": "",
        }
    )
    index.add([updated])

    assert len(index) == 1
    assert index.search("car", top_k=1) == []
    assert index.search("bus", top_k=1)


def test_filter_supports_fields_video_and_candidate_subset(tmp_path):
    index = InvertedIndex(store_path=str(tmp_path))
    index.add(_records())

    assert index.filter({"object": {"contains": "person"}}) == [
        "L21_V001#001",
        "L21_V002#001",
    ]
    assert index.filter(
        {"metadata": "Nguyễn Huệ", "video_id": "L21_V002"},
        candidate_ids=["L22_V003#001", "L21_V002#001"],
    ) == ["L21_V002#001"]
    assert index.filter({"video_id": ["L21_V002"]}) == ["L21_V002#001"]


def test_filter_rejects_unknown_constraints(tmp_path):
    index = InvertedIndex(store_path=str(tmp_path))

    with pytest.raises(ValueError, match="Unknown filter constraint"):
        index.filter({"unknown": "value"})

    with pytest.raises(ValueError, match="Unknown constraint operator"):
        index.filter({"object": {"equals": "car"}})

    with pytest.raises(ValueError, match="string or collection"):
        index.filter({"object": 42})

    with pytest.raises(ValueError, match="non-empty text"):
        index.filter({"object": ""})

    with pytest.raises(ValueError, match="non-empty text"):
        index.filter({"video_id": []})


@pytest.mark.parametrize("top_k", [0, 501])
def test_search_rejects_invalid_top_k(tmp_path, top_k):
    index = InvertedIndex(store_path=str(tmp_path))

    with pytest.raises(ValueError, match="top_k must be between"):
        index.search("car", top_k=top_k)


def test_search_rejects_empty_query_and_unknown_fields(tmp_path):
    index = InvertedIndex(store_path=str(tmp_path))

    with pytest.raises(ValueError, match="at least one searchable term"):
        index.search("  !!!  ", top_k=1)

    with pytest.raises(ValueError, match="Unknown search field"):
        index.search("car", fields=["title"], top_k=1)

    with pytest.raises(ValueError, match="at least one search field"):
        index.search("car", fields=[], top_k=1)


def test_search_empty_store_returns_no_candidates(tmp_path):
    index = InvertedIndex(store_path=str(tmp_path))

    assert index.search("car", top_k=1) == []


def test_search_returns_empty_when_all_documents_have_no_terms(tmp_path):
    index = InvertedIndex(store_path=str(tmp_path))
    index.add(
        [
            InvertedRecord(
                keyframe_id="L21_V001#001",
                video_id="L21_V001",
            )
        ]
    )

    assert index.search("car", top_k=1) == []


def test_save_and_load_preserve_search_and_filters(tmp_path):
    index = InvertedIndex(store_path=str(tmp_path))
    index.add(_records())
    index.save()

    loaded = InvertedIndex.load(store_path=str(tmp_path))

    assert len(loaded) == 3
    assert [candidate.keyframe_id for candidate in loaded.search("bicycle", top_k=1)] == [
        "L21_V002#001"
    ]
    assert loaded.filter({"object_labels": "tree"}) == ["L22_V003#001"]
    assert (tmp_path / "inverted_records.json").is_file()


def test_load_rejects_non_list_payload(tmp_path):
    (tmp_path / "inverted_records.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON list"):
        InvertedIndex.load(store_path=str(tmp_path))
