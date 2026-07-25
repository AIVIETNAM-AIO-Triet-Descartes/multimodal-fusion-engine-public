import faiss
import numpy as np
import pytest

from mfe.contracts.records import VectorRecord
from mfe.stores.vector.index import VectorIndex


def test_new_vector_index_is_empty(tmp_path):
    index = VectorIndex(
        store_path=str(tmp_path),
        embed_dim=3,
    )

    assert len(index) == 0


def test_add_records_updates_faiss_and_sidecar(tmp_path):
    index = VectorIndex(store_path=str(tmp_path), embed_dim=3)
    records = [
        VectorRecord(
            internal_id=0,
            keyframe_id="L21_V001#001",
            video_id="L21_V001",
            embedding=[1.0, 0.0, 0.0],
        ),
        VectorRecord(
            internal_id=1,
            keyframe_id="L22_V002#001",
            video_id="L22_V002",
            embedding=[0.0, 1.0, 0.0],
        ),
    ]

    index.add(records)

    assert len(index) == 2
    assert index._sidecar.to_keyframe_id(0) == "L21_V001#001"
    assert index._sidecar.video_id_of(1) == "L22_V002"


def test_add_rejects_wrong_embedding_dimension_without_mutating_index(tmp_path):
    index = VectorIndex(store_path=str(tmp_path), embed_dim=3)
    record = VectorRecord(
        internal_id=0,
        keyframe_id="L21_V001#001",
        video_id="L21_V001",
        embedding=[1.0, 0.0],
    )

    with pytest.raises(ValueError, match="expected 3"):
        index.add([record])

    assert len(index) == 0
    assert len(index._sidecar) == 0


def test_add_rejects_unexpected_internal_id_without_mutating_index(tmp_path):
    index = VectorIndex(store_path=str(tmp_path), embed_dim=3)
    record = VectorRecord(
        internal_id=1,
        keyframe_id="L21_V001#001",
        video_id="L21_V001",
        embedding=[1.0, 0.0, 0.0],
    )

    with pytest.raises(ValueError, match="Expected internal_id 0"):
        index.add([record])

    assert len(index) == 0
    assert len(index._sidecar) == 0


def test_add_rejects_duplicate_keyframe_before_mutating_index(tmp_path):
    index = VectorIndex(store_path=str(tmp_path), embed_dim=2)
    records = [
        VectorRecord(
            internal_id=0,
            keyframe_id="L21_V001#001",
            video_id="L21_V001",
            embedding=[1.0, 0.0],
        ),
        VectorRecord(
            internal_id=1,
            keyframe_id="L21_V001#001",
            video_id="L21_V001",
            embedding=[0.0, 1.0],
        ),
    ]

    with pytest.raises(ValueError, match="Duplicate keyframe_id in batch"):
        index.add(records)

    assert len(index) == 0
    assert len(index._sidecar) == 0


def test_add_rejects_keyframe_already_in_index(tmp_path):
    index = VectorIndex(store_path=str(tmp_path), embed_dim=2)
    first = VectorRecord(
        internal_id=0,
        keyframe_id="L21_V001#001",
        video_id="L21_V001",
        embedding=[1.0, 0.0],
    )
    duplicate = VectorRecord(
        internal_id=1,
        keyframe_id="L21_V001#001",
        video_id="L21_V001",
        embedding=[0.0, 1.0],
    )
    index.add([first])

    with pytest.raises(ValueError, match="Duplicate keyframe_id"):
        index.add([duplicate])

    assert len(index) == 1
    assert len(index._sidecar) == 1


def test_add_rejects_non_finite_embedding(tmp_path):
    index = VectorIndex(store_path=str(tmp_path), embed_dim=2)
    record = VectorRecord(
        internal_id=0,
        keyframe_id="L21_V001#001",
        video_id="L21_V001",
        embedding=[float("nan"), 0.0],
    )

    with pytest.raises(ValueError, match="contains NaN or infinity"):
        index.add([record])

    assert len(index) == 0


def test_add_empty_batch_is_a_no_op(tmp_path):
    index = VectorIndex(store_path=str(tmp_path), embed_dim=3)

    index.add([])

    assert len(index) == 0
    assert len(index._sidecar) == 0


def test_add_detects_unexpected_sidecar_assignment(tmp_path, monkeypatch):
    index = VectorIndex(store_path=str(tmp_path), embed_dim=2)
    record = VectorRecord(
        internal_id=0,
        keyframe_id="L21_V001#001",
        video_id="L21_V001",
        embedding=[1.0, 0.0],
    )
    monkeypatch.setattr(index._sidecar, "append", lambda **_kwargs: 99)

    with pytest.raises(AssertionError, match="Sidecar assigned internal_id 99"):
        index.add([record])


def test_save_and_load_preserve_faiss_and_sidecar(tmp_path):
    index = VectorIndex(store_path=str(tmp_path), embed_dim=3)
    index.add(
        [
            VectorRecord(
                internal_id=0,
                keyframe_id="L21_V001#001",
                video_id="L21_V001",
                embedding=[1.0, 0.0, 0.0],
            ),
            VectorRecord(
                internal_id=1,
                keyframe_id="L22_V002#001",
                video_id="L22_V002",
                embedding=[0.0, 1.0, 0.0],
            ),
        ]
    )

    index.save()
    loaded = VectorIndex.load(store_path=str(tmp_path), embed_dim=3)

    assert len(loaded) == 2
    assert loaded._sidecar.to_keyframe_id(0) == "L21_V001#001"
    assert loaded._sidecar.video_id_of(1) == "L22_V002"
    assert (tmp_path / "index.faiss").is_file()
    assert (tmp_path / "sidecar.parquet").is_file()


def test_load_rejects_wrong_embedding_dimension(tmp_path):
    index = VectorIndex(store_path=str(tmp_path), embed_dim=3)
    index.save()

    with pytest.raises(ValueError, match="Stored Faiss dimension 3"):
        VectorIndex.load(store_path=str(tmp_path), embed_dim=2)


def test_load_rejects_non_inner_product_index(tmp_path):
    faiss.write_index(faiss.IndexFlatL2(2), str(tmp_path / "index.faiss"))

    with pytest.raises(ValueError, match="is not inner product"):
        VectorIndex.load(store_path=str(tmp_path), embed_dim=2)


def test_search_normalizes_query_and_returns_ranked_candidates(tmp_path):
    index = VectorIndex(store_path=str(tmp_path), embed_dim=3)
    index.add(
        [
            VectorRecord(
                internal_id=0,
                keyframe_id="L21_V001#001",
                video_id="L21_V001",
                embedding=[1.0, 0.0, 0.0],
            ),
            VectorRecord(
                internal_id=1,
                keyframe_id="L22_V002#001",
                video_id="L22_V002",
                embedding=[0.0, 1.0, 0.0],
            ),
        ]
    )

    candidates = index.search(query_embedding=[10.0, 0.0, 0.0], top_k=2)

    assert [candidate.keyframe_id for candidate in candidates] == [
        "L21_V001#001",
        "L22_V002#001",
    ]
    assert candidates[0].score == pytest.approx(1.0)
    assert candidates[0].evidence_id == "vector_search:store:0"


def test_search_filters_by_video_without_losing_lower_ranked_hits(tmp_path):
    index = VectorIndex(store_path=str(tmp_path), embed_dim=2)
    index.add(
        [
            VectorRecord(
                internal_id=0,
                keyframe_id="L21_V001#001",
                video_id="L21_V001",
                embedding=[1.0, 0.0],
            ),
            VectorRecord(
                internal_id=1,
                keyframe_id="L22_V002#001",
                video_id="L22_V002",
                embedding=[0.8, 0.6],
            ),
        ]
    )

    candidates = index.search(
        query_embedding=[1.0, 0.0],
        top_k=1,
        video_id="L22_V002",
    )

    assert [candidate.keyframe_id for candidate in candidates] == ["L22_V002#001"]


@pytest.mark.parametrize("top_k", [0, 501])
def test_search_rejects_invalid_top_k(tmp_path, top_k):
    index = VectorIndex(store_path=str(tmp_path), embed_dim=2)

    with pytest.raises(ValueError, match="top_k must be between"):
        index.search(query_embedding=[1.0, 0.0], top_k=top_k)


def test_search_rejects_wrong_dimension_and_zero_vector(tmp_path):
    index = VectorIndex(store_path=str(tmp_path), embed_dim=2)

    with pytest.raises(ValueError, match="dimension 3"):
        index.search(query_embedding=[1.0, 0.0, 0.0], top_k=1)

    with pytest.raises(ValueError, match="zero vector"):
        index.search(query_embedding=[0.0, 0.0], top_k=1)

    with pytest.raises(ValueError, match="contains NaN or infinity"):
        index.search(query_embedding=[float("nan"), 0.0], top_k=1)


def test_search_empty_index_returns_no_candidates(tmp_path):
    index = VectorIndex(store_path=str(tmp_path), embed_dim=2)

    assert index.search(query_embedding=[1.0, 0.0], top_k=1) == []


def test_search_ignores_faiss_missing_id_sentinel(tmp_path, monkeypatch):
    index = VectorIndex(store_path=str(tmp_path), embed_dim=2)
    index.add(
        [
            VectorRecord(
                internal_id=0,
                keyframe_id="L21_V001#001",
                video_id="L21_V001",
                embedding=[1.0, 0.0],
            )
        ]
    )

    def return_missing_id(_query, _top_k):
        return (
            np.array([[0.0]], dtype=np.float32),
            np.array([[-1]], dtype=np.int64),
        )

    monkeypatch.setattr(index._index, "search", return_missing_id)

    assert index.search(query_embedding=[1.0, 0.0], top_k=1) == []
