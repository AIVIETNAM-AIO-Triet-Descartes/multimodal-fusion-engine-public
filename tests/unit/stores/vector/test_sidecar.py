import pytest

from mfe.stores.vector.sidecar import Sidecar


# T01: Keyframe -> Index
def test_append_assigns_internal_id(tmp_path):
    sidecar = Sidecar(store_path=str(tmp_path))

    first_id = sidecar.append(keyframe_id="L21_V001#001", video_id="L21_V001")
    second_id = sidecar.append(keyframe_id="L21_V001#002", video_id="L21_V001")

    assert first_id == 0
    assert second_id == 1
    assert len(sidecar) == 2


# T02: Internal ID -> keyframe ID
def test_to_keyframe_id_returns_matching_keyframe(tmp_path):
    sidecar = Sidecar(store_path=str(tmp_path))
    internal_id = sidecar.append(keyframe_id="L21_V001#001", video_id="L21_V001")

    res = sidecar.to_keyframe_id(internal_id)
    assert res == "L21_V001#001"


def test_to_keyframe_id_reject_unknown_id(tmp_path):
    sidecar = Sidecar(store_path=str(tmp_path))

    with pytest.raises(IndexError):
        sidecar.to_keyframe_id(0)

    sidecar.append(keyframe_id="L21_V001#001", video_id="L21_V001")

    with pytest.raises(IndexError):
        sidecar.to_keyframe_id(-1)


# T03: Keyframe ID -> internal ID
def test_to_internal_id_returns_matching_id(tmp_path):
    sidecar = Sidecar(store_path=str(tmp_path))

    sidecar.append(
        keyframe_id="L21_V001#001",
        video_id="L21_V001",
    )

    expected_id = sidecar.append(
        keyframe_id="L21_V001#002",
        video_id="L21_V001",
    )

    result = sidecar.to_internal_id("L21_V001#002")

    assert result == expected_id


def test_append_rejects_duplicate_keyframe_id(tmp_path):
    sidecar = Sidecar(store_path=str(tmp_path))
    sidecar.append(keyframe_id="L21_V001#001", video_id="L21_V001")

    with pytest.raises(ValueError, match="Duplicate keyframe_id"):
        sidecar.append(keyframe_id="L21_V001#001", video_id="L99_V999")

    assert len(sidecar) == 1
    assert sidecar.to_internal_id("L21_V001#001") == 0


# T04: Internal ID -> video ID
def test_video_id_of_returns_matching_video(tmp_path):
    sidecar = Sidecar(store_path=str(tmp_path))

    sidecar.append(
        keyframe_id="L21_V001#001",
        video_id="L21_V001",
    )

    internal_id = sidecar.append(
        keyframe_id="L22_V002#001",
        video_id="L22_V002",
    )

    result = sidecar.video_id_of(internal_id)

    assert result == "L22_V002"


def test_video_id_of_rejects_unknown_internal_id(tmp_path):
    sidecar = Sidecar(store_path=str(tmp_path))

    with pytest.raises(IndexError, match="Unknown internal_id"):
        sidecar.video_id_of(0)


def test_verify_alignment_checks_faiss_size(tmp_path):
    sidecar = Sidecar(store_path=str(tmp_path))

    sidecar.append(
        keyframe_id="L21_V001#001",
        video_id="L21_V001",
    )

    sidecar.verify_alignment(faiss_size=1)

    with pytest.raises(AssertionError):
        sidecar.verify_alignment(faiss_size=2)

def test_to_internal_id_rejects_unknown_keyframe(tmp_path):
    sidecar = Sidecar(store_path=str(tmp_path))

    sidecar.append(
        keyframe_id="L21_V001#001",
        video_id="L21_V001",
    )

    with pytest.raises(KeyError):
        sidecar.to_internal_id("L99_V999#999")
