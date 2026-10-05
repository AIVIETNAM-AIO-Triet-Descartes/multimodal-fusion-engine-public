import pytest

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.contracts.records import AssetRecord, AssetType
from mfe.core.backends.local import LocalBackend
from mfe.stores.asset.manifest import AssetStore


def _asset_record(
    *,
    keyframe_id: str = "L21_V001#001",
    asset_type: AssetType = AssetType.IMAGE,
    asset_path: str = "data/keyframes/L21_V001/001.jpg",
    byte_size: int = 100,
) -> AssetRecord:
    return AssetRecord(
        keyframe_id=keyframe_id,
        asset_type=asset_type,
        asset_path=asset_path,
        width=1280,
        height=720,
        byte_size=byte_size,
    )


def test_add_and_get_record_use_keyframe_and_asset_type(tmp_path):
    store = AssetStore(store_path=str(tmp_path))
    image = _asset_record()
    segment = _asset_record(
        asset_type=AssetType.VIDEO_SEGMENT,
        asset_path="data/videos/L21_V001.mp4",
        byte_size=1_000,
    )

    store.add([image, segment])

    assert store.get_record(image.keyframe_id, AssetType.IMAGE) == image
    assert store.get_record(segment.keyframe_id, AssetType.VIDEO_SEGMENT) == segment


def test_add_updates_an_existing_manifest_key(tmp_path):
    store = AssetStore(store_path=str(tmp_path))
    original = _asset_record(byte_size=100)
    updated = _asset_record(
        asset_path="data/keyframes/L21_V001/001-new.jpg",
        byte_size=150,
    )

    store.add([original])
    store.add([updated])

    assert store.get_record(updated.keyframe_id, AssetType.IMAGE) == updated
    assert store.total_bytes() == 150


def test_get_record_raises_structured_error_on_a_miss(tmp_path):
    store = AssetStore(store_path=str(tmp_path))

    with pytest.raises(ContractError) as error:
        store.get_record("L99_V999#999", AssetType.IMAGE)

    assert error.value.code is ErrorCode.KEYFRAME_NOT_FOUND


def test_total_bytes_sums_current_manifest_rows(tmp_path):
    store = AssetStore(store_path=str(tmp_path))
    store.add(
        [
            _asset_record(byte_size=100),
            _asset_record(
                keyframe_id="L21_V001#002",
                asset_path="data/keyframes/L21_V001/002.jpg",
                byte_size=250,
            ),
        ]
    )

    assert store.total_bytes() == 350


def test_get_asset_reads_registered_bytes_through_backend(tmp_path):
    backend = LocalBackend(root=str(tmp_path))
    backend.write_bytes("data/keyframes/L21_V001/001.jpg", b"original-image")
    store = AssetStore(store_path=str(tmp_path), backend=backend)
    store.add([_asset_record()])

    result = store.get_asset("L21_V001#001")

    assert result == b"original-image"


def test_get_asset_preserves_structured_errors(tmp_path):
    backend = LocalBackend(root=str(tmp_path))
    store = AssetStore(store_path=str(tmp_path), backend=backend)

    with pytest.raises(ContractError) as missing_record:
        store.get_asset("L99_V999#999")
    assert missing_record.value.code is ErrorCode.KEYFRAME_NOT_FOUND

    store.add([_asset_record()])
    with pytest.raises(ContractError) as missing_bytes:
        store.get_asset("L21_V001#001")
    assert missing_bytes.value.code is ErrorCode.STORE_UNAVAILABLE
