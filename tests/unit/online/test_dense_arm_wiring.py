"""Ba bảng phải khớp nhau, nếu không một arm dense biến mất IM LẶNG.

Một arm dense sống nhờ ba chỗ khai riêng biệt:

    DENSE_ARM_SPECS   tên arm -> (collection, chiều, checkpoint, ngôn ngữ, mô tả)
    DENSE_FILE        tên arm -> tên file `.npz` trong kho phẳng
    Artifact          tên artifact -> chính tên file đó, ở `mfe.offline.bundle`

Lệch một chỗ thì không có gì nổ. `FlatBundle.load` chỉ ghi `missing[name]`, arm
báo "không có vector trong bản gộp", và người đọc kết luận là **tầng offline
chưa chạy xong** — trong khi file nằm sẵn ở đó dưới một cái tên khác.

`SOURCE_FILE` đã có một phép đối chiếu như thế ngay trong `FlatBundle.load`
(nó raise khi lệch `sota_payloads.SOURCE_FILES`). Phía dense chưa có, nên chỗ
canh là ở đây.
"""

from __future__ import annotations

import pytest

from mfe.offline.bundle import ARTIFACTS
from mfe.online.dense_arms import DENSE_ARM_SPECS
from mfe.online.flat_arms import DENSE_FILE


def test_moi_arm_dense_deu_co_file_phang():
    """Thiếu ở `DENSE_FILE` = arm vĩnh viễn không nạp được trên đường phẳng."""
    assert set(DENSE_ARM_SPECS) == set(DENSE_FILE), (
        "DENSE_ARM_SPECS va DENSE_FILE lech nhau: "
        f"chi co spec {set(DENSE_ARM_SPECS) - set(DENSE_FILE)}, "
        f"chi co file {set(DENSE_FILE) - set(DENSE_ARM_SPECS)}"
    )


@pytest.mark.parametrize("arm,filename", sorted(DENSE_FILE.items()))
def test_ten_file_khop_mot_artifact_that(arm: str, filename: str):
    """Tên file phải là tên một `Artifact` sinh ra thật, không phải gõ theo trí nhớ."""
    names = {a.flat_name for a in ARTIFACTS.values()}
    assert filename in names, (
        f"arm {arm} doc {filename!r}, nhung khong Artifact nao sinh ra ten do. "
        f"Co: {sorted(names)}"
    )


@pytest.mark.parametrize("arm,filename", sorted(DENSE_FILE.items()))
def test_so_chieu_khop_artifact(arm: str, filename: str):
    """Chiều khai ở arm phải khớp chiều artifact ghi ra.

    Lệch chiều thì `encode_query` ném — nhưng chỉ ở lượt truy vấn ĐẦU TIÊN có
    arm đó bật, tức là giữa một lượt chạy cả gói.
    """
    artifact = next(a for a in ARTIFACTS.values() if a.flat_name == filename)
    _collection, dim, *_ = DENSE_ARM_SPECS[arm]
    assert artifact.dim == dim, (
        f"arm {arm}: spec khai {dim} chieu, artifact {artifact.name} ghi "
        f"{artifact.dim}"
    )


def test_hai_arm_bge_m3_doc_HAI_file_khac_nhau():
    """`bge_m3` nhúng caption, `subtitle_bge_m3` nhúng phụ đề.

    Cùng model, cùng 1024 chiều, khác nguồn — nên trỏ nhầm nhau thì mọi thứ vẫn
    chạy, cosine vẫn hợp lý, và không có gì báo. Đây là đúng loại lỗi mà chỉ một
    phép so tên file bắt được.
    """
    assert DENSE_FILE["bge_m3"] != DENSE_FILE["subtitle_bge_m3"]
    assert DENSE_FILE["bge_m3"] == "dense_vectors.npz"
    assert DENSE_FILE["subtitle_bge_m3"] == "subtitles_dense.npz"


def test_arm_moi_phai_co_trong_so_khai_TUONG_MINH():
    """Arm không khai trọng số sẽ chạy ở 1,0 — tức đổi hệ đem thi mà không ai quyết.

    `build_flat_dense_arms` và `build_flat_text_arms` đều mặc định
    `weights.get(name, 1.0)`. Với một arm CHƯA ĐO thì mặc định đó là sai hướng:
    nó bật lên đầy đủ trọng số ngay lượt chạy sau khi dữ liệu về.
    """
    from mfe.core.config import load_config  # noqa: PLC0415

    # `FusionWeights` khong phai dict — duyet no cho ra cap `(ten, so)`.
    weights = dict(load_config("config/btc_b1.yaml").retrieval.fusion_weights)
    missing = [name for name in DENSE_ARM_SPECS if name not in weights]
    assert not missing, (
        f"arm dense chua khai trong so trong config: {missing}. "
        "Khai 0.0 neu chua do."
    )
