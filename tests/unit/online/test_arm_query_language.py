"""Mỗi arm văn bản phải khai ngôn ngữ KHỚP với ngôn ngữ của chỉ mục nó đọc.

Vì sao có file này
-------------------
`TEXT_ARM_SPECS` trước 21-09-2026 chỉ có `(source, phrase)` — **không có chỗ khai
ngôn ngữ** — nên mọi arm văn bản nhận mặc định `query_language="vi"` của dataclass
`Arm`. Nhưng `caption_qwen` đọc `qwen_captions.json`, là nửa `[CAPTION]` **viết
tiếng Anh** theo quyết định 25-08.

Kết quả: một chỉ mục tiếng Anh bị tìm bằng truy vấn tiếng Việt, **trọng số 1,0**,
suốt gần một tháng. Đo 21-09 trên 12 câu dev đầu, chỉ đổi chuỗi truy vấn:
`vi` 6/12 · hạng TB 37,2  so với  `en` 10/12 · hạng TB 20,8.

Điều làm nó sống lâu: cả `_DESCRIPTIONS` lẫn CLAUDE.md §4 đều ghi arm này "cần
bản dịch truy vấn". Tài liệu đúng, code không có cách nào nói ra. Test này là chỗ
hai thứ đó buộc phải khớp nhau.
"""

from __future__ import annotations

import pytest

from mfe.online.arms import Arm
from mfe.online.text_arms import TEXT_ARM_SPECS

#: Arm nào đọc chỉ mục TIẾNG ANH. Suy thẳng từ prompt sinh ra nguồn đó:
#: nửa `[CAPTION]` viết tiếng Anh, nửa `[TEXT]` cấm dịch.
ARM_TIENG_ANH = {"caption", "caption_qwen"}


def test_moi_spec_co_du_ba_cot():
    for ten, spec in TEXT_ARM_SPECS.items():
        assert len(spec) == 3, f"{ten}: spec phải là (source, phrase, query_language)"


@pytest.mark.parametrize("ten", sorted(TEXT_ARM_SPECS))
def test_ngon_ngu_nam_trong_tap_dong(ten):
    assert TEXT_ARM_SPECS[ten][2] in {"vi", "en", "any"}


@pytest.mark.parametrize("ten", sorted(ARM_TIENG_ANH))
def test_arm_doc_chi_muc_tieng_ANH_phai_khai_en(ten):
    """Đây là ca đã trả giá. Đừng đổi về `vi` mà không đo lại."""
    assert TEXT_ARM_SPECS[ten][2] == "en", (
        f"{ten} đọc caption tiếng Anh; khai `vi` là tìm chỉ mục Anh bằng truy vấn Việt"
    )


@pytest.mark.parametrize(
    "ten", sorted(set(TEXT_ARM_SPECS) - ARM_TIENG_ANH)
)
def test_arm_con_lai_doc_chi_muc_TIENG_VIET(ten):
    """Phụ đề, OCR và nhãn vật thể đều là tiếng Việt.

    `object` cũng `vi` vì `labels_vi.translate_labels` đã dịch nhãn OpenImages
    sang tiếng Việt **lúc dựng chỉ mục** — không dịch thì giữa chỉ mục và truy
    vấn không còn một từ nào chung.
    """
    assert TEXT_ARM_SPECS[ten][2] == "vi"


def test_arm_khai_en_bi_BO_QUA_khi_thieu_ban_dich():
    """Thiếu bản dịch thì arm phải ngồi ngoài VÀ NÓI TO, không được lặng lẽ
    thay bằng tiếng Việt.

    Nếu nó âm thầm dùng chuỗi tiếng Việt thì lỗi này sẽ không bao giờ lộ ra —
    đúng cách nó đã sống gần một tháng.
    """
    from mfe.online.arms import ArmRegistry

    reg = ArmRegistry()
    reg.register(
        Arm(
            name="chi_tieng_anh",
            search=lambda q, k, run: [],
            query_language="en",
            description="x",
        )
    )
    ket = reg.run("câu tiếng Việt", top_k=5, run_id="t", query_text_en=None)
    assert "chi_tieng_anh" not in ket


def test_arm_khai_en_CHAY_khi_co_ban_dich():
    from mfe.online.arms import ArmRegistry

    nhan = {}

    def search(q, k, run):
        nhan["q"] = q
        return []

    reg = ArmRegistry()
    reg.register(
        Arm(name="chi_tieng_anh", search=search, query_language="en", description="x")
    )
    reg.run("câu tiếng Việt", top_k=5, run_id="t", query_text_en="an english query")
    assert nhan["q"] == "an english query", "arm `en` phải nhận CHUỖI ĐÃ DỊCH"
