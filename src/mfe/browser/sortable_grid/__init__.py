"""Lưới ảnh kéo thả — xếp hạng ứng viên bằng cách kéo chính tấm ảnh.

Vì sao tự viết thay vì dùng `streamlit-sortables`
--------------------------------------------------
`sort_items` render mỗi item ra **text thuần** (kiểm trong bundle của nó:
`dangerouslySetInnerHTML` chỉ xuất hiện trong nội bộ React, không phải chỗ nó
dựng item). Đặt một danh sách `keyframe_id` kéo được cạnh lưới ảnh nghĩa là
người vận hành phải tự đối chiếu chữ với hình — với 100 khung thì đó là chính
cái việc mà giao diện đáng lẽ phải làm hộ.

Thứ tự dòng có tính điểm (Final Score = trung bình R@k tại k ∈ {1,5,20,50,100}),
nên xếp tay là thao tác **trực tiếp ăn điểm**, không phải trang trí. Nó xứng
đáng có một component riêng.

Không có bước build. Giao thức component của Streamlit là `postMessage` thuần,
và `declare_component(path=…)` phục vụ thư mục tĩnh — nên đây là một file HTML,
đọc được, sửa được, không kéo theo Node.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_FRONTEND = Path(__file__).parent / "frontend"
_component: Any = None


def _declare() -> Any:
    """Khai báo một lần rồi giữ. Khai lại mỗi lần vẽ sẽ mất trạng thái kéo."""
    global _component
    if _component is None:
        import streamlit.components.v1 as components  # noqa: PLC0415

        _component = components.declare_component(
            "mfe_sortable_grid", path=str(_FRONTEND)
        )
    return _component


#: Ảnh nhúng thẳng vào trang dưới dạng data URI, không phục vụ qua HTTP.
#: 100 ô × ~12 KB ≈ 1,2 MB — nặng nhưng một lần, và đổi lại là **không có
#: đường mạng nào** giữa lưới và ổ đĩa. Kho phẳng chạy offline được thì giao
#: diện cũng nên vậy.
THUMB_WIDTH = 320


def _data_uri(path: Path, width: int = THUMB_WIDTH) -> str | None:
    """JPEG thu nhỏ, base64. `None` khi ảnh hỏng hoặc không mở được.

    Trả `None` **im lặng** là đúng ở đây: thiếu một ảnh thì ô đó hiện chữ
    "khong co anh" và vẫn kéo được. Ném sẽ biến một ảnh lỗi thành một lưới
    trắng, tức mất luôn 99 ô còn xem được.
    """
    try:
        from io import BytesIO  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        with Image.open(path) as raw:
            image = raw.convert("RGB")
            image.thumbnail((width, width))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=72)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    except Exception:  # noqa: BLE001 - anh hong thi o do khong co hinh, the thoi
        return None


def sortable_grid(
    keyframe_ids: Sequence[str],
    image_of,
    *,
    machine_rank: dict[str, int] | None = None,
    cols: int = 5,
    key: str | None = None,
) -> list[str] | None:
    """Bày lưới ảnh kéo thả. Trả thứ tự MỚI, hoặc `None` khi chưa ai kéo.

    `None` khác `[]` và khác "thứ tự cũ": phần gọi phải phân biệt được "người
    dùng chưa đụng vào" với "người dùng đã kéo và thứ tự tình cờ giống cũ" —
    cái đầu không được ghi đè gì cả.

    `image_of(keyframe_id) -> Path | None` để component này không biết gì về
    Modal, cache hay đĩa; nó chỉ biết có ảnh hay không.

    `machine_rank` là hạng do máy xếp, hiện trên ô **chỉ khi khác** hạng hiện
    tại — một cột số trùng nhau ở mọi ô là nhiễu, một con số lệch là thông tin.
    """
    machine_rank = machine_rank or {}
    items = []
    for keyframe_id in keyframe_ids:
        path = image_of(keyframe_id)
        items.append({
            "id": keyframe_id,
            "src": _data_uri(path) if path is not None else None,
            "was": machine_rank.get(keyframe_id),
        })
    return _declare()(items=items, cols=cols, key=key, default=None)
