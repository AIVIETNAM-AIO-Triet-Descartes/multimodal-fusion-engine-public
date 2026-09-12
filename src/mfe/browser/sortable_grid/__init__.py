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

    Đo được **14 ms/ảnh** (1,43 s cho 100 ảnh) — nhỏ một lần, nhưng Streamlit
    vẽ lại toàn bộ trang sau **mỗi** thao tác, nên không cache thì mỗi lần kéo
    một tấm là trả lại 1,4 giây cho 99 tấm không đổi. `_cached_uri` bọc nó.
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


def _cached_uri(path_str: str, mtime: float, width: int) -> str | None:
    """`_data_uri` có nhớ, khoá theo (đường dẫn, mtime, bề rộng).

    `mtime` nằm trong khoá chứ không phải để trang trí: ảnh keyframe được tải
    dần từ Modal vào đúng đường dẫn cũ, nên một khoá chỉ-đường-dẫn sẽ giữ mãi
    bản đọc hỏng của lần tải dở.
    """
    import streamlit as st  # noqa: PLC0415

    @st.cache_data(show_spinner=False, max_entries=2048)
    def _inner(path_str: str, mtime: float, width: int) -> str | None:
        return _data_uri(Path(path_str), width)

    return _inner(path_str, mtime, width)


def sortable_grid(
    keyframe_ids: Sequence[str],
    image_of,
    *,
    machine_rank: dict[str, int] | None = None,
    cols: int = 5,
    key: str | None = None,
) -> dict[str, Any] | None:
    """Bày lưới ảnh kéo thả. Trả `{"order": [...], "click": id|None}`, hoặc
    `None` khi chưa ai đụng vào.

    `None` khác `[]` và khác "thứ tự cũ": phần gọi phải phân biệt được "người
    dùng chưa đụng vào" với "người dùng đã kéo và thứ tự tình cờ giống cũ" —
    cái đầu không được ghi đè gì cả.

    **`click` là ô người dùng vừa bấm**, để phần gọi mở đúng khung đó ra xem.
    Kéo và bấm đi chung một đường trả về vì component của Streamlit chỉ có
    **một** giá trị: tách thành hai component là tách luôn trạng thái kéo.

    `image_of(keyframe_id) -> Path | None` để component này không biết gì về
    Modal, cache hay đĩa; nó chỉ biết có ảnh hay không.

    `machine_rank` là hạng do máy xếp, hiện trên ô **chỉ khi khác** hạng hiện
    tại — một cột số trùng nhau ở mọi ô là nhiễu, một con số lệch là thông tin.
    """
    machine_rank = machine_rank or {}
    items = []
    for keyframe_id in keyframe_ids:
        path = image_of(keyframe_id)
        src = None
        if path is not None:
            try:
                src = _cached_uri(str(path), path.stat().st_mtime, THUMB_WIDTH)
            except OSError:
                src = None
        items.append({
            "id": keyframe_id,
            "src": src,
            "was": machine_rank.get(keyframe_id),
        })
    return _declare()(items=items, cols=cols, key=key, default=None)
