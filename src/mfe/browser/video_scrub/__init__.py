"""Trình phát video có nút "lấy đúng khung đang dừng" gửi thẳng về Python.

Vì sao phải là component, không phải `components.html`
------------------------------------------------------
`st.components.v1.html` chỉ **vẽ** một trang tĩnh: nó không có đường trả giá trị
về. Nên bản đầu của bàn nộp phải bày thêm hai ô số để người vận hành **chép tay**
số khung từ trình phát sang — đúng thứ thừa mà một giao diện không nên bắt người
dùng làm, và cũng là chỗ dễ nộp nhầm nhất: đổi ứng viên mà quên đổi ô số thì màn
hình chiếu một khung còn bài nộp ghi một khung khác.

`declare_component` thì có `setComponentValue`. Cùng cơ chế `postMessage` mà
`sortable_grid` đang dùng, cùng cách phục vụ thư mục tĩnh, **không** bước build.

Vì sao có `nonce`
------------------
Giá trị của một component Streamlit **sống qua mọi lượt vẽ lại**, không chỉ lượt
ngay sau thao tác. Không có số đếm thì chọn lại đúng khung cũ sẽ không được ghi
nhận (giá trị không đổi ⇒ Streamlit coi như không có gì xảy ra), còn phần gọi lại
không phân biệt được "vừa bấm" với "giá trị cũ còn đó" nên dễ rơi vào vòng lặp
`st.rerun()` vô hạn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_FRONTEND = Path(__file__).parent / "frontend"
_component: Any = None


def _declare() -> Any:
    """Khai báo một lần rồi giữ. Khai lại mỗi lượt vẽ là mất vị trí đang tua."""
    global _component
    if _component is None:
        import streamlit.components.v1 as components  # noqa: PLC0415

        _component = components.declare_component("mfe_video_scrub", path=str(_FRONTEND))
    return _component


def video_scrub(
    url: str,
    fps: float,
    start_seconds: float,
    *,
    action_label: str = "Thay khung này",
    key: str | None = None,
) -> dict[str, Any] | None:
    """Bày trình phát, trả `{"frame", "seconds", "nonce"}` khi người dùng chọn.

    `None` nghĩa là chưa ai bấm chọn lần nào trong phiên này.

    `seconds` is the presented frame's mediaTime from requestVideoFrameCallback.
    Selection is blocked without reliable PTS. `frame` is an FPS-based display
    estimate only; it is NOT an authoritative decoded index for VFR videos.
    """
    return _declare()(
        url=url,
        fps=float(fps),
        start=float(start_seconds),
        label=action_label,
        key=key,
        default=None,
    )
