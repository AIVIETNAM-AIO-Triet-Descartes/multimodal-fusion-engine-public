"""Bộ tuỳ chọn truy hồi có PHẠM VI — cả gói < một dạng < một câu.

Giao diện nộp cần hỏi "câu này nếu tắt `vlm` thì sao", và cần hỏi ở ba mức: cả
gói, riêng một dạng đề, riêng một câu. Ba dạng cần thứ khác nhau — QA sống nhờ
OCR và phụ đề, KIS nhiều cảnh cần thứ tự cảnh, còn với TRAKE thì `vlm` vô nghĩa
vì mỗi dòng là một **chuỗi** chứ không phải một khung.

Vì sao ở thư viện chứ không trong `app/submit_lab.py`
------------------------------------------------------
Nó quyết định arm nào chạy cho câu nào, tức nó quyết định **nội dung file nộp**.
Một script Streamlit chạy mã ở cấp module ngay khi import nên không test nào với
tới được, và logic nộp không có test là logic nộp sẽ sai lặng lẽ. Cùng lý do
`mfe.eval.submission_bundle` được tách ra.

`None` nghĩa là "theo cấp trên", KHÔNG phải "tắt"
--------------------------------------------------
Đây là chỗ dễ sai nhất và nó sai theo hướng khó thấy. Nếu một ô bỏ trống ở cấp
"dạng" ghi đè bằng `None` thì nó **xoá** lựa chọn của cấp "gói" thay vì kế thừa
— người vận hành đặt `vlm` cho cả gói, mở bảng của dạng KIS ra xem rồi đóng lại,
và `vlm` biến mất khỏi KIS mà không có gì báo.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

#: Ba dạng đề của vòng sơ tuyển. Thứ tự này cũng là thứ tự tab trong giao diện.
QUERY_TYPES = ("KIS", "QA", "TRAKE")


@dataclass(frozen=True)
class Options:
    """Một bộ tuỳ chọn. Trường `None` = chưa chọn, để cấp trên quyết."""

    #: `{tên arm: bật}`. Chỉ chứa arm đã được chọn tường minh ở phạm vi này.
    arms: dict[str, bool] = field(default_factory=dict)
    #: Danh sách tầng rerank theo tên (`mfe.online.rerank_tiers`).
    tiers: tuple[str, ...] | None = None
    scene_order: float | None = None

    def merge(self, other: Options) -> Options:
        """`other` đè lên `self`, nhưng trường `None` của `other` thì KHÔNG đè.

        `arms` gộp theo từng khoá chứ không thay cả dict: phạm vi hẹp thường chỉ
        muốn nói về một arm, và thay cả dict sẽ lặng lẽ tắt mọi arm nó không
        nhắc tới.
        """
        return Options(
            arms={**self.arms, **other.arms},
            tiers=other.tiers if other.tiers is not None else self.tiers,
            scene_order=(
                other.scene_order if other.scene_order is not None else self.scene_order
            ),
        )

    def with_tiers(self, tiers) -> Options:
        return replace(self, tiers=None if tiers is None else tuple(tiers))


def scope_keys(query_id: str, query_type: str) -> tuple[str, str, str]:
    """Ba khoá phạm vi cho một câu, **từ rộng tới hẹp**.

    Trả về theo đúng thứ tự áp dụng để phần gọi không phải tự nhớ chiều — đảo
    chiều là câu thua dạng, và người vận hành sẽ thấy lựa chọn riêng của mình bị
    bỏ qua mà không hiểu vì sao.
    """
    return ("goi", f"dang_{query_type}", f"cau_{query_id}")


def resolve(store, query_id: str, query_type: str) -> Options:
    """Bộ tuỳ chọn hiệu lực cho một câu: **gói < dạng < câu**.

    `store` là bất cứ mapping nào (`st.session_state`, hay một dict trong test).
    Khoá thiếu thì coi như chưa chọn gì ở phạm vi đó.
    """
    resolved = Options()
    for key in scope_keys(query_id, query_type):
        resolved = resolved.merge(store.get(key) or Options())
    return resolved
