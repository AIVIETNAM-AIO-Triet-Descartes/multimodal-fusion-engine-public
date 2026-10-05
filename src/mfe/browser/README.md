# browser/ — UI tối thiểu

**Owner:** Phạm Trọng. **Spec:** `docs/04-online-retrieval.md` §4.

## Stack Sprint 1: Streamlit, in-repo

```bash
streamlit run src/mfe/browser/app.py
```

Chọn Python in-repo vì đúng tinh thần "MVP, không gold-plate". Cài bằng extra `browser`:

```bash
pip install -e ".[browser]"
```

## Giữ ở mức MVP

docs/00 §8 xếp **gold-plate browser** là rủi ro thứ 2 của sprint: làm hết 45 slide Buổi 2 thay vì tiến tới đủ MVP.

| Khả năng | Trạng thái |
|---|---|
| Grid ổn định, diversity, không tràn layout | Bắt buộc |
| Timeline/neighbor — mở trước/sau **một thao tác** | Bắt buộc |
| Query refine | Bắt buộc |
| Positive/negative feedback | Nên có |
| Sketch / pose / scene-graph | **Để sau — đừng bắt đầu** |

## UI đẹp không có điểm — UI nhanh thì có

docs/context/buoi-1.md §7: kết quả đúng frame mới là trọng tâm. Nhưng UI hữu dụng giúp tìm nhanh hơn → giảm TTCA → **có** điểm. Tối ưu cho tay của operator, không cho ảnh chụp màn hình.

## Hai luật cứng

**1. Tách nút "xem" và "submit"** (docs/04 §4). Submit phải qua bước xác nhận **và** đi qua validator Nhật Bảo. Nút submit một-click cạnh nút xem một-click là cách một đáp án sai rời khỏi phòng thi ở tốc độ cao.

**2. Log mọi tương tác** qua `mfe.core.logging` — query, candidate đã xem, click, thời điểm. Thiếu thì Vỹ Triết không tính được TTCA (docs/04 §5), mà TTCA là thứ cả cái browser này sinh ra để cải thiện.

## Đừng viết gì vào đây

- **Logic retrieval** → `online/`. Browser là *consumer* của tool, không phải chỗ chứa tool.
- **Tool mới** → `online/tools/`. docs/04 §2: tool "không dính chặt vào UI" — agent Buổi 3 phải gọi lại được.

## Nếu sau này cần frontend riêng

Không sửa package này. Thêm:

```
apps/web/          frontend (React/...)
src/mfe/api/       service layer (FastAPI) — gọi cùng online/tools/
```

`browser/` chỉ là một consumer trong nhiều consumer. Cây thư mục không vỡ.
