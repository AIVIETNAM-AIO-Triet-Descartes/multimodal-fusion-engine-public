# agent/ — CHỖ ĐÃ CHỪA cho Buổi 3. Chưa code gì trong sprint này.

**Owner:** chưa phân. **Spec:** `docs/context/buoi-3.md`. **Trạng thái:** ngoài scope Sprint 1 (docs/00 §1).

Thư mục này cố ý rỗng. Nó tồn tại để trả lời câu hỏi "sau này agent đặt ở đâu" **một lần**, ngay bây giờ, thay vì để nó thành một cuộc tái cấu trúc lúc sprint sau.

---

## Điều kiện tiên quyết — đọc trước khi viết dòng đầu tiên

docs/context/buoi-3.md §3: *"Agent chỉ đáng thêm sau khi từng tool đơn lẻ có API rõ, test và benchmark. Nếu search cơ sở chưa tốt, agent chỉ tự động hóa một pipeline yếu."*

Nghĩa là: `online/tools/` phải xong, có test, có số benchmark **trước**. Không có ngoại lệ.

## Luật số một: agent GỌI LẠI tool, không viết lại

docs/context/buoi-3.md §2: *"agent không nên thay retrieval engine. Agent nên điều phối các công cụ đã đo được chất lượng."*

```python
from mfe.online.tools import registry   # ĐÚNG — dùng lại tool Phạm Trọng đã đo
# KHÔNG: tự viết search trong agent/
```

Đây chính là lý do `online/tools/` tách khỏi `browser/` từ Sprint 1. Nếu tool dính UI state, agent không gọi được, và cả thiết kế này vô nghĩa.

## Khi tới lúc, đặt vào đâu

| Thành phần | File dự kiến | Nguồn |
|---|---|---|
| Planner (ReAct) | `planner.py` | buoi-3 §4, §8 |
| Memory: episodic/semantic/procedural | `memory.py` | buoi-3 §4 |
| World model | `world_model.py` | buoi-3 §4 |
| VQA reader | `reader.py` | buoi-3 §6.2 |
| Guardrails: step/time/cost budget | `guardrails.py` | buoi-3 §7 |
| Routing: query nào cần agent | `router.py` | buoi-3 §9 |

Tool **spatial/temporal** mới (`select_segment`, `crop`, `ocr` vùng ảnh) → `online/tools/`, **không** vào đây. Chúng là năng lực retrieval, agent chỉ là người gọi.

## Guardrail tối thiểu (buoi-3 §7) — không phải tuỳ chọn

- Giới hạn tổng step, thời gian, chi phí mỗi query
- **Agent không được gọi submit trực tiếp** khi chưa qua validator/human approval ở interactive track
- Bắt buộc answer trỏ về frame/segment bằng chứng
- Retry có giới hạn; tool lỗi phải fail rõ
- Log plan, tool arguments, result, latency, final answer
- Có fallback về browser thủ công

## Exit criteria (buoi-3 §10)

Agent chỉ được coi là sẵn sàng khi:
- Không giảm final accuracy so với browser trên query **đơn giản**
- Cải thiện rõ trên query **đa bước**
- 100% câu trả lời có evidence hợp lệ
- Không vượt step/time budget
- Mọi tool trace tái hiện được

## Cảnh báo

buoi-3 §8: giảng viên nói ở interactive track, trợ lý ảo/chatbot là **tùy chọn**, không bắt buộc. Thông tin về automatic track: không có (`TO_CONFIRM`).

buoi-3 §9: *"query KIS đơn giản đi thẳng browser; query QA đa sự kiện mới dùng agent. Không dùng một pipeline đắt cho mọi câu."* Và: với event retrieval đơn giản, **người quen hệ thống có thể nhanh hơn agent**.
