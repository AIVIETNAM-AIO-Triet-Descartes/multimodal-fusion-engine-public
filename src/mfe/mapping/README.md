# mapping/ — Relational Map & Validator (P0)

**Owner:** Nhật Bảo. **Spec:** `docs/05-validator-mapping.md`.

## Đây là P0 — làm sớm nhất

docs/01 §4 xếp package này **bước 1** trong thứ tự tích hợp, trước cả ingestion. Lý do: mọi thứ khác đều nối vào `keyframe_id`, và đây là nơi định nghĩa `keyframe_id` nghĩa là gì.

docs/context/buoi-1.md §8 xếp mapping sai là chỗ **số một** các đội mất điểm: *"tìm đúng hình nhưng submit sai ID/frame là lỗi trắng."*

## Vì sao mapping và validator do một người

docs/05 mở đầu: kho quan hệ **chính là** validator core. Tách hai người → schema lệch.

Và: phần này **generalize 100%** sang data mới — nó là logic ID, không phụ thuộc pretrained model nào. Mỗi giờ đầu tư ở đây sống sót qua data 2026; nhiều phần khác thì không.

## Luồng — retrieval KHÔNG đọc thẳng vào đây

```
Phạm Trọng cascade ──> chọn keyframe_id ──> [validator] ──> [submit adapter] ──> submit
                                          │
                                          └── tra Relational Map (bước cuối, ở đây)
```

docs/00 §2: *"retrieval không query thẳng vào Relational Map"*. Ba kho tìm kiếm phục vụ **tìm**; kho này phục vụ **xác minh**.

Ngoại lệ duy nhất: `get_neighbors` — nó trả ngữ cảnh, không phải xác minh.

## Modules

| File | Vai trò |
|---|---|
| `schema.sql` | 2 bảng: `videos`, `keyframes` |
| `relational.py` | build + tra cứu nguồn sự thật |
| `validator.py` | 5 luật chặn submit (docs/05 §3) |
| `submit.py` + `adapters/` | dịch `keyframe_id` → record submit |
| `backends/` | SQLite giờ, Postgres khi scale |

## Nguyên tắc của validator

**Submit sai còn tệ hơn không submit** (docs/05 §3). Mất thời gian tìm lại, có thể bị phạt. Nên: *"thà chặn nhầm còn hơn cho lọt."* Luật nào mơ hồ → chặn.

## Vì sao 4 adapter cho một schema

Schema BTC còn `TO_CONFIRM` (docs/00 §3). docs/context/buoi-1.md §7 ghi giảng viên nói có thể là timestamp mili-giây, **hoặc** frame number. Giữ cả 4 sống, config chọn. BTC công bố → thêm một file, không sửa code khác.

## Đừng viết gì vào đây

- **Logic retrieval/ranking** → `online/` (Phạm Trọng).
- **Suy `timestamp_ms` từ `frame_number` lúc submit.** Đọc thẳng từ bảng. docs/00 §3 lưu cả hai cột chính vì lý do này — xem `docs/adr/0002-id-format.md`, ví dụ trong docs từng lệch nhau vì rounding.

## Checklist bàn giao (docs/05 §5)

- [ ] Relational Map dựng từ output Quốc Đạt, mapping đúng **100%** trên mẫu
- [ ] `build_submission` xuất được cả timestamp và frame_number
- [ ] Validator chặn đủ 5 luật
- [ ] `keyframe_id` không tồn tại → báo lỗi, **không đoán**
- [ ] Query set ≥ 20 query + ground truth ở `eval/queryset.jsonl`
