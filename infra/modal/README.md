# infra/modal/ — STUB. Không kích hoạt Sprint 1.

**Owner:** Lead. **Spec:** `docs/01-lead-coordination-infra.md` §3.2.

docs/00 §1 xếp Modal ở mức **stub** trong sprint này. Thư mục này giữ chỗ.

## Mục tiêu thiết kế

> "Đổi từ local sang Modal chỉ bằng đổi config, không sửa logic" — docs/01 §3.2

Ba mảnh đã dựng sẵn để lúc đó diff nằm gọn:

| Mảnh | Ở đâu | Trạng thái |
|---|---|---|
| Storage backend | `src/mfe/core/backends/modal.py` | Stub — raise NOT_IMPLEMENTED |
| Config | `config/modal.yaml` | Khung đủ, không active |
| Deploy | `infra/modal/` (đây) | Trống |

Stub **fail to**, không âm thầm fallback về local. Fallback âm thầm khiến người ta tưởng đang test Modal trong khi không.

## TODO khi scale — ghi ở đây, KHÔNG làm trong Sprint 1

docs/01 §3.2 liệt kê thứ phải ước lượng trước khi bắt đầu:

- [ ] Trích keyframe hàng trăm giờ video — bao lâu? (dùng số `eval/throughput.json` của TV1)
- [ ] Kích thước Faiss index — vào RAM nổi không? Cần IVF/PQ thay Flat?
- [ ] RAM/VRAM — batch size CLIP/detector trần ở đâu?
- [ ] Asset Store — dung lượng volume?

Số cho những câu này do TV1 đo từ mẫu nhỏ ngay Sprint 1 (docs/02 §5). Đó là lý do đo sớm: *"thứ hay vỡ khi scale không phải logic mà là thông lượng và lưu trữ."*

## Cảnh báo scope

docs/00 §8 rủi ro 3: **cấm** chạy full offline trước khi mapping đúng 100% trên mẫu nhỏ. Đừng dùng thư mục này để lách luật đó.
