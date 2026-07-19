# config/

**Owner:** Vỹ Triết. **Spec:** `docs/01-lead-coordination-infra.md` §3.

## Quy tắc

**Không hardcode.** Mọi đường dẫn, tên model, ngưỡng, top-k, trọng số fusion đọc từ đây. Nếu bạn đang gõ một literal path hoặc một con số magic vào code, nó thuộc về file này.

**Không hardcode FPS.** FPS đọc từ metadata từng video (`mfe.offline.probe`), không bao giờ từ config — mỗi video một khác (docs/00 §7, docs/05 §1.1).

## Files

| File | Trạng thái | Dùng khi nào |
|---|---|---|
| `local.yaml` | **Active** | Sprint 1 — cái duy nhất chạy |
| `modal.yaml` | Stub, không kích hoạt | Sprint sau, khi scale-out (docs/01 §3.2) |

## Thêm key mới

1. Thêm vào **cả** `local.yaml` và `modal.yaml` — hai file phải cùng shape, khác value.
2. Thêm field tương ứng vào `src/mfe/core/config.py` để nó được validate lúc load, không phải lúc dùng.
3. Key ảnh hưởng contract giữa 2 workstream (ví dụ `retrieval.embed_dim`, `submit.schema_format`) → báo Vỹ Triết, xem docs/01 §1.

## Vì sao paths ở đây khớp docs/00 §7 từng ký tự

Code nằm ở `src/mfe/`, artifact nằm ở top-level (`data/`, `stores/`, `eval/`). Tách vậy nên đường dẫn artifact trong docs không phải đổi khi ta chọn src layout. Xem `docs/adr/0001-src-layout.md`.
