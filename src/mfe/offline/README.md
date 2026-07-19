# offline/ — Ingestion pipeline

**Owner:** Quốc Đạt. **Spec:** `docs/02-offline-ingestion.md`.

## Việc của package này

Biến một thư mục video bất kỳ thành dữ liệu cho 4 kho, bằng **một lệnh**:

```bash
python -m mfe.offline.ingest --input data/raw/ --config config/local.yaml
# hoặc
mfe-ingest --input data/raw/ --config config/local.yaml
```

## Tiêu chí thành công — không phải accuracy

docs/02 §1 nói thẳng: giai đoạn này **không** để đạt độ chính xác cao. Nó là lớp orchestrate pretrained models sao cho **chạy lại được**. Data 2026 sẽ khác data mẫu (thời sự / camera tĩnh / lifelog).

- [ ] Thư mục video bất kỳ → chạy hết, không crash
- [ ] Output đúng record format 4 kho đọc được
- [ ] Đổi input path / bật-tắt module chỉ bằng config, không sửa code
- [ ] Ghi throughput ra file cho Vỹ Triết

## Modules

| File | Bước docs/02 §2 | Ghi chú |
|---|---|---|
| `ingest.py` | entrypoint | orchestrate 1→4 |
| `probe.py` | [1] | **fps vào hệ thống ở đây** — đọc từng video, không đoán |
| `keyframes.py` | [2] | sinh 1 lúc 2 record: Asset + Relational |
| `features/` | [3] | cắm-rút qua config (docs/02 §3) |
| `writers.py` | [4] | seam Quốc Đạt→Công Quang/Nhật Bảo |
| `throughput.py` | ngang [1-4] | đo để ước lượng scale (docs/02 §5) |

## Hai chỗ dễ sai nhất

**1. `keyframe_id`.** Luôn dùng `mfe.contracts.ids.build_keyframe_id`, không tự f-string. docs/02 §4: *"Đây là điểm dễ sai nhất — sai một chỗ là cả pipeline lệch."*

**2. `frame_number` là index trong video gốc, 0-indexed** (docs/00 §3) — không phải thứ tự trong danh sách keyframe đã trích. Hai cái này khác nhau ngay khi sampling không phải every-frame.

## Đừng viết gì vào đây

- **Định nghĩa schema** → `contracts/records.py`.
- **Logic đọc/ghi kho** → `stores/` (Công Quang) và `mapping/` (Nhật Bảo). Package này *tạo ra* record, gọi API của họ để ghi.
- **`open()` lên đường dẫn store** → đi qua `mfe.core.storage`, xem `core/README.md`.

## Thêm modality mới (ASR, ...)

Một file mới trong `features/`, một key mới trong `config/*.yaml` dưới `models:`. Không đụng chỗ khác — đó là điểm của `features/base.py`.
