# stores/ — 3 kho TÌM KIẾM

**Owner:** TV2. **Spec:** `docs/03-storage-indexes.md`.

## Việc của package này

Hiện thực 3 kho, mở API **ghi** cho TV1 và API **đọc** cho TV3.

| Kho | Vai trò trong cascade | Backend đề xuất |
|---|---|---|
| `vector/` | Late-fusion coarse (tầng A) — semantic | Faiss + sidecar |
| `inverted/` | Lọc / rerank theo chữ, nhãn, số | BM25 |
| `asset/` | Bytes ảnh gốc — cho early-fusion rerank (tầng C) + browser | file/volume |

## Kho thứ 4 KHÔNG ở đây

Relational Map là của TV4 (`src/mfe/mapping/`). Nó không phải kho tìm kiếm — nó là **nguồn sự thật để xác minh**. docs/00 §2: *"retrieval không query thẳng vào Relational Map"*.

## Đừng nhầm `stores/` (code) với `stores/` (artifact)

| Đây | Là gì |
|---|---|
| `src/mfe/stores/` | **code** — package này |
| `stores/` ở gốc repo | **artifact** — Faiss index, BM25 index, sqlite. Gitignored |

Đường dẫn artifact đọc từ `config.paths.*_store`, không hardcode. Xem `docs/adr/0001-src-layout.md`.

## Chỗ kho này hỏng âm thầm nhất

**Sidecar lệch một dòng.** Faiss chỉ biết `internal_id → embedding`. Nếu sidecar lệch, mọi query trả về một keyframe **có thật, hợp lý, và sai** — không có gì trong kết quả trông hỏng cả. docs/03 §4 đòi chính xác **100%**, không phải 99%.

Gọi `Sidecar.verify_alignment()` mỗi lần `load`.

## Score không cùng thang

BM25 unbounded và phụ thuộc corpus; CLIP cosine ∈ [-1,1]. **Đừng pre-normalize ở đây** — TV3 chuẩn hóa lúc fusion (docs/04 §3.1) và cần thông tin gốc để làm việc đó.

## Đồng bộ phải chốt sớm

| Với ai | Chốt gì | Vỡ gì nếu lệch |
|---|---|---|
| TV1 | Vocabulary nhãn object, normalization của OCR text | Term không bao giờ match |
| TV3 | `get_asset` trả bytes hay path? Ảnh cỡ nào? | Cross-encoder và UI đều kẹt |
| TV3 | Cấu trúc `evidence_id` trên Candidate | Validator không map ngược được |

## Checklist bàn giao (docs/03 §4)

- [ ] Vector Index build từ output TV1, `search` trả đúng `keyframe_id`
- [ ] Sidecar dịch `internal_id ↔ keyframe_id` chính xác 100%
- [ ] `search` + `filter` chạy trên object/ocr/metadata
- [ ] `get_asset` trả đúng ảnh cho một `keyframe_id`
- [ ] API đọc cho TV3 và API ghi cho TV1 đã thống nhất

## Thêm kho mới (scene graph, ...)

Thư mục mới trong `src/mfe/stores/`, path mới trong `config.paths`. Không đụng kho có sẵn.
