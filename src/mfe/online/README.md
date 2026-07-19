# online/ — Retrieval cascade

**Owner:** Phạm Trọng. **Spec:** `docs/04-online-retrieval.md`.

## Đây là critical path dài nhất của sprint

docs/04 mở đầu nói thẳng: xếp người mạnh code nhất, hoặc ghép Vỹ Triết hỗ trợ giai đoạn cao điểm.

## Cascade

```
query text
  -> [A] late-fusion coarse: vector + text trên TOÀN kho   -> top 200-500
  -> [B] dedup + temporal grouping                          -> top 50-100
  -> [C] early-fusion rerank: cross-encoder trên ảnh gốc    -> top 10-20
  -> [D] browser: grid + timeline
  -> candidate (+ evidence_id) -> Validator (Nhật Bảo)
```

Tinh thần: model rẻ/nhanh trên toàn kho (A), model đắt/chính xác chỉ trên tập nhỏ (C). Kích thước candidate là **điểm khởi đầu, không phải luật** — benchmark theo GPU/time budget thật.

## Thứ tự làm (docs/01 §4)

`search_text` end-to-end **trước**, rồi B, rồi C. Một lát cắt mỏng chạy được hơn một hệ rộng dựng dở.

## Fallback không phải tính năng tuỳ chọn

docs/04 §1: khi C lỗi hoặc chậm, hệ **tự** quay về kết quả coarse. Timeout đúng lúc cần là mất vòng thi. Ghi log khi fallback nổ — degradation âm thầm trông y hệt regression chất lượng.

## `tools/` — vì sao tách khỏi UI

docs/00 §5 đặt tool-contract ở phần **cross-cutting**, không phải file Phạm Trọng. Lý do: hai consumer — browser (Sprint 1) và agent (Buổi 3).

| Chỗ | Gì |
|---|---|
| `contracts/tools.py` | **shape** — Vỹ Triết sở hữu, ổn định |
| `online/tools/*.py` | **impl** — Phạm Trọng sở hữu |

docs/context/buoi-3.md §2: *"agent không nên thay retrieval engine"*. Test để biết một tool có thuộc đây không: **agent gọi được nó khi không có browser nào chạy không?** Nếu nó cần UI state, nó sai chỗ.

Ngược lại, §9 cảnh báo: *"Action space gọn"* — quá nhiều tool trùng chức năng làm planner chọn sai. Sprint 1: **5 tool, không hơn**.

## Cảnh báo lớn nhất của package này

**Score CLIP và BM25 khác thang — không cộng thẳng** (docs/04 §3.1). CLIP cosine ∈ [-1,1]; BM25 unbounded, phụ thuộc corpus. Cộng thẳng = BM25 âm thầm quyết định mọi ranking.

Trọng số ở config (`retrieval.fusion_weights`). docs/00 §8 rủi ro 4: **đừng calibrate sâu vào data năm ngoái** — hằng số đã tune là một khoản nợ trông giống một cải tiến.

## Đo lường (docs/04 §5)

- Log mỗi query: text, candidate đã xem, click, thời điểm → Vỹ Triết tính **TTCA**
- Đo latency **riêng từng tầng** A/B/C để biết chỗ nghẽn
- Nhớ: *"một model tăng Recall nhưng làm operator chậm hơn có thể khiến tổng điểm giảm"* — tối ưu TTCA, không chỉ Recall

## Đừng viết gì vào đây

- **Query thẳng vào Relational Map** → docs/00 §2 cấm. Ngoại lệ duy nhất: `get_neighbors` (trả ngữ cảnh, không phải xác minh).
- **Logic submit** → `mapping/` (Nhật Bảo). Phạm Trọng chuyển candidate + `evidence_id` sang, dừng ở đó.
- **Tool dính UI state** → xem trên.

## Checklist bàn giao (docs/04 §6)

- [ ] `search_text` end-to-end: query → top-k keyframe có `evidence_id`
- [ ] Cascade A→B→C chạy, có fallback khi C lỗi
- [ ] Fusion weights ở config, không hardcode
- [ ] Browser: grid + mở timeline một thao tác
- [ ] Nút submit tách riêng, đi qua validator Nhật Bảo
- [ ] Log đủ để Vỹ Triết tính TTCA
