# eval/ — DATA (không phải code)

**Owner:** `queryset.jsonl` → Nhật Bảo. `results/` → sinh bởi harness.

| Đây | Là gì |
|---|---|
| `eval/` (thư mục này) | **data** — query set + kết quả benchmark |
| `src/mfe/eval/` | **code** — harness |

Chính va chạm tên này là một trong hai lý do chọn src layout. Xem `docs/adr/0001-src-layout.md`.

## Files

| File | Git | Ai sinh |
|---|---|---|
| `queryset.jsonl` | **Tracked** | Nhật Bảo — ground truth, nhỏ, giá trị cao |
| `results/` | Gitignored | `mfe-eval` mỗi lần chạy |
| `throughput.json` | Gitignored | Quốc Đạt (`offline/throughput.py`) |

## Format `queryset.jsonl`

Một dòng một query (docs/01 §2, docs/05 §4):

```json
{"query_id": "q001", "text": "người đàn ông cầm túi tím bước xuống xe", "gt_keyframe_ids": ["L01_V001#001234"], "gt_video_id": "L01_V001", "type": "KIS"}
```

| Field | Nghĩa |
|---|---|
| `query_id` | Định danh query |
| `text` | Mô tả khoảnh khắc cần tìm |
| `gt_keyframe_ids` | Ground truth — **nhiều** id, vì một khoảng frame có thể đều đúng |
| `gt_video_id` | Video chứa đáp án |
| `type` | `KIS` trước. QA đa bước để sau |

## Mục tiêu Sprint 1

**≥ 20 query** (docs/05 §5), KIS trước. docs/00 §6 nói 20–50 query đa bước.

Nhật Bảo phụ trách vì hiểu mapping/ground truth rõ nhất. Đây là điểm đồng bộ Nhật Bảo↔Vỹ Triết (docs/01 §1): lệch format → harness không chấm được gì.

## Vì sao `gt_keyframe_ids` là list

docs/context/buoi-1.md §7: BTC có thể định nghĩa **một khoảng frame liên tiếp** là hợp lệ. Chấm bằng so khớp đúng một id có thể chặt hơn scoring thật, và làm hệ trông kém hơn thực tế.

## Chạy

```bash
mfe-eval --queryset eval/queryset.jsonl --config config/local.yaml --out eval/results/
```
