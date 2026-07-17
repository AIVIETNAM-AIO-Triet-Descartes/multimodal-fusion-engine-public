# eval/ (code) — Benchmark harness

**Owner:** Lead. **Spec:** `docs/01-lead-coordination-infra.md` §2.

## Đây là code. Data ở chỗ khác.

| Đây | Là gì |
|---|---|
| `src/mfe/eval/` | **code** — package này |
| `eval/` ở gốc repo | **data** — `queryset.jsonl` (tracked), `results/` (gitignored) |

Xem `docs/adr/0001-src-layout.md` — chính va chạm tên này là lý do chọn src layout.

## Chạy

```bash
python -m mfe.eval.run --queryset eval/queryset.jsonl --config config/local.yaml --out eval/results/
# hoặc
mfe-eval --queryset eval/queryset.jsonl --config config/local.yaml --out eval/results/
```

## Vai hay bị bỏ quên nhất

docs/01 §2 nói thẳng: đây là thứ cho nhóm biết **có tiến bộ thật hay không** — và là vai hay bị bỏ quên nhất. Không ai thấy workstream của mình bị chặn khi thiếu nó. Đó chính là vấn đề.

## Output

Một dòng mỗi query:

| query_id | type | Recall@1 | Recall@5 | latency_ms | TTCA_s | submit_valid |
|---|---|---|---|---|---|---|
| q001 | KIS | 0/1 | 1/1 | 180 | 12.3 | ✓ |

Cộng dồn: **median** TTCA (không phải mean — một query operator bỏ cuộc sẽ kéo mean thành vô nghĩa), mean Recall@K theo `type`, tỷ lệ submit hợp lệ.

Cộng: in **PASS/FAIL** từng tiêu chí "Done" ở docs/00 §1.

## Input — do TV4 cung cấp

```json
{"query_id": "q001", "text": "người đàn ông cầm túi tím bước xuống xe",
 "gt_keyframe_ids": ["L01_V001#001234"], "gt_video_id": "L01_V001", "type": "KIS"}
```

TV4 hiểu mapping/ground truth rõ nhất nên phụ trách query set (docs/05 §4). Mục tiêu ≥20 query, KIS trước.

Đây là điểm đồng bộ TV4↔Lead (docs/01 §1): lệch format → harness không chấm được gì.

## Query crash không phải lý do dừng

Bắt, ghi lại, chạy tiếp. *"Chạy được ≥30 query nội bộ không crash"* chỉ đo được nếu run chạy tới cuối.

## Metric định nghĩa ở đâu

`metrics.py` — một chỗ duy nhất (docs/00 §6). File khác chỉ tham chiếu, không định nghĩa lại.

Riêng **throughput** do TV1 sinh (`offline/throughput.py` → `eval/throughput.json`), Lead chỉ tổng hợp.
