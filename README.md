# multimodal-fusion-engine

Hệ truy xuất video cho **AIC 2026**. Nhận một mô tả bằng chữ, trả về đúng khoảnh khắc trong một kho video lớn, dưới áp lực thời gian.

Đây không phải bài toán "nhét video vào AI rồi hỏi". Nó là: tìm keyframe ứng viên → xác định đúng video và đúng khoảnh khắc → kiểm ngữ cảnh trước/sau → trả `video_id` + frame/timestamp đúng schema BTC.

> **Trạng thái: Sprint 1 — scaffold.** `src/mfe/contracts/` có code chạy được và test. Phần còn lại là stub có signature + README ownership, chờ 4 workstream điền.

---

## Bắt đầu

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate trên Unix
pip install -e ".[dev]"

pytest tests/unit/contracts -v    # 52 test, không cần torch/faiss
```

Cài thêm theo workstream của bạn: `pip install -e ".[offline]"` (Quốc Đạt), `".[stores]"` (Công Quang), `".[online,browser]"` (Phạm Trọng). Nhật Bảo và Vỹ Triết chỉ cần `dev`.

Hai lệnh của sprint (chưa hiện thực):

```bash
mfe-ingest --input data/sample/ --config config/local.yaml
mfe-eval --queryset eval/queryset.jsonl --config config/local.yaml --out eval/results/
```

---

## Đọc gì trước khi code

**Bắt buộc, theo thứ tự:**

1. [`docs/00-overview.md`](docs/00-overview.md) — **contract**. Nguồn duy nhất. Đọc hết trước khi gõ dòng đầu tiên.
2. File cá nhân của bạn (bảng dưới).
3. README của package bạn sở hữu — nó nói *đừng* viết gì vào đó.

**Nền:** [`docs/context/`](docs/context/) — ghi chú Buổi 1 (bài toán & dữ liệu), Buổi 2 (browser tương tác & fusion), Buổi 3 (agent).

**Quyết định:** [`docs/adr/`](docs/adr/) — vì sao lệnh chạy là `mfe.*`, và chuẩn ID/timestamp.

---

## Bản đồ workstream

| Người | Vai | Spec | Code |
|---|---|---|---|
| Vỹ Triết | Contract, eval, infra | [`docs/01`](docs/01-lead-coordination-infra.md) | [`contracts/`](src/mfe/contracts/) [`core/`](src/mfe/core/) [`eval/`](src/mfe/eval/) |
| Quốc Đạt | Offline ingestion | [`docs/02`](docs/02-offline-ingestion.md) | [`offline/`](src/mfe/offline/) |
| Công Quang | Storage (3 kho tìm kiếm) | [`docs/03`](docs/03-storage-indexes.md) | [`stores/`](src/mfe/stores/) |
| Phạm Trọng | Online retrieval + browser | [`docs/04`](docs/04-online-retrieval.md) | [`online/`](src/mfe/online/) [`browser/`](src/mfe/browser/) |
| Nhật Bảo | Validator / Mapping **(P0)** | [`docs/05`](docs/05-validator-mapping.md) | [`mapping/`](src/mfe/mapping/) |

**`src/mfe/<workstream>/` = một owner.** Đó là luật giữ cây thư mục không rối khi thêm sprint.

---

## Kiến trúc

```
                          OFFLINE
video mẫu (local)
  └─ [Quốc Đạt] ingestion: keyframe + pretrained (CLIP/object/OCR/metadata)
       └─ ghi ra 4 KHO:

  [Công Quang] Vector Index   [Công Quang] Inverted Index   [Công Quang] Asset Store   [Nhật Bảo] Relational Map
     (semantic)          (text/exact match)      (ảnh gốc/blob)      (validator core, P0)
  └──────────── phục vụ TÌM KIẾM ────────────┘   └── nguồn SỰ THẬT, chỉ dùng lúc xác minh ──┘

                          ONLINE
query text
  └─ [Phạm Trọng] Cascade: late-fusion coarse → dedup/temporal → early-fusion rerank → browser
       └─ [Nhật Bảo] Validator + submit adapter → Submit hợp lệ
```

**Luồng quan trọng nhất:** retrieval **không** query thẳng vào Relational Map. Ba kho đầu phục vụ *tìm*; kho thứ tư phục vụ *xác minh*, chỉ ở bước cuối trước submit.

---

## Ba thứ cả nhóm phải thuộc

**1. `keyframe_id` là khóa nối mọi kho.**

```python
from mfe.contracts.ids import build_keyframe_id
build_keyframe_id("L21_V001", 1)   # 'L21_V001#001'  (keyframe_index, khớp file 001.jpg)
```

Không tự f-string. docs/02 §4 gọi đây là *"điểm dễ sai nhất — sai một chỗ là cả pipeline lệch"*.

**2. `frame_idx` và `pts_time` đọc thẳng từ `map-keyframes` CSV, KHÔNG tự tính.** `round(pts_time×fps)` đã được chứng minh sai trên 83–93% dòng video 29.97fps (NTSC drop-frame). BTC chưa chốt submit theo cái nào nên lưu đủ — xem [`adr/0002`](docs/adr/0002-id-format.md).

**3. Submit sai còn tệ hơn không submit.** Validator thà chặn nhầm còn hơn cho lọt (docs/05 §3).

---

## Cấu trúc thư mục

```
src/mfe/          CODE — mọi thứ chạy được
config/           local.yaml (active) · modal.yaml (stub)
docs/             contract · context 3 buổi · adr
tests/            unit/<workstream>/ · integration/ · fixtures/

data/             video, keyframe, segment          ← gitignored
stores/           Faiss · BM25 · sqlite             ← gitignored
eval/             queryset.jsonl (tracked) · results/ (gitignored)
infra/modal/      stub deploy
scripts/          tiện ích dev
```

Code ở `src/`, artifact ở gốc — nên `src/mfe/stores/` (code) và `stores/` (Faiss index) không đụng nhau. Lý do đầy đủ: [`adr/0001`](docs/adr/0001-src-layout.md).

---

## Sprint 1 — Definition of Done

Từ docs/00 §1. `tests/integration/test_vertical_slice.py` là bản code của danh sách này (hiện skip toàn bộ — mỗi test được unskip bởi workstream sở hữu nó).

- [ ] Offline chạy trên một thư mục video bất kỳ bằng **một lệnh**, đổ ra đủ 4 kho
- [ ] Online nhận query text → trả top-k keyframe → mở đúng video/timestamp
- [ ] Mapping ID↔frame↔timestamp đúng **100%** trên tập mẫu
- [ ] Mọi submit sai schema bị validator **chặn**
- [ ] Chạy ≥ 30 query nội bộ không crash, có log tái hiện
- [ ] Có số liệu Recall@K, TTCA, latency, throughput

**Thứ tự tích hợp** (docs/01 §4): Nhật Bảo dựng mapping+validator → Quốc Đạt ingest → Công Quang mở API → Phạm Trọng nối `search_text` → Vỹ Triết chạy harness → bổ sung dần.

---

## Đổi contract thế nào

docs/01 §1: muốn đổi schema một kho hoặc chữ ký một tool → **báo Vỹ Triết**. Vỹ Triết cập nhật `docs/00`, ghi ADR, thông báo nhóm. Không ai tự sửa contract trong file cá nhân.
