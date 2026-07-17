# tests/

## Layout gương với package

```
tests/unit/contracts/    [Lead] ✅ chạy được ngay — khóa ID model + records
tests/unit/core/         [Lead]
tests/unit/offline/      [TV1]
tests/unit/stores/       [TV2]
tests/unit/mapping/      [TV4]
tests/unit/online/       [TV3]
tests/integration/       [Lead] vertical slice — hiện skip toàn bộ
tests/fixtures/          video tí hon + expected records
```

`tests/unit/<workstream>/` thuộc owner của workstream đó. Cross-workstream → `tests/integration/`.

## Chạy

```bash
pytest                              # tất cả
pytest tests/unit/contracts -v      # chỉ contract — không cần torch/faiss
```

## `tests/integration/` skip toàn bộ — cố ý

Các skip **chính là** Definition of Done của Sprint 1, viết dưới dạng code. Khi tất cả chạy được, sprint xong.

Mỗi test unskip bởi workstream sở hữu nó. docs/01 §4: *"luôn giữ nhánh main ở trạng thái chạy vertical slice được"* — chỉ kiểm được nếu có thứ đi kiểm.

## `tests/fixtures/` cần gì

Một video **tí hon** (vài giây, vài trăm KB) + expected records. Không commit video mẫu thật.

Fixture nên có **ít nhất 2 video khác fps** — 29.97 và 25.00 như ví dụ docs/05 §1.1(c). fps phân số là chỗ bug rounding sống; fixture 30.0 sẽ giấu chúng.

## Test contract nói gì

`tests/unit/contracts/test_ids.py` có một test **cố ý khác docs**:

```python
def test_disagrees_with_two_stale_docs_examples():
    assert frame_to_timestamp_ms(1234, 29.97) == 41175
```

docs/05 §1.2(c) ghi 41174, docs/00 §3 ghi 41400. Cả hai sai. Xem `docs/adr/0002-id-format.md`. Test khóa **công thức** (normative), docs sửa theo.
