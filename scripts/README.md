# scripts/

**Owner:** ai cũng được. Script dev một lần, không phải sản phẩm.

## Viết gì vào đây

Tiện ích chạy tay: tải video mẫu, reset stores, dump một index để xem, sinh fixture.

## Đừng viết gì vào đây

**Logic sản phẩm.** Nếu một script bắt đầu chứa thứ pipeline cần, nó thuộc `src/mfe/`. Test để phân biệt: *có ai chạy nó trong lúc thi không?* Có → nó là code, không phải script.

Script không có test và không ai review kỹ. Đừng để đường đi tới submit chạy qua đây.

## Gợi ý ban đầu

| Script | Việc |
|---|---|
| `reset_stores.py` | Xoá `stores/*` để ingest lại sạch |
| `make_fixture_video.py` | Cắt vài giây từ video thật → `tests/fixtures/` |
| `inspect_index.py` | Dump n dòng đầu Faiss sidecar để mắt thường soi |
