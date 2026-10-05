# tests/fixtures/

Dữ liệu test — commit vào git, **khác** với `data/` (gitignored, video thật).

## Cần gì

| Fixture | Yêu cầu |
|---|---|
| Video tí hon | Vài giây, vài trăm KB. Đủ nhỏ để commit |
| ≥ 2 video khác fps | 29.97 và 25.00 (như docs/05 §1.1(c)) |
| ≥ 2 video khác resolution | 1280×720 và 1920×1080 |
| Expected records | Output đúng của ingest trên fixture trên |

## Vì sao phải có fps phân số

29.97 là chỗ bug rounding sống. Fixture toàn 30.0 sẽ **giấu** đúng loại lỗi mà `docs/adr/0002-id-format.md` ghi lại — ví dụ trong docs từng lệch nhau chính vì chuyện này.

docs/context/buoi-1.md §7: BTC có thể phát video khác fps và khác resolution. Fixture phải phản ánh điều đó, không được giả định đồng nhất.

## Đừng commit

- Video mẫu thật → `data/raw/` (gitignored)
- Faiss index, sqlite → `stores/` (gitignored)
