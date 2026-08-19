# `btc/` — dùng lại bộ keyframe của BTC

Volume Modal **`keyframes-btc`** (v2) · bundle `aic_btc_b1_v1` · `keyframe_set: btc`

Nếu host BTC tải chậm hoặc liên tục đứt, notebook 02 và 03 có thể đọc trực tiếp ZIP nguyên bản từ
private Kaggle Dataset đã attach. Xem [`kaggle-zip-cache.md`](kaggle-zip-cache.md) để tạo cache,
kiểm tra kích thước và nhận biết notebook đang dùng cache hay fallback về website.

> **Người làm: Trọng & Quang.** Tài liệu giao việc đầy đủ — phân công, model từng feature, hợp
> đồng ID/schema, cách gộp sau khi xong: [`giao-viec-trong-quang.md`](giao-viec-trong-quang.md).
> Không sinh caption ở lượt này.

Chạy theo thứ tự, đổi `MEMBER` ở ô đầu thành `"trong"` hoặc `"quang"`. Cả ba lượt **dùng chung
một cách chia người**, nên tên bạn cho ra cùng tập video ở mọi lượt.

| | Notebook | Tầng | Requirements |
|---|---|---|---|
| 1 | `02-keyframes-btc-import.ipynb` | `n0_import_btc` | `../requirements-kaggle-nomodel.txt` |
| 2 | `03-btc-siglip-ocr.ipynb` | `n4_siglip`, `n5_ocr_easyocr`, `n5_ocr_qwen` | `../requirements-kaggle-btc-images.txt` |
| 3 | `04-btc-subtitles-audio.ipynb` | `n2_subtitles`, `n3_audio` | `../requirements-kaggle-btc-video.txt` |

Bộ keyframe của BTC **cố định** — không phụ thuộc ρ hay ngưỡng mật độ nào của DAKE. Nên đây
không phải bản nháp chờ chốt tham số: mọi thứ trích ở đây là kết quả cuối. Phương án DAKE sẽ
ra một volume **khác**, xem `../dake/`.

`n4_objects` không có ở lượt nào — BTC phát hành sẵn nhãn và lượt 1 nạp chúng. `n6_recap` /
`n6_bge_m3` do Lead hoãn.

Chi tiết và các bất biến phải giữ: [`docs/triet/ke-hoach-offline-ingest.md`](../../docs/triet/ke-hoach-offline-ingest.md) §7.1.
