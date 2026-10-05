# data/ — dataset thật (KHÔNG commit nội dung)

Thư mục này **gitignored** (chỉ file này + các `.gitkeep` được track). Không ai push dữ liệu thật lên GitHub — nó quá lớn. Repo chỉ giữ **khung thư mục** để cấu trúc làm việc giữa các thành viên không bị lệch.

## Cách setup (mỗi thành viên làm 1 lần)

Giải nén file zip dataset (**Kaggle "AI Challenge 2025"** — `aresusayhi/ai-challenge-2025`) vào đúng thư mục:

```
data/sample/
```

Sau khi giải nén, `data/sample/` phải có cấu trúc:

```
data/sample/
├── videos/          # .mp4 gốc, mỗi video một file
├── keyframes/       # ảnh keyframe (n.jpg), cấp sẵn — không tự trích
├── clip-features/   # .npy mỗi video, shape (n_keyframe, 512), float16
├── objects/         # JSON object detection mỗi keyframe
├── map-keyframes/   # CSV mỗi video: cột n, pts_time, fps, frame_idx
│                    #   ← NGUỒN SỰ THẬT cho frame/timestamp (docs/00 §3, adr/0002)
├── media-info/      # JSON metadata mỗi video
└── manifest.json
```

## Vì sao phải cùng một chỗ

`config/local.yaml` trỏ đường dẫn vào đây. Ai extract sai chỗ → code không tìm thấy data, hoặc lệch path so với người khác. Giải nén đúng `data/sample/` là xong.

> **Lưu ý:** `map-keyframes/*.csv` là nguồn duy nhất nối `n` với `frame_idx` — có dòng thì đọc thẳng. Hàm bị cấm là `round(pts_time×fps)` (sai 83–93% ở video 29.97fps); `floor(pts_time×fps)` đúng 100% và là cách đặt id cho khung ngoài danh mục BTC. Xem `CLAUDE.md` §2.
