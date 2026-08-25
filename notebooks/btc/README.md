# `btc/` — dùng lại bộ keyframe của BTC

Volume Modal **`keyframes-btc`** (v2) · bundle `aic_btc_b1_v1` · `keyframe_set: btc`

Nếu host BTC tải chậm hoặc liên tục đứt, notebook 02 và 03 có thể đọc trực tiếp ZIP nguyên bản từ
private Kaggle Dataset đã attach. Xem [`kaggle-zip-cache.md`](kaggle-zip-cache.md) để tạo cache,
kiểm tra kích thước và nhận biết notebook đang dùng cache hay fallback về website.

> **Người làm: Trọng & Quang.** Tài liệu giao việc đầy đủ — phân công, model từng feature, hợp
> đồng ID/schema, cách gộp sau khi xong: [`giao-viec-trong-quang.md`](giao-viec-trong-quang.md).
> Caption: KHÔNG sinh ở ba lượt đầu; lượt 5 (`08`) sinh caption + chữ trên màn hình bằng
> Qwen2.5-VL, theo đúng phương pháp Bảo nghiên cứu (`../reference/caption.ipynb`).

Chạy theo thứ tự, đổi `MEMBER` ở ô đầu thành `"trong"` hoặc `"quang"`. Cả ba lượt **dùng chung
một cách chia người**, nên tên bạn cho ra cùng tập video ở mọi lượt.

| | Notebook | Tầng | Shards | Requirements |
|---|---|---|---|---|
| 1 | `02-keyframes-btc-import.ipynb` | `n0_import_btc` | `…_btc.yaml` | `../requirements-kaggle-nomodel.txt` |
| 2 | `03-btc-siglip-ocr.ipynb` | `n4_siglip`, `n5_ocr_easyocr` | `…_btc_images.yaml` | `../requirements-kaggle-btc-images.txt` |
| 3 | `04-btc-subtitles-audio.ipynb` | `n2_subtitles`, `n3_audio` | `…_btc_video.yaml` | `../requirements-kaggle-btc-video.txt` |
| 4 | `06-btc-paddle-vietocr.ipynb` | `n5_ocr_paddle`, `n5_ocr_vietocr` | `…_btc_ocr2.yaml` | `../requirements-kaggle-btc-paddle-vietocr.txt` |
| 5 | `08-btc-caption-ocr-qwen25.ipynb` | `n6_caption_qwen`, `n6_ocr_qwen25` | `…_btc_caption.yaml` | `../requirements-kaggle-btc-caption.txt` |

**Mỗi lượt một file shards riêng, và đó không phải sự lộn xộn.** `merge_bundle.py` đòi ĐỦ mảnh
cho mọi tầng có trong `plan.stages`; gộp hết vào một file là bundle bị từ chối cho tới khi lượt
cuối cùng chạy xong toàn bộ 873 video. Tách ra thì gộp được độc lập, và bốn lượt không đụng nhau
vì `merge` lọc artifact theo `spec.stage in plan.stages`.

**Lượt 5 (`08`) có hai thứ bốn lượt kia không có:**

* **Cắt lát bằng `PART` / `NUM_PARTS`** để nhiều notebook Kaggle chạy song song cho **cùng một**
  `MEMBER`. Chia theo *gói zip*, cân theo số ảnh, tất định. `author` ghi ra vẫn là `MEMBER` —
  thêm tên mới vào `assignment` để chạy song song là cách chắc chắn làm hỏng bundle
  (`merge.check_ownership`).
* **Cổng đối chiếu `PROBE_ONLY`** — chạy `L21_V001` bằng **cả hai** biến thể prompt, in bảng
  năng lực, **đo throughput**, so với 855 keyframe Bảo đã chạy
  (`../reference/dense_captions_qwen.json`) qua `scripts/compare_qwen_captions.py`, rồi **dừng**.
  Lượt đầy đủ là 177.321 ảnh trên một model 7B; phát hiện lệch sau lượt đó là mất cả lượt.

**Prompt của lượt 5 nằm ở `src/mfe/offline/caption_prompt.py`, hai biến thể:**

| biến thể | tiếng | dùng khi |
|---|---|---|
| `bao` | Anh | probe — phép kiểm **đường ống** duy nhất đối chiếu được với bản của Bảo |
| `aic` | **Việt** | chạy thật (mặc định) — có few-shot chỉ-văn-bản + chỉ dẫn theo `diem-yeu-pipeline-theo-49-de.md` §3 |

Đo trên chính 855 keyframe của Bảo, caption **tiếng Anh** của bạn ấy đạt **màu 1,2% · vị trí 0,1%**
trên phép đếm từ khoá tiếng Việt (`mfe/offline/caption_audit.py`) — trong khi 49 đề thật đòi
**63%** và **45%**. Arm `caption_qwen` là BM25 trên truy vấn tiếng Việt, nên đó là bài học
`labels_vi` đo được bằng số: không đổi ngôn ngữ thì arm vô dụng.

⚠️ **Notebook 08 không được chạy chung phiên Kaggle với 03 hay 06**: vLLM ghim `torch` riêng và
sẽ kéo phiên bản khác về khi cài. Một phiên, một môi trường.

Bộ keyframe của BTC **cố định** — không phụ thuộc ρ hay ngưỡng mật độ nào của DAKE. Nên đây
không phải bản nháp chờ chốt tham số: mọi thứ trích ở đây là kết quả cuối. Phương án DAKE sẽ
ra một volume **khác**, xem `../dake/`.

`n4_objects` không có ở lượt nào — BTC phát hành sẵn nhãn và lượt 1 nạp chúng. `n5_ocr_qwen`
đã bỏ (10,8 giây/ảnh, ~532 giờ GPU cho batch 1). `n6_recap` / `n6_bge_m3` do Lead hoãn —
lượt 5 KHÔNG dùng lại artifact `dense_captions` của ReCap, nó có `qwen_captions` riêng.

Chi tiết và các bất biến phải giữ: [`docs/triet/ke-hoach-offline-ingest.md`](../../docs/triet/ke-hoach-offline-ingest.md) §7.1.
