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

Notebook `10` là research producer riêng: nó đọc `keyframes-btc` chỉ để lấy mapping/FPS và ghi
shot proposals sang volume `keyframe-augmentation-research`, không sửa bundle BTC. Notebook này
dùng `MEMBER` là owner thật và `WORKER` là slice disjoint; Trọng dùng `trong1`/`trong2`, Quang
dùng `quang1`–`quang4`, các owner còn lại dùng hai worker theo config. Nếu BTC host bị block,
attach private Kaggle Dataset chứa MP4 đúng kích thước; notebook sẽ ưu tiên đọc `/kaggle/input`
read-only trước HTTP Range.

| | Notebook | Tầng | Shards | Requirements |
|---|---|---|---|---|
| 1 | `02-keyframes-btc-import.ipynb` | `n0_import_btc` | `…_btc.yaml` | `../requirements-kaggle-nomodel.txt` |
| 2 | `03-btc-siglip-ocr.ipynb` | `n4_siglip`, `n5_ocr_easyocr` | `…_btc_images.yaml` | `../requirements-kaggle-btc-images.txt` |
| 3 | `04-btc-subtitles-audio.ipynb` | `n2_subtitles`, `n3_audio` | `…_btc_video.yaml` | `../requirements-kaggle-btc-video.txt` |
| 4 | `06-btc-paddle-vietocr.ipynb` | `n5_ocr_paddle`, `n5_ocr_vietocr` | `…_btc_ocr2.yaml` | `../requirements-kaggle-btc-paddle-vietocr.txt` |
| 5 | `08-btc-caption-ocr-qwen25.ipynb` | `n6_caption_qwen`, `n6_ocr_qwen25` | `…_btc_caption.yaml` | `../requirements-kaggle-btc-caption.txt` |
| 6 | `09-btc-bge-m3-embed.ipynb` | `n6_bge_m3` | `…_btc_bge.yaml` | `../requirements-kaggle-btc-bge.txt` |

Notebook producer TransNetV2 dùng tạm để chạy GPU theo worker nằm ở
`tmp-kaggle-keyframe-producer-v2.ipynb`. Nó không phải một lượt được đánh số trong
pipeline và không thay thế notebook chính `10-btc-keyframe-augmentation.ipynb`.

Sau khi producer đủ **12/12 worker và 873/873 COMPLETE marker**, chạy
`11-btc-shot-coverage-audit.ipynb` bằng **CPU / Accelerator None**. Notebook 11
không chạy lại TransNetV2 và không tải MP4: nó băm lại manifest, marker, report,
shot artifact và mapping BTC; sau đó xuất C0/C1/COVERED/CT/CE vào thư mục research
`gate1-audit-v1`. Đây chỉ là temporal diagnostic O1-T, không được dùng riêng để
kết luận `NO_HEADROOM`; O1-S/O1-R và B0–B4 vẫn là các gate tiếp theo.

Khi notebook 11 đã publish audit manifest/corpus lock được lead chấp nhận, chạy
`12-btc-b0-b4-candidate-freeze.ipynb` bằng CPU để sinh B1–B4 và khóa hard delta
budget. Notebook 12 chỉ xuất candidate manifests cùng feature worklist sang
`gate1-selection-v1`; nó không decode ảnh, chưa tạo feature/index và không mutate
BTC. Nếu một arm underfill, exact-collide BTC hoặc sai selector contract thì không
có freeze hợp lệ.

Sau selection freeze, `13-btc-delta-siglip-producer.ipynb` chạy GPU theo đúng
hai runner `quang`/`trong` để decode union frame một lần, lưu deterministic JPEG tar và vector
SigLIP research. Mặc định `PREFLIGHT`: notebook so một BTC JPEG/vector đã lưu với
pipeline hiện tại và chặn shard nếu numeric canary hoặc package environment lệch.
SigLIP PASS chưa phải full feature parity; OCR/caption/BGE và downstream vẫn bị
ghi `BLOCKED`, và không artifact nào của lượt 13 được ghi vào `keyframes-btc`.

Sau khi notebook 17 materialize đủ oracle frame, `18-btc-keyframe-augmentation-oracle-eval.ipynb`
đo DEV B0, natural SigLIP probe và O1-R final-candidate ceiling. Đây là diagnostic-only:
probe dương không được dùng để bật production, còn O1-R là trần xếp hạng chứ không phải
selector uplift.

`19-btc-o1s-representative-scan-producer.ipynb` là bước tiếp theo để gỡ blocker O1-S.
Notebook chia 90 video cho đúng hai runner `quang`/`trong`, lấy tối đa 64 source frame
đều trên mỗi inclusive DEV GT interval và union còn 12.982 frame. Interval dài bị sample
nên artifact phải gọi `O1-S-LB64`, không gọi full/best O1-S. Chạy `PREFLIGHT` trước rồi
đổi sang `SHARD`; chỉ khi cả hai worker manifest `COMPLETE` mới aggregate/stage-trace.

Khi notebook 19 có đúng **90/90 COMPLETE marker**, chạy
`20-btc-o1s-lb64-aggregate.ipynb` một lần bằng GPU. Notebook 20 không chia runner và
khi BTC flat chưa được publish thì cần attach đúng Dataset `mfe-data-v2`: nó xác minh lại hai manifest cùng hash của mọi source artifact,
đo 71 DEV query-event trên toàn bộ 176.707 vector BTC, rồi xuất O1-S-LB64, dense-stage
rank/score trace, math ledger, hypothesis backlog, adversarial counterexamples và workspace
safety manifest. Kết quả vẫn `diagnostic_only`; fusion/reranker/scorer, full feature parity
và official B0–B4 DEV A/B chưa được phép gọi là đã hoàn tất.

Sau notebook 20, chạy `21-btc-o1s-exact-interval-scan.ipynb` bằng GPU. Notebook 21
chỉ quét chính xác từng frame trong 61 DEV interval dài hơn 64 frame, chia video
disjoint cho `quang`/`trong`, rồi lưu JPEG + SigLIP artifact có marker resume. Chạy
`PREFLIGHT` trước, sau đó mỗi người chạy `SHARD` với `CONFIRM_SHARD` đúng runner;
hai runner có thể chạy song song. Chỉ khi cả hai manifest `COMPLETE` mới chạy
notebook 22 aggregate exact O1-S; notebook 23 chỉ được chạy sau notebook 22 và
full feature parity OCR/caption/BGE/downstream.

Khi hai manifest notebook 21 đều `COMPLETE`, chạy
`22-btc-o1s-exact-aggregate.ipynb` đúng một lần bằng GPU. Notebook 22 fail-closed
trên 35 marker/32.867 exact frame, giữ 10 short interval đã full-scan ở notebook
20 và thay 61 LB64 interval bằng exact best-source result. Nó xuất đủ 71 DEV
query-event, bảng exact-vs-LB64, SigLIP score/rank headroom và failure-stage trace
sang `gate1-o1s-exact-aggregate-v1`. Đây vẫn là diagnostic-only; notebook 23 bị
khóa cho tới khi OCR/caption/BGE và downstream fingerprint parity hoàn tất.

Kết quả notebook 22 đã quan sát được SigLIP headroom trên DEV, nên
`23-btc-feature-parity-worklist-freeze.ipynb` bắt đầu gỡ parity blocker bằng CPU.
Notebook 23 xác minh selection freeze, hai runner notebook 13, mọi marker và bốn
artifact nguồn của từng video; sau đó khóa đúng 17.399 frame thành ba nhóm
`ocr_classical`, `qwen25`, `bge_m3`. OCR cổ điển và Qwen25 có thể chạy song song;
BGE-M3 chỉ chạy sau Qwen25 COMPLETE. Đây là workload freeze, chưa phải full parity
PASS và chưa phải official B0–B4 DEV A/B.

Sau notebook 23, chạy `24-btc-siglip-b0-b4-screen.ipynb` bằng GPU để sàng lọc
B1–B4 trước khi tạo feature đắt tiền. Notebook dùng đúng 5.301 candidate mỗi arm,
71 DEV event và vector SigLIP đã có; 846 NPZ được tải song song. Kết quả chỉ chọn
arm để chạy OCR/Qwen/BGE, chưa phải official full-feature uplift.

Khi notebook 24 chọn B1, `25-btc-b1-easyocr-producer.ipynb` và
`26-btc-b1-qwen25-producer.ipynb` có thể chạy song song ở runtime riêng. Mỗi file
chia đúng hai runner `quang`/`trong`, chỉ xử lý 5.301 frame B1 và resume theo
marker từng video. Qwen cần T4 x2; không chạy chung kernel với OCR.
`27-btc-b1-paddle-vietocr-producer.ipynb` là runtime thứ ba và cũng chạy song
song; nó giữ Paddle GPU cu126 tách khỏi Torch/vLLM để tránh ABI conflict.

Khi Qwen runner manifest của **chính runner đó** đã `COMPLETE`, chạy
`28-btc-b1-bge-m3-producer.ipynb` trong một BGE-only runtime với GPU T4. Notebook
28 chỉ tải `qwen_captions` JSON và nhúng bằng `BAAI/bge-m3`; nó không đọc ảnh,
không chạy lại Qwen/OCR và không sửa BTC base. Chạy `PREFLIGHT` trước để khóa
lineage, model fingerprint, package version và numeric runtime sanity; sau đó
mới đổi sang `SHARD` với `CONFIRM_SHARD` đúng runner. Quang không cần chờ Qwen
manifest của Trọng để chạy shard Quang, nhưng aggregate full parity phải đợi cả
hai BGE runner manifest `COMPLETE`. Kết quả notebook 28 vẫn chỉ là research
feature shard — chưa phải full parity, official A/B hoặc production GO.

Khi đủ 8 runner manifest của notebook 25–28, chạy
`29-btc-b1-feature-aggregate.ipynb` đúng một lần bằng CPU. Notebook 29 xác minh
mọi manifest, marker và hash artifact, lọc đúng 5.301 frame B1 rồi gộp SigLIP,
EasyOCR, PaddleOCR, VietOCR, Qwen caption/OCR và BGE-M3 thành delta flat bundle
research-only tại `gate1-b1-feature-aggregate-v1`. Nó không sửa BTC và không đọc
held-out. SQLite snapshot/index B0+B1 được hoãn sang notebook 30 vì bước đó cần
attach baseline BTC bất biến; notebook 29 `COMPLETE` chỉ có nghĩa bundle delta
đã sẵn sàng cho paired DEV A/B, chưa phải bằng chứng điểm tăng.

**Mỗi lượt một file shards riêng, và đó không phải sự lộn xộn.** `merge_bundle.py` đòi ĐỦ mảnh
cho mọi tầng có trong `plan.stages`; gộp hết vào một file là bundle bị từ chối cho tới khi lượt
cuối cùng chạy xong toàn bộ 873 video. Tách ra thì gộp được độc lập, và bốn lượt không đụng nhau
vì `merge` lọc artifact theo `spec.stage in plan.stages`.

**Lượt 6 (`09`) phải chạy SAU lượt 5** — nó nhúng chính caption lượt 5 sinh ra. Đây là lượt rẻ
nhất của cả bundle vì nó **không đọc ảnh**: mỗi video là một file JSON caption tải từ Volume.
Không tải `Keyframes_*.zip` (28,7 GB), không cần `map-keyframes`, và **không có `PART`** — cắt
lát tồn tại để tránh tải trùng gói zip, mà ở đây không có gói nào để tải.

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
| `aic` | Anh | chạy thật (mặc định) — có few-shot chỉ-văn-bản + chỉ dẫn theo `diem-yeu-pipeline-theo-49-de.md` §3 |

**Caption viết tiếng Anh** vì nó đi vào **hai** arm: `bge_m3` (lượt 6, BGE-M3 đa ngữ — truy vấn
tiếng Việt khớp thẳng không cần dịch) và BM25 `caption_qwen` (dùng bản dịch truy vấn đã có sẵn
cho `siglip`). Tiếng Anh phục vụ cả hai; tiếng Việt chỉ một. Nửa `[TEXT]` thì **cấm dịch** — nó
là chữ trên màn hình và là nguồn của arm `ocr_qwen25`.

Cái đổi so với Bảo là **nội dung chỉ dẫn**. Đo trên chính 855 keyframe của bạn ấy
(`mfe/offline/caption_audit.py`, bộ dò song ngữ), hai chỗ hụt thật là **quan hệ hành động**
(50,5% so với 57,1% đề thật hỏi) và **mở đầu sáo mòn** ("The image shows…" ở 59,9% caption — một
cụm lặp ở đầu mọi caption là một stopword mới).

⚠️ **Notebook 08 không được chạy chung phiên Kaggle với 03 hay 06**: vLLM ghim `torch` riêng và
sẽ kéo phiên bản khác về khi cài. Một phiên, một môi trường.

Bộ keyframe của BTC **cố định** — không phụ thuộc ρ hay ngưỡng mật độ nào của DAKE. Nên đây
không phải bản nháp chờ chốt tham số: mọi thứ trích ở đây là kết quả cuối. Phương án DAKE sẽ
ra một volume **khác**, xem `../dake/`.

⚠️ **Notebook 09 cũng không được chạy chung phiên với 03 / 06 / 08** — cùng lý do ngược lại:
`sentence-transformers` và `torch` quyết định con số trong vector BGE-M3, và vLLM sẽ kéo torch
sang bản khác. Ô Step 0a của notebook 09 **dừng ngay** nếu thấy `vllm` trong phiên.

`n4_objects` không có ở lượt nào — BTC phát hành sẵn nhãn và lượt 1 nạp chúng. `n5_ocr_qwen`
đã bỏ (10,8 giây/ảnh, ~532 giờ GPU cho batch 1). **`n6_recap` đã XOÁ HẲN 25-08-2026** cùng cả
module `mfe.offline.recap`: nhóm đổi sang Qwen2.5-VL, nên đường ReCap (Gemini/Qwen2-VL, có bộ
nhớ theo chuỗi shot) không còn người dùng nào. Artifact `dense_captions` cũng đi theo —
`qwen_captions` là nguồn caption duy nhất của bundle.

Chi tiết và các bất biến phải giữ: [`docs/triet/ke-hoach-offline-ingest.md`](../../docs/triet/ke-hoach-offline-ingest.md) §7.1.
