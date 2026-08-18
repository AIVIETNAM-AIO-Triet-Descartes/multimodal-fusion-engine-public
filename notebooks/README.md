# Notebook trích xuất — cái nào sinh ra cái gì

Kế hoạch: [ke-hoach-offline-ingest.md](../docs/triet/ke-hoach-offline-ingest.md) — §7.1 nói riêng
về việc chia thư mục dưới đây.

## Bố cục — chia theo PHƯƠNG ÁN, không theo số thứ tự

Hai phương án keyframe, **hai Modal Volume riêng**, và không bao giờ dùng chung bundle: ảnh và
`objects/` đều đánh địa chỉ theo số thứ tự trích `n`, nên trộn hai bộ là bundle không còn khớp
`map-keyframes` của chính nó (§1.2). Thư mục phản ánh đúng ranh giới đó.

```
notebooks/
├── btc/         ĐANG CHẠY — dùng lại bộ keyframe của BTC   -> volume keyframes-btc
├── dake/        chờ chốt ρ  — tự trích bằng DAKE            -> volume aic2026
├── reference/   CHỈ ĐỂ ĐỌC — bản gốc Bảo gửi, có lỗi đã ghi
└── requirements-kaggle*.txt        (dùng chung, KHÔNG nằm trong btc/ hay dake/)
```

**Vì sao requirements ở gốc.** `requirements-kaggle-nomodel.txt` được **cả hai** phương án dùng
(`dake/01` và `btc/02` đều không nạp model nào), nên nó không thuộc phương án nào — tên cũ của nó
là `-keyframes` và cái tên đó nói sai. Bản `.lock` của §2.5 cũng nằm ở đây: một môi trường cho
cả năm người, không phải một môi trường mỗi thư mục.

| File | Dùng bởi |
|---|---|
| `requirements-kaggle-nomodel.txt` | `dake/01`, `btc/02` — không model |
| `requirements-kaggle-btc-images.txt` | `btc/03` — torch + transformers + easyocr |
| `requirements-kaggle-btc-video.txt` | `btc/04` — faster-whisper + CLAP |
| `requirements-kaggle.txt` | `dake/00` — cả stack; là nguồn của `requirements-kaggle.lock.txt` |

> Số đầu tên file **không** phải thứ tự toàn cục nữa, chỉ là thứ tự trong thư mục của nó. Trong
> `dake/` thì `01` chạy TRƯỚC `00` (00 là notebook 9 tầng, dùng sau khi ρ chốt).

---

## `btc/` — dùng lại bộ keyframe của BTC, ba lượt theo thứ tự này

Volume `keyframes-btc` (v2), bundle `aic_btc_b1_v1`. **Cả ba dùng chung một cách chia người**;
đổi `MEMBER` ở ô đầu là ra đúng phần việc quen thuộc ở mọi lượt.

| # | Notebook | Sinh ra | Gói cần | Đầu vào lấy ở đâu |
|---|---|---|---|---|
| 1 | `btc/02-keyframes-btc-import.ipynb` | ảnh keyframe, `objects/`, `map-keyframes` | `-nomodel` | `Keyframes_*.zip` + `objects-*.zip` |
| 2 | `btc/03-btc-siglip-ocr.ipynb` | `siglip_vectors`, `ocr_text`, `sviptr_ocr` | `-btc-images` | tải lại `Keyframes_*.zip` |
| 3 | `btc/04-btc-subtitles-audio.ipynb` | `subtitle_segments`, `subtitles`, `audio_features` | `-btc-video` | **từng video** qua HTTP Range |

`n4_objects` không có ở lượt nào: BTC phát hành sẵn nhãn, lượt 1 nạp chúng. Chạy detector của ta
lên chính bộ keyframe này là phép đo khác (§3.6). `n6_recap`/`n6_bge_m3` do Lead hoãn.

**Lượt 2 và 3 không đọc ảnh từ Volume.** Ở đường BTC, Volume là ghi-một-chiều: kéo ảnh về là
~177.000 lần `read_file`, còn tải lại gói là 1,4–5,7 GB có resume.

> ⚠️ **Lượt 3 chia người theo gói KEYFRAME dù nó ăn video.** Không phải nhầm: nó lấy từng video
> bằng Range nên gói không còn ảnh hưởng băng thông, còn `part-manifest` chỉ ghi **một**
> `author` và `check_ownership` từ chối nếu author khác người được giao. Chia lại theo gói
> video (`zips_batch1.yaml`) lệch tới mức `bao` trùng **0** video — làm thế là hỏng cả bundle.
> Khoá bằng `tests/unit/offline/test_multipass_bundle.py`.

---

## `dake/` — tự trích keyframe bằng DAKE

Volume `aic2026` (v2), bundle `aic_dake_r0015_b1_v1`. **Cách chia người khác `btc/`** (theo gói
`Videos_*.zip` thay vì `Keyframes_*.zip`) — hai bundle khác nhau nên không sao, nhưng đừng trộn
hai bảng phân công vào cùng một bundle (§7.1).

| Thứ tự | Notebook | Dùng khi | Gói cần |
|---|---|---|---|
| 1 | `dake/01-keyframes-dake-kaggle-modal.ipynb` | N1/DAKE trên batch 1 — **chờ chốt ρ** | `-nomodel`; GPU T4 qua ffmpeg/NVDEC, có đường lùi CPU |
| 2 | `dake/00-master-ingest-kaggle-modal.ipynb` | 9 tầng trong một vòng — ⛔ **ô fetch đang `raise`** | `requirements-kaggle.txt`, cả stack |

`00` chưa chuyển sang đường tải trực tiếp nên chưa chạy được. Nó vẫn ở đây vì nó là chỗ duy nhất
chạy cả 9 tầng theo một vòng-theo-video, và đó là hình dạng cần cho các tầng model của bộ DAKE.

**Model nạp một lần cho cả phiên — nhưng chỉ ở `btc/`.** `stages.py` có cache theo checkpoint.
`btc/03` và `btc/04` đổi **video** mỗi bước nên giữ model qua các video, chỉ `unload_models()` khi
hết phiên; nạp lại theo từng video tốn **5–16 giờ mỗi người chỉ để nạp** (đo trên số video thật).
`dake/00` đổi **tầng** mỗi bước nên gọi `unload_models()` sau mỗi tầng — ở đó TF và torch nằm
chung process, đúng trường hợp §4.2 mô tả. Dùng `free_gpu()` ở vòng lặp theo tầng là **sai**: nó
không thả cache, và 9 model sẽ tích lại.

Notebook `dake/01` tồn tại riêng vì N1 không nạp model nào — 5 gói thay vì cả stack.
**Bật Accelerator = GPU không tự làm nó nhanh hơn**: `opencv-python` không có CUDA, nên GPU
chỉ giúp nếu ffmpeg có NVDEC. Marker ghi `decode_backend` để biết đường nào đã chạy —
xem [§10.4](../docs/triet/ke-hoach-offline-ingest.md).

**Nguồn dữ liệu (18-08): tải thẳng từ `aic-data.ledo.io.vn`, ẩn danh — không Drive, không
rclone, không token.** Đo bằng HTTP Range: 14 gói `Videos_*.zip`, 77,29 GB, đúng 873 video.
Mọi mp4 đều STORED nên notebook lấy **từng video** bằng Range, không tải gói nào — đĩa cao
nhất ~160 MB thay vì 24 GB.
Kiểm kê ở `config/zips_batch1.yaml`, sinh lại bằng `scripts/make_zip_inventory.py`. §5.6
ghi lại nguyên cái hiểu nhầm Drive/rclone để không ai đi lại đường đó.

---

## `reference/` — bản gốc, CHỈ ĐỂ ĐỌC

Đọc ngày 17-08-2026. Bốn notebook `bao-*` do Nhật Bảo gửi; `01-triet-dake-keyframes` do Claude web
viết cho nhóm ta. Tên file cũ là tên Kaggle tự đặt (`notebook53b1574439`…), đã đổi theo thứ hạng
trong chuỗi phụ thuộc.

> **Cả năm cái đều KHÔNG chạy được như đang có**, và đó là lý do chúng nằm riêng một thư mục thay
> vì lẫn vào `btc/` hoặc `dake/`. Lỗi của từng cái ghi bên dưới: notebook 03 chưa từng chạy thành
> công (3/3 video lỗi mà vẫn in "HOÀN TẤT!"), notebook 04 ghi file pickle, `01-triet-dake` lệch
> hợp đồng ở 5 chỗ kể cả công thức DAKE. Chúng là **bằng chứng và tài liệu tham khảo**, không phải
> đường chạy.
>
> `01-triet-dake-keyframes` bị thay bởi `dake/01-keyframes-dake-kaggle-modal.ipynb`.

| # | Notebook | Sinh ra | Model |
|---|---|---|---|
| 01 | `01-triet-dake-keyframes` | `keyframes/`, `manifest.csv`, `schema.sql`, `meta.json` → HF private | DAKE (không model) |
| 02 | `02-bao-subtitles-whisper` | `subtitles.json` | faster-whisper `large-v3`, VAD |
| 03 | `03-bao-audio-features-clap` | `audio_features.json` | `laion/clap-htsat-unfused` |
| 04 | `04-bao-siglip-vectors-plus-qwen2vl-caption-ocr` | `siglip_vectors.npy` **+** `dense_captions.json` **+** `sviptr_ocr.json` | `google/siglip-so400m-patch14-384`; `Qwen/Qwen2-VL-7B-Instruct` 4-bit |
| 05 | `05-bao-colpali-vectors-plus-offline-pipeline` | `colpali_vectors.npy`, và chạy cả `run_pipeline.py --mode offline` (⇒ `objects_text.json`, `ocr_text.json`) | ColPali (PaliGemma) |

Không có notebook riêng cho `ocr_text.json` và `objects_text.json` — chúng được sinh bởi
`extract_ocr.py` / `extract_objects.py` **bên trong** repo của Bảo, mà notebook 05 gọi qua
`run_pipeline.py`.

---

## Điều quan trọng nhất học được: `sviptr_ocr.json` không phải SVIPTR

`sviptr_ocr.json` là **đầu ra của Qwen2-VL-7B**, không phải của SVIPTR. Notebook 04 dùng một
prompt hai nhiệm vụ rồi cắt chuỗi kết quả:

```
Task 1: Describe all visual details … in English.
Task 2: Transcribe ALL visible text … If no text, write 'None'.
Format: [CAPTION] … [TEXT] …
```

`[CAPTION]` → `dense_captions.json`, `[TEXT]` → `sviptr_ocr.json`. Cái tên là di sản, giữ lại
vì đã ăn vào `SOURCE_FILES` và vào các bảng đo, nhưng **đừng đọc nó như tên model**.

Việc này trả lời §8 câu 2 và câu 5 của [ke-hoach-merge-sota.md](../docs/triet/ke-hoach-merge-sota.md),
và nó **đổi kết luận** ở hai chỗ:

**1. `caption` và `ocr_sviptr` không phải hai bằng chứng độc lập.** Chúng là hai trường cắt ra
từ **một lượt sinh của một model**. Đo được trên bộ pilot: `ocr_sviptr` đứng riêng hạng 1.0
(3/9) nhưng đóng góp biên **âm**, còn `caption` 4/9 hạng 6.8 — giờ đã biết vì sao hai con số
này đi cùng nhau. Fusion coi chúng là hai lá phiếu, thực tế là một.

**2. Kế hoạch N5 phải sửa.** Đã sửa: [ke-hoach-offline-ingest.md](../docs/triet/ke-hoach-offline-ingest.md)
§2.2 ghi thẳng `sviptr_ocr` = Qwen2-VL. Muốn tái lập thì dùng Qwen2-VL; muốn **thật sự** có
nguồn OCR thứ hai độc lập thì mới cần model khác — và lúc đó nó là *quyết định mới*, không
phải tái lập (câu 5 của §12 bản kế hoạch).

Kèm một cái bẫy định dạng trong notebook 04: nếu đầu ra của model thiếu `[CAPTION]`/`[TEXT]`
thì `caption` nhận **toàn bộ** chuỗi và `ocr` để rỗng. Không có cảnh báo, không có bộ đếm.

---

## Notebook 03 (audio) — chưa từng chạy được, và file đang có không tái lập được

> **Đã sửa ngày 17-08** (5 chỗ, xem cell markdown đầu notebook). Phần dưới là bằng chứng.

**Bản gốc chưa từng chạy thành công, và chính output lưu trong notebook nói thế:** cả **3/3
video** đều lỗi ở `processor(audios=…)` — `transformers` 5.x ném `ValueError` với tên tham số
cũ. `try/except` quanh từng video nuốt lỗi, dòng cuối in `"HOÀN TẤT!"`. Nó ghi ra một file
**rỗng** và báo thành công.

> **Bài học về quy trình, cho chính ta:** câu trả lời nằm sẵn trong **output pane** của
> notebook. Tôi đã chạy ba lượt CLAP để suy ra một phiên bản yếu hơn của cùng kết luận trước
> khi đọc nó. Đọc output đã lưu **trước** khi tái lập.
>
> Hai phát hiện vẫn bổ sung cho nhau chứ không thay thế: output chỉ nói *notebook này* không
> sinh ra gì, còn phép tái lập dưới đây mới nói `audio_features.json` **đang có** là không
> kiểm chứng được — điều mà output không nói.

### Phần dưới đây vẫn đúng và vẫn cần: file đang có không tái lập được

Notebook lấy mốc thời gian của keyframe như sau:

```python
frame_idx = int(kf_id.split('_')[-1])   # comment trong code: "L21_V001_001523 -> 1523"
t = frame_idx / fps
```

Nhưng hậu tố của `kf_id` là **số thứ tự keyframe**, không phải frame. Đo trên `L21_V001`
(307 keyframe, fps 30):

| | |
|---|---|
| keyframe cuối `n=307` | `pts_time` thật **1257,2s**, `frame_idx` 37716 |
| notebook tính `t = 307/30` | **10,2s** |
| lệch | **1247s** |

Tức mọi cửa sổ audio của cả video 21 phút đều nằm trong **10,2 giây đầu**.

Phép thử đầu: nếu `t = n/fps` thì `n = 1..60` đều cho cửa sổ `[0, ~4s]` sau padding, nên các
vector phải gần trùng khít. Đo thật: cosine trung bình **0,493** — không trùng khít, tức file
cũng **không** mang dấu vết công thức sai đó.

Nên đừng đoán công thức nữa, đi tìm mốc thời gian thật: nhúng một cửa sổ 4 giây **mỗi 2 giây
suốt 21 phút** của `L21_V001` (630 cửa sổ) rồi hỏi từng vector của Bảo xem nó khớp cửa sổ nào.
Một lượt lưới phục vụ cả 307 keyframe.

| | max-cosine trung vị | phân vị 90 | >0.75 |
|---|---|---|---|
| 307 vector `L21_V001` vs audio **của chính nó** | 0.753 | 0.829 | 50.8% |
| 548 vector `L21_V002/V003` vs audio **V001** — đối chứng | **0.730** | 0.789 | 34.1% |

**Đối chứng là chỗ quyết định.** Vector của hai video khác *không thể* khớp audio của V001,
vậy mà chúng cũng đạt max-cosine 0.730 — nên 0.753 chỉ là hiệu ứng của phép **lấy max trên
630 cửa sổ**, không phải khớp thật. Tương quan giữa `t*` tìm được và mọi công thức
(`pts_time`, `n/fps`, `n` giây, `n × dur/n_kf`) đều **≈ 0**, |lệch| trung vị 549s / 254s / 14s.

Ngoại lệ duy nhất: `n=1` khớp **hạng 1/855**, `cos 0.833`, tại `t*=0`. Điều đó quan trọng theo
hai chiều — nó chứng minh pipeline này **tái lập được** vector của Bảo khi mốc thời gian đúng,
nên việc mọi keyframe khác không khớp **không thể** quy cho khác biệt tiền xử lý.

**Kết luận:** `audio_features.json` đang có **không kiểm chứng được** — không mốc thời gian nào
giải thích được nó, và notebook được gửi thì chưa từng sinh ra gì. Đừng dùng nó làm dữ liệu
tầng 1c; sinh lại bằng bản đã sửa. Chi phí thực tế lúc này bằng **0**, vì arm CLAP chưa được
dựng trong tầng online (việc 7+13).

> **Một lưu ý về đọc cosine của CLAP:** vector CLAP audio rất dị hướng — shipped-vs-shipped
> trung vị ~0.50, và max trên 630 cửa sổ ngẫu nhiên đã cho 0.73. Nên với CLAP, **0.75 không
> có nghĩa là "giống"**. Đừng đặt ngưỡng cosine cho tầng này mà không có đối chứng.

Ba thông số khác của notebook 03, ghi lại vì chúng là lựa chọn thật: cửa sổ **±2 giây**
(4s, padding bằng 0 nếu thiếu), `sr = 48000`, và vector CLAP **512 chiều** — đo trực tiếp:
`audio_features.json` có 855 khoá, mỗi khoá một list 512 số. Con số 512 trong
[ke-hoach-offline-ingest.md](../docs/triet/ke-hoach-offline-ingest.md) §1.5 giờ là
**đo được**, không còn là "theo tài liệu".

---

## Notebook 02 (subtitle) — làm đúng, và đáng chép lại

Đây là notebook chắc tay nhất trong bốn cái:

- **Đọc `pts_time` từ CSV của BTC**, không tự tính từ `n / fps`. Đúng hợp đồng của ta.
- **`vad_filter=True`** với `min_silence_duration_ms=500`. Whisper hay "ảo giác" trên đoạn
  nhạc nền hoặc im lặng — VAD chặn đúng chỗ đó.
- `language="vi"` khai tường minh, không để tự nhận diện.
- Chỉ ghi keyframe có subtitle khác rỗng → 847/855 khoá, 8 keyframe không có tiếng nói.

Một chỗ khác kế hoạch của ta: Bảo lấy cửa sổ **±3 giây cố định** quanh `pts_time`, còn
[§4.5](../docs/triet/ke-hoach-offline-ingest.md) đề xuất cửa sổ `[pts_time(n), pts_time(n+1))`
vì khoảng cách keyframe biến thiên (BTC: trung vị 84 frame, xa nhất 210). Chưa đo cái nào hơn.

Không chuẩn hoá **NFC** trước khi ghi. Chưa gây lỗi (`StorageRecord.validate_text` chuẩn hoá ở
biên ingest), nhưng file trên đĩa thì không bảo đảm.

---

## Notebook 04 (SigLIP) — đây là chỗ sinh ra file pickle

```python
np.save("siglip_vectors.npy", siglip_vectors)   # siglip_vectors là dict
```

Đúng nguồn của vấn đề `allow_pickle=True` mà §9b kế hoạch merge đã xin Bảo đổi sang `savez`.
Nạp file này là **thực thi pickle**, mà nhóm truyền artifact giữa 5 máy.

Phần còn lại làm đúng: `get_image_features`, chuẩn hoá L2 ngay trong notebook, và có nhánh
`hasattr(outputs, …)` đề phòng API `transformers` đổi hình dạng trả về.

`kf_id = f"{parent_dir}_{filename_stem}"` — tức **khoá phụ thuộc vào tên file ảnh**. Ghi lại vì
nó là ràng buộc ngầm: chạy notebook này trên bộ keyframe DAKE của ta (ảnh đặt tên
`{frame_idx:06d}.jpg`) sẽ ra khoá `L21_V001_031124`, và `SotaIdMap` sẽ đọc `031124` **như số
thứ tự** rồi báo `KEYFRAME_NOT_FOUND`. Hỏng to, không hỏng im lặng — nhưng phải biết trước.

---

## Notebook 01 (DAKE, của ta) — bốn chỗ lệch với repo, phải sửa trước khi dùng

Notebook này do Claude web viết, chưa đối chiếu với contract trong repo. Bốn điểm lệch:

**1. Tên file ảnh.** Notebook ghi `keyframes/{video_id}/{frame_idx:06d}.jpg`, còn
`provided.keyframe_image_path` dựng `{n:03d}.jpg`. `ProvidedDataset` **không đọc được** đầu ra
này.

**2. Không sinh `map-keyframes/{video_id}.csv`.** Nó sinh `manifest.csv` theo schema
Relational Map (`keyframe_id, video_id, keyframe_index, frame_idx, pts_time, timestamp_ms,
shot_id`) — thiếu hẳn cột **`fps`**, mà `read_map_keyframes` đòi bốn cột `n, pts_time, fps,
frame_idx`.

**3. `pts_time` bị suy ra từ `frame_idx`, ngược chiều hợp đồng.** Notebook đặt
`pts_time = frame_idx / fps` và ghi trong `meta.json` rằng `frame_idx` lấy từ biến đếm chứ
không tính từ `pts_time`. Lấy từ biến đếm thì **tốt hơn** — nó là sự thật của bộ giải mã. Cái
giá là phép kiểm `frame_idx == floor(pts_time × fps)` của ta trở thành **tautology**: đảo
ngược công thức thì nó luôn đúng, nên nó không còn bắt được lỗi gì. Phải biết là mình đã mất
một phép kiểm, chứ không phải mình vừa qua được nó.

**4. Công thức DAKE không phải công thức trong paper.** Notebook dùng

```python
abs(s_norm[j] - s_norm[i]) / (j - i)
```

và markdown tự thừa nhận "S(i,j) là bản tái dựng" vì người viết không đọc được công thức từ
PDF. Nhưng **ta có công thức** — [ke-hoach-offline-ingest.md](../docs/triet/ke-hoach-offline-ingest.md)
§3.2 chép nguyên văn:

```
S(i,j) = (100·(sⱼ−sᵢ)/s_max) / √[ (j−i)² + (100·(sⱼ−sᵢ)/s_max)² ]
```

Hai khác biệt có hậu quả: paper **giữ dấu** (nên frame có kích thước *giảm* mạnh xếp hạng
thấp), notebook lấy `abs()` (coi tăng và giảm như nhau); và paper là **sin của góc dốc** nên
bị chặn trong `[−1,1]`, notebook là độ dốc thô nên không chặn.

**Thêm điểm thứ năm:** ràng buộc "**≥1 keyframe mỗi cửa sổ 2×fps**" — thứ tạo ra lời hứa
"khoảng cách tối đa 2,0s" ở §3.1 kế hoạch offline ingest, tức **lý do chính** ta muốn DAKE — **không được
cài**. Notebook chỉ có `MIN_GAP=5` (khoảng cách *tối thiểu*), là ràng buộc ngược hướng.

Phần notebook 01 làm tốt và nên giữ: đọc **tuần tự** cả hai lượt thay vì `cap.set(POS_FRAMES)`
(seek trên video có B-frame lệch frame), cell kiểm toàn vẹn trước khi upload (trùng
`keyframe_id`, thiếu ảnh, ảnh thừa, `frame_idx` không tăng), `meta.json` ghi lại tham số, và
token HF không bao giờ bị in ra.

---

## Việc phải làm, xếp theo mức chặn

| | Việc | Vì sao |
|---|---|---|
| 1 | **Hỏi Bảo: `audio_features.json` sinh ra bằng gì?** Notebook 03 chắc chắn không phải nó (3/3 video lỗi), và không mốc thời gian nào tái lập được file | Có notebook mà không khớp artifact còn tệ hơn không có notebook. Notebook 03 đã sửa và chạy được, nhưng **chưa chạy lại trên Kaggle** |
| 2 | Sửa 5 điểm lệch của notebook 01 trước khi chạy 16 video | Đầu ra hiện tại `ProvidedDataset` không đọc được |
| 3 | Cập nhật §N5 kế hoạch: `sviptr_ocr` = Qwen2-VL | Đang ghi sai là "chưa biết" |
| 4 | Đo lại xem `caption` và `ocr_sviptr` có nên là **một** arm | Chúng vốn là một lượt sinh; `tune_weights --ablation` trả lời được |
| 5 | Đổi `np.save(dict)` → `np.savez` ở notebook 04 | Bỏ pickle khỏi đường truyền artifact giữa 5 máy |
