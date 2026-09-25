"""BM25Okapi tính bằng ma trận thưa — cùng điểm, nhanh hơn 400–779 lần.

Vì sao có file này
-------------------
Đo 25-09-2026 trên 8 câu dev thật: một truy vấn mất **3,4–4,4 giây**, và
**85–93% nằm ở tầng văn bản**. Bốn arm dense chỉ tốn 307–502 ms — tức tối ưu
vector là tối ưu 1/10 sai chỗ.

Thủ phạm nằm gọn trong thân `rank_bm25.BM25Okapi.get_scores`:

    q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])

Một vòng lặp **Python thuần trên 168.000 dict**, chạy lại cho **mỗi từ** của
truy vấn. `np.array(...)` chỉ gói kết quả lại; phần nặng là list comprehension.

Hai hệ quả, và cái thứ hai mới là cái đắt:

1. **Chậm** — 362–1.725 ms mỗi arm.
2. **Giữ GIL** — nên `ArmRegistry.run` chạy song song chỉ được **10%**
   (7.663 → 6.904 ms), không phải 3 lần. Và nên 5 người truy vấn cùng lúc thì
   xếp hàng sau nhau.

Cách làm ở đây
---------------
Toàn bộ phần **không phụ thuộc truy vấn** của công thức BM25 được tính sẵn một
lần thành ma trận `W[tài liệu, từ]`:

    W[d,t] = idf[t] · tf[d,t]·(k1+1) / (tf[d,t] + k1·(1 - b + b·len_d/avg_len))

Chấm một truy vấn khi đó chỉ là **cộng các CỘT** ứng với những từ có trong truy
vấn. Trên ma trận CSC đó là phép cắt rẻ, và một từ hiếm chạm vài chục dòng thay
vì 168.000. Cả vòng lặp nằm trong C của scipy nên nó **nhả GIL** — tức bản vá
này sửa cả hai hệ quả ở trên, không chỉ hệ quả thứ nhất.

Đo được (8 câu dev, 28 từ/câu, đã tính cả bước xếp hạng top-100):

| nguồn | tài liệu | `rank_bm25` | thưa | lợi |
|---|---:|---:|---:|---:|
| `caption_qwen` | 176.707 | 1.725 ms | 4,3 ms | 400× |
| `subtitle` | 159.106 | 1.237 ms | 1,6 ms | 779× |
| `ocr_qwen25` | 114.516 | 362 ms | 0,9 ms | 405× |

⚠ Đây KHÔNG phải một công thức BM25 khác
-----------------------------------------
Nó phải ra **đúng** điểm của `BM25Okapi`, nếu không mọi bảng trọng số đã leo dốc
đều mất nghĩa. Ba chỗ dễ lệch, cả ba đều im lặng:

* **`k1`, `b`** phải là mặc định của `rank_bm25` (1.5 / 0.75), không phải mặc
  định của một thư viện khác.
* **Luật idf âm.** `BM25Okapi` không kẹp idf âm về 0 — nó thay bằng
  `eps × idf dương trung bình` (`eps = 0.25`). Kẹp về 0 làm mọi từ phổ biến
  biến mất khỏi điểm thay vì bị phạt nhẹ, và bảng xếp hạng đổi ở đuôi.
* **`float64`.** `rank_bm25` tính bằng float64. Bản thử `float32` đo ra đủ tốt
  (top-100 trùng 100/100, lệch ≤ 9,1e-06) nhưng float64 xoá hẳn nhóm nghi vấn
  đó với giá ~66 MB cho nguồn lớn nhất — rẻ so với một buổi truy vì sao Final
  lệch 0,001.

Test khoá ở `tests/unit/online/test_bm25_thua.py`: đối chiếu THẲNG với
`rank_bm25` trên cùng corpus, cả điểm lẫn thứ hạng.

⚠ **Không có test tự động cho việc "nhả GIL"** — đã thử và bỏ. Ở quy mô một
unit test (60.000 tài liệu, 0,08 ms mỗi lượt chấm) phép đo bắt **tranh chấp
cấp phát bộ nhớ** giữa hai luồng chứ không bắt GIL, và ra 5,6× tức tệ hơn cả
tuần tự. Ở quy mô thật (150.000 tài liệu, 1,0 ms mỗi lượt) cùng phép đo ra
**1,13×** so với 2,00× của tuần tự hoàn toàn. Bằng chứng đúng cho tính chất này
là số đo end-to-end trong `scripts/measure_latency.py`, không phải một test
thời gian sẽ đỏ ngẫu nhiên trên CI.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

#: Mặc định của `rank_bm25.BM25Okapi`. Đổi ba số này là đổi mọi điểm đã đo.
K1 = 1.5
B = 0.75
EPSILON = 0.25


class BM25Thua:
    """BM25Okapi trên ma trận thưa. Giao diện trùng phần `get_scores` đang dùng.

    Cố ý **không** kế thừa hay bắt chước toàn bộ API của `BM25Okapi`: chỗ duy
    nhất trong dự án gọi tới nó là `get_scores`, và một lớp giả vờ tương thích
    hơn mức đó sẽ mời người sau gọi một phương thức không có.
    """

    __slots__ = ("W", "tu", "so_tai_lieu")

    def __init__(self, corpus: list[list[str]]) -> None:
        """`corpus` là danh sách tài liệu đã tách từ, cùng thứ tự với `text_keys`."""
        self.tu: dict[str, int] = {}
        hang: list[int] = []
        cot: list[int] = []
        dem: list[int] = []
        do_dai = np.empty(len(corpus), dtype=np.float64)

        for i, doc in enumerate(corpus):
            do_dai[i] = len(doc)
            # Đếm tần suất trong một dict rồi mới đổ ra: ghi thẳng vào COO cho
            # mỗi lần xuất hiện sẽ sinh ra khoá trùng, và `csr_matrix` CỘNG các
            # khoá trùng — đúng kết quả, nhưng tốn gấp nhiều lần bộ nhớ trung
            # gian trên tài liệu dài.
            tan: dict[int, int] = {}
            for t in doc:
                j = self.tu.get(t)
                if j is None:
                    j = self.tu[t] = len(self.tu)
                tan[j] = tan.get(j, 0) + 1
            hang.extend([i] * len(tan))
            cot.extend(tan)
            dem.extend(tan.values())

        n = len(corpus)
        v = len(self.tu)
        self.so_tai_lieu = n
        if n == 0 or v == 0:
            self.W = sp.csc_matrix((n, max(v, 1)), dtype=np.float64)
            return

        tf = sp.coo_matrix(
            (np.asarray(dem, dtype=np.float64), (hang, cot)), shape=(n, v)
        )
        avg = float(do_dai.mean()) or 1.0

        # `df` đếm số TÀI LIỆU chứa từ, không phải tổng số lần xuất hiện. COO ở
        # đây đã gộp theo (tài liệu, từ) nên mỗi mục là một tài liệu ⇒ đếm mục
        # theo cột là đúng df.
        df = np.bincount(tf.col, minlength=v).astype(np.float64)
        idf = np.log(n - df + 0.5) - np.log(df + 0.5)
        duong = idf[idf > 0]
        idf[idf <= 0] = EPSILON * (float(duong.mean()) if duong.size else 0.0)

        mau_hang = K1 * (1.0 - B + B * do_dai / avg)
        tu_so = tf.data * (K1 + 1.0)
        mau = tf.data + mau_hang[tf.row]
        self.W = sp.csc_matrix(
            (tu_so / mau * idf[tf.col], (tf.row, tf.col)), shape=(n, v)
        )

    def get_scores(self, tokens: list[str]) -> np.ndarray:
        """Điểm BM25 của mọi tài liệu cho `tokens`, cùng thứ tự với corpus.

        Một từ lặp trong truy vấn được cộng đúng số lần nó xuất hiện — giống
        `BM25Okapi`, vốn lặp `for q in query` chứ không khử trùng. Nên
        `["mưa", "mưa"]` cho điểm gấp đôi `["mưa"]`, và đó là hành vi phải giữ.
        """
        cols = [j for j in (self.tu.get(t) for t in tokens) if j is not None]
        if not cols:
            return np.zeros(self.so_tai_lieu, dtype=np.float64)
        return np.asarray(self.W[:, cols].sum(axis=1)).ravel()
