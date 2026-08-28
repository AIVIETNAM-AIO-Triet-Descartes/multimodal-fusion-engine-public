"""Câu QA nên đi TÌM bằng chuỗi nào — bốn chiến lược, cùng một kho.

    python scripts/measure_qa_search_text.py --flat-root stores/flat

Vì sao có script này
---------------------
`PackagedQuery` **không có** trường `retrieval_text`, và `searches()` trả thẳng
`[self.text]` cho QA. Nhưng **48/48** câu QA của `queryset_dev.jsonl` và **15/15**
của `queryset_heldout.jsonl` đều có `retrieval_text` viết tay khác `text`.

Nghĩa là mọi con số QA từng công bố đều đo trên một chuỗi mà **đường nộp không
dựng được**. Script này đặt giá cho khoảng cách đó, rồi thử hai cách khép nó lại
mà không cần người viết tay.

Bốn chiến lược
---------------
* `text` — nguyên văn câu đề. Đúng thứ `app/submit_lab.py` đang tìm bằng.
* `cat_cau_hoi` — bỏ các câu kết thúc bằng `?`. Không gọi API, không tốn gì.
* `quest` — `quest_decompose` tách thành `search_hint` (TIẾNG ANH, cho tháp dense)
  và `ocr_keywords` (tiếng Việt, cho các arm BM25). `search_hint` đi **thẳng**
  vào SigLIP, không qua envit5 — đó là điểm mấu chốt, vì envit5 chính là một
  trong hai nguyên nhân.
* `retrieval_text` — bản viết tay. **Trần trên**, không phải một lựa chọn: lúc
  thi không ai đưa nó cho ta.

Đo cả hạng của arm `siglip` (trọng số 2,0) lẫn hạng sau hợp nhất, vì hai con số
trả lời hai câu khác nhau: arm nào hỏng, và hỏng đó có sống sót qua RRF không.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfe.contracts.tools import Candidate  # noqa: E402
from mfe.core.cli import setup_console  # noqa: E402
from mfe.core.config import load_config  # noqa: E402
from mfe.offline.transport import load_dotenv  # noqa: E402
from mfe.online.flat_arms import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    FlatBundle,
    load_cached_bm25,
    load_cached_vectors,
)
from mfe.online.fusion import weighted_fusion  # noqa: E402
from mfe.online.query_text import strip_questions  # noqa: E402
from mfe.stores.inverted.index import _normalize_text, _tokenize  # noqa: E402

logger = logging.getLogger(__name__)

SIGLIP_CHECKPOINT = "google/siglip-so400m-patch14-384"
POOL = 500

#: Arm văn bản -> khoá nguồn trong `FlatBundle`. Cùng bảng `query_flat.py` dùng.
TEXT_ARMS = {
    "ocr_vietocr": "ocr_vietocr",
    "ocr_paddle": "ocr_paddle",
    "ocr": "ocr",
    "subtitle_partial": "subtitle",
    "object": "object",
}


@dataclass
class Strategy:
    """Một chiến lược: chuỗi tiếng Việt cho BM25, chuỗi tiếng Anh cho dense."""

    name: str
    vi: str
    #: Đã là tiếng Anh sẵn thì khỏi qua envit5. `None` = dịch `vi`.
    en: str | None = None
    #: Đường dự phòng đã chạy thay vì đường thật (quest hỏng, thiếu khoá…).
    fell_back: bool = False


def build_strategies(row: dict, use_quest: bool) -> list[Strategy]:
    text = row["text"]
    out = [
        Strategy("text", text),
        Strategy("cat_cau_hoi", strip_questions(text)),
    ]
    if use_quest:
        from mfe.online.quest import quest_decompose  # noqa: PLC0415

        parts = quest_decompose(text)
        hint = (parts.get("search_hint") or "").strip()
        keywords = (parts.get("ocr_keywords") or "").strip()
        # `quest_decompose` rơi về câu hỏi gốc KHÔNG BÁO GÌ khi API hỏng. Một
        # chiến lược không chạy in ra y hệt một chiến lược chạy mà vô ích; cột
        # `roi ve` là chỗ duy nhất phân biệt.
        fell_back = hint == text or not hint
        out.append(Strategy(
            "quest", keywords or strip_questions(text),
            en=None if fell_back else hint, fell_back=fell_back,
        ))
    out.append(Strategy("retrieval_text", row["retrieval_text"]))
    return out


class Engine:
    """Kho phẳng + hai tháp text, nạp một lần."""

    def __init__(self, flat_root: str, weights: dict[str, float]) -> None:
        self.base = Path(flat_root) / "data" / "vectors"
        if not self.base.is_dir():
            self.base = Path(flat_root)
        self.weights = weights
        self.bundle = FlatBundle.load(flat_root)
        self.ids, matrix = load_cached_vectors(self.base, "siglip", DEFAULT_CACHE_DIR)
        self.matrix = np.asarray(matrix)
        #: `keyframe_id -> chỉ số`. `list.index` trong vòng lặp là O(n) trên
        #: 176.707 phần tử, và vòng lặp chạy 4 chiến lược × 47 câu.
        self.position = {kid: i for i, kid in enumerate(self.ids)}
        #: BM25 đọc từ pickle mất 0,4–1,5s MỖI nguồn; giữ lại thì 4 chiến lược ×
        #: 47 câu chỉ trả giá đó một lần thay vì 188 lần.
        self._bm25: dict[str, object] = {}
        self._tok = None
        self._siglip = None
        self._mt = None

    def bm25(self, source: str):
        if source not in self._bm25:
            self._bm25[source] = load_cached_bm25(
                self.bundle, source, self.base, DEFAULT_CACHE_DIR
            )
        return self._bm25[source]

    def encode(self, text_en: str) -> np.ndarray:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        if self._siglip is None:
            self._tok = AutoTokenizer.from_pretrained(SIGLIP_CHECKPOINT)
            self._siglip = AutoModel.from_pretrained(SIGLIP_CHECKPOINT).eval()
        # `padding="max_length"` bat buoc voi SigLIP: thap cua no huan luyen o do
        # dai co dinh 64, padding dong cho ra vector khac.
        batch = self._tok(
            [text_en], padding="max_length", max_length=64,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            out = self._siglip.get_text_features(**batch)
        pooled = getattr(out, "pooler_output", None)
        vector = (pooled if pooled is not None else out).numpy()
        return (vector / np.linalg.norm(vector, axis=1, keepdims=True))[0]

    def translate(self, text_vi: str) -> str:
        from mfe.online.translate import MachineTranslator  # noqa: PLC0415

        if self._mt is None:
            self._mt = MachineTranslator()
        return self._mt.translate(text_vi)

    def run(self, strategy: Strategy) -> tuple[list[Candidate], np.ndarray]:
        """`(danh sach da hop nhat, diem siglip toan cuc)`."""
        text_en = strategy.en or self.translate(strategy.vi)
        scores = self.matrix @ self.encode(text_en)
        order = np.argsort(-scores)

        runs: dict[str, list[Candidate]] = {
            "siglip": _to_candidates([self.ids[i] for i in order[:POOL]], "siglip")
        }
        for arm, source in TEXT_ARMS.items():
            if source not in self.bundle.texts:
                continue
            model = self.bm25(source)
            if model is None:
                continue
            keys = self.bundle.text_keys[source]
            bm = model.get_scores(_tokenize(strategy.vi))
            picked = [i for i in np.argsort(-bm)[:POOL] if bm[i] > 0]
            runs[arm] = _to_candidates([keys[i] for i in picked], arm)

        if "subtitle" in self.bundle.texts:
            needle = _normalize_text(strategy.vi)
            hits = [
                k for k in self.bundle.text_keys["subtitle"]
                if needle in _normalize_text(self.bundle.texts["subtitle"][k])
            ]
            runs["subtitle_exact"] = _to_candidates(hits[:POOL], "subtitle_exact")

        return weighted_fusion(runs, weights=self.weights), scores


def _to_candidates(keys, arm: str) -> list[Candidate]:
    return [
        Candidate(keyframe_id=k, video_id=k.split("#")[0], score=1.0 / rank,
                  evidence_id=f"{arm}:local:{rank}")
        for rank, k in enumerate(keys, start=1)
    ]


def gt_positions(ids: list[str], row: dict) -> tuple[set[int], set[str]]:
    """Chỉ số và `keyframe_id` của mọi keyframe nằm trong khoảng GT."""
    lo, hi = row["gt_frame_range"]
    video = row["gt_video_id"]
    hit_idx, hit_ids = set(), set()
    for pos, kid in enumerate(ids):
        vid, _, frame = kid.partition("#")
        if vid == video and lo <= int(frame) <= hi:
            hit_idx.add(pos)
            hit_ids.add(kid)
    return hit_idx, hit_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queryset", default="eval/queryset_dev.jsonl")
    parser.add_argument("--flat-root", default="stores/flat")
    parser.add_argument("--config", default="config/btc_b1.yaml")
    parser.add_argument("--no-quest", action="store_true",
                        help="Bo chien luoc quest (khong goi API).")
    parser.add_argument("--limit", type=int, default=0, help="Chi N cau dau.")
    args = parser.parse_args()
    setup_console()

    # Khong nap .env la `quest` bao "khong co GEMINI_API_KEY" trong khi khoa nam
    # san trong file — dung loi da lam mat mot luot do `hf_cross` hom 24-08.
    for key, value in load_dotenv(ROOT / ".env").items():
        os.environ.setdefault(key, value)

    rows = [
        json.loads(line)
        for line in Path(args.queryset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    qa = [r for r in rows if r.get("type") == "QA" and r.get("gt_video_id")]
    if args.limit:
        qa = qa[: args.limit]

    config = load_config(args.config)
    weights = dict(config.retrieval.fusion_weights)
    engine = Engine(args.flat_root, weights)
    print(f"trong so: { {k: v for k, v in weights.items() if v} }")
    print(f"{len(qa)} cau QA co ground truth\n")

    use_quest = not args.no_quest
    if use_quest:
        from mfe.online.quest import GEMINI_MODEL, QuestClient  # noqa: PLC0415

        ok, why = QuestClient().available()
        print(f"quest: {'san sang' if ok else 'KHONG chay duoc — ' + str(why)}"
              f"  model={GEMINI_MODEL}\n")
        use_quest = ok

    table: dict[str, list[tuple[int, int]]] = {}
    fallbacks: dict[str, int] = {}
    #: Số câu chiến lược thật sự ĐỔI chuỗi so với `text`. Một luật không chạm câu
    #: nào in ra y hệt `text` — cột này là chỗ duy nhất phân biệt "luật vô hại"
    #: với "luật không bao giờ kích hoạt".
    touched: dict[str, int] = {}
    for row in qa:
        idx, _ = gt_positions(engine.ids, row)
        if not idx:
            print(f"  bo qua {row['query_id']}: khong keyframe nao trong khoang GT")
            continue
        line = [f"  {row['query_id']:16}"]
        for strategy in build_strategies(row, use_quest):
            fused, scores = engine.run(strategy)
            order = np.argsort(-scores)
            sig = next(
                (r for r, i in enumerate(order, 1) if int(i) in idx), len(engine.ids)
            )
            hit = next(
                (r for r, c in enumerate(fused, 1)
                 if engine.position[c.keyframe_id] in idx),
                len(engine.ids),
            )
            table.setdefault(strategy.name, []).append((sig, hit))
            fallbacks[strategy.name] = fallbacks.get(strategy.name, 0) + int(
                strategy.fell_back
            )
            touched[strategy.name] = touched.get(strategy.name, 0) + int(
                strategy.vi.strip() != row["text"].strip() or strategy.en is not None
            )
            line.append(f"{strategy.name}={sig}/{hit}")
        print("  ".join(line))

    n = len(next(iter(table.values()), []))
    print(f"\n{'=' * 92}\nTONG KET tren {n} cau  (hang siglip / hang sau hop nhat)")
    print(f"{'chien luoc':16} {'pool 500':>10} {'R@100':>7} {'R@20':>6} "
          f"{'R@5':>5} {'trung vi':>9} {'cham':>6} {'roi ve':>8}")
    print("-" * 92)
    for name, pairs in table.items():
        fused = [f for _, f in pairs]
        print(
            f"{name:16} {sum(1 for r in fused if r <= POOL):>7}/{n} "
            f"{sum(1 for r in fused if r <= 100):>7} "
            f"{sum(1 for r in fused if r <= 20):>6} "
            f"{sum(1 for r in fused if r <= 5):>5} "
            f"{int(np.median(fused)):>9} {touched.get(name, 0):>6} "
            f"{fallbacks.get(name, 0):>8}"
        )
    print("\ncham   = so cau chien luoc thuc su doi chuoi so voi `text`")
    print("roi ve = so cau chien luoc HONG va chay bang duong du phong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
