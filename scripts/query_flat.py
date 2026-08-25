"""Tìm keyframe trên kho phẳng, chạy thẳng ở laptop — không Docker, không Modal.

    python scripts/query_flat.py "hai con lân múa trên cột"
    python scripts/query_flat.py "sạt lở đèo" --top 20 --arms ocr_vietocr,subtitle
    python scripts/query_flat.py "..." --fetch out/     # kéo ảnh về xem luôn

Vì sao tồn tại
---------------
Cả `siglip` (176.707 vector × 1152 chiều) lẫn hai kho văn bản đều **vừa RAM**, và
phép tìm là một `matmul` cộng vài BM25. Đo được: 24,8 ms cho một truy vấn vector
trên toàn bộ chỉ mục, ở CPU. Nên tầng truy vấn không cần Milvus, không cần ES, và
không cần GPU — ba thứ mà việc dựng chúng lên chiếm phần lớn thời gian vận hành
tuần trước.

Cái đắt là **nạp**, không phải **tìm**. Đo được trên chính máy này:

    dựng BM25 (169k dòng)      0,6s   -> cache pickle    0,5s
    đọc npz 777 MB             4,2s   -> mmap .npy       0,0s
    nạp tháp text SigLIP      ~10s    -> KHÔNG cache được giữa hai tiến trình
    ------------------------------------------------------------------
    truy vấn chỉ arm văn bản          ~1s toàn bộ
    truy vấn có siglip                ~14s, gần hết là nạp model

Nên: **một truy vấn lẻ thì dùng script này; dò nhiều truy vấn thì dùng notebook**
và giữ model trong phiên. Cache ở `.cache/flat/` đã lo hai dòng đầu; dòng thứ ba
là chi phí khởi động của tiến trình và không có cách nào tránh trong một CLI.

Trọng số lấy từ config
-----------------------
`retrieval.fusion_weights` của `config/btc_b1.yaml` — cùng con số mà lượt nộp
dùng. Nếu chỗ này khai lại trọng số riêng thì công cụ chẩn đoán sẽ đo một hệ
khác với hệ đem đi thi, và mọi kết luận rút ra ở đây đều không chuyển được.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from mfe.contracts.tools import Candidate
from mfe.core.cli import setup_console
from mfe.online.flat_arms import (
    DEFAULT_CACHE_DIR,
    FlatBundle,
    load_cached_bm25,
    load_cached_vectors,
)
from mfe.online.fusion import weighted_fusion
from mfe.stores.inverted.index import _normalize_text, _tokenize

logger = logging.getLogger(__name__)

DEFAULT_ROOT = "stores/flat"
DEFAULT_CONFIG = "config/btc_b1.yaml"
SIGLIP_CHECKPOINT = "google/siglip-so400m-patch14-384"
CLAP_CHECKPOINT = "laion/clap-htsat-unfused"

#: Arm bật mặc định. `ocr_vietocr` chứ không `ocr`: đo trên 873 video, vietocr có
#: 85,4% token nằm trong vốn từ tiếng Việt so với 75,7% của easyocr và 68,9% của
#: paddle (paddle đọc đúng chữ cái nhưng rụng dấu thanh).
DEFAULT_ARMS = ("siglip", "ocr_vietocr", "subtitle", "subtitle_exact")

#: Tên arm -> khoá trong `retrieval.fusion_weights`. `subtitle_exact` không phải
#: một nguồn văn bản riêng: nó là cách tra KHÁC trên cùng `subtitles.json`.
WEIGHT_KEY = {
    "siglip": "siglip",
    "clap": "clap",
    "ocr": "ocr",
    "ocr_vietocr": "ocr_vietocr",
    "ocr_paddle": "ocr_paddle",
    "subtitle": "subtitle_partial",
    "subtitle_exact": "subtitle_exact",
    "object": "object",
    "caption": "caption",
}


def encode_query_siglip(text_en: str) -> np.ndarray:
    """Tháp text SigLIP. Dùng chính `get_text_features` mà `n4_siglip` dùng cho ảnh.

    `padding="max_length"` là bắt buộc với SigLIP — tháp của nó huấn luyện với
    độ dài cố định 64, và padding động cho ra vector khác.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SIGLIP_CHECKPOINT)
    model = AutoModel.from_pretrained(SIGLIP_CHECKPOINT).eval()
    batch = tokenizer(
        [text_en], padding="max_length", max_length=64,
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        out = model.get_text_features(**batch)
    pooled = getattr(out, "pooler_output", None)
    vector = (pooled if pooled is not None else out).numpy()
    return (vector / np.linalg.norm(vector, axis=1, keepdims=True))[0]


def encode_query_clap(text_en: str) -> np.ndarray:
    """Tháp text CLAP. Cùng checkpoint `n3_audio` dùng cho phía âm thanh.

    CLAP **nghe**, không nhìn: truy vấn phải mô tả tiếng động ("tiếng trống",
    "còi xe"), không phải cảnh. Đưa một mô tả thị giác vào đây vẫn ra kết quả,
    xếp hạng vẫn trông hợp lý, và nó không có nghĩa gì.
    """
    import torch
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(CLAP_CHECKPOINT).eval()
    processor = ClapProcessor.from_pretrained(CLAP_CHECKPOINT)
    batch = processor(text=[text_en], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.get_text_features(**batch)
    pooled = getattr(out, "pooler_output", None)
    vector = (pooled if pooled is not None else out).numpy()
    return (vector / np.linalg.norm(vector, axis=1, keepdims=True))[0]


#: Arm dense -> hàm mã hoá truy vấn. Cả hai đều cần tiếng Anh.
DENSE_ENCODER = {"siglip": encode_query_siglip, "clap": encode_query_clap}


def to_candidates(keys, arm: str) -> list[Candidate]:
    return [
        Candidate(keyframe_id=k, video_id=k.split("#")[0], score=1.0 / rank,
                  evidence_id=f"{arm}:local:{rank}")
        for rank, k in enumerate(keys, start=1)
    ]


def run_arms(bundle, base, query_vi, query_en, arms, pool, use_cache, vectors=None):
    """`{ten_arm: [Candidate]}`. Arm không chạy được thì vắng mặt, kèm log lý do."""
    runs: dict[str, list[Candidate]] = {}

    vectors = vectors or {}
    for arm in arms:
        if arm in DENSE_ENCODER:
            if arm not in vectors:
                logger.warning("%s: khong nap duoc ma tran", arm)
                continue
            if not query_en:
                logger.warning(
                    "%s: khong co ban dich tieng Anh — arm ngoi ngoai. "
                    "Dung --en de dua tay, hoac --translate de goi envit5.", arm
                )
                continue
            vec_ids, matrix = vectors[arm]
            vector = DENSE_ENCODER[arm](query_en)
            scores = np.asarray(matrix) @ vector
            order = np.argsort(-scores)[:pool]
            runs[arm] = to_candidates([vec_ids[i] for i in order], arm)
            continue

        source = "subtitle" if arm == "subtitle_exact" else arm
        if source not in bundle.texts:
            logger.warning("%s: %s", arm, bundle.missing.get(source, "khong co nguon"))
            continue

        keys = bundle.text_keys[source]
        if arm == "subtitle_exact":
            needle = _normalize_text(query_vi)
            hits = [k for k in keys if needle in _normalize_text(bundle.texts[source][k])]
            runs[arm] = to_candidates(hits[:pool], arm)
            continue

        model = (
            load_cached_bm25(bundle, source, base, DEFAULT_CACHE_DIR)
            if use_cache
            else bundle.build_bm25(source)
        )
        if model is None:
            continue
        scores = model.get_scores(_tokenize(query_vi))
        order = [i for i in np.argsort(-scores)[:pool] if scores[i] > 0]
        runs[arm] = to_candidates([keys[i] for i in order], arm)

    return runs


def _load_uncached(base: Path, model: str):
    """`--no-cache`: đọc thẳng npz, không đụng `.cache/flat/`."""
    from mfe.online.flat_arms import DENSE_FILE

    with np.load(base / DENSE_FILE[model], allow_pickle=False) as payload:
        ids = [str(x) for x in payload["ids"].tolist()]
        matrix = np.asarray(payload["vectors"], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return ids, matrix / norms


def apply_quest(query_vi: str, query_en: str | None) -> tuple[str, str | None]:
    """Gọi tên thứ truy vấn nhắc tới mà không nêu tên — `mfe.online.quest`.

    Đụng **7 trong 49 đề chính thức**: *"phim của đạo diễn Steven Spielberg
    1975"* phải thành *"cá mập"* trước khi tìm, vì từ khoá cần tìm không nằm
    trong câu hỏi. Đo ngày 24-08 trên đúng 7 câu đó: 6 câu được gọi tên đúng,
    và cả 3 câu đối chứng (đề đã nêu đủ tên) trả về **rỗng** — tính chất quan
    trọng thứ hai, vì một từ bịa thêm là recall mất đi.

    In ra thứ nó thêm và **vì sao**. Một truy vấn âm thầm bị viết lại là một
    bảng kết quả không giải thích được.
    """
    from mfe.offline.transport import load_dotenv
    from mfe.online.quest import expanded_queries, quest_expand

    for key, value in load_dotenv(".env").items():
        os.environ.setdefault(key, value)

    terms = quest_expand(query_vi)
    if not terms:
        logger.info("quest: khong co gi de goi ten — tim bang nguyen cau")
        return query_vi, query_en
    for term in terms:
        logger.info("quest them: %r / %r — %s", term.vi, term.en, term.why)
    return expanded_queries(query_vi, query_en, terms)


def load_weights(config_path: str) -> dict[str, float]:
    import yaml

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return dict(raw.get("retrieval", {}).get("fusion_weights", {}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("query", help="Truy van tieng Viet.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Goc kho phang.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Nguon trong so.")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                        help="Danh sach arm, cach nhau bang dau phay.")
    parser.add_argument("--top", type=int, default=20, help="So dong in ra.")
    parser.add_argument("--pool", type=int, default=200,
                        help="Moi arm nop bao nhieu ung vien vao hop nhat.")
    parser.add_argument("--en", help="Ban dich tieng Anh, dua tay (tot nhat).")
    parser.add_argument("--translate", action="store_true",
                        help="Dich bang envit5 neu khong co --en.")
    parser.add_argument("--quest", action="store_true",
                        help="Goi LLM GOI TEN thu de nhac toi ma khong neu ten "
                             "(7/49 de chinh thuc). Chi THEM tu vao cuoi, khong thay cau.")
    parser.add_argument("--fetch", metavar="THU_MUC",
                        help="Keo anh cua top ket qua tu Modal Volume ve day.")
    parser.add_argument("--json", action="store_true", help="In JSON thay vi bang.")
    parser.add_argument("--no-cache", action="store_true", help="Bo qua cache BM25.")
    return parser


def main() -> None:
    setup_console()
    args = build_parser().parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    root = Path(args.root)
    base = root / "data" / "vectors"
    if not base.is_dir():
        base = root
    if not base.is_dir():
        raise SystemExit(f"{base} khong ton tai — kho phang nam o dau?")

    models = {a for a in arms if a in DENSE_ENCODER}
    sources = {("subtitle" if a == "subtitle_exact" else a) for a in arms} - models

    started = time.perf_counter()
    bundle = FlatBundle.load(root, sources=sources, models=set())
    vectors = {
        model: load_cached_vectors(base, model, DEFAULT_CACHE_DIR)
        if not args.no_cache
        else _load_uncached(base, model)
        for model in sorted(models)
    }
    logger.info("nap xong %.1fs · %s", time.perf_counter() - started, bundle.report())

    query_vi = args.query
    query_en = args.en
    if not query_en and args.translate:
        from mfe.online.translate import MachineTranslator

        query_en = MachineTranslator().translate(query_vi)
        logger.info("dich: %s", query_en)

    if args.quest:
        query_vi, query_en = apply_quest(query_vi, query_en)

    runs = run_arms(bundle, base, query_vi, query_en, arms, args.pool,
                    not args.no_cache, vectors)
    if not runs:
        raise SystemExit("khong arm nao chay duoc — xem canh bao o tren.")

    all_weights = load_weights(args.config)
    weights = {a: all_weights.get(WEIGHT_KEY.get(a, a), 1.0) for a in runs}
    fused = weighted_fusion(runs, weights=weights, run_id="query_flat",
                            normalizer="rrf", rrf_k=60)[:args.top]

    # Arm nao dong gop cho tung ket qua — cot quan trong nhat khi chan doan, va
    # thu ma bang xep hang goi gon lai lam mat.
    contributors: dict[str, list[str]] = {}
    for arm, candidates in runs.items():
        for candidate in candidates:
            contributors.setdefault(candidate.keyframe_id, []).append(arm)

    if args.json:
        print(json.dumps([
            {"rank": i, "keyframe_id": c.keyframe_id, "video_id": c.video_id,
             "score": round(c.score, 6), "arms": contributors.get(c.keyframe_id, [])}
            for i, c in enumerate(fused, start=1)
        ], ensure_ascii=False, indent=1))
    else:
        print(f"\ntrong so: {weights}")
        print("arm chay: " + ", ".join(f"{a}({len(c)})" for a, c in runs.items()))
        print(f"\n{'#':>3}  {'keyframe_id':<20} {'diem':>8}  arm")
        for i, c in enumerate(fused, start=1):
            print(f"{i:>3}  {c.keyframe_id:<20} {c.score:>8.5f}  "
                  f"{','.join(contributors.get(c.keyframe_id, []))}")

    if args.fetch:
        fetch_images([c.keyframe_id for c in fused], Path(args.fetch))


def fetch_images(keyframe_ids, out_dir: Path) -> None:
    """Kéo ảnh của các keyframe này từ Modal Volume về `out_dir`.

    `keyframe_id` mang `frame_idx`, còn tên file ảnh là số thứ tự `n` — hai con
    số khác nhau, và `map-keyframes` là chỗ duy nhất nối chúng lại.
    """
    import csv
    from concurrent.futures import ThreadPoolExecutor

    from mfe.offline.bundle import BundleLayout
    from mfe.offline.transport import build_transport, ensure_modal_credentials

    ensure_modal_credentials(".env")
    layout = BundleLayout("aic_btc_b1_v1")
    transport = build_transport("modal", volume="keyframes-btc",
                                local_root="bundle_local", version=2)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_dir = Path("data/btc_b1/map-keyframes")

    jobs = []
    for rank, keyframe_id in enumerate(keyframe_ids, start=1):
        video_id, _, frame = keyframe_id.partition("#")
        ordinal = None
        with (map_dir / f"{video_id}.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row["frame_idx"]) == int(frame):
                    ordinal = int(row["n"])
                    break
        if ordinal is None:
            logger.warning("%s: khong co dong map-keyframes", keyframe_id)
            continue
        jobs.append((f"{layout.keyframes_dir(video_id)}/{ordinal:03d}.jpg",
                     out_dir / f"{rank:02d}_{video_id}_{frame}_n{ordinal:03d}.jpg"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda job: transport.get(*job), jobs))
    print(f"\nkeo {len(jobs)} anh -> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
