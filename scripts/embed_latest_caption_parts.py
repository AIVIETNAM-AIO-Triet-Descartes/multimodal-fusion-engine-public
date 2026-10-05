#!/usr/bin/env python3
"""Embed the newest Gemini caption parts from both BTC batches.

Batch 1's newest captions are on the research volume as one JSON part per
video.  Batch 2's newest captions are the output of the raw split/embed stage
and are read from ``parts/dense_vectors`` after the three-video repair.  The
model is loaded once for the complete run and the resulting flat file is
written atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch1-captions",
        type=Path,
        default=Path(
            "/data/multimodal-fusion-engine/modal/keyframe-augmentation-research/"
            "research/caption-gemini-trammay-v1/parts/captions"
        ),
    )
    parser.add_argument(
        "--batch2-root",
        type=Path,
        default=Path(
            "/data/multimodal-fusion-engine/modal/keyframes-batch-2-v2/"
            "bundle/aic_btc_b2_v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/data/multimodal-fusion-engine/runtime/btc_b12/flat/data/vectors/"
            "dense_vectors.npz"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--checkpoint", default="BAAI/bge-m3")
    parser.add_argument(
        "--batch1-dense-parts",
        type=Path,
        default=None,
        help=(
            "Dùng trực tiếp 873 part dense_vectors B1 đã kiểm định; đây là "
            "đường aggregate từ part, không encode lại caption."
        ),
    )
    parser.add_argument(
        "--batch1-dense",
        type=Path,
        default=None,
        help=(
            "Dùng dense_vectors B1 đã kiểm định thay vì encode lại 176k caption. "
            "Caption Gemini B1 vẫn được aggregate riêng cho arm BM25; bỏ option "
            "để chạy nâng cấp BGE đầy đủ."
        ),
    )
    parser.add_argument("--dotenv", type=Path, default=Path("/home/ubuntu/.env"))
    return parser.parse_args()


def load_caption_parts(root: Path) -> list[tuple[str, list[str], list[str]]]:
    files = sorted(root.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Không có caption part ở {root}")
    output: list[tuple[str, list[str], list[str]]] = []
    seen: set[str] = set()
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: caption part phải là object")
        keys = sorted(k for k, value in payload.items() if isinstance(value, str) and value.strip())
        duplicate = seen.intersection(keys)
        if duplicate:
            raise ValueError(f"Caption key trùng giữa các part: {sorted(duplicate)[:5]}")
        seen.update(keys)
        output.append((path.stem, keys, [str(payload[k]).strip() for k in keys]))
    return output


def load_vector_parts(
    directory: Path, expected_files: int, label: str
) -> tuple[list[str], list[np.ndarray]]:
    files = sorted(directory.glob("*.npz"))
    if len(files) != expected_files:
        raise RuntimeError(
            f"{label} dense_vectors có {len(files)}/{expected_files} part; "
            "chạy đủ repair trước khi aggregate."
        )
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    seen: set[str] = set()
    for path in files:
        with np.load(path, allow_pickle=False) as payload:
            part_ids = [str(value) for value in payload["ids"].tolist()]
            matrix = np.asarray(payload["vectors"], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != 1024 or matrix.shape[0] != len(part_ids):
            raise ValueError(f"{path}: shape {matrix.shape} không khớp ids")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{path}: có NaN/Inf")
        duplicate = seen.intersection(part_ids)
        if duplicate:
            raise ValueError(f"{label} dense key trùng: {sorted(duplicate)[:5]}")
        seen.update(part_ids)
        ids.extend(part_ids)
        vectors.append(matrix)
    return ids, vectors


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.threads <= 0:
        raise SystemExit("--batch-size và --threads phải lớn hơn 0")
    if not args.batch1_captions.is_dir():
        raise SystemExit(f"Không thấy Batch 1 Gemini parts: {args.batch1_captions}")

    from mfe.offline.transport import load_dotenv

    for key, value in load_dotenv(args.dotenv).items():
        os.environ.setdefault(key, value)

    b1_parts = load_caption_parts(args.batch1_captions)
    b2_ids, b2_matrices = load_vector_parts(
        args.batch2_root / "parts" / "dense_vectors", 614, "Batch 2"
    )
    print(f"B1 Gemini caption parts: {len(b1_parts)}/873", flush=True)
    print(f"B2 dense parts: {len(b2_ids):,} vectors from 614/614 files", flush=True)
    ids: list[str] = []
    matrices: list[np.ndarray] = []
    seen: set[str] = set()
    started = time.perf_counter()
    rows = 0
    if args.batch1_dense_parts is not None and args.batch1_dense is not None:
        raise SystemExit("Chỉ chọn một trong --batch1-dense-parts hoặc --batch1-dense")
    if args.batch1_dense_parts is not None:
        b1_ids, b1_matrices = load_vector_parts(
            args.batch1_dense_parts, 873, "Batch 1"
        )
        ids.extend(b1_ids)
        matrices.extend(b1_matrices)
        seen.update(b1_ids)
        rows = len(b1_ids)
        print(
            f"B1 dense: aggregate từ {len(b1_matrices)}/873 part, "
            f"{rows:,} rows (không encode lại)",
            flush=True,
        )
    elif args.batch1_dense is not None:
        with np.load(args.batch1_dense, allow_pickle=False) as payload:
            b1_ids = [str(value) for value in payload["ids"].tolist()]
            b1_matrix = np.asarray(payload["vectors"], dtype=np.float32)
        if b1_matrix.ndim != 2 or b1_matrix.shape != (len(b1_ids), 1024):
            raise ValueError(f"B1 dense legacy shape {b1_matrix.shape}")
        ids.extend(b1_ids)
        matrices.append(b1_matrix)
        seen.update(b1_ids)
        rows = len(b1_ids)
        print(f"B1 dense: dùng file đã kiểm định {rows:,} rows (không encode lại)", flush=True)
    else:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, args.threads // 4)))
        except RuntimeError:
            pass
        print(f"Loading {args.checkpoint} once", flush=True)
        model = SentenceTransformer(args.checkpoint, device="cpu")
        for index, (video_id, keys, texts) in enumerate(b1_parts, 1):
            vectors = np.asarray(
                model.encode(
                    texts,
                    normalize_embeddings=True,
                    batch_size=args.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            if vectors.ndim != 2 or vectors.shape != (len(keys), 1024):
                raise ValueError(f"{video_id}: embedding shape {vectors.shape}")
            duplicate = seen.intersection(keys)
            if duplicate:
                raise ValueError(f"B1 caption key trùng: {sorted(duplicate)[:5]}")
            seen.update(keys)
            ids.extend(keys)
            matrices.append(vectors)
            rows += len(keys)
            if index == 1 or index % 25 == 0 or index == len(b1_parts):
                elapsed = time.perf_counter() - started
                rate = rows / max(elapsed, 0.001)
                print(
                    f"B1 [{index:>3}/{len(b1_parts)}] {rows:,} rows, "
                    f"{rate:,.0f} rows/s",
                    flush=True,
                )

    batch1_dense_rows = len(ids)
    duplicate = seen.intersection(b2_ids)
    if duplicate:
        raise ValueError(f"B1/B2 caption key trùng: {sorted(duplicate)[:5]}")
    ids.extend(b2_ids)
    matrices.extend(b2_matrices)
    matrix = np.concatenate(matrices, axis=0).astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(matrix).all() or np.any(norms <= 0):
        raise ValueError("dense_vectors cuối có NaN/Inf/zero norm")
    matrix /= norms[:, None]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez_compressed(temporary, ids=np.asarray(ids, dtype=np.str_), vectors=matrix)
    temporary.replace(args.output)
    report = {
        "checkpoint": args.checkpoint,
        "source": "latest Gemini caption parts",
        "batch1_parts": len(b1_parts),
        "batch1_rows": sum(len(keys) for _, keys, _ in b1_parts),
        "batch1_dense_rows": batch1_dense_rows,
        "batch1_dense_source": (
            str(args.batch1_dense_parts)
            if args.batch1_dense_parts
            else str(args.batch1_dense)
            if args.batch1_dense
            else "encoded from latest Gemini parts"
        ),
        "batch2_parts": 614,
        "batch2_rows": len(b2_ids),
        "batch2_dense_rows": len(b2_ids),
        "rows": len(ids),
        "dim": int(matrix.shape[1]),
        "created_at_unix": time.time(),
    }
    args.output.with_name("dense_vectors.manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
