#!/usr/bin/env python3
"""Generate missing BGE-M3 window parts from the complete window text map.

The Modal B2 workers produced ``window_text`` for all videos but only uploaded
``window_dense`` for a subset.  This script fills only videos with no dense
part, using the same BAAI/bge-m3 SentenceTransformer settings as the existing
parts.  It never overwrites a downloaded part.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-text",
        type=Path,
        default=Path(
            "/data/multimodal-fusion-engine/modal/keyframes-batch-2-v2/"
            "bundle/aic_btc_b2_v1/data/vectors/window_text.json"
        ),
    )
    parser.add_argument(
        "--existing-dir",
        type=Path,
        default=Path(
            "/data/multimodal-fusion-engine/modal/keyframes-batch-2-v2/"
            "bundle/aic_btc_b2_v1/parts/window_dense"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/multimodal-fusion-engine/runtime/btc_b12/"
            "generated/window_dense_b2"
        ),
    )
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.window_text.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in payload.items()
    ):
        raise ValueError("window_text phải là map string -> string")

    existing_videos = {path.stem for path in args.existing_dir.glob("*.npz")}
    by_video: dict[str, list[str]] = {}
    for window_id, _text in payload.items():
        video, marker = window_id.split("#", 1)
        if not marker.startswith("w"):
            raise ValueError(f"window id không hợp lệ: {window_id}")
        by_video.setdefault(video, []).append(window_id)
    missing_videos = sorted(set(by_video) - existing_videos)
    if not missing_videos:
        print("Không có video thiếu window_dense", flush=True)
        return

    missing_ids = [
        window_id
        for video in missing_videos
        for window_id in sorted(
            by_video[video], key=lambda value: int(value.rsplit("#w", 1)[1])
        )
    ]
    texts = [payload[window_id] for window_id in missing_ids]
    print(
        f"Sinh bù {len(missing_videos)} video / {len(missing_ids)} cửa sổ "
        f"bằng {args.model}",
        flush=True,
    )

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    vectors = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(missing_ids), 1024):
        raise ValueError(f"BGE-M3 trả shape bất thường: {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or np.any(norms <= 0):
        raise ValueError("vector sinh bù có NaN/Inf/zero norm")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    offset = 0
    for video in missing_videos:
        ids = sorted(
            by_video[video], key=lambda value: int(value.rsplit("#w", 1)[1])
        )
        matrix = vectors[offset : offset + len(ids)]
        offset += len(ids)
        path = args.output_dir / f"{video}.npz"
        if path.exists():
            raise FileExistsError(f"Không ghi đè part đã sinh: {path}")
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            ids=np.asarray(ids, dtype=np.str_),
            vectors=matrix,
        )
        os.replace(temporary, path)
        written += len(ids)

    manifest = {
        "model": args.model,
        "source": str(args.window_text),
        "existing_dir": str(args.existing_dir),
        "videos": len(missing_videos),
        "rows": written,
        "dim": int(vectors.shape[1]),
        "videos_written": missing_videos,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
