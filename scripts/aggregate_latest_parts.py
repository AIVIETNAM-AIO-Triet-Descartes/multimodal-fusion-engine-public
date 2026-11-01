#!/usr/bin/env python3
"""Aggregate the newest parts from BTC batch 1 and batch 2.

The output is a staged flat bundle under ``/data``.  It is deliberately built
from per-video parts and checked before the web process sees it:

* Gemini captions: B1 research parts + B2 raw parts split deterministically;
* subtitles: both bundles' subtitle parts;
* SigLIP and subtitle BGE: all current per-video vector parts;
* caption BGE: produced separately by ``embed_latest_caption_parts.py``;
* legacy Qwen B1 files are copied for rollback and comparison only.

No old flat file is silently mixed into a new artifact.  Every output receives
a JSON manifest with source paths and counts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch1-root", type=Path,
        default=Path("/data/multimodal-fusion-engine/modal/keyframes-btc/bundle/aic_btc_b1_v1"),
    )
    parser.add_argument(
        "--batch1-gemini", type=Path,
        default=Path(
            "/data/multimodal-fusion-engine/modal/keyframe-augmentation-research/"
            "research/caption-gemini-trammay-v1/parts/captions"
        ),
    )
    parser.add_argument(
        "--batch2-root", type=Path,
        default=Path(
            "/data/multimodal-fusion-engine/modal/keyframes-batch-2-v2/"
            "bundle/aic_btc_b2_v1"
        ),
    )
    parser.add_argument(
        "--batch1-window-text", type=Path,
        default=Path(
            "/data/multimodal-fusion-engine/modal/keyframe-augmentation-research/"
            "research/window-records-v1/data/vectors/window_text.json"
        ),
    )
    parser.add_argument(
        "--batch1-window-root", type=Path,
        default=Path(
            "/data/multimodal-fusion-engine/modal/keyframe-augmentation-research"
        ),
        help="Volume/root chứa parts window_dense của batch 1",
    )
    parser.add_argument(
        "--batch2-window-text", type=Path,
        default=Path(
            "/data/multimodal-fusion-engine/modal/keyframes-batch-2-v2/"
            "bundle/aic_btc_b2_v1/data/vectors/window_text.json"
        ),
    )
    parser.add_argument(
        "--batch2-window-dense-extra", type=Path, default=None,
        help="Part window_dense B2 sinh bù ngoài Modal (nếu có)",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/data/multimodal-fusion-engine/runtime/btc_b12/flat"),
    )
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    return parser.parse_args()


def read_json_parts(directory: Path) -> tuple[dict[str, str], int]:
    files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Không có JSON parts ở {directory}")
    merged: dict[str, str] = {}
    duplicates: list[str] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: phải là object")
        for key, value in payload.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(f"{path}: key/value không hợp lệ")
            if key in merged:
                duplicates.append(key)
            elif value.strip():
                merged[key] = value.strip()
    if duplicates:
        raise ValueError(f"JSON part trùng key ở {directory}: {duplicates[:5]}")
    return merged, len(files)


def merge_json_files(*paths: Path) -> tuple[dict[str, str], int]:
    """Merge already-flat JSON maps and reject duplicate window ids."""
    merged: dict[str, str] = {}
    for path in paths:
        payload = read_json_file(path)
        for key, value in payload.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(f"{path}: key/value không hợp lệ")
            if key in merged:
                raise ValueError(f"JSON phẳng trùng key {key} giữa các nguồn")
            if value.strip():
                merged[key] = value.strip()
    return merged, len(paths)


def read_json_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Không có JSON phẳng: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: phải là object")
    if any(not isinstance(key, str) or not isinstance(value, str)
           for key, value in payload.items()):
        raise ValueError(f"{path}: key/value không hợp lệ")
    return payload


def split_b2_raw(directory: Path) -> tuple[dict[str, str], dict[str, str], int]:
    from mfe.offline.caption_prompt import loc_khoi_ocr, tach_caption_ocr

    files = sorted(directory.glob("*.json"))
    if len(files) != 614:
        raise RuntimeError(f"B2 gemini_raw mới có {len(files)}/614 part")
    captions: dict[str, str] = {}
    ocr: dict[str, str] = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, raw in payload.items():
            cap, text = tach_caption_ocr(str(raw))
            if cap.strip():
                if key in captions:
                    raise ValueError(f"B2 caption trùng key {key}")
                captions[key] = cap.strip()
            text = loc_khoi_ocr(text).strip()
            if text:
                if key in ocr:
                    raise ValueError(f"B2 OCR trùng key {key}")
                ocr[key] = text
    return captions, ocr, len(files)


def merge_vector_parts(
    *directories: Path, expected_files: tuple[int, ...]
) -> tuple[list[str], np.ndarray, int]:
    paths: list[Path] = []
    for directory in directories:
        paths.extend(sorted(directory.glob("*.npz")))
    if len(paths) != sum(expected_files):
        raise RuntimeError(
            f"Vector parts không đủ: có {len(paths)}, cần {sum(expected_files)} "
            f"({expected_files}) ở {directories}"
        )
    ids: list[str] = []
    matrices: list[np.ndarray] = []
    positions: dict[str, int] = {}
    duplicates = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            part_ids = [str(value) for value in payload["ids"].tolist()]
            matrix = np.asarray(payload["vectors"], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(part_ids):
            raise ValueError(f"{path}: vectors {matrix.shape} không khớp ids")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{path}: có NaN/Inf")
        keep: list[int] = []
        for index, key in enumerate(part_ids):
            if key in positions:
                duplicates += 1
                continue
            positions[key] = len(ids) + len(keep)
            keep.append(index)
        if keep:
            ids.extend(part_ids[index] for index in keep)
            matrices.append(matrix[keep])
    if not matrices:
        return [], np.zeros((0, 0), dtype=np.float32), duplicates
    matrix = np.concatenate(matrices, axis=0).astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("vector aggregate có NaN/Inf/zero norm")
    matrix /= norms
    return ids, matrix, duplicates


def write_json(path: Path, payload: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_npz(path: Path, ids: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, ids=np.asarray(ids, dtype=np.str_), vectors=matrix)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    from mfe.offline.transport import load_dotenv

    for key, value in load_dotenv(args.dotenv).items():
        os.environ.setdefault(key, value)
    b1 = args.batch1_root
    b2 = args.batch2_root
    if not b1.is_dir() or not b2.is_dir():
        raise SystemExit("Thiếu bundle local B1 hoặc B2")

    b1_gemini, b1_gemini_files = read_json_parts(args.batch1_gemini)
    b2_gemini, b2_ocr, b2_gemini_files = split_b2_raw(b2 / "parts" / "gemini_raw")
    if set(b1_gemini).intersection(b2_gemini):
        raise RuntimeError("Key caption B1/B2 bị trùng")
    gemini_captions = {**b1_gemini, **b2_gemini}

    b1_subtitles, b1_subtitle_files = read_json_parts(b1 / "parts" / "subtitles")
    b2_subtitles, b2_subtitle_files = read_json_parts(b2 / "parts" / "subtitles")
    if set(b1_subtitles).intersection(b2_subtitles):
        raise RuntimeError("Key subtitle B1/B2 bị trùng")
    subtitles = {**b1_subtitles, **b2_subtitles}

    b1_siglip_files = len(list((b1 / "parts" / "siglip_vectors").glob("*.npz")))
    b2_siglip_files = len(list((b2 / "parts" / "siglip_vectors").glob("*.npz")))
    siglip_ids, siglip, siglip_duplicates = merge_vector_parts(
        b1 / "parts" / "siglip_vectors", b2 / "parts" / "siglip_vectors",
        expected_files=(b1_siglip_files, b2_siglip_files),
    )
    b1_subdense_files = len(list((b1 / "parts" / "subtitles_dense").glob("*.npz")))
    b2_subdense_files = len(list((b2 / "parts" / "subtitles_dense").glob("*.npz")))
    subdense_ids, subdense, subdense_duplicates = merge_vector_parts(
        b1 / "parts" / "subtitles_dense", b2 / "parts" / "subtitles_dense",
        expected_files=(b1_subdense_files, b2_subdense_files),
    )

    b1_window_text = read_json_file(args.batch1_window_text)
    b2_window_text = read_json_file(args.batch2_window_text)
    window_text, _window_text_sources = merge_json_files(
        args.batch1_window_text, args.batch2_window_text
    )
    b1_window_dir = args.batch1_window_root / "bundle" / "aic_btc_b1_v1" / "parts" / "window_dense"
    b2_window_dir = b2 / "parts" / "window_dense"
    b1_window_files = len(list(b1_window_dir.glob("*.npz")))
    b2_window_files = len(list(b2_window_dir.glob("*.npz")))
    window_dense_dirs = [b1_window_dir, b2_window_dir]
    window_dense_file_counts = [b1_window_files, b2_window_files]
    extra_window_files = 0
    if args.batch2_window_dense_extra is not None:
        extra_window_files = len(list(args.batch2_window_dense_extra.glob("*.npz")))
        window_dense_dirs.append(args.batch2_window_dense_extra)
        window_dense_file_counts.append(extra_window_files)
    window_ids, window_dense, window_duplicates = merge_vector_parts(
        *window_dense_dirs,
        expected_files=tuple(window_dense_file_counts),
    )
    if window_duplicates:
        raise RuntimeError(
            f"window_dense bị trùng {window_duplicates} window_id; không gộp âm thầm"
        )
    if window_dense.ndim != 2 or window_dense.shape[1] != 1024:
        raise RuntimeError(
            f"window_dense phải có 1024 chiều, nhận {window_dense.shape}"
        )
    text_ids = set(window_text)
    dense_ids = set(window_ids)
    if text_ids != dense_ids:
        missing_dense = sorted(text_ids - dense_ids)
        extra_dense = sorted(dense_ids - text_ids)
        raise RuntimeError(
            "window_text/window_dense lệch khóa: "
            f"thiếu dense {len(missing_dense)}, dư dense {len(extra_dense)}; "
            f"mẫu thiếu {missing_dense[:3]}, mẫu dư {extra_dense[:3]}"
        )

    vector_dir = args.output_root / "data" / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    write_json(vector_dir / "gemini_captions.json", gemini_captions)
    write_json(vector_dir / "gemini_ocr.json", b2_ocr)
    write_json(vector_dir / "subtitles.json", subtitles)
    write_npz(vector_dir / "siglip_vectors.npz", siglip_ids, siglip)
    write_npz(vector_dir / "subtitles_dense.npz", subdense_ids, subdense)
    write_json(vector_dir / "window_text.json", window_text)
    write_npz(vector_dir / "window_dense.npz", window_ids, window_dense)

    # Keep the already validated B1 Qwen files available as a rollback arm. The
    # B12 config disables caption_qwen, so these files cannot silently replace
    # the newest Gemini caption source.
    legacy_dir = b1 / "data" / "vectors"
    for filename in ("qwen_captions.json", "qwen_ocr.json"):
        source = legacy_dir / filename
        if source.is_file():
            shutil.copyfile(source, vector_dir / filename)

    manifest = {
        "created_at_unix": time.time(),
        "source": "latest per-video parts",
        "batch1": {
            "gemini_caption_parts": b1_gemini_files,
            "gemini_caption_rows": len(b1_gemini),
            "subtitle_parts": b1_subtitle_files,
            "subtitle_rows": len(b1_subtitles),
            "siglip_parts": b1_siglip_files,
            "subtitles_dense_parts": b1_subdense_files,
            "window_text_sources": 1,
            "window_text_rows": len(b1_window_text),
            "window_dense_parts": b1_window_files,
        },
        "batch2": {
            "gemini_raw_parts": b2_gemini_files,
            "gemini_caption_rows": len(b2_gemini),
            "gemini_ocr_rows": len(b2_ocr),
            "subtitle_parts": b2_subtitle_files,
            "subtitle_rows": len(b2_subtitles),
            "siglip_parts": b2_siglip_files,
            "subtitles_dense_parts": b2_subdense_files,
            "window_text_rows": len(b2_window_text),
            "window_dense_parts": b2_window_files,
            "window_dense_generated_parts": extra_window_files,
        },
        "artifacts": {
            "gemini_captions": len(gemini_captions),
            "gemini_ocr": len(b2_ocr),
            "subtitles": len(subtitles),
            "siglip_vectors": {
                "rows": len(siglip_ids),
                "dim": int(siglip.shape[1]),
                "duplicates_dropped": siglip_duplicates,
            },
            "subtitles_dense": {
                "rows": len(subdense_ids),
                "dim": int(subdense.shape[1]),
                "duplicates_dropped": subdense_duplicates,
            },
            "window_text": len(window_text),
            "window_dense": {
                "rows": len(window_ids),
                "dim": int(window_dense.shape[1]),
                "duplicates_dropped": window_duplicates,
            },
            "dense_vectors": "generated by embed_latest_caption_parts.py",
        },
        "disabled_arms": ["caption_qwen", "ocr", "ocr_paddle", "ocr_vietocr", "clap", "object"],
    }
    (args.output_root / "aggregate-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
