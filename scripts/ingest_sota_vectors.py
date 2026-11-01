"""Load SOTA_AIC's precomputed image/text embeddings into Milvus.

    python scripts/ingest_sota_vectors.py --model bge-m3 --sota-root path/to/SOTA_AIC \
        --config config/pilot.yaml --videos L21_V001,L21_V002,L21_V003

Bảo already ran both models over the 855 pilot keyframes, so this costs no GPU —
it is the cheap way to test the whole Milvus dense path before committing to a
Kaggle run (ke-hoach-merge-sota.md §6.0 P2).

**Two files, two formats.** `dense_vectors.npz` is a clean `ids` + `vectors`
pair; `siglip_vectors.npy` is a pickled Python dict and needs
`allow_pickle=True`, which executes pickle on load. That is why this script
refuses to read it from anywhere but a path the caller named explicitly, and why
the merge plan asks Bảo to standardise on `savez` (§9b).

One collection per model: Milvus cannot mix dimensions, and the
`model_fingerprint` field makes a model swap visible instead of silent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from mfe.core.config import load_config
from mfe.offline.adapters.sota_ids import SotaIdMap
from mfe.offline.provided import ProvidedDataset
from mfe.stores.production.config import MilvusConfig
from mfe.stores.production.contracts import StorageRecord
from mfe.stores.production.milvus import MilvusVectorStore

#: model key -> (filename, dimension, collection, fingerprint, format)
MODELS: dict[str, tuple[str, int, str, str, str]] = {
    "bge-m3": (
        "dense_vectors.npz",
        1024,
        "aic_keyframe_bge_m3_v1",
        "bge-m3-dense-caption-fp32",
        "npz",
    ),
    "siglip": (
        "siglip_vectors.npy",
        1152,
        "aic_keyframe_siglip_v1",
        "siglip-so400m-patch14-384-fp32",
        "npy_dict",
    ),
}


def _read_vectors(path: Path, fmt: str) -> dict[str, np.ndarray]:
    if fmt == "npz":
        payload = np.load(path)
        return dict(zip(payload["ids"].tolist(), payload["vectors"], strict=True))
    # Pickled dict. Only reached for a path the caller typed.
    return dict(np.load(path, allow_pickle=True).item())


def main() -> None:
    parser = argparse.ArgumentParser(prog="ingest_sota_vectors", description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--sota-root", required=True)
    parser.add_argument("--config", default="config/pilot.yaml")
    parser.add_argument("--videos", required=True)
    parser.add_argument("--collection", default=None, help="Override the default collection")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    filename, dimension, default_collection, fingerprint, fmt = MODELS[args.model]
    collection = args.collection or default_collection

    config = load_config(args.config)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    dataset = ProvidedDataset(
        keyframes_dir=config.paths.keyframes,
        map_keyframes_dir=config.paths.map_keyframes,
        clip_features_dir=config.paths.clip_features,
        objects_dir=config.paths.objects,
        media_info_dir=config.paths.media_info,
        videos_dir=config.paths.videos,
    )
    id_map = SotaIdMap(dataset, videos)

    path = Path(args.sota_root) / "data" / "vectors" / filename
    if not path.is_file():
        print(f"khong tim thay {path}")
        sys.exit(1)

    raw = _read_vectors(path, fmt)
    print(f"\n--- {args.model} ---")
    print(f"  file       : {path.name} ({len(raw)} vector)")

    records: list[StorageRecord] = []
    unknown: list[str] = []
    wrong_dim: list[str] = []
    for sota_id, vector in raw.items():
        try:
            keyframe_id = id_map.to_keyframe_id(sota_id)
        except Exception:  # noqa: BLE001 — ids outside the pilot set are expected
            unknown.append(sota_id)
            continue
        array = np.asarray(vector, dtype=np.float32).ravel()
        if array.shape[0] != dimension:
            wrong_dim.append(sota_id)
            continue
        video_id, frame_id = keyframe_id.split("#")
        records.append(
            StorageRecord.from_values(
                video_id=video_id, frame_id=int(frame_id), embedding=array
            )
        )

    print(f"  dich duoc  : {len(records)}")
    if unknown:
        print(f"  ngoai bo   : {len(unknown)} (vd {unknown[:3]})")
    if wrong_dim:
        print(f"  SAI CHIEU  : {len(wrong_dim)} — cho {dimension}")
    if not records:
        sys.exit(1)

    norms = np.linalg.norm([r.embedding for r in records], axis=1)
    print(f"  norm L2    : {norms.min():.4f} .. {norms.max():.4f}")

    if args.dry_run:
        print("\n--dry-run: khong ghi gi.")
        return

    milvus_config = MilvusConfig.from_env().model_copy(
        update={
            "collection": collection,
            "dimension": dimension,
            "model_fingerprint": fingerprint,
        }
    )
    store = MilvusVectorStore(milvus_config)
    health = store.health()
    if not health.connection_ok:
        print(f"\nMilvus khong ket noi duoc: {health.message}")
        sys.exit(1)

    store.ensure_schema()
    report = store.ingest(records)

    print("\n--- ghi vao Milvus ---")
    print(f"  collection : {collection} ({dimension} chieu)")
    print(f"  fingerprint: {fingerprint}")
    print(f"  inserted   : {report.inserted}   updated: {report.updated}   failed: {report.failed}")
    for error in report.errors[:5]:
        print(f"    ! {error.logical_key} {error.message[:160]}")
    print(f"  tong       : {store.count()}")
    if report.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
