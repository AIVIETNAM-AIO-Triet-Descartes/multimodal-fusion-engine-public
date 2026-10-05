"""Prove the CLIP checkpoint matches the one BTC used — do not assume it.

    python scripts/verify_clip_checkpoint.py --config config/local.yaml -n 5

Re-encodes a few already-ingested keyframes with `clip-ViT-B-32` and compares
each result against the vector BTC shipped for that same keyframe. Both are
unit-norm, so the dot product is the cosine similarity.

Why this script exists: a mismatched checkpoint does not raise. Text queries
would still return 100 confidently-ranked results, all meaningless, and nothing
in the pipeline would look broken. The only way to know is to measure.

Caveat worth keeping in mind (Vỹ Triết): the local sample is AIC 2025 batch 1.
The official PDF §3 states batch 1 of AIC 2026 *is* that same data, so a pass
here is real evidence — but batch 2 has not shipped, and this check should be
re-run against it before anyone treats the question as permanently closed.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from mfe.core.cli import setup_console
from mfe.core.config import load_config
from mfe.online.encoder import (
    CHECKPOINT_MATCH_THRESHOLD,
    DEFAULT_CLIP_CHECKPOINT,
    verify_against_reference,
)
from mfe.stores.asset.manifest import AssetStore
from mfe.stores.vector.index import VectorIndex


def main() -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--checkpoint", default=DEFAULT_CLIP_CHECKPOINT)
    parser.add_argument("-n", type=int, default=5, help="How many keyframes to test")
    args = parser.parse_args()

    config = load_config(args.config)
    index = VectorIndex.load(config.paths.vector_store, config.retrieval.embed_dim)
    assets = AssetStore.load(config.paths.asset_store)

    total = len(index)
    if total == 0:
        print("Vector Index is empty — run mfe-ingest first.")
        return 2

    # Spread the sample across the store rather than taking the first N: the
    # first N are all one video, and a per-video preprocessing difference would
    # hide behind that.
    positions = np.linspace(0, total - 1, num=min(args.n, total), dtype=int)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Nguong chap nhan: {CHECKPOINT_MATCH_THRESHOLD}\n")
    print(f"{'keyframe_id':<20} {'cosine':>9}   ket qua")
    print("-" * 48)

    similarities: list[float] = []
    for position in positions:
        keyframe_id = index._sidecar.to_keyframe_id(int(position))  # noqa: SLF001
        reference = np.asarray(index._index.reconstruct(int(position)))  # noqa: SLF001
        image_path = assets.get_record(keyframe_id, asset_type="image").asset_path

        matches, similarity = verify_against_reference(
            image_path, reference, checkpoint=args.checkpoint
        )
        similarities.append(similarity)
        print(f"{keyframe_id:<20} {similarity:>9.4f}   {'OK' if matches else 'LECH'}")

    mean = float(np.mean(similarities))
    worst = float(np.min(similarities))
    print("-" * 48)
    print(f"{'trung binh':<20} {mean:>9.4f}")
    print(f"{'thap nhat':<20} {worst:>9.4f}\n")

    if worst >= CHECKPOINT_MATCH_THRESHOLD:
        print("=> KHOP. Dung checkpoint BTC da dung; nhanh vector dung duoc.")
        return 0
    print("=> LECH. KHONG cam encoder nay vao search_fn:")
    print("   query text se duoc cham diem trong mot khong gian khac voi anh.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
