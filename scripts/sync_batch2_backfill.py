#!/usr/bin/env python3
"""Copy the three repaired B2 per-video parts from Modal to the /data snapshot."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfe.offline.bundle import BundleLayout  # noqa: E402
from mfe.offline.transport import build_transport, load_dotenv  # noqa: E402

VIDEOS = ("N025-V002", "N041-V001", "S01-V007")
ARTIFACTS = ("dense_vectors", "subtitles_dense", "gemini_captions", "gemini_ocr", "manifest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default="keyframes-batch-2-v2")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/data/multimodal-fusion-engine/modal/keyframes-batch-2-v2"),
    )
    parser.add_argument("--dotenv", type=Path, default=Path("/home/ubuntu/.env"))
    args = parser.parse_args()
    for key, value in load_dotenv(args.dotenv).items():
        os.environ.setdefault(key, value)
    transport = build_transport("modal", volume=args.volume, version=2)
    layout = BundleLayout("aic_btc_b2_v1")
    root = args.destination / "bundle" / "aic_btc_b2_v1"
    for video_id in VIDEOS:
        for artifact in ARTIFACTS:
            if artifact == "manifest":
                remote = layout.part_manifest(video_id)
                local = root / "parts" / "manifest" / f"{video_id}.json"
            else:
                remote = layout.part(artifact, video_id)
                suffix = "npz" if artifact in {"dense_vectors", "subtitles_dense"} else "json"
                local = root / "parts" / artifact / f"{video_id}.{suffix}"
            if not transport.exists(remote):
                raise SystemExit(f"Modal thiếu {remote}")
            transport.get(remote, local)
            print(f"synced {local}", flush=True)


if __name__ == "__main__":
    main()
