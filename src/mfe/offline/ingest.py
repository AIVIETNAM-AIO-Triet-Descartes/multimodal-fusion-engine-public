"""Ingestion entrypoint — docs/02 §2.

    python -m mfe.offline.ingest --input data/raw/ --config config/local.yaml
    mfe-ingest --input data/raw/ --config config/local.yaml

Pipeline (docs/02 §2):

    data/raw/*.mp4
      -> [1] probe metadata (fps, duration, resolution)
      -> [2] extract keyframes  -> Asset Store + Relational Map rows
      -> [3] extract features per keyframe:
             CLIP embedding   (required) -> Vector Index
             object detection (required) -> Inverted Index
             OCR              (optional) -> Inverted Index
             metadata text    (if any)   -> Inverted Index
      -> [4] write the 4 stores per the record format in docs/00 §4

Success here is not accuracy — it is reproducibility. docs/02 §1: "chĩa vào thư
mục bất kỳ, chạy một lệnh, ra đủ 4 kho, không crash".
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mfe-ingest",
        description="Turn a directory of videos into the 4 stores (docs/02).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory of source videos. Any directory — no assumptions about naming.",
    )
    parser.add_argument(
        "--config",
        default="config/local.yaml",
        help="Config file. Paths, models and module on/off switches all come from here.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N videos. For the small-sample vertical slice (docs/00 §8 risk 3).",
    )
    return parser


def run(input_dir: str, config_path: str, limit: int | None = None) -> None:
    """Run the full offline pipeline.

    Owner: TV1. TODO(TV1): implement steps 1-4 above.

    Two things that are easy to get wrong and expensive to fix:

    - `keyframe_id` must come from `mfe.contracts.ids.build_keyframe_id`. docs/02
      §4 calls this "điểm dễ sai nhất — sai một chỗ là cả pipeline lệch". Do not
      format the id by hand.
    - fps is read per video from `probe.py`, never assumed (docs/00 §7).
    """
    raise NotImplementedError("TODO(TV1): orchestrate probe -> keyframes -> features -> writers")


def main() -> None:
    args = build_parser().parse_args()
    run(input_dir=args.input, config_path=args.config, limit=args.limit)


if __name__ == "__main__":
    main()
