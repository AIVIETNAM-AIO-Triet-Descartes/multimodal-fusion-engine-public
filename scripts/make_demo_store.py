"""Build a tiny SYNTHETIC flat store so the text-retrieval path can be exercised.

The real store used in the competition (~4.8 GB of vectors and text extracted
from the organizers' videos) is not redistributed. This script writes a few
hand-written, clearly fake records in the same layout that
`mfe.online.flat_arms.FlatBundle` reads, so the lexical arms (BM25 over
captions/OCR, subtitle matching) and weighted RRF can be run end to end on a
laptop without models, GPUs or competition data.

    python scripts/make_demo_store.py --out stores/demo
    python scripts/query_flat.py "người đàn ông đạp xe" --root stores/demo \
        --config config/btc_b12.yaml --en "a man riding a bicycle" \
        --arms caption_gemini,ocr_qwen25,subtitle,subtitle_exact

Nothing produced here is a competition result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# keyframe_id = {video_id}#{frame_idx:06d}; DEMO_* videos do not exist anywhere.
CAPTIONS = {
    "DEMO_V001#000120": "A man in a red jacket rides a bicycle along a riverside road at sunset.",
    "DEMO_V001#000480": "Two cyclists stop near a bridge; one drinks water from a green bottle.",
    "DEMO_V002#000075": "A news anchor sits at a desk before a blue screen showing a city map.",
    "DEMO_V002#000300": "Heavy rain floods a street; motorbikes push through knee-deep water.",
    "DEMO_V003#000050": "A chef slices green onions on a wooden board in a small kitchen.",
    "DEMO_V003#000210": "Noodles are stirred in a pot while steam rises; a sign reads 'PHO 24'.",
    "DEMO_V004#000090": "A traffic light turns red at a crossroads with five cars waiting.",
    "DEMO_V004#000360": "A white bus passes a pedestrian crossing under a 'SCHOOL ZONE' sign.",
}
OCR = {
    "DEMO_V002#000075": "BẢN TIN THỜI SỰ",
    "DEMO_V003#000210": "PHO 24",
    "DEMO_V004#000360": "SCHOOL ZONE",
}
SUBTITLES = {
    "DEMO_V001#000120": "người đàn ông đạp xe dọc bờ sông lúc hoàng hôn",
    "DEMO_V002#000300": "mưa lớn khiến nhiều tuyến đường ngập sâu",
    "DEMO_V003#000050": "đầu bếp thái hành lá trên thớt gỗ",
    "DEMO_V004#000090": "đèn giao thông chuyển sang màu đỏ",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("stores/demo"))
    args = parser.parse_args()
    vectors = args.out / "data" / "vectors"
    vectors.mkdir(parents=True, exist_ok=True)
    files = {
        "gemini_captions.json": CAPTIONS,
        "qwen_ocr.json": OCR,
        "subtitles.json": SUBTITLES,
    }
    for name, payload in files.items():
        (vectors / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"wrote {len(files)} synthetic sources to {vectors}")


if __name__ == "__main__":
    main()
