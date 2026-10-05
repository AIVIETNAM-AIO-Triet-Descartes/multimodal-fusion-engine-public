import argparse
import csv
import logging
from pathlib import Path

from mfe.offline.dake import (
    decode_frames,
    jpeg_sizes,
    pts_for_frame,
    select_keyframes,
    video_fps,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Supplement BTC keyframes with DAKE frames.")
    parser.add_argument("--video", required=True, type=Path, help="Path to video file")
    parser.add_argument("--btc-map", required=True, type=Path, help="Path to BTC map-keyframes.csv")
    parser.add_argument("--out-map", required=True, type=Path,
                        help="Path to save output map-keyframes.csv")
    parser.add_argument("--rho", type=float, default=0.02, help="DAKE rho (default: 0.02)")
    parser.add_argument("--max-gap", type=float, default=8.0,
                        help="DAKE max gap in seconds (default: 8.0)")
    parser.add_argument("--min-gap", type=float, default=1.0,
                        help="Minimum gap in seconds (default: 1.0)")
    return parser.parse_args()


def read_btc_frames(map_path: Path) -> list[int]:
    """Reads frame_idx from the BTC map-keyframes.csv"""
    frames = []
    with map_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame_idx"]))
    return sorted(frames)


def main():
    args = parse_args()
    
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.btc_map.is_file():
        raise FileNotFoundError(f"BTC map not found: {args.btc_map}")

    # 1. Read existing BTC frames
    btc_frames = read_btc_frames(args.btc_map)
    if not btc_frames:
        logger.warning("No BTC frames found.")
    
    # 2. Extract DAKE frames
    used_fps, _ = video_fps(str(args.video))
    logger.info(f"Decoding video {args.video.name} at {used_fps} fps...")
    frames_iter = decode_frames(str(args.video))
    sizes = list(jpeg_sizes(frames_iter))
    
    dake_frames = select_keyframes(
        sizes, 
        fps=used_fps, 
        rho=args.rho, 
        max_gap_seconds=args.max_gap, 
        min_gap_seconds=args.min_gap
    )
    
    # 3. Filter DAKE frames that are too close to BTC frames
    min_gap_frames = max(1, int(args.min_gap * used_fps))
    supplemented = []
    
    for df in dake_frames:
        if btc_frames:
            closest_dist = min(abs(df - bf) for bf in btc_frames)
            if closest_dist >= min_gap_frames:
                supplemented.append(df)
        else:
            supplemented.append(df)
            
    # 4. Merge and generate new CSV
    hybrid_frames = sorted(list(set(btc_frames) | set(supplemented)))
    
    args.out_map.parent.mkdir(parents=True, exist_ok=True)
    with args.out_map.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "pts_time", "frame_idx"])
        for n, frame_idx in enumerate(hybrid_frames, start=1):
            pts = pts_for_frame(frame_idx, used_fps)
            writer.writerow([f"{n:03d}", f"{pts:.4f}", frame_idx])
            
    logger.info(f"Original BTC frames: {len(btc_frames)}")
    logger.info(f"DAKE frames selected: {len(dake_frames)}")
    logger.info(f"DAKE frames added (after min_gap filter): {len(supplemented)}")
    logger.info(f"Total Hybrid frames: {len(hybrid_frames)}")
    logger.info(f"Saved to {args.out_map}")

if __name__ == "__main__":
    main()
