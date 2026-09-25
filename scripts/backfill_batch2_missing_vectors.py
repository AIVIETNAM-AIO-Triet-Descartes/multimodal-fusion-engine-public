#!/usr/bin/env python3
"""Rebuild the two missing B2 vector layers for the three late videos.

The three videos were imported after the original embedding pass.  Their
Gemini raw captions and subtitle parts are present, so this script deliberately
runs only the deterministic split/embed and subtitle-embed stages:

    N025-V002  N041-V001  S01-V007

It uses the same stage implementation as the notebooks, keeps the BGE model in
memory for the whole run, writes a done marker only after a verified upload,
and can work against either a local /data snapshot or the Modal volume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfe.offline.bundle import BundleLayout, BundleWriter, StageStatus  # noqa: E402
from mfe.offline.stages import STAGES, StageContext, unload_models  # noqa: E402
from mfe.offline.transport import build_transport, load_dotenv  # noqa: E402

DEFAULT_VIDEOS = ("N025-V002", "N041-V001", "S01-V007")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("modal", "local"), default="modal")
    parser.add_argument("--volume", default="keyframes-batch-2-v2")
    parser.add_argument("--local-root", type=Path,
                        default=Path("/data/multimodal-fusion-engine/modal/keyframes-batch-2-v2"))
    parser.add_argument("--bundle", default="aic_btc_b2_v1")
    parser.add_argument("--member", default="missing1")
    parser.add_argument("--workdir", type=Path, default=Path(".cache/backfill-b2"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--videos", nargs="+", default=list(DEFAULT_VIDEOS))
    parser.add_argument("--redo", action="store_true",
                        help="Chạy lại dù marker đã có; mặc định chỉ bổ sung phần còn thiếu.")
    parser.add_argument("--dotenv", type=Path, default=Path("/home/ubuntu/.env"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_keyframe_count(transport, layout: BundleLayout, workdir: Path, video_id: str) -> int:
    """Read the count imported by n0; never replace it with vector row count."""
    local = workdir / f"manifest-{video_id}.json"
    try:
        transport.get(layout.part_manifest(video_id), local)
        payload = json.loads(local.read_text(encoding="utf-8"))
        return int(payload["counts"]["keyframes"])
    except Exception as exc:  # noqa: BLE001 - old manifests may be absent
        raise RuntimeError(f"{video_id}: không đọc được part manifest n0: {exc}") from exc
    finally:
        local.unlink(missing_ok=True)


def _preflight(transport, layout: BundleLayout, videos: list[str]) -> None:
    required = []
    for video_id in videos:
        required.extend([
            layout.part("gemini_raw", video_id),
            layout.part("subtitles", video_id),
        ])
    missing = [path for path in required if not transport.exists(path)]
    if missing:
        raise SystemExit(
            "Preflight thất bại; không chạy dở dang:\n" + "\n".join(f"  {p}" for p in missing)
        )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size phải lớn hơn 0")
    if sorted(set(args.videos)) != sorted(args.videos):
        raise SystemExit("--videos chứa id trùng")
    if set(args.videos) != set(DEFAULT_VIDEOS):
        unknown = sorted(set(args.videos) - set(DEFAULT_VIDEOS))
        if unknown:
            raise SystemExit(f"Chỉ cho phép ba video late-addition: {unknown}")

    for key, value in load_dotenv(args.dotenv).items():
        os.environ.setdefault(key, value)

    transport = build_transport(
        args.transport,
        volume=args.volume,
        local_root=str(args.local_root),
        version=2,
    )
    layout = BundleLayout(args.bundle)
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    _preflight(transport, layout, args.videos)

    print(f"bundle={args.bundle} transport={args.transport} member={args.member}", flush=True)
    print(f"videos={','.join(args.videos)}", flush=True)
    print("required inputs: gemini_raw + subtitles: OK", flush=True)
    if args.dry_run:
        print("dry-run: không ghi artifact", flush=True)
        return

    status = StageStatus.load(transport, layout, args.member)
    writer = BundleWriter(transport, layout, args.member, workdir)
    params = {
        "n6_gemini_split_embed": {"batch_size": args.batch_size},
        "n7_bge_m3_subtitles": {"batch_size": args.batch_size},
    }
    stages = ("n6_gemini_split_embed", "n7_bge_m3_subtitles")
    started = time.perf_counter()
    failures: list[tuple[str, str, str]] = []

    try:
        for video_index, video_id in enumerate(args.videos, 1):
            keyframes = _read_keyframe_count(transport, layout, workdir, video_id)
            print(
                f"[{video_index}/{len(args.videos)}] {video_id}: {keyframes} keyframes",
                flush=True,
            )
            for stage_name in stages:
                if not args.redo and status.is_done(stage_name, video_id):
                    print(f"  {stage_name}: marker đã có, bỏ qua", flush=True)
                    continue
                print(f"  {stage_name}: bắt đầu", flush=True)
                try:
                    ctx = StageContext(
                        video_id, Path("/dev/null"), [], writer, layout, workdir,
                        params=dict(params[stage_name]),
                    )
                    marker = STAGES[stage_name](ctx)
                    writer.write_part_manifest(
                        video_id,
                        [marker],
                        keyframes=keyframes,
                        merge=True,
                    )
                    status.done.add((stage_name, video_id))
                    print(
                        f"  {stage_name}: OK rows={marker.get('rows', 0)} "
                        f"elapsed={marker.get('elapsed_s')}s",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - keep other video resumable
                    failures.append((video_id, stage_name, repr(exc)))
                    print(f"  {stage_name}: ERROR {exc}", flush=True)
                    traceback.print_exc(limit=4)
                finally:
                    for pattern in (
                        f"gemini_raw_{video_id}.json",
                        f"gemini_captions_{video_id}.json",
                        f"gemini_ocr_{video_id}.json",
                        f"subtitles_{video_id}.json",
                        f"dense_vectors_{video_id}.npz",
                        f"subtitles_dense_{video_id}.npz",
                    ):
                        (workdir / pattern).unlink(missing_ok=True)
    finally:
        unload_models()

    print(f"finished in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    if failures:
        print("failures:", flush=True)
        for failure in failures:
            print("  " + " | ".join(failure), flush=True)
        raise SystemExit(1)
    print("BACKFILL OK: cả ba video đã có hai stage", flush=True)


if __name__ == "__main__":
    main()
