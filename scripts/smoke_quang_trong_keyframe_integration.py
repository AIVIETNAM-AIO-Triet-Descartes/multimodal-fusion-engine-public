"""One-video local plumbing smoke for the integrated keyframe pipeline.

This command deliberately uses synthetic boundary scores.  It validates real
video decode, coordinate flow, coverage, selectors, budget guards, and blocked
oracle schema without pretending to measure TransNetV2 or retrieval quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

from mfe.offline.augmentation.acceptance import (
    ARMS,
    build_blocked_oracle_evidence,
    validate_primary_equal_budget,
    validate_video_assignment,
    validate_worker_assignment,
    write_json_atomic,
)
from mfe.offline.augmentation.coverage import (
    btc_in_shot,
    classify_coverage,
    read_btc_mapping,
)
from mfe.offline.augmentation.io import canonical_hash, sha256_file, write_jsonl_atomic
from mfe.offline.augmentation.selectors import (
    apply_global_cap,
    select_b1,
    select_b2,
    select_b3,
    select_b4,
)
from mfe.offline.augmentation.transnet import (
    decode_for_transnet,
    proposals_from_scores,
    verify_shot_partition,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument(
        "--assignment-config", type=Path, default=Path("config/zips_batch1.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-cut-every", type=int, default=150)
    parser.add_argument("--target-gap", type=float, default=6.0)
    parser.add_argument("--k-max", type=int, default=3)
    parser.add_argument("--cap-ratio", type=float, default=0.03)
    parser.add_argument(
        "--allow-synthetic-detector",
        action="store_true",
        help="required acknowledgement that this is plumbing-only",
    )
    return parser


def _events_for_video(path: Path, video_id: str) -> list[dict]:
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        query = json.loads(line)
        if str(query.get("gt_video_id")) != video_id:
            continue
        query_type = str(query.get("type") or query.get("query_type") or "KIS").upper()
        if query_type in {"Q&A", "QNA"}:
            query_type = "QA"
        intervals = (
            query.get("gt_events_strict")
            or query.get("gt_events")
            or [query.get("gt_frame_range_strict") or query.get("gt_frame_range")]
        )
        for interval in intervals:
            if interval:
                events.append(
                    {
                        "query_id": str(query["query_id"]),
                        "query_type": query_type,
                        "video_id": video_id,
                        "start_frame": int(interval[0]),
                        "end_frame": int(interval[1]),
                    }
                )
    return events


def _literal_fps(path: Path) -> float:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        values = {float(row["fps"]) for row in csv.DictReader(stream)}
    if len(values) != 1:
        raise ValueError(f"mapping must contain one literal FPS, got {sorted(values)}")
    return next(iter(values))


def main() -> None:
    args = _parser().parse_args()
    if not args.allow_synthetic_detector:
        raise RuntimeError("pass --allow-synthetic-detector for plumbing-only smoke")
    validate_worker_assignment(args.member, args.worker)
    assigned_zips = validate_video_assignment(
        args.assignment_config, args.member, args.video_id
    )
    if args.synthetic_cut_every < 2:
        raise ValueError("--synthetic-cut-every must be >= 2")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    timeline = decode_for_transnet(args.video)
    scores = [0.0] * timeline.total_frames
    for frame_idx in range(
        args.synthetic_cut_every - 1,
        timeline.total_frames - 1,
        args.synthetic_cut_every,
    ):
        scores[frame_idx] = 1.0
    detector_fingerprint = canonical_hash(
        {
            "kind": "synthetic_boundaries_plumbing_only",
            "cut_every": args.synthetic_cut_every,
            "video_sha256": sha256_file(args.video),
        }
    )
    shots = proposals_from_scores(
        args.video_id,
        scores,
        timeline.pts_times,
        timeline.duration_seconds,
        0.5,
        detector_fingerprint,
    )
    partition_problems = verify_shot_partition(shots, timeline.total_frames)
    if partition_problems:
        raise RuntimeError(f"shot partition failed: {partition_problems}")

    btc = read_btc_mapping(args.video_id, args.mapping)
    literal_fps = _literal_fps(args.mapping)
    audit = [
        classify_coverage(
            shot,
            btc,
            target_gap_seconds=args.target_gap,
            k_max=args.k_max,
            micro_shot_seconds=0.5,
            uncertain_below=0.6,
        )
        for shot in shots
    ]
    pts_by_video = {
        args.video_id: {
            frame_idx: pts_time for frame_idx, pts_time in enumerate(timeline.pts_times)
        }
    }
    btc_by_shot = {shot.shot_id: btc_in_shot(shot, btc) for shot in shots}
    common = {
        "decoder_fingerprints": {args.video_id: timeline.decoder_fingerprint},
        "literal_fps_by_video": {args.video_id: literal_fps},
        "config_hash": canonical_hash(
            {
                "target_gap": args.target_gap,
                "k_max": args.k_max,
                "cap_ratio": args.cap_ratio,
            }
        ),
    }
    proposed = {
        "B1": select_b1(audit, pts_by_video, btc_by_shot, **common),
        "B2": select_b2(audit, pts_by_video, btc_by_shot, **common),
        "B3": select_b3(audit, pts_by_video, btc_by_shot, **common),
    }
    proposed["B4"] = select_b4(
        proposed["B3"],
        audit,
        pts_by_video,
        btc_by_shot,
        **common,
        seed=2026,
    )
    selected = {
        arm: apply_global_cap(
            proposed[arm],
            audit,
            base_keyframes=len(btc),
            cap_ratio=args.cap_ratio,
        )
        for arm in ARMS
    }
    coordinate_errors = [
        {
            "arm": arm,
            "frame_idx": row.frame_idx,
            "pts_time": row.pts_time,
            "floor_pts_fps": math.floor(row.pts_time * literal_fps),
        }
        for arm in ARMS
        for row in selected[arm]
        if math.floor(row.pts_time * literal_fps) != row.frame_idx
    ]
    if coordinate_errors:
        raise RuntimeError(f"candidate coordinate floor-roundtrip failed: {coordinate_errors[:3]}")
    budget_error = None
    try:
        budget = validate_primary_equal_budget(
            base_keyframes=len(btc),
            cap_ratio=args.cap_ratio,
            arm_rows={
                arm: [row.model_dump(mode="json") for row in selected[arm]]
                for arm in ARMS
            },
        )
    except RuntimeError as error:
        budget_error = str(error)
        budget = json.loads(str(error))

    arm_paths = {}
    for arm in ARMS:
        arm_paths[arm] = write_jsonl_atomic(output / f"arm-{arm}.jsonl", selected[arm])
    events = _events_for_video(args.dev, args.video_id)
    oracle_rows = build_blocked_oracle_evidence(
        events,
        provenance_hashes={
            "video_sha256": sha256_file(args.video),
            "mapping_sha256": sha256_file(args.mapping),
            "dev_sha256": sha256_file(args.dev),
            "decoder_fingerprint": timeline.decoder_fingerprint,
        },
    )
    oracle_path = write_json_atomic(output / "o1s-blocked.json", oracle_rows)
    repo_root = Path(__file__).resolve().parents[1]
    implementation_paths = {
        "smoke_script": Path(__file__).resolve(),
        "acceptance": repo_root / "src/mfe/offline/augmentation/acceptance.py",
        "models": repo_root / "src/mfe/offline/augmentation/models.py",
        "selectors": repo_root / "src/mfe/offline/augmentation/selectors.py",
        "materialize": repo_root / "src/mfe/offline/augmentation/materialize.py",
        "transnet": repo_root / "src/mfe/offline/augmentation/transnet.py",
    }
    report = {
        "schema_version": 1,
        "status": "PASS_PLUMBING_ONLY",
        "scientific_authority": "NONE_SYNTHETIC_DETECTOR_NO_FEATURES_NO_RETRIEVAL",
        "member": args.member,
        "worker": args.worker,
        "video_id": args.video_id,
        "assignment": {
            "config": str(args.assignment_config.resolve()),
            "config_sha256": sha256_file(args.assignment_config),
            "assigned_zips": assigned_zips,
            "video_assignment_status": "PASS",
        },
        "command": [sys.executable, *sys.argv],
        "source_script_sha256": sha256_file(Path(__file__)),
        "implementation_sha256": {
            name: sha256_file(path) for name, path in implementation_paths.items()
        },
        "inputs": {
            "video": str(args.video.resolve()),
            "video_sha256": sha256_file(args.video),
            "mapping": str(args.mapping.resolve()),
            "mapping_sha256": sha256_file(args.mapping),
            "dev": str(args.dev.resolve()),
            "dev_sha256": sha256_file(args.dev),
        },
        "decoded_frames": timeline.total_frames,
        "duration_seconds": timeline.duration_seconds,
        "literal_fps": literal_fps,
        "candidate_coordinate_floor_roundtrip": "PASS",
        "decoder_fingerprint": timeline.decoder_fingerprint,
        "shots": len(shots),
        "shot_class_counts": dict(
            sorted(Counter(str(row.coverage_class) for row in audit).items())
        ),
        "proposed_counts": {arm: len(rows) for arm, rows in proposed.items()},
        "selected_counts": {arm: len(rows) for arm, rows in selected.items()},
        "equal_budget": budget,
        "equal_budget_error": budget_error,
        "oracle": {
            "O1-T": "DIAGNOSTIC_ONLY",
            "O1-S": "BLOCKED_MISSING_STAGE_ARTIFACT",
            "O1-R": "BLOCKED_MISSING_STAGE_ARTIFACT",
            "events": len(events),
            "artifact": str(oracle_path),
        },
        "feature_parity": "BLOCKED_MISSING_NUMERIC_CANARY",
        "heldout_accessed": False,
        "outputs": {
            arm: {"path": str(path), "sha256": sha256_file(path)}
            for arm, path in arm_paths.items()
        }
        | {"o1s": {"path": str(oracle_path), "sha256": sha256_file(oracle_path)}},
    }
    report["run_hash"] = canonical_hash(report)
    report_path = write_json_atomic(output / "smoke-report.json", report)
    print(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
