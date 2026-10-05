"""Command line pipeline for Gate 0-2 and immutable delta snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from mfe.core.config import load_config
from mfe.offline.keyframe_augmentation.bundle import (
    activate_snapshot,
    materialize_bundle,
    rollback_snapshot,
)
from mfe.offline.keyframe_augmentation.common import atomic_write_json, atomic_write_jsonl
from mfe.offline.keyframe_augmentation.gate0 import (
    audit_mapping_math,
    build_dataset_lock,
    read_and_validate_mapping,
)
from mfe.offline.keyframe_augmentation.gate1 import (
    build_oracles,
    load_query_events,
    load_shot_proposals,
    run_gate1,
)
from mfe.offline.keyframe_augmentation.gate2 import (
    load_arm_run,
    load_oracle_run,
    run_gate2,
    run_oracle_diagnostic,
)
from mfe.offline.keyframe_augmentation.models import (
    AugmentationConfig,
    BtcFrame,
    CandidateFrame,
)
from mfe.offline.keyframe_augmentation.transnet import detect_video


def _augmentation_config(path: str | Path) -> AugmentationConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("augmentation config must be a YAML mapping")
    return AugmentationConfig.model_validate(raw)


def _load_lock(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "dataset_hash" not in value:
        raise ValueError("invalid Gate 0 lock")
    return value


def _mappings_from_lock(lock: dict[str, Any]) -> dict[str, list[BtcFrame]]:
    mappings: dict[str, list[BtcFrame]] = {}
    for video in lock["videos"]:
        frames, _ = read_and_validate_mapping(
            video["mapping_path"],
            video_id=video["video_id"],
            total_frames=int(video["probe"]["total_frames"]),
        )
        mappings[video["video_id"]] = frames
    return mappings


def _gate0(args: argparse.Namespace) -> None:
    config = _augmentation_config(args.config)
    core = load_config(args.core_config)
    lock, _ = build_dataset_lock(
        config=config,
        config_path=args.config,
        map_dir=core.paths.map_keyframes,
        video_dir=core.paths.videos,
        dev_path=args.dev,
        heldout_path=args.heldout,
        video_ids=args.video_id,
        hash_videos=args.hash_videos,
    )
    atomic_write_json(args.output, lock)
    print(
        f"Gate 0 PASS: {len(lock['videos'])} video, dataset_hash={lock['dataset_hash']}, "
        f"run_id={lock['run_id']}"
    )


def _math_audit(args: argparse.Namespace) -> None:
    paths: list[Path] = []
    for raw_root in args.map_dir:
        root = Path(raw_root)
        paths.extend(root.glob("*.csv"))
    if not paths:
        raise ValueError("no mapping CSV found")
    report = audit_mapping_math(paths)
    atomic_write_json(args.output, report)
    print(
        f"Mapping math audit: files={report['files']}, rows={report['rows']}, "
        f"floor_wrong={report['floor_mismatches']}, round_wrong={report['round_mismatches']}"
    )


def _detect(args: argparse.Namespace) -> None:
    lock = _load_lock(args.gate0)
    locked = {row["video_id"]: row for row in lock["videos"]}
    proposals = []
    reports = []
    for raw_video in args.video:
        video_id = Path(raw_video).stem
        if video_id not in locked:
            raise ValueError(f"{video_id}: video is absent from the Gate 0 lock")
        shots, report = detect_video(
            raw_video,
            repo=args.transnet_repo,
            weights=args.weights,
            literal_fps=float(locked[video_id]["mapping"]["literal_fps"]),
            threshold=args.threshold,
            device=args.device,
        )
        proposals.extend(shots)
        reports.append(report)
    atomic_write_jsonl(
        args.output,
        (proposal.model_dump(mode="json") for proposal in proposals),
    )
    atomic_write_json(Path(args.output).with_suffix(".report.json"), {"videos": reports})
    print(f"TransNetV2: {len(proposals)} inclusive shot proposal(s) -> {args.output}")


def _gate1(args: argparse.Namespace) -> None:
    config = _augmentation_config(args.config)
    lock = _load_lock(args.gate0)
    mappings = _mappings_from_lock(lock)
    proposals = load_shot_proposals(args.shots)
    locked_ids = set(mappings)
    proposals = [shot for shot in proposals if shot.video_id in locked_ids]
    for shot in proposals:
        literal_fps = mappings[shot.video_id][0].fps
        if shot.fps != literal_fps:
            raise ValueError(
                f"{shot.shot_id}: proposal FPS {shot.fps} != locked literal FPS {literal_fps}"
            )
    events = [
        event
        for event in load_query_events(lock["split"]["dev"]["path"])
        if event["video_id"] in locked_ids
    ]
    report, arms, oracle_manifests = run_gate1(
        config=config,
        proposals=proposals,
        mappings=mappings,
        dev_events=events,
        dataset_hash=lock["dataset_hash"],
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "gate1-report.json", report)
    atomic_write_jsonl(output / "arm-B0.jsonl", [])
    for arm, candidates in arms.items():
        atomic_write_jsonl(
            output / f"arm-{arm}.jsonl",
            (candidate.model_dump(mode="json") for candidate in candidates),
        )
    for oracle_name, rows in oracle_manifests.items():
        atomic_write_jsonl(output / f"oracle-{oracle_name}.jsonl", rows)
    print(
        f"Gate 1 {report['decision']}: "
        f"hard_budget={report['budget']['locked_hard_delta_budget']}, "
        f"primary_equal_budget={report['budget']['primary_equal_budget_pass']}, "
        f"run_hash={report['run_hash']}"
    )


def _diagnostic(args: argparse.Namespace) -> None:
    lock = _load_lock(args.gate0)
    mappings = _mappings_from_lock(lock)
    events = [
        event
        for event in load_query_events(lock["split"]["dev"]["path"])
        if event["video_id"] in mappings
    ]
    oracle_manifests, oracle = build_oracles(events, mappings)
    query_ids = {event["query_id"] for event in events}
    report = {
        "schema_version": 1,
        "dataset_hash": lock["dataset_hash"],
        "split": "dev",
        "pilot_videos": sorted({event["video_id"] for event in events}),
        "queries": len(query_ids),
        "oracle": oracle,
        "decision": "BLOCKED_PENDING_REPRESENTATIVE_RETRIEVAL_ORACLE",
        "evidence": {
            "strict_btc_coverage": "MEASURED",
            "o1_t_candidate_manifest": "DERIVED",
            "o1_t_end_to_end": "UNVERIFIED",
            "o1_s_best_source_frame": "UNVERIFIED",
            "o1_r_force_rank": "UNVERIFIED",
        },
    }
    output = Path(args.output)
    atomic_write_json(output, report)
    for oracle_name, rows in oracle_manifests.items():
        atomic_write_jsonl(output.with_name(f"diagnostic-oracle-{oracle_name}.jsonl"), rows)
    print(
        f"Gate 1 diagnostic {report['decision']}: "
        f"events={oracle['events']}, temporal_headroom="
        f"{oracle['temporal_coverage_headroom_events']}"
    )


def _bundle(args: argparse.Namespace) -> None:
    config = _augmentation_config(args.config)
    lock = _load_lock(args.gate0)
    mappings = _mappings_from_lock(lock)
    candidates = [CandidateFrame.model_validate(row) for row in _read_rows(args.candidates)]
    manifest = materialize_bundle(
        candidates=candidates,
        mappings=mappings,
        config=config,
        core_config_path=args.core_config,
        output_dir=args.output,
        parent_bundle=args.parent_bundle,
        run_id=lock["run_id"],
        dataset_hash=lock["dataset_hash"],
        delta_vectors_path=args.delta_vectors,
        delta_inverted_path=args.delta_inverted,
    )
    print(
        f"Bundle {manifest['bundle_id']}: delta={manifest['origin_counts']['supplemental']}, "
        f"index_ready={manifest['readiness']['index_ready']}"
    )


def _read_rows(path: str | Path):
    from mfe.offline.keyframe_augmentation.common import read_jsonl  # noqa: PLC0415

    return read_jsonl(path)


def _gate2(args: argparse.Namespace) -> None:
    config = _augmentation_config(args.config)
    lock = _load_lock(args.gate0)
    runs = [load_arm_run(path) for path in args.arm_run]
    query_rows = load_query_events(lock["split"]["dev"]["path"])
    query_to_video = {row["query_id"]: row["video_id"] for row in query_rows}
    oracle_path = Path(args.gate1).resolve().parent / "oracle-O1-T.jsonl"
    coverage_gap_query_ids = {row["query_id"] for row in _read_rows(oracle_path)}
    report = run_gate2(
        runs=runs,
        config=config,
        dataset_hash=lock["dataset_hash"],
        query_to_video=query_to_video,
        coverage_gap_query_ids=coverage_gap_query_ids,
    )
    atomic_write_json(args.output, report)
    print(f"Gate 2 {report['decision']}: run_hash={report['run_hash']}")


def _oracle(args: argparse.Namespace) -> None:
    lock = _load_lock(args.gate0)
    baseline = load_arm_run(args.baseline_run)
    source_oracle = load_oracle_run(args.source_run)
    force_rank_oracle = load_oracle_run(args.force_rank_run)
    report = run_oracle_diagnostic(
        baseline=baseline,
        source_oracle=source_oracle,
        force_rank_oracle=force_rank_oracle,
        dataset_hash=lock["dataset_hash"],
    )
    atomic_write_json(args.output, report)
    print(
        f"Oracle measured: O1-S headroom={report['o1_s_headroom_queries']}, "
        f"O1-R headroom={report['o1_r_headroom_queries']}, run_hash={report['run_hash']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mfe-keyframe-augmentation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gate0 = subparsers.add_parser("gate0", help="lock data/mapping/splits and verify contracts")
    gate0.add_argument("--config", required=True)
    gate0.add_argument("--core-config", required=True)
    gate0.add_argument("--dev", required=True)
    gate0.add_argument("--heldout", required=True)
    gate0.add_argument("--video-id", action="append")
    gate0.add_argument("--hash-videos", action=argparse.BooleanOptionalAction, default=True)
    gate0.add_argument("--output", required=True)
    gate0.set_defaults(function=_gate0)

    math_audit = subparsers.add_parser(
        "math-audit", help="measure floor/round against canonical mapping columns"
    )
    math_audit.add_argument("--map-dir", action="append", required=True)
    math_audit.add_argument("--output", required=True)
    math_audit.set_defaults(function=_math_audit)

    detect = subparsers.add_parser("detect", help="run pinned official TransNetV2")
    detect.add_argument("--video", action="append", required=True)
    detect.add_argument("--gate0", required=True)
    detect.add_argument("--transnet-repo", required=True)
    detect.add_argument("--weights", required=True)
    detect.add_argument("--threshold", type=float, default=0.5)
    detect.add_argument("--device", default="cpu")
    detect.add_argument("--output", required=True)
    detect.set_defaults(function=_detect)

    diagnostic = subparsers.add_parser(
        "diagnostic", help="measure BTC coverage and emit O1-T/O1-S DEV manifests"
    )
    diagnostic.add_argument("--gate0", required=True)
    diagnostic.add_argument("--output", required=True)
    diagnostic.set_defaults(function=_diagnostic)

    gate1 = subparsers.add_parser(
        "gate1", help="audit DEV/grid candidates and emit equal-budget B0-B4"
    )
    gate1.add_argument("--config", required=True)
    gate1.add_argument("--gate0", required=True)
    gate1.add_argument("--shots", required=True)
    gate1.add_argument("--output", required=True)
    gate1.set_defaults(function=_gate1)

    bundle = subparsers.add_parser("bundle", help="materialize an immutable delta snapshot")
    bundle.add_argument("--config", required=True)
    bundle.add_argument("--core-config", required=True)
    bundle.add_argument("--gate0", required=True)
    bundle.add_argument("--candidates", required=True)
    bundle.add_argument("--parent-bundle", required=True)
    bundle.add_argument("--delta-vectors")
    bundle.add_argument("--delta-inverted")
    bundle.add_argument("--output", required=True)
    bundle.set_defaults(function=_bundle)

    gate2 = subparsers.add_parser("gate2", help="compare official B0-B4 runs")
    gate2.add_argument("--config", required=True)
    gate2.add_argument("--gate0", required=True)
    gate2.add_argument("--gate1", required=True)
    gate2.add_argument("--arm-run", action="append", required=True)
    gate2.add_argument("--output", required=True)
    gate2.set_defaults(function=_gate2)

    oracle = subparsers.add_parser(
        "oracle", help="measure B0 vs best-source O1-S and force-rank O1-R"
    )
    oracle.add_argument("--gate0", required=True)
    oracle.add_argument("--baseline-run", required=True)
    oracle.add_argument("--source-run", required=True)
    oracle.add_argument("--force-rank-run", required=True)
    oracle.add_argument("--output", required=True)
    oracle.set_defaults(function=_oracle)

    activate = subparsers.add_parser("activate", help="atomically point at a snapshot config")
    activate.add_argument("--pointer", required=True)
    activate.add_argument("--config", required=True)
    activate.set_defaults(function=lambda args: print(activate_snapshot(args.pointer, args.config)))

    rollback = subparsers.add_parser("rollback", help="restore the previous snapshot config")
    rollback.add_argument("--pointer", required=True)
    rollback.set_defaults(function=lambda args: print(rollback_snapshot(args.pointer)))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
