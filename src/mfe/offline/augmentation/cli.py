"""Command-line workflow for the audit-gated BTC augmentation experiment."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mfe.contracts.ids import build_keyframe_id
from mfe.offline.augmentation.config import AugmentationConfig, load_augmentation_config
from mfe.offline.augmentation.coverage import (
    BoundaryLabel,
    btc_in_shot,
    choose_threshold,
    classify_coverage,
    read_btc_mapping,
)
from mfe.offline.augmentation.dedup import ProtectedReference, protected_dedup
from mfe.offline.augmentation.evaluation import (
    VariantRun,
    dev_gate,
    heldout_gate,
    oracle_gate,
    run_force_rank_variant,
    run_overlay_variant,
)
from mfe.offline.augmentation.io import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from mfe.offline.augmentation.locking import build_gate0_lock, load_inventory, read_lock
from mfe.offline.augmentation.materialize import (
    assign_augmented_ordinals,
    build_stage_view,
    extract_delta_images,
)
from mfe.offline.augmentation.models import (
    AugmentationManifestRow,
    CandidateRow,
    CoverageAuditRow,
    OracleTargetRow,
    ShotProposal,
    VideoInventoryRow,
)
from mfe.offline.augmentation.oracle import build_oracle_targets
from mfe.offline.augmentation.selectors import (
    apply_global_cap,
    match_random_control,
    select_b1,
    select_b2,
    select_b3,
    select_b4,
)
from mfe.offline.augmentation.snapshot import build_relational_snapshot
from mfe.offline.augmentation.store import ExperimentStore
from mfe.offline.augmentation.transnet import (
    TransNetV2Adapter,
    decode_for_transnet,
    proposals_from_scores,
    verify_shot_partition,
)
from mfe.offline.transport import build_transport

logger = logging.getLogger(__name__)
_VARIANT_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")


def _config(args: argparse.Namespace) -> AugmentationConfig:
    return load_augmentation_config(args.config)


def _workdir(config: AugmentationConfig) -> Path:
    path = config.resolve(config.paths.workdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _store(config: AugmentationConfig) -> ExperimentStore:
    transport = build_transport(
        config.transport.kind,
        volume=config.transport.volume,
        local_root=str(config.resolve(config.transport.local_root)),
        version=config.transport.version,
    )
    return ExperimentStore(transport, config.experiment.id, _workdir(config))


def _publish(config: AugmentationConfig, path: Path, relative: str) -> None:
    _store(config).publish(path, relative)


def _lock_path(config: AugmentationConfig) -> Path:
    return _workdir(config) / "gate0" / "lock.json"


def _load_lock(config: AugmentationConfig) -> dict[str, Any]:
    lock = read_lock(_lock_path(config))
    if lock["config_hash"] != config.canonical_hash():
        raise RuntimeError("config changed after Gate 0; create a new experiment id")
    return lock


def _read_freeze(config: AugmentationConfig, path: str | None) -> dict[str, Any]:
    if not path:
        raise RuntimeError("held-out access requires --freeze from a passing DEV gate")
    freeze = json.loads(Path(path).read_text(encoding="utf-8"))
    if not freeze.get("dev_go") or freeze.get("config_hash") != config.canonical_hash():
        raise RuntimeError("freeze is not a passing decision for this exact config")
    return freeze


def _enforce_heldout_grid(
    config: AugmentationConfig,
    cohort: str,
    freeze_path: str | None,
    threshold: float | None,
    target_gap: float | None,
    k_max: int | None,
) -> None:
    """Reject grid sweeps or policy drift after the held-out split is opened."""

    if cohort != "heldout":
        return
    freeze = _read_freeze(config, freeze_path)
    actual = (threshold, target_gap, k_max)
    expected = (
        freeze.get("threshold"),
        freeze.get("target_gap_seconds"),
        freeze.get("k_max"),
    )
    if None in actual or actual != expected:
        raise RuntimeError(f"held-out must use the frozen grid {expected}, got {actual}")


def _cohort_rows(
    config: AugmentationConfig,
    cohort: str,
    *,
    freeze: str | None = None,
) -> list[VideoInventoryRow]:
    lock = _load_lock(config)
    if cohort == "heldout":
        _read_freeze(config, freeze)
        selected = set(lock["heldout_videos_excluded"])
    else:
        selected = set(lock["cohort"])
    rows = [VideoInventoryRow.model_validate(row) for row in lock["inventory"]]
    return [row for row in rows if row.video_id in selected]


def _resolve_locked_file(
    row: VideoInventoryRow,
    attribute: str,
    override_root: str | None,
) -> Path:
    recorded = Path(str(getattr(row, attribute)))
    path = Path(override_root) / recorded.name if override_root else recorded
    expected_hash = row.sha256 if attribute == "source_path" else row.map_sha256
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise RuntimeError(f"{row.video_id}: {attribute} is missing or its hash changed: {path}")
    if attribute == "source_path" and path.stat().st_size != row.byte_size:
        raise RuntimeError(f"{row.video_id}: source video byte size changed")
    return path


def _token(value: float) -> str:
    return str(value).replace(".", "p")


def _variant(value: str) -> str:
    if not _VARIANT_PATTERN.fullmatch(value):
        raise ValueError("variant may contain only letters, digits, dot, underscore, hyphen")
    return value


def _grid_name(threshold: float, gap: float, k_max: int) -> str:
    return f"t{_token(threshold)}_g{_token(gap)}_k{k_max}"


def _raw_path(config: AugmentationConfig, cohort: str, video_id: str) -> Path:
    return _workdir(config) / "gate1" / cohort / "raw" / f"{video_id}.npz"


def _audit_path(
    config: AugmentationConfig,
    cohort: str,
    threshold: float,
    gap: float,
    k_max: int,
) -> Path:
    return (
        _workdir(config)
        / "gate1"
        / cohort
        / "audit"
        / f"{_grid_name(threshold, gap, k_max)}.jsonl"
    )


def _selector_dir(
    config: AugmentationConfig,
    cohort: str,
    threshold: float,
    gap: float,
    k_max: int,
) -> Path:
    return (
        _workdir(config)
        / "gate2"
        / cohort
        / "selectors"
        / _grid_name(threshold, gap, k_max)
    )


def _load_raw(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as artifact:
        return {
            "scores": artifact["scores"],
            "pts_times": tuple(float(item) for item in artifact["pts_times"]),
            "duration_seconds": float(artifact["duration_seconds"].item()),
            "decoder_fingerprint": str(artifact["decoder_fingerprint"].item()),
            "detector_fingerprint": str(artifact["detector_fingerprint"].item()),
        }


def _write_raw(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _filter_rows(
    rows: list[VideoInventoryRow], video_ids: list[str] | None, limit: int | None
) -> list[VideoInventoryRow]:
    if video_ids:
        requested = set(video_ids)
        known = {row.video_id for row in rows}
        if missing := requested - known:
            raise ValueError(f"video ids are outside the locked cohort: {sorted(missing)}")
        rows = [row for row in rows if row.video_id in requested]
    rows = sorted(rows, key=lambda row: row.video_id)
    return rows[:limit] if limit is not None else rows


def command_inventory(args: argparse.Namespace) -> None:
    """Create the file-hashed Gate-0 inventory from canonical videos/maps."""

    videos_root = Path(args.videos_root).resolve()
    maps_root = Path(args.maps_root).resolve()
    rows: list[VideoInventoryRow] = []
    try:
        import av
    except ImportError as error:  # pragma: no cover - integration dependency
        raise RuntimeError("inventory requires PyAV; install mfe[offline]") from error
    for video in sorted(videos_root.rglob(args.video_glob)):
        video_id = video.stem
        mapping = maps_root / f"{video_id}.csv"
        if not mapping.is_file():
            raise FileNotFoundError(f"{video_id}: missing mapping {mapping}")
        btc = read_btc_mapping(video_id, mapping)
        with mapping.open("r", encoding="utf-8-sig", newline="") as stream:
            fps_values = {
                float(row["fps"])
                for row in csv.DictReader(stream)
                if str(row.get("fps", "")).strip()
            }
        if len(fps_values) != 1:
            raise ValueError(f"{mapping}: expected one literal FPS, got {fps_values}")
        literal_fps = fps_values.pop()
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            if stream.duration is None or stream.time_base is None:
                raise RuntimeError(f"{video}: missing duration metadata")
            duration = float(stream.duration * stream.time_base)
        rows.append(
            VideoInventoryRow(
                video_id=video_id,
                source_path=str(video),
                sha256=sha256_file(video),
                byte_size=video.stat().st_size,
                fps=literal_fps,
                duration_seconds=duration,
                btc_keyframes=len(btc),
                map_path=str(mapping),
                map_sha256=sha256_file(mapping),
            )
        )
    output = Path(args.output)
    write_jsonl_atomic(output, rows)
    print(f"inventory: {len(rows)} video -> {output}")


def command_lock(args: argparse.Namespace) -> None:
    """Freeze code/config/query/input hashes before looking at detector output."""

    config = _config(args)
    inventory = load_inventory(config.resolve(config.dataset.video_inventory))
    store = _store(config)
    remote_manifest = f"/bundle/{config.experiment.parent_bundle}/manifest.json"
    temporary = _workdir(config) / "gate0" / "parent-manifest.json"
    store.transport.get(remote_manifest, temporary)
    lock = build_gate0_lock(
        config,
        inventory,
        parent_manifest_sha256=sha256_file(temporary),
    )
    target = write_json_atomic(_lock_path(config), lock)
    store.publish(target, "gate0/lock.json")
    print(f"Gate 0 locked {len(lock['cohort'])} pilot/DEV videos; held-out remains sealed")


def command_sync(args: argparse.Namespace) -> None:
    """Restore durable experiment artifacts into the current runner's workdir.

    Kaggle creates a fresh ``/kaggle/working`` directory for every Version.
    Keeping this operation as an explicit command makes a restart observable
    and prevents a missing local file from being mistaken for a fresh run.
    """

    config = _config(args)
    store = _store(config)
    destination = Path(args.destination) if args.destination else _workdir(config)
    restored = store.restore_prefix(args.prefix, destination, verify=not args.skip_verify)
    print(
        json.dumps(
            {
                "experiment_id": config.experiment.id,
                "prefix": args.prefix,
                "destination": str(destination),
                "restored": len(restored),
                "verified": not args.skip_verify,
            },
            ensure_ascii=False,
        )
    )


def command_status(args: argparse.Namespace) -> None:
    """Print local/remote artifact counts without changing experiment state."""

    config = _config(args)
    store = _store(config)
    prefix = args.prefix.strip("/")
    remote_prefix = store.paths.remote(prefix) if prefix else store.paths.root
    remote_files = store.transport.listdir(remote_prefix)
    workdir = _workdir(config)
    local_files = sorted(
        path.relative_to(workdir).as_posix()
        for path in workdir.rglob("*")
        if path.is_file() and not path.name.endswith(".download")
    )
    payload = {
        "experiment_id": config.experiment.id,
        "remote_prefix": remote_prefix,
        "remote_files": len(remote_files),
        "local_files": len(local_files),
        "lock_present": _lock_path(config).is_file(),
        "freeze_present": (_workdir(config) / "gate2" / "freeze.json").is_file(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_detect(args: argparse.Namespace) -> None:
    """Decode locked videos once and persist raw TransNet scores plus direct PTS."""

    config = _config(args)
    rows = _filter_rows(
        _cohort_rows(config, args.cohort, freeze=args.freeze), args.video, args.limit
    )
    adapter = TransNetV2Adapter(
        config.detector.factory,
        config.resolve(config.detector.checkpoint),
        config.detector.checkpoint_sha256,
    )
    for number, row in enumerate(rows, 1):
        target = _raw_path(config, args.cohort, row.video_id)
        if target.is_file() and not args.overwrite:
            logger.info("resume %s: raw artifact exists", row.video_id)
            continue
        source = _resolve_locked_file(row, "source_path", args.video_root)
        timeline = decode_for_transnet(source)
        scores = adapter.predict(timeline)
        _write_raw(
            target,
            scores=scores,
            pts_times=timeline.pts_times,
            duration_seconds=timeline.duration_seconds,
            decoder_fingerprint=timeline.decoder_fingerprint,
            detector_fingerprint=adapter.fingerprint,
        )
        _publish(config, target, f"gate1/{args.cohort}/raw/{target.name}")
        print(f"[{number}/{len(rows)}] {row.video_id}: {timeline.total_frames} frames")


def command_audit(args: argparse.Namespace) -> None:
    """Build all frozen detector/density grid audits without extracting images."""

    config = _config(args)
    _enforce_heldout_grid(
        config,
        args.cohort,
        args.freeze,
        args.threshold,
        args.target_gap,
        args.k_max,
    )
    rows = _filter_rows(
        _cohort_rows(config, args.cohort, freeze=args.freeze), args.video, args.limit
    )
    thresholds = (args.threshold,) if args.threshold is not None else config.detector.threshold_grid
    gaps = (
        (args.target_gap,)
        if args.target_gap is not None
        else config.coverage.target_gap_seconds_grid
    )
    k_values = (args.k_max,) if args.k_max is not None else config.coverage.k_max_grid
    summary: dict[str, Any] = {"cohort": args.cohort, "grids": {}}
    for threshold in thresholds:
        shots_by_video: dict[str, list[ShotProposal]] = {}
        btc_by_video = {}
        for row in rows:
            raw = _load_raw(_raw_path(config, args.cohort, row.video_id))
            shots = proposals_from_scores(
                row.video_id,
                raw["scores"],
                raw["pts_times"],
                raw["duration_seconds"],
                threshold,
                raw["detector_fingerprint"],
            )
            if problems := verify_shot_partition(shots, len(raw["pts_times"])):
                raise RuntimeError(f"{row.video_id}: invalid shot partition: {problems}")
            shots_by_video[row.video_id] = shots
            mapping = _resolve_locked_file(row, "map_path", args.maps_root)
            btc_by_video[row.video_id] = read_btc_mapping(row.video_id, mapping)
        for gap in gaps:
            for k_max in k_values:
                audit = [
                    classify_coverage(
                        shot,
                        btc_by_video[video_id],
                        target_gap_seconds=gap,
                        k_max=k_max,
                        micro_shot_seconds=config.detector.micro_shot_seconds,
                        uncertain_below=config.detector.uncertain_below,
                    )
                    for video_id in sorted(shots_by_video)
                    for shot in shots_by_video[video_id]
                ]
                target = _audit_path(config, args.cohort, threshold, gap, k_max)
                write_jsonl_atomic(target, audit)
                _publish(
                    config,
                    target,
                    f"gate1/{args.cohort}/audit/{target.name}",
                )
                counts = Counter(item.coverage_class.value for item in audit)
                summary["grids"][_grid_name(threshold, gap, k_max)] = {
                    "shots": len(audit),
                    "coverage_classes": dict(sorted(counts.items())),
                    "eligible_shots": sum(item.eligible for item in audit),
                }
    target = write_json_atomic(
        _workdir(config) / "gate1" / args.cohort / "audit-summary.json", summary
    )
    _publish(config, target, f"gate1/{args.cohort}/audit-summary.json")
    print(f"Gate 1 audit: {len(summary['grids'])} grids -> {target}")


def command_calibrate(args: argparse.Namespace) -> None:
    """Choose a detector threshold from manually reviewed DEV boundaries."""

    config = _config(args)
    cohort = {row.video_id for row in _cohort_rows(config, "dev")}
    labelled = read_jsonl(args.labels)
    if not labelled:
        raise ValueError("manual boundary label file is empty")
    unknown = {str(row.get("video_id")) for row in labelled} - cohort
    if unknown:
        raise ValueError(f"manual labels contain non-DEV/pilot videos: {sorted(unknown)}")
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in labelled:
        if not isinstance(row.get("is_boundary"), bool):
            raise ValueError("manual is_boundary values must be JSON booleans")
        video_id = str(row.get("video_id") or "")
        if not video_id:
            raise ValueError("manual boundary labels require a non-empty video_id")
        if "frame_idx" not in row:
            raise ValueError("manual boundary labels require frame_idx")
        by_video[video_id].append(row)
    combined_scores: list[float] = []
    combined_labels: list[BoundaryLabel] = []
    has_positive = False
    has_negative = False
    for video_id in sorted(by_video):
        scores = _load_raw(_raw_path(config, "dev", video_id))["scores"]
        offset = len(combined_scores)
        combined_scores.extend(float(item) for item in scores)
        for row in by_video[video_id]:
            try:
                frame_idx = int(row["frame_idx"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{video_id}: manual frame_idx must be an integer"
                ) from error
            if not 0 <= frame_idx < len(scores):
                raise ValueError(
                    f"{video_id}: manual frame_idx {frame_idx} is outside "
                    f"the {len(scores)} decoded frames"
                )
            is_boundary = bool(row["is_boundary"])
            has_positive |= is_boundary
            has_negative |= not is_boundary
            combined_labels.append(
                BoundaryLabel(frame_idx=offset + frame_idx, is_boundary=is_boundary)
            )
    if not has_positive or not has_negative:
        raise ValueError(
            "manual calibration must contain at least one positive and one negative label"
        )
    selected, metrics = choose_threshold(
        combined_scores,
        combined_labels,
        config.detector.threshold_grid,
    )
    report = {
        "split": "dev",
        "config_hash": config.canonical_hash(),
        "label_file_sha256": sha256_file(args.labels),
        "videos_reviewed": sorted(by_video),
        "labels": len(combined_labels),
        "selected_threshold": selected.threshold,
        "metrics": [row.__dict__ for row in metrics],
    }
    target = write_json_atomic(
        _workdir(config) / "gate1" / "dev" / "detector-calibration.json", report
    )
    _publish(config, target, "gate1/dev/detector-calibration.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _selection_inputs(
    config: AugmentationConfig,
    args: argparse.Namespace,
) -> tuple[
    list[CoverageAuditRow],
    dict[str, dict[int, float]],
    dict[str, list[Any]],
    dict[str, list[Any]],
    dict[str, str],
    dict[str, float],
]:
    rows = _filter_rows(
        _cohort_rows(config, args.cohort, freeze=args.freeze), args.video, args.limit
    )
    audit = [
        CoverageAuditRow.model_validate(row)
        for row in read_jsonl(
            _audit_path(
                config,
                args.cohort,
                args.threshold,
                args.target_gap,
                args.k_max,
            )
        )
    ]
    wanted = {row.video_id for row in rows}
    audit = [row for row in audit if row.video_id in wanted]
    pts_by_video: dict[str, dict[int, float]] = {}
    btc_by_video = {}
    fingerprints = {}
    literal_fps_by_video = {}
    for row in rows:
        raw = _load_raw(_raw_path(config, args.cohort, row.video_id))
        pts_by_video[row.video_id] = dict(enumerate(raw["pts_times"]))
        fingerprints[row.video_id] = raw["decoder_fingerprint"]
        literal_fps_by_video[row.video_id] = row.fps
        mapping = _resolve_locked_file(row, "map_path", args.maps_root)
        btc_by_video[row.video_id] = read_btc_mapping(row.video_id, mapping)
    shots = {
        row.shot_id: ShotProposal(
            video_id=row.video_id,
            shot_id=row.shot_id,
            start_frame=row.start_frame,
            end_frame_exclusive=row.end_frame_exclusive,
            start_pts_time=row.start_pts_time,
            end_pts_time=row.end_pts_time,
            boundary_confidence=row.detector_confidence,
            detector_fingerprint="audit-locked",
        )
        for row in audit
    }
    btc_by_shot = {
        shot_id: btc_in_shot(shot, btc_by_video[shot.video_id])
        for shot_id, shot in shots.items()
    }
    return (
        audit,
        pts_by_video,
        btc_by_shot,
        btc_by_video,
        fingerprints,
        literal_fps_by_video,
    )


def command_select(args: argparse.Namespace) -> None:
    """Generate B1--B4 allowlists and enforce the global supplemental cap."""

    config = _config(args)
    _enforce_heldout_grid(
        config,
        args.cohort,
        args.freeze,
        args.threshold,
        args.target_gap,
        args.k_max,
    )
    audit, pts, btc_by_shot, btc_by_video, fingerprints, literal_fps = (
        _selection_inputs(config, args)
    )
    common = {
        "decoder_fingerprints": fingerprints,
        "literal_fps_by_video": literal_fps,
        "config_hash": config.canonical_hash(),
    }
    proposed_b1 = select_b1(audit, pts, btc_by_shot, **common)
    proposed_b2 = select_b2(audit, pts, btc_by_shot, **common)
    proposed_b3 = select_b3(audit, pts, btc_by_shot, **common)
    base_count = len({item.keyframe_id for rows in btc_by_video.values() for item in rows})
    # Every selector is an A/B arm, so every arm must consume the same global
    # budget.  Keeping B1/B2 uncapped would make their optional evaluation
    # incomparable with B3/B4 and could silently exceed the pilot guardrail.
    b1 = apply_global_cap(
        proposed_b1,
        audit,
        base_keyframes=base_count,
        cap_ratio=config.coverage.delta_cap_ratio,
    )
    b2 = apply_global_cap(
        proposed_b2,
        audit,
        base_keyframes=base_count,
        cap_ratio=config.coverage.delta_cap_ratio,
    )
    b3 = apply_global_cap(
        proposed_b3,
        audit,
        base_keyframes=base_count,
        cap_ratio=config.coverage.delta_cap_ratio,
    )
    b4 = select_b4(
        b3,
        audit,
        pts,
        btc_by_shot,
        **common,
        seed=config.experiment.seed,
    )
    b4_pool = select_b4(
        b3,
        audit,
        pts,
        btc_by_shot,
        **common,
        seed=config.experiment.seed,
        quota_multiplier=3,
    )
    if len(b4) != len(b3):
        raise RuntimeError(
            "B4 random control cannot match B3's pre-dedup budget: "
            f"{len(b4)} != {len(b3)}; inspect shots with too few non-BTC frames"
        )
    if len(b4_pool) < len(b3):
        raise RuntimeError(
            "B4 over-sampled pool is smaller than B3's budget: "
            f"{len(b4_pool)} < {len(b3)}; increase audit coverage or review the pool"
        )
    target_dir = _selector_dir(
        config, args.cohort, args.threshold, args.target_gap, args.k_max
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    variants = {
        "b1": b1,
        "b2": b2,
        "b3": b3,
        "b4": b4,
        "b4_pool": b4_pool,
    }
    variant_paths: dict[str, Path] = {}
    for name, candidates in variants.items():
        target = write_jsonl_atomic(target_dir / f"{name}.jsonl", candidates)
        variant_paths[name] = target
        _publish(
            config,
            target,
            f"gate2/{args.cohort}/selectors/{target_dir.name}/{target.name}",
        )
    summary = {
        "cohort": args.cohort,
        "grid": target_dir.name,
        "threshold": args.threshold,
        "target_gap_seconds": args.target_gap,
        "k_max": args.k_max,
        "config_hash": config.canonical_hash(),
        "base_keyframes": base_count,
        "proposed_b1": len(proposed_b1),
        "proposed_b2": len(proposed_b2),
        "proposed_b3": len(proposed_b3),
        "cap_ratio": config.coverage.delta_cap_ratio,
        "variants": {key: len(value) for key, value in variants.items()},
        "variant_sha256": {
            key: sha256_file(path) for key, path in variant_paths.items()
        },
        "b3_b4_equal_budget": len(b3) == len(b4),
    }
    target = write_json_atomic(target_dir / "summary.json", summary)
    _publish(
        config,
        target,
        f"gate2/{args.cohort}/selectors/{target_dir.name}/summary.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_oracle(args: argparse.Namespace) -> None:
    """Build DEV-only oracle targets; this command refuses held-out inputs."""

    config = _config(args)
    if args.cohort != "dev":
        raise RuntimeError("oracle labels are DEV-only")
    audit, pts, _, btc_by_video, fingerprints, literal_fps = _selection_inputs(
        config, args
    )
    audit_by_video: dict[str, list[CoverageAuditRow]] = defaultdict(list)
    for row in audit:
        audit_by_video[row.video_id].append(row)
    queries = read_jsonl(config.resolve(config.dataset.dev_queryset))
    targets = build_oracle_targets(
        queries,
        pts,
        btc_by_video,
        audit_by_video,
        literal_fps_by_video=literal_fps,
        decoder_fingerprints=fingerprints,
        config_hash=config.canonical_hash(),
    )
    target = write_jsonl_atomic(
        _workdir(config) / "gate1" / "dev" / "oracles" / "targets.jsonl",
        targets,
    )
    unique_candidates = {
        item.target.keyframe_id: item.target for item in targets
    }
    candidate_target = write_jsonl_atomic(
        target.parent / "candidates.jsonl",
        [unique_candidates[key] for key in sorted(unique_candidates)],
    )
    _publish(config, target, "gate1/dev/oracles/targets.jsonl")
    _publish(config, candidate_target, "gate1/dev/oracles/candidates.jsonl")
    print(
        f"DEV oracle: {len(targets)} labels / {len(unique_candidates)} unique frames "
        f"-> {target.parent}"
    )


def _candidate_rows(path: str | Path) -> list[CandidateRow]:
    return [CandidateRow.model_validate(row) for row in read_jsonl(path)]


def _assert_candidate_config(
    candidates: Iterable[CandidateRow], config: AugmentationConfig, *, label: str
) -> None:
    """Reject candidate artifacts produced under a different experiment config."""

    expected = config.canonical_hash()
    mismatched = sorted(
        {candidate.keyframe_id for candidate in candidates if candidate.config_hash != expected}
    )
    if mismatched:
        raise RuntimeError(
            f"{label} candidates were built by another config; "
            f"examples: {mismatched[:3]}"
        )


def command_materialize(args: argparse.Namespace) -> None:
    """Extract a selector's source frames and publish a provenance sidecar."""

    config = _config(args)
    variant = _variant(args.variant)
    if args.cohort == "heldout":
        _read_freeze(config, args.freeze)
    rows = _cohort_rows(config, args.cohort, freeze=args.freeze)
    inventory = {row.video_id: row for row in rows}
    candidates = _candidate_rows(args.candidates)
    if not candidates:
        raise ValueError("cannot materialize an empty candidate allowlist")
    _assert_candidate_config(candidates, config, label=args.cohort)
    unknown = {item.video_id for item in candidates} - set(inventory)
    if unknown:
        raise ValueError(f"candidate videos are outside locked cohort: {sorted(unknown)}")
    by_video: dict[str, list[CandidateRow]] = defaultdict(list)
    for candidate in candidates:
        by_video[candidate.video_id].append(candidate)
    output_root = _workdir(config) / "pool" / args.cohort / variant / "images"
    materialized = []
    manifest: list[AugmentationManifestRow] = []
    upload_pairs: list[tuple[str | Path, str]] = []
    for video_id in sorted(by_video):
        row = inventory[video_id]
        source = _resolve_locked_file(row, "source_path", args.video_root)
        mapping = _resolve_locked_file(row, "map_path", args.maps_root)
        btc = read_btc_mapping(video_id, mapping)
        images = extract_delta_images(source, by_video[video_id], output_root)
        ordinals = assign_augmented_ordinals(btc, by_video[video_id])
        stage_dir = (
            _workdir(config)
            / "pool"
            / args.cohort
            / variant
            / "stage-view"
            / video_id
        )
        stage_paths = build_stage_view(images, stage_dir)
        stage_map = stage_dir.parent.parent / "stage-maps" / f"{video_id}.csv"
        stage_map.parent.mkdir(parents=True, exist_ok=True)
        with stage_map.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("n", "pts_time", "fps", "frame_idx")
            )
            writer.writeheader()
            ordered = sorted(images, key=lambda item: item.candidate.pts_time)
            for number, image in enumerate(ordered, 1):
                writer.writerow(
                    {
                        "n": number,
                        "pts_time": image.candidate.pts_time,
                        "fps": row.fps,
                        "frame_idx": image.candidate.frame_idx,
                    }
                )
                if stage_paths[image.candidate.keyframe_id].name != f"{number:03d}.jpg":
                    raise RuntimeError("stage-view ordinal drift")
        # The legacy offline stages consume the ordinal view and its literal
        # CSV, not the source-indexed materialized manifest.  They are part of
        # the durable unit: a fresh Kaggle Version must be able to resume
        # feature extraction without decoding the (large) source video again.
        for stage_path in sorted(stage_dir.rglob("*")):
            if stage_path.is_file():
                upload_pairs.append(
                    (
                        stage_path,
                        f"pool/{args.cohort}/{variant}/stage-view/"
                        f"{video_id}/{stage_path.name}",
                    )
                )
        upload_pairs.append(
            (
                stage_map,
                f"pool/{args.cohort}/{variant}/stage-maps/{stage_map.name}",
            )
        )
        for image in images:
            candidate = image.candidate
            materialized.append(
                {
                    "keyframe_id": candidate.keyframe_id,
                    "path": str(image.path),
                    "image_sha256": image.image_sha256,
                    "width": image.width,
                    "height": image.height,
                    "byte_size": image.byte_size,
                }
            )
            manifest.append(
                AugmentationManifestRow(
                    keyframe_id=candidate.keyframe_id,
                    video_id=candidate.video_id,
                    frame_idx=candidate.frame_idx,
                    literal_fps=candidate.literal_fps,
                    pts_time=candidate.pts_time,
                    decoder_pts_time=candidate.decoder_pts_time,
                    frame_idx_source=candidate.frame_idx_source,
                    pts_time_source=candidate.pts_time_source,
                    ordinal=ordinals[candidate.keyframe_id],
                    origin="supplemental",
                    selector=candidate.selector,
                    shot_id=candidate.shot_id,
                    parent_bundle=config.experiment.parent_bundle,
                    run_id=config.experiment.id,
                    decoder_fingerprint=candidate.decoder_fingerprint,
                    config_hash=candidate.config_hash,
                    image_sha256=image.image_sha256,
                    audit_status=candidate.audit_status,
                    coverage_class=candidate.coverage_class,
                    selection_round=candidate.selection_round,
                )
            )
            upload_pairs.append(
                (
                    image.path,
                    f"pool/{args.cohort}/{variant}/images/{video_id}/{image.path.name}",
                )
            )
    pool_dir = _workdir(config) / "pool" / args.cohort / variant
    materialized_path = write_jsonl_atomic(pool_dir / "materialized.jsonl", materialized)
    manifest_path = write_jsonl_atomic(pool_dir / "augmentation_manifest.jsonl", manifest)
    store = _store(config)
    store.publish_many(upload_pairs)
    store.publish(
        materialized_path,
        f"pool/{args.cohort}/{variant}/materialized.jsonl",
    )
    store.publish(
        manifest_path,
        f"pool/{args.cohort}/{variant}/augmentation_manifest.jsonl",
    )
    print(f"materialized and hash-verified {len(manifest)} supplemental frames")


def command_extract_features(args: argparse.Namespace) -> None:
    """Run existing offline stages on delta-only stage views and flatten them."""

    from mfe.offline.bundle import BundleLayout, BundleWriter, ShardPlan
    from mfe.offline.merge import merge_bundle
    from mfe.offline.stages import STAGES, StageContext, read_keyframes, unload_models
    from mfe.offline.transport import LocalDirTransport

    config = _config(args)
    variant = _variant(args.variant)
    if args.cohort == "heldout":
        _read_freeze(config, args.freeze)
    candidates = _candidate_rows(args.candidates)
    if not candidates:
        raise ValueError("cannot extract features for an empty candidate allowlist")
    _assert_candidate_config(candidates, config, label=args.cohort)
    video_ids = sorted({item.video_id for item in candidates})
    inventory = {
        row.video_id: row
        for row in _cohort_rows(config, args.cohort, freeze=args.freeze)
    }
    if missing := set(video_ids) - set(inventory):
        raise ValueError(f"candidate videos are outside locked cohort: {sorted(missing)}")
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    allowed = {
        "n4_objects",
        "n4_siglip",
        "n5_ocr_easyocr",
        "n5_ocr_qwen",
        "n5_ocr_paddle",
        "n5_ocr_vietocr",
        "n6_bge_m3",
        "n6_caption_qwen",
        "n6_ocr_qwen25",
    }
    unknown = set(stages) - allowed
    if not stages or unknown:
        raise ValueError(
            f"unknown/unsafe delta stages: {sorted(unknown) if stages else stages}"
        )
    if "n4_siglip" not in stages:
        raise ValueError("delta feature parity requires n4_siglip")
    raw_params = (
        json.loads(Path(args.stage_params).read_text(encoding="utf-8"))
        if args.stage_params
        else {}
    )
    if not isinstance(raw_params, dict):
        raise ValueError("--stage-params must be a JSON object keyed by stage")

    root = _workdir(config)
    pool = root / "pool" / args.cohort / variant
    transport = LocalDirTransport(root / "delta-transport" / args.cohort / variant)
    bundle_name = f"{config.experiment.id}_{args.cohort}_{variant}_delta"
    layout = BundleLayout(bundle_name)
    writer = BundleWriter(
        transport,
        layout,
        "augmentation",
        root / "delta-work" / args.cohort / variant,
    )
    contexts: dict[str, StageContext] = {}
    markers: dict[str, list[dict]] = defaultdict(list)
    for video_id in video_ids:
        map_csv = pool / "stage-maps" / f"{video_id}.csv"
        image_dir = pool / "stage-view" / video_id
        keyframes = read_keyframes(map_csv, image_dir)
        expected = {item.keyframe_id for item in candidates if item.video_id == video_id}
        if {
            build_keyframe_id(video_id, keyframe.frame_idx) for keyframe in keyframes
        } != expected:
            raise RuntimeError(f"{video_id}: stage view does not match candidate allowlist")
        source = _resolve_locked_file(inventory[video_id], "source_path", args.video_root)
        writer.write_file(map_csv, layout.map_keyframes(video_id))
        writer.write_directory(image_dir, layout.keyframes_dir(video_id))
        contexts[video_id] = StageContext(
            video_id=video_id,
            video_path=source,
            keyframes=keyframes,
            writer=writer,
            layout=layout,
            workdir=root / "delta-work" / args.cohort / variant,
        )

    cache_pairs = [
        ("n5_ocr_paddle", "n5_ocr_vietocr"),
        ("n6_caption_qwen", "n6_ocr_qwen25"),
    ]
    units: list[tuple[str, ...]] = []
    index = 0
    while index < len(stages):
        pair = tuple(stages[index : index + 2])
        if pair in cache_pairs:
            units.append(pair)
            index += 2
        else:
            units.append((stages[index],))
            index += 1
    for unit in units:
        unload_models()
        if len(unit) == 2:
            # These stages share a per-video scan.  Keep the outer loop on
            # video so the second stage consumes the first stage's cache, while
            # the model itself remains cached across all videos.
            for video_id in video_ids:
                context = contexts[video_id]
                for stage in unit:
                    context.params = dict(raw_params.get(stage, {}))
                    markers[video_id].append(STAGES[stage](context))
        else:
            stage = unit[0]
            for video_id in video_ids:
                context = contexts[video_id]
                context.params = dict(raw_params.get(stage, {}))
                markers[video_id].append(STAGES[stage](context))
    unload_models()
    for video_id in video_ids:
        writer.write_part_manifest(
            video_id,
            markers[video_id],
            len(contexts[video_id].keyframes),
        )

    plan = ShardPlan(
        bundle=bundle_name,
        keyframe_set="augmentation_delta",
        calibration_video=video_ids[0],
        members={"augmentation": video_ids},
        stages=stages,
    )
    merge_root = root / "delta-merge" / args.cohort / variant
    report = merge_bundle(
        transport,
        plan,
        merge_root,
        emit="single",
        layout=layout,
        workers=1,
    )
    report_path = write_json_atomic(merge_root / "merge-report.json", report.to_dict())
    if not report.ok:
        raise RuntimeError(f"delta merge refused: {report.refusals}")
    out = merge_root / "out"
    files = [
        (path, path.relative_to(out).as_posix())
        for path in out.rglob("*")
        if path.is_file()
    ]
    store = _store(config)
    store.publish_many(
        [
            (path, f"gate2/{args.cohort}/delta-flat/{variant}/{relative}")
            for path, relative in files
        ]
    )
    store.publish(
        report_path,
        f"gate2/{args.cohort}/delta-flat/{variant}/merge-report.json",
    )
    print(f"delta flat bundle: {len(candidates)} frames -> {out}")


def _load_embedding_map(path: str | Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as artifact:
        ids = [str(item) for item in artifact["ids"].tolist()]
        vectors = np.asarray(artifact["vectors"], dtype=np.float32)
    if vectors.ndim != 2 or len(ids) != len(vectors) or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: invalid or duplicate id/vector rows")
    return dict(zip(ids, vectors, strict=True))


def _load_phash_map(path: str | Path) -> dict[str, int]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected keyframe_id -> integer pHash mapping")
    return {
        str(key): int(value, 0) if isinstance(value, str) else int(value)
        for key, value in raw.items()
    }


def _flat_vector_path(root: str | Path, filename: str) -> Path:
    root = Path(root)
    nested = root / "data" / "vectors" / filename
    return nested if nested.is_file() else root / filename


def command_prepare_dedup(args: argparse.Namespace) -> None:
    """Build reproducible BTC/candidate pHash and SigLIP inputs for dedup."""

    import cv2
    import numpy as np

    from mfe.offline.dedup_keyframes import phash

    config = _config(args)
    variant = _variant(args.variant)
    if args.cohort == "heldout":
        _read_freeze(config, args.freeze)
    candidates = _candidate_rows(args.candidates)
    _assert_candidate_config(candidates, config, label=args.cohort)
    audit = {
        row.shot_id: row
        for row in (
            CoverageAuditRow.model_validate(item) for item in read_jsonl(args.audit)
        )
    }
    candidate_shots = {item.shot_id for item in candidates}
    missing_shots = candidate_shots - set(audit)
    if missing_shots:
        raise ValueError(f"candidate shots are absent from audit: {sorted(missing_shots)}")
    inventory = {
        row.video_id: row
        for row in _cohort_rows(config, args.cohort, freeze=args.freeze)
    }
    btc_rows: list[dict] = []
    btc_images: dict[str, Path] = {}
    for video_id in sorted({item.video_id for item in candidates}):
        row = inventory[video_id]
        mapping = _resolve_locked_file(row, "map_path", args.maps_root)
        seen: set[str] = set()
        with mapping.open("r", encoding="utf-8-sig", newline="") as stream:
            for source in csv.DictReader(stream):
                frame_idx = int(source["frame_idx"])
                keyframe_id = build_keyframe_id(video_id, frame_idx)
                if keyframe_id in seen:
                    continue
                seen.add(keyframe_id)
                matching = [
                    audit[shot_id]
                    for shot_id in candidate_shots
                    if audit[shot_id].video_id == video_id
                    and audit[shot_id].start_frame
                    <= frame_idx
                    < audit[shot_id].end_frame_exclusive
                ]
                if not matching:
                    continue
                shot = min(matching, key=lambda item: item.shot_id)
                ordinal = int(source["n"])
                root = Path(args.btc_keyframes_root)
                direct = root / video_id / f"{ordinal:03d}.jpg"
                nested = root / "keyframes" / video_id / f"{ordinal:03d}.jpg"
                image_path = direct if direct.is_file() else nested
                if not image_path.is_file():
                    raise FileNotFoundError(f"BTC image unavailable for {keyframe_id}")
                btc_rows.append({"keyframe_id": keyframe_id, "shot_id": shot.shot_id})
                btc_images[keyframe_id] = image_path

    materialized = {
        str(row["keyframe_id"]): Path(row["path"])
        for row in read_jsonl(args.materialized)
    }
    missing_images = {item.keyframe_id for item in candidates} - set(materialized)
    if missing_images:
        raise ValueError(f"materialized report misses candidates: {sorted(missing_images)[:3]}")

    def hashes(images: dict[str, Path]) -> dict[str, int]:
        out = {}
        for keyframe_id, path in images.items():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot decode image for pHash: {path}")
            out[keyframe_id] = phash(image)
        return out

    candidate_images = {item.keyframe_id: materialized[item.keyframe_id] for item in candidates}
    btc_phashes = hashes(btc_images)
    candidate_phashes = hashes(candidate_images)
    base_embeddings = _load_embedding_map(
        _flat_vector_path(args.base_flat_root, "siglip_vectors.npz")
    )
    delta_embeddings = _load_embedding_map(
        _flat_vector_path(args.delta_flat_root, "siglip_vectors.npz")
    )
    btc_ids = sorted(btc_images)
    candidate_ids = sorted(candidate_images)
    for keyframe_id in btc_ids:
        if keyframe_id not in base_embeddings:
            raise KeyError(f"base SigLIP misses protected BTC id {keyframe_id}")
    for keyframe_id in candidate_ids:
        if keyframe_id not in delta_embeddings:
            raise KeyError(f"delta SigLIP misses candidate id {keyframe_id}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "btc_shots": write_jsonl_atomic(output / "btc-shots.jsonl", btc_rows),
        "btc_phashes": write_json_atomic(output / "btc-phashes.json", btc_phashes),
        "candidate_phashes": write_json_atomic(
            output / "candidate-phashes.json", candidate_phashes
        ),
    }
    np.savez_compressed(
        output / "btc-embeddings.npz",
        ids=np.asarray(btc_ids, dtype=np.str_),
        vectors=np.asarray([base_embeddings[key] for key in btc_ids], dtype=np.float32),
    )
    np.savez_compressed(
        output / "candidate-embeddings.npz",
        ids=np.asarray(candidate_ids, dtype=np.str_),
        vectors=np.asarray(
            [delta_embeddings[key] for key in candidate_ids], dtype=np.float32
        ),
    )
    files["btc_embeddings"] = output / "btc-embeddings.npz"
    files["candidate_embeddings"] = output / "candidate-embeddings.npz"
    report = {
        "variant": variant,
        "candidates": len(candidate_ids),
        "protected_btc": len(btc_ids),
        "config_hash": config.canonical_hash(),
        "files": {key: sha256_file(path) for key, path in files.items()},
    }
    report_path = write_json_atomic(output / "report.json", report)
    store = _store(config)
    relative = f"gate2/{args.cohort}/dedup-inputs/{variant}"
    store.publish_many(
        [(path, f"{relative}/{path.name}") for path in (*files.values(), report_path)]
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_dedup(args: argparse.Namespace) -> None:
    """Apply protected BTC-first pHash/SigLIP dedup to one selector pool."""

    config = _config(args)
    if args.cohort == "heldout":
        _read_freeze(config, args.freeze)
    candidates = _candidate_rows(args.candidates)
    _assert_candidate_config(candidates, config, label=args.cohort)
    btc_rows = read_jsonl(args.btc_shots)
    btc_phashes = _load_phash_map(args.btc_phashes)
    candidate_phashes = _load_phash_map(args.candidate_phashes)
    btc_embeddings = _load_embedding_map(args.btc_embeddings)
    candidate_embeddings = _load_embedding_map(args.candidate_embeddings)
    protected = []
    for row in btc_rows:
        keyframe_id = str(row["keyframe_id"])
        if keyframe_id not in btc_phashes or keyframe_id not in btc_embeddings:
            raise KeyError(f"missing protected BTC pHash/SigLIP for {keyframe_id}")
        protected.append(
            ProtectedReference(
                keyframe_id=keyframe_id,
                shot_id=str(row["shot_id"]),
                phash=btc_phashes[keyframe_id],
                embedding=btc_embeddings[keyframe_id],
            )
        )
    result = protected_dedup(
        candidates,
        protected,
        candidate_phashes,
        candidate_embeddings,
        phash_max_distance=config.dedup.phash_max_distance,
        embedding_max_cosine=config.dedup.embedding_max_cosine,
    )
    output = Path(args.output)
    dropped = output.with_name(output.stem + "-dropped.jsonl")
    write_jsonl_atomic(output, result.kept)
    write_jsonl_atomic(dropped, [item.__dict__ for item in result.dropped])
    report = {
        "config_hash": config.canonical_hash(),
        "input_sha256": sha256_file(args.candidates),
        "output_sha256": sha256_file(output),
        "kept": len(result.kept),
        "dropped": len(result.dropped),
    }
    report_path = write_json_atomic(
        output.with_name(output.stem + "-report.json"), report
    )
    relative_root = f"gate2/{args.cohort}/dedup"
    _publish(config, output, f"{relative_root}/{output.name}")
    _publish(config, dropped, f"{relative_root}/{dropped.name}")
    _publish(config, report_path, f"{relative_root}/{report_path.name}")
    print(f"dedup: kept {len(result.kept)}, dropped {len(result.dropped)}")


def command_match_control(args: argparse.Namespace) -> None:
    """Match post-dedup B4 to B3's exact per-shot budget."""

    config = _config(args)
    if args.cohort == "heldout":
        _read_freeze(config, args.freeze)
    b3_candidates = _candidate_rows(args.b3)
    b4_pool_candidates = _candidate_rows(args.b4_pool)
    _assert_candidate_config(b3_candidates, config, label=f"{args.cohort} B3")
    _assert_candidate_config(
        b4_pool_candidates, config, label=f"{args.cohort} B4 pool"
    )
    matched = match_random_control(
        b3_candidates,
        b4_pool_candidates,
    )
    target = write_jsonl_atomic(args.output, matched)
    report = {
        "config_hash": config.canonical_hash(),
        "b3_sha256": sha256_file(args.b3),
        "b4_pool_sha256": sha256_file(args.b4_pool),
        "output_sha256": sha256_file(target),
        "frames": len(matched),
    }
    report_path = write_json_atomic(
        target.with_name(target.stem + "-report.json"), report
    )
    _publish(config, target, f"gate2/{args.cohort}/dedup/{target.name}")
    _publish(config, report_path, f"gate2/{args.cohort}/dedup/{report_path.name}")
    print(f"matched B4 control: {len(matched)} frames -> {target}")


def command_snapshot(args: argparse.Namespace) -> None:
    """Build and audit a disposable augmented RelationalMap snapshot."""

    config = _config(args)
    if args.cohort == "heldout":
        _read_freeze(config, args.freeze)
    candidates = _candidate_rows(args.candidates)
    _assert_candidate_config(candidates, config, label=args.cohort)
    report = build_relational_snapshot(
        args.base_db,
        args.output,
        candidates,
        overwrite=args.overwrite,
    )
    report_path = Path(args.output).with_suffix(".report.json")
    write_json_atomic(report_path, report)
    relative = f"gate2/{args.cohort}/snapshots"
    _publish(config, Path(args.output), f"{relative}/{Path(args.output).name}")
    _publish(config, report_path, f"{relative}/{report_path.name}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _variant_payload(run: VariantRun) -> dict[str, Any]:
    return {
        "name": run.name,
        "query_ids": run.query_ids,
        "query_types": run.query_types,
        "scores": run.scores,
        "latency_ms": run.latency_ms,
        "mean_score": run.mean_score,
        "p95_latency_ms": run.p95_latency_ms,
        "by_type": run.by_type,
    }


def _json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _verify_b3_lineage(
    config: AugmentationConfig,
    *,
    cohort: str,
    selection_summary: str,
    dedup_report: str,
    candidates: str,
) -> dict[str, Any]:
    summary = _json_object(selection_summary)
    dedup = _json_object(dedup_report)
    expected_config = config.canonical_hash()
    if summary.get("cohort") != cohort or summary.get("config_hash") != expected_config:
        raise RuntimeError("selection summary belongs to another cohort/config")
    if dedup.get("config_hash") != expected_config:
        raise RuntimeError("dedup report belongs to another config")
    if dedup.get("input_sha256") != summary.get("variant_sha256", {}).get("b3"):
        raise RuntimeError("B3 dedup input does not match the selected B3 allowlist")
    if dedup.get("output_sha256") != sha256_file(candidates):
        raise RuntimeError("B3 candidate bytes do not match the dedup report")
    return summary


def _run_variant(
    config: AugmentationConfig,
    name: str,
    queryset: Path,
    relational: str,
    delta_root: str,
    candidates: str | None,
    repeats: int,
) -> VariantRun:
    include_ids = [item.keyframe_id for item in _candidate_rows(candidates)] if candidates else []
    return run_overlay_variant(
        name=name,
        config_path=str(config.resolve(config.paths.eval_config)),
        queryset_path=str(queryset),
        relational_db=relational,
        base_flat_root=str(config.resolve(config.paths.base_flat_root)),
        delta_flat_root=delta_root,
        include_ids=include_ids,
        latency_repeats=repeats,
    )


def command_evaluate_dev(args: argparse.Namespace) -> None:
    """Run causal B0/B3/B4 official-score evaluation and the frozen DEV gate."""

    config = _config(args)
    queryset = config.resolve(config.dataset.dev_queryset)
    b3_candidates = _candidate_rows(args.b3)
    b4_candidates = _candidate_rows(args.b4)
    _assert_candidate_config(b3_candidates, config, label="dev B3")
    _assert_candidate_config(b4_candidates, config, label="dev B4")
    if len(b3_candidates) != len(b4_candidates):
        raise RuntimeError("B3 and B4 must have the same post-dedup frame budget")
    summary = _verify_b3_lineage(
        config,
        cohort="dev",
        selection_summary=args.selection_summary,
        dedup_report=args.b3_dedup_report,
        candidates=args.b3,
    )
    b4_pool_dedup = _json_object(args.b4_pool_dedup_report)
    b4_match = _json_object(args.b4_match_report)
    if any(
        report.get("config_hash") != config.canonical_hash()
        for report in (b4_pool_dedup, b4_match)
    ):
        raise RuntimeError("B4 lineage report belongs to another config")
    if b4_pool_dedup.get("input_sha256") != summary["variant_sha256"]["b4_pool"]:
        raise RuntimeError("B4 pool dedup input does not match selection summary")
    if b4_match.get("b3_sha256") != sha256_file(args.b3):
        raise RuntimeError("B4 match report was built against another B3")
    if b4_match.get("b4_pool_sha256") != b4_pool_dedup.get("output_sha256"):
        raise RuntimeError("B4 match report was built against another deduped pool")
    if b4_match.get("output_sha256") != sha256_file(args.b4):
        raise RuntimeError("B4 candidate bytes do not match the match report")
    b0 = _run_variant(
        config,
        "B0",
        queryset,
        args.base_relational,
        args.b3_delta_root,
        None,
        args.repeats,
    )
    b3 = _run_variant(
        config,
        "B3",
        queryset,
        args.b3_relational,
        args.b3_delta_root,
        args.b3,
        args.repeats,
    )
    b4 = _run_variant(
        config,
        "B4",
        queryset,
        args.b4_relational,
        args.b4_delta_root,
        args.b4,
        args.repeats,
    )
    optional_runs = []
    for name in ("b1", "b2"):
        candidates = getattr(args, name)
        relational = getattr(args, f"{name}_relational")
        delta_root = getattr(args, f"{name}_delta_root")
        if len({bool(candidates), bool(relational), bool(delta_root)}) != 1:
            raise ValueError(
                f"--{name}, --{name}-relational and --{name}-delta-root "
                "must be supplied together"
            )
        if candidates:
            optional_runs.append(
                _run_variant(
                    config,
                    name.upper(),
                    queryset,
                    relational,
                    delta_root,
                    candidates,
                    args.repeats,
                )
            )
    target_ids = {
        OracleTargetRow.model_validate(row).query_id for row in read_jsonl(args.oracle_targets)
    }
    gate = dev_gate(
        b0,
        b3,
        b4,
        target_query_ids=target_ids,
        delta_frames=len(b3_candidates),
        base_frames=int(summary["base_keyframes"]),
        cap_ratio=config.coverage.delta_cap_ratio,
        noninferiority_margin=config.evaluation.noninferiority_margin,
        max_latency_ratio=config.evaluation.max_latency_ratio,
        bootstrap_samples=config.evaluation.bootstrap_samples,
        bootstrap_seed=config.evaluation.bootstrap_seed,
    )
    report = {
        "split": "dev",
        "config_hash": config.canonical_hash(),
        "policy": {
            "threshold": summary["threshold"],
            "target_gap_seconds": summary["target_gap_seconds"],
            "k_max": summary["k_max"],
        },
        "variants": {
            run.name: _variant_payload(run)
            for run in (b0, *optional_runs, b3, b4)
        },
        "gate": gate,
    }
    target = write_json_atomic(_workdir(config) / "gate2" / "dev" / "report.json", report)
    _publish(config, target, "gate2/dev/report.json")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


def command_evaluate_oracle(args: argparse.Namespace) -> None:
    """Evaluate O1-T retrieval headroom and the O1-R system ceiling."""

    config = _config(args)
    _assert_candidate_config(
        _candidate_rows(args.oracle_candidates), config, label="oracle"
    )
    queryset = config.resolve(config.dataset.dev_queryset)
    b0 = _run_variant(
        config,
        "B0",
        queryset,
        args.base_relational,
        args.oracle_delta_root,
        None,
        args.repeats,
    )
    o1t = _run_variant(
        config,
        "O1-T",
        queryset,
        args.oracle_relational,
        args.oracle_delta_root,
        args.oracle_candidates,
        args.repeats,
    )
    targets = [
        OracleTargetRow.model_validate(row) for row in read_jsonl(args.oracle_targets)
    ]
    _assert_candidate_config((target.target for target in targets), config, label="oracle")
    o1r = run_force_rank_variant(
        name="O1-R",
        config_path=str(config.resolve(config.paths.eval_config)),
        queryset_path=str(queryset),
        relational_db=args.oracle_relational,
        base_flat_root=str(config.resolve(config.paths.base_flat_root)),
        targets=targets,
    )
    gate = oracle_gate(
        b0,
        o1t,
        min_mean_delta=config.evaluation.min_oracle_mean_delta,
        min_rescues=config.evaluation.min_oracle_rescues,
    )
    report = {
        "split": "dev",
        "config_hash": config.canonical_hash(),
        "variants": {run.name: _variant_payload(run) for run in (b0, o1t, o1r)},
        "gate": gate,
        "o1r_is_ceiling_only": True,
    }
    target = write_json_atomic(
        _workdir(config) / "gate1" / "dev" / "oracles" / "report.json", report
    )
    _publish(config, target, "gate1/dev/oracles/report.json")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


def command_freeze(args: argparse.Namespace) -> None:
    """Freeze a passing DEV policy before any held-out detector/audit run."""

    config = _config(args)
    report = json.loads(Path(args.dev_report).read_text(encoding="utf-8"))
    if (
        report.get("split") != "dev"
        or report.get("config_hash") != config.canonical_hash()
        or not report["gate"].get("go")
    ):
        raise RuntimeError("cannot freeze: DEV gate did not pass for this config")
    expected_policy = (
        report.get("policy", {}).get("threshold"),
        report.get("policy", {}).get("target_gap_seconds"),
        report.get("policy", {}).get("k_max"),
    )
    requested_policy = (args.threshold, args.target_gap, args.k_max)
    if requested_policy != expected_policy:
        raise RuntimeError(
            f"freeze policy {requested_policy} differs from DEV report {expected_policy}"
        )
    candidates = Path(args.b3)
    decision = {
        "schema_version": 1,
        "experiment_id": config.experiment.id,
        "config_hash": config.canonical_hash(),
        "dev_go": True,
        "threshold": args.threshold,
        "target_gap_seconds": args.target_gap,
        "k_max": args.k_max,
        "selector": "b3_dynamic_density",
        "candidate_sha256": sha256_file(candidates),
        "dev_report_sha256": sha256_file(args.dev_report),
    }
    target = write_json_atomic(_workdir(config) / "gate2" / "freeze.json", decision)
    _publish(config, target, "gate2/freeze.json")
    print(f"DEV decision frozen -> {target}; held-out may now be opened once")


def command_evaluate_heldout(args: argparse.Namespace) -> None:
    """Run frozen B0/B3 on held-out and publish aggregate-only output."""

    config = _config(args)
    _enforce_heldout_grid(
        config,
        "heldout",
        args.freeze,
        args.threshold,
        args.target_gap,
        args.k_max,
    )
    summary = _verify_b3_lineage(
        config,
        cohort="heldout",
        selection_summary=args.selection_summary,
        dedup_report=args.b3_dedup_report,
        candidates=args.b3,
    )
    selected_policy = (
        summary.get("threshold"),
        summary.get("target_gap_seconds"),
        summary.get("k_max"),
    )
    if selected_policy != (args.threshold, args.target_gap, args.k_max):
        raise RuntimeError("held-out selection summary differs from frozen policy")
    if any(
        candidate.config_hash != config.canonical_hash()
        for candidate in _candidate_rows(args.b3)
    ):
        raise RuntimeError("held-out candidates were built by another config")
    queryset = config.resolve(config.dataset.heldout_queryset)
    b0 = _run_variant(
        config,
        "B0",
        queryset,
        args.base_relational,
        args.b3_delta_root,
        None,
        args.repeats,
    )
    b3 = _run_variant(
        config,
        "B3",
        queryset,
        args.b3_relational,
        args.b3_delta_root,
        args.b3,
        args.repeats,
    )
    gate = heldout_gate(
        b0,
        b3,
        by_type_margin=config.evaluation.heldout_by_type_margin,
        max_latency_ratio=config.evaluation.max_latency_ratio,
    )
    report = {
        "split": "heldout",
        "config_hash": config.canonical_hash(),
        "gate": gate,
    }
    target = write_json_atomic(
        _workdir(config) / "gate2" / "heldout" / "aggregate-report.json", report
    )
    _publish(config, target, "gate2/heldout/aggregate-report.json")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", default="config/keyframe_augmentation_runtime_v1.yaml"
    )


def _add_cohort(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cohort", choices=("dev", "heldout"), default="dev")
    parser.add_argument("--freeze", help="required before any held-out access")


def _add_grid(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument("--threshold", type=float, required=required)
    parser.add_argument("--target-gap", type=float, required=required)
    parser.add_argument("--k-max", type=int, required=required)


def _add_input_filter(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", action="append", help="repeat to select locked video ids")
    parser.add_argument("--limit", type=int, help="deterministic smoke-test prefix")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mfe-keyframe-augment", description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="hash canonical videos and mappings")
    inventory.add_argument("--videos-root", required=True)
    inventory.add_argument("--maps-root", required=True)
    inventory.add_argument("--video-glob", default="*.mp4")
    inventory.add_argument("--output", default="data/augmentation/video_inventory.jsonl")
    inventory.set_defaults(handler=command_inventory)

    lock = commands.add_parser("lock", help="create and publish the Gate-0 lock")
    _add_common(lock)
    lock.set_defaults(handler=command_lock)

    sync = commands.add_parser(
        "sync", help="restore durable experiment artifacts from Modal/local transport"
    )
    _add_common(sync)
    sync.add_argument(
        "--prefix",
        default="",
        help="experiment-relative prefix to restore (default: the whole experiment)",
    )
    sync.add_argument("--destination", help="local destination (default: configured workdir)")
    sync.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip remote SHA verification (only for a trusted local transport)",
    )
    sync.set_defaults(handler=command_sync)

    status = commands.add_parser("status", help="summarize local and durable experiment state")
    _add_common(status)
    status.add_argument("--prefix", default="", help="experiment-relative prefix to inspect")
    status.set_defaults(handler=command_status)

    detect = commands.add_parser("detect", help="persist raw TransNet scores and PTS")
    _add_common(detect)
    _add_cohort(detect)
    _add_input_filter(detect)
    detect.add_argument("--video-root", help="rebase locked video filenames on this runner")
    detect.add_argument("--overwrite", action="store_true")
    detect.set_defaults(handler=command_detect)

    audit = commands.add_parser("audit", help="sweep detector and coverage grids")
    _add_common(audit)
    _add_cohort(audit)
    _add_input_filter(audit)
    _add_grid(audit, required=False)
    audit.add_argument("--maps-root", help="rebase locked mapping filenames on this runner")
    audit.set_defaults(handler=command_audit)

    calibrate = commands.add_parser(
        "calibrate", help="choose TransNet threshold from manual DEV labels"
    )
    _add_common(calibrate)
    calibrate.add_argument("--labels", required=True)
    calibrate.set_defaults(handler=command_calibrate)

    select = commands.add_parser("select", help="generate capped B1--B4 allowlists")
    _add_common(select)
    _add_cohort(select)
    _add_input_filter(select)
    _add_grid(select, required=True)
    select.add_argument("--maps-root", help="rebase locked mapping filenames on this runner")
    select.set_defaults(handler=command_select)

    oracle = commands.add_parser("oracle", help="build DEV-only oracle targets")
    _add_common(oracle)
    _add_cohort(oracle)
    _add_input_filter(oracle)
    _add_grid(oracle, required=True)
    oracle.add_argument("--maps-root")
    oracle.set_defaults(handler=command_oracle)

    materialize = commands.add_parser("materialize", help="extract and publish delta images")
    _add_common(materialize)
    _add_cohort(materialize)
    materialize.add_argument("--candidates", required=True)
    materialize.add_argument("--variant", required=True)
    materialize.add_argument("--video-root")
    materialize.add_argument("--maps-root")
    materialize.set_defaults(handler=command_materialize)

    extract = commands.add_parser(
        "extract-features", help="run existing model stages on delta-only images"
    )
    _add_common(extract)
    _add_cohort(extract)
    extract.add_argument("--variant", required=True)
    extract.add_argument("--candidates", required=True)
    extract.add_argument("--stages", default="n4_siglip,n5_ocr_easyocr")
    extract.add_argument("--stage-params", help="JSON object keyed by stage name")
    extract.add_argument("--video-root")
    extract.set_defaults(handler=command_extract_features)

    prepare_dedup = commands.add_parser(
        "prepare-dedup", help="compute BTC/candidate pHash and SigLIP dedup inputs"
    )
    _add_common(prepare_dedup)
    _add_cohort(prepare_dedup)
    prepare_dedup.add_argument("--variant", required=True)
    prepare_dedup.add_argument("--candidates", required=True)
    prepare_dedup.add_argument("--audit", required=True)
    prepare_dedup.add_argument("--materialized", required=True)
    prepare_dedup.add_argument("--btc-keyframes-root", required=True)
    prepare_dedup.add_argument("--base-flat-root", required=True)
    prepare_dedup.add_argument("--delta-flat-root", required=True)
    prepare_dedup.add_argument("--maps-root")
    prepare_dedup.add_argument("--output-dir", required=True)
    prepare_dedup.set_defaults(handler=command_prepare_dedup)

    dedup = commands.add_parser("dedup", help="protected BTC-first pHash/SigLIP dedup")
    _add_common(dedup)
    dedup.add_argument("--cohort", choices=("dev", "heldout"), default="dev")
    dedup.add_argument("--freeze", help="required before held-out dedup")
    dedup.add_argument("--candidates", required=True)
    dedup.add_argument("--btc-shots", required=True)
    dedup.add_argument("--btc-phashes", required=True)
    dedup.add_argument("--btc-embeddings", required=True)
    dedup.add_argument("--candidate-phashes", required=True)
    dedup.add_argument("--candidate-embeddings", required=True)
    dedup.add_argument("--output", required=True)
    dedup.set_defaults(handler=command_dedup)

    match = commands.add_parser(
        "match-control", help="match post-dedup B4 to B3 per-shot quotas"
    )
    _add_common(match)
    match.add_argument("--cohort", choices=("dev", "heldout"), default="dev")
    match.add_argument("--freeze", help="required before held-out matching")
    match.add_argument("--b3", required=True)
    match.add_argument("--b4-pool", required=True)
    match.add_argument("--output", required=True)
    match.set_defaults(handler=command_match_control)

    snapshot = commands.add_parser("snapshot", help="build augmented SQLite snapshot")
    _add_common(snapshot)
    snapshot.add_argument("--cohort", choices=("dev", "heldout"), default="dev")
    snapshot.add_argument("--freeze", help="required before held-out snapshot")
    snapshot.add_argument("--base-db", required=True)
    snapshot.add_argument("--candidates", required=True)
    snapshot.add_argument("--output", required=True)
    snapshot.add_argument("--overwrite", action="store_true")
    snapshot.set_defaults(handler=command_snapshot)

    oracle_eval = commands.add_parser(
        "evaluate-oracle", help="run O1-T retrieval and O1-R force-rank ceilings"
    )
    _add_common(oracle_eval)
    oracle_eval.add_argument("--oracle-delta-root", required=True)
    oracle_eval.add_argument("--base-relational", required=True)
    oracle_eval.add_argument("--oracle-relational", required=True)
    oracle_eval.add_argument("--oracle-candidates", required=True)
    oracle_eval.add_argument("--oracle-targets", required=True)
    oracle_eval.add_argument("--repeats", type=int, default=3)
    oracle_eval.set_defaults(handler=command_evaluate_oracle)

    evaluate = commands.add_parser("evaluate-dev", help="run B0/B3/B4 and DEV Go/No-Go")
    _add_common(evaluate)
    evaluate.add_argument("--b1-delta-root")
    evaluate.add_argument("--b2-delta-root")
    evaluate.add_argument("--b3-delta-root", required=True)
    evaluate.add_argument("--b4-delta-root", required=True)
    evaluate.add_argument("--base-relational", required=True)
    evaluate.add_argument("--b1-relational")
    evaluate.add_argument("--b2-relational")
    evaluate.add_argument("--b3-relational", required=True)
    evaluate.add_argument("--b4-relational", required=True)
    evaluate.add_argument("--b1")
    evaluate.add_argument("--b2")
    evaluate.add_argument("--b3", required=True)
    evaluate.add_argument("--b4", required=True)
    evaluate.add_argument("--oracle-targets", required=True)
    evaluate.add_argument("--selection-summary", required=True)
    evaluate.add_argument("--b3-dedup-report", required=True)
    evaluate.add_argument("--b4-pool-dedup-report", required=True)
    evaluate.add_argument("--b4-match-report", required=True)
    evaluate.add_argument("--repeats", type=int, default=3)
    evaluate.set_defaults(handler=command_evaluate_dev)

    freeze = commands.add_parser("freeze", help="freeze a passing DEV policy")
    _add_common(freeze)
    freeze.add_argument("--dev-report", required=True)
    freeze.add_argument("--b3", required=True)
    _add_grid(freeze, required=True)
    freeze.set_defaults(handler=command_freeze)

    heldout = commands.add_parser("evaluate-heldout", help="aggregate-only held-out gate")
    _add_common(heldout)
    heldout.add_argument("--freeze", required=True)
    heldout.add_argument("--b3-delta-root", required=True)
    heldout.add_argument("--base-relational", required=True)
    heldout.add_argument("--b3-relational", required=True)
    heldout.add_argument("--b3", required=True)
    heldout.add_argument("--selection-summary", required=True)
    heldout.add_argument("--b3-dedup-report", required=True)
    _add_grid(heldout, required=True)
    heldout.add_argument("--repeats", type=int, default=3)
    heldout.set_defaults(handler=command_evaluate_heldout)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args.handler(args)


if __name__ == "__main__":  # pragma: no cover
    main()
