"""Materialize a protected-BTC delta bundle and offline local snapshots."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import yaml

from mfe.contracts.ids import build_keyframe_id, pts_time_to_timestamp_ms
from mfe.contracts.records import (
    AssetRecord,
    AssetType,
    InvertedRecord,
    KeyframeRow,
    VectorRecord,
    VideoRow,
)
from mfe.mapping.relational import RelationalMap
from mfe.offline.keyframe_augmentation.common import (
    atomic_write_json,
    atomic_write_jsonl,
    pts_time_for_frame_idx,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from mfe.offline.keyframe_augmentation.models import (
    AugmentationConfig,
    BtcFrame,
    CandidateFrame,
    ProvenanceRow,
)
from mfe.offline.provided import read_jpeg_size
from mfe.stores.asset.manifest import AssetStore


def _phash(path: Path) -> int:
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot decode image for pHash: {path}")
    resized = cv2.resize(image, (32, 32)).astype(np.float32)
    low = cv2.dct(resized)[:8, :8]
    values = low.flatten()[1:]
    median = float(np.median(values))
    result = 0
    for index, value in enumerate(values):
        if value > median:
            result |= 1 << index
    return result


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _probe_video(path: Path, literal_fps: float) -> VideoRow:
    import cv2  # noqa: PLC0415

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    return VideoRow(
        video_id=path.stem,
        source_path=str(path.resolve()),
        fps=literal_fps,
        duration_ms=round(total_frames / literal_fps * 1000),
        total_frames=total_frames,
        width=width,
        height=height,
    )


def _decode_selected(
    video_path: Path,
    selected: set[int],
    output_dir: Path,
) -> dict[int, tuple[Path, float | None, str]]:
    try:
        import av  # noqa: PLC0415
    except ImportError:
        return _decode_selected_opencv(video_path, selected, output_dir)

    result: dict[int, tuple[Path, float | None, str]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        decoder_detail = {
            "library": f"pyav-{av.__version__}",
            "codec": stream.codec_context.name,
            "time_base": str(stream.time_base),
            "average_rate": str(stream.average_rate),
        }
        decoder_fingerprint = f"decoder:{sha256_json(decoder_detail)}"
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx not in selected:
                if result.keys() >= selected:
                    break
                continue
            target = output_dir / f"frame_{frame_idx:06d}.jpg"
            frame.to_image().save(target, format="JPEG", quality=95)
            decoder_pts = float(frame.pts * stream.time_base) if frame.pts is not None else None
            result[frame_idx] = (target, decoder_pts, decoder_fingerprint)
            if result.keys() >= selected:
                break
    missing = sorted(selected - set(result))
    if missing:
        raise ValueError(f"decoder did not produce requested frame(s): {missing[:10]}")
    return result


def _decode_selected_opencv(
    video_path: Path,
    selected: set[int],
    output_dir: Path,
) -> dict[int, tuple[Path, float | None, str]]:
    """CPU fallback when PyAV is absent; frame identity is the decode ordinal."""
    import cv2  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    result: dict[int, tuple[Path, float | None, str]] = {}
    detail = {
        "library": f"opencv-{cv2.__version__}",
        "backend": capture.getBackendName() if capture.isOpened() else "unavailable",
    }
    fingerprint = f"decoder:{sha256_json(detail)}"
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        frame_idx = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if frame_idx in selected:
                target = output_dir / f"frame_{frame_idx:06d}.jpg"
                if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise ValueError(f"cannot write decoded frame: {target}")
                decoder_pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                result[frame_idx] = (target, decoder_pts, fingerprint)
                if result.keys() >= selected:
                    break
            frame_idx += 1
    finally:
        capture.release()
    missing = sorted(selected - set(result))
    if missing:
        raise ValueError(f"decoder did not produce requested frame(s): {missing[:10]}")
    return result


def _base_image_path(keyframes_root: Path, frame: BtcFrame) -> Path:
    return keyframes_root / frame.video_id / f"{frame.ordinal:03d}.jpg"


def _audit_candidate_coordinate(
    frame_idx: int,
    pts_time: float,
    base_frames: list[BtcFrame],
) -> dict[str, Any]:
    """Validate a decoder coordinate against neighbouring canonical BTC rows."""
    previous = max(
        (row for row in base_frames if row.pts_time <= pts_time),
        key=lambda row: row.pts_time,
        default=None,
    )
    following = min(
        (row for row in base_frames if row.pts_time > pts_time),
        key=lambda row: row.pts_time,
        default=None,
    )
    if previous is not None and frame_idx < previous.frame_idx:
        raise ValueError(f"frame {frame_idx} precedes canonical previous {previous.keyframe_id}")
    if following is not None and frame_idx > following.frame_idx:
        raise ValueError(f"frame {frame_idx} overtakes canonical next {following.keyframe_id}")
    error: float | None = None
    status = "WARNING_OPEN_BRACKET"
    if previous is not None and following is not None:
        gap_seconds = following.pts_time - previous.pts_time
        if gap_seconds > 0:
            ratio = (pts_time - previous.pts_time) / gap_seconds
            expected = previous.frame_idx + ratio * (
                following.frame_idx - previous.frame_idx
            )
            error = abs(frame_idx - expected)
        status = "PASS" if error is None or error <= 1.0 else "WARNING_INTERPOLATION"
    return {
        "mapping_previous_keyframe_id": previous.keyframe_id if previous else None,
        "mapping_next_keyframe_id": following.keyframe_id if following else None,
        "interpolation_frame_error": error,
        "mapping_audit_status": status,
    }


def _load_delta_vectors(path: Path, start_internal_id: int) -> list[VectorRecord]:
    records = []
    for offset, row in enumerate(read_jsonl(path)):
        row["internal_id"] = start_internal_id + offset
        records.append(VectorRecord.model_validate(row))
    return records


def _build_optional_indexes(
    *,
    output: Path,
    base_vector_store: Path,
    base_inverted_store: Path,
    delta_vectors: Path | None,
    delta_inverted: Path | None,
    embed_dim: int,
) -> dict[str, Any]:
    readiness = {"faiss": "UNVERIFIED", "bm25": "UNVERIFIED"}
    vector_target = output / "stores" / "vector"
    inverted_target = output / "stores" / "inverted"

    if delta_vectors is not None:
        if not (base_vector_store / "index.faiss").is_file():
            raise ValueError("delta vectors supplied but the BTC Faiss snapshot is missing")
        from mfe.stores.vector.index import VectorIndex  # noqa: PLC0415

        shutil.copytree(base_vector_store, vector_target)
        vector = VectorIndex.load(str(vector_target), embed_dim)
        vector.add(_load_delta_vectors(delta_vectors, len(vector)))
        vector.save()
        readiness["faiss"] = "PASS"
        readiness["faiss_count"] = len(vector)

    if delta_inverted is not None:
        if not (base_inverted_store / "inverted_records.json").is_file():
            raise ValueError("delta text supplied but the BTC BM25 snapshot is missing")
        from mfe.stores.inverted.index import InvertedIndex  # noqa: PLC0415

        shutil.copytree(base_inverted_store, inverted_target)
        inverted = InvertedIndex.load(str(inverted_target))
        inverted.add([InvertedRecord.model_validate(row) for row in read_jsonl(delta_inverted)])
        inverted.save()
        readiness["bm25"] = "PASS"
        readiness["bm25_count"] = len(inverted)

    readiness["index_ready"] = readiness["faiss"] == readiness["bm25"] == "PASS"
    if not readiness["index_ready"]:
        readiness["reason"] = (
            "delta feature/text artifacts were not supplied; snapshot is fail-closed for Gate 2"
        )
    return readiness


def materialize_bundle(
    *,
    candidates: list[CandidateFrame],
    mappings: dict[str, list[BtcFrame]],
    config: AugmentationConfig,
    core_config_path: str | Path,
    output_dir: str | Path,
    parent_bundle: str,
    run_id: str,
    dataset_hash: str,
    delta_vectors_path: str | Path | None = None,
    delta_inverted_path: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty bundle directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    raw_core = yaml.safe_load(Path(core_config_path).read_text(encoding="utf-8"))
    if not isinstance(raw_core, dict) or not isinstance(raw_core.get("paths"), dict):
        raise ValueError("core config must contain a paths mapping")
    repo_root = Path(core_config_path).resolve().parent.parent

    def resolve(raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else (repo_root / path).resolve()

    paths = raw_core["paths"]
    video_root = resolve(paths["videos"])
    keyframes_root = resolve(paths["keyframes"])
    config_hash = sha256_json(config.model_dump(mode="json"))

    protected = {
        (video_id, frame.frame_idx) for video_id, frames in mappings.items() for frame in frames
    }
    unique_candidates: dict[tuple[str, int], CandidateFrame] = {}
    dropped: list[dict[str, Any]] = []
    for candidate in candidates:
        base_rows = mappings.get(candidate.video_id)
        if not base_rows:
            raise ValueError(f"candidate video {candidate.video_id} is absent from Gate 0")
        if candidate.fps != base_rows[0].fps:
            raise ValueError(
                f"{candidate.keyframe_id}: candidate FPS {candidate.fps} != "
                f"locked literal FPS {base_rows[0].fps}"
            )
        key = (candidate.video_id, candidate.frame_idx)
        if key in protected:
            dropped.append({"candidate": candidate.model_dump(mode="json"), "reason": "btc_exact"})
            continue
        if key in unique_candidates:
            dropped.append(
                {"candidate": candidate.model_dump(mode="json"), "reason": "delta_exact"}
            )
            continue
        unique_candidates[key] = candidate

    decoded: dict[tuple[str, int], tuple[Path, float | None, str]] = {}
    for video_id in sorted({candidate.video_id for candidate in unique_candidates.values()}):
        selected = {
            candidate.frame_idx
            for candidate in unique_candidates.values()
            if candidate.video_id == video_id
        }
        rows = mappings.get(video_id)
        if not rows:
            raise ValueError(f"candidate video {video_id} is absent from the locked mapping")
        video_result = _decode_selected(
            video_root / f"{video_id}.mp4",
            selected,
            output / "assets" / video_id,
        )
        decoded.update({(video_id, frame_idx): value for frame_idx, value in video_result.items()})

    base_hashes: dict[str, tuple[str, Path]] = {}
    base_phashes: dict[str, tuple[int, Path]] = {}
    for _video_id, frames in mappings.items():
        for frame in frames:
            image = _base_image_path(keyframes_root, frame)
            if not image.is_file():
                raise ValueError(f"BTC asset missing: {image}")
            digest = sha256_file(image)
            base_hashes.setdefault(digest, (frame.keyframe_id, image))
            if config.phash_max_distance is not None:
                base_phashes[frame.keyframe_id] = (_phash(image), image)

    kept: list[CandidateFrame] = []
    delta_hashes: dict[str, CandidateFrame] = {}
    delta_phashes: list[tuple[int, CandidateFrame]] = []
    phash_distance_by_key: dict[tuple[str, int], int | None] = {}
    for candidate in sorted(
        unique_candidates.values(), key=lambda row: (row.video_id, row.frame_idx)
    ):
        image, _, _ = decoded[(candidate.video_id, candidate.frame_idx)]
        digest = sha256_file(image)
        if digest in base_hashes:
            dropped.append(
                {
                    "candidate": candidate.model_dump(mode="json"),
                    "reason": "btc_image_sha256",
                    "matched_keyframe_id": base_hashes[digest][0],
                }
            )
            continue
        if digest in delta_hashes:
            dropped.append(
                {
                    "candidate": candidate.model_dump(mode="json"),
                    "reason": "delta_image_sha256",
                    "matched_keyframe_id": delta_hashes[digest].keyframe_id,
                }
            )
            continue
        minimum: int | None = None
        if config.phash_max_distance is not None:
            signature = _phash(image)
            base_matches = [
                (_hamming(signature, base_signature), keyframe_id)
                for keyframe_id, (base_signature, _) in base_phashes.items()
            ]
            delta_matches = [
                (_hamming(signature, delta_signature), row.keyframe_id)
                for delta_signature, row in delta_phashes
            ]
            matches = base_matches + delta_matches
            if matches:
                minimum, matched_id = min(matches)
                if minimum <= config.phash_max_distance:
                    dropped.append(
                        {
                            "candidate": candidate.model_dump(mode="json"),
                            "reason": "btc_phash" if matched_id in base_phashes else "delta_phash",
                            "matched_keyframe_id": matched_id,
                            "phash_distance": minimum,
                        }
                    )
                    continue
            delta_phashes.append((signature, candidate))
        phash_distance_by_key[(candidate.video_id, candidate.frame_idx)] = minimum
        delta_hashes[digest] = candidate
        kept.append(candidate)

    relational_path = output / "stores" / "relational" / "app.db"
    relational_path.parent.mkdir(parents=True, exist_ok=True)
    relational = RelationalMap(str(relational_path))
    relational.init_schema()
    asset_store = AssetStore(str(output / "stores" / "asset"))
    provenance: list[ProvenanceRow] = []
    all_asset_rows: list[AssetRecord] = []
    try:
        by_video_delta: dict[str, dict[int, CandidateFrame]] = {}
        for candidate in kept:
            by_video_delta.setdefault(candidate.video_id, {})[candidate.frame_idx] = candidate
        for video_id, base_frames in sorted(mappings.items()):
            fps = base_frames[0].fps
            video = _probe_video(video_root / f"{video_id}.mp4", fps)
            relational.upsert_video(video)
            combined = sorted(
                {frame.frame_idx for frame in base_frames} | set(by_video_delta.get(video_id, {}))
            )
            base_by_idx = {frame.frame_idx: frame for frame in base_frames}
            keyframe_rows: list[KeyframeRow] = []
            for ordinal, frame_idx in enumerate(combined, 1):
                base = base_by_idx.get(frame_idx)
                candidate = by_video_delta.get(video_id, {}).get(frame_idx)
                if base is not None:
                    pts_time = base.pts_time
                    shot_id = None
                    image = _base_image_path(keyframes_root, base)
                    selector = "btc"
                    origin = "btc"
                    decoder_fingerprint = "btc-provided"
                    decoder_pts_time = None
                    audit_status = "protected"
                    frame_idx_source = "map-keyframes.csv"
                    pts_time_source = "map-keyframes.csv"
                    coordinate_audit = {
                        "mapping_previous_keyframe_id": base.keyframe_id,
                        "mapping_next_keyframe_id": base.keyframe_id,
                        "interpolation_frame_error": 0.0,
                        "mapping_audit_status": "PASS",
                    }
                elif candidate is not None:
                    shot_id = candidate.shot_id
                    image, decoder_pts_time, decoder_fingerprint = decoded[(video_id, frame_idx)]
                    if (
                        decoder_pts_time is not None
                        and math.floor(decoder_pts_time * fps) == frame_idx
                    ):
                        pts_time = decoder_pts_time
                        pts_time_source = "decoder_pts"
                    else:
                        pts_time = pts_time_for_frame_idx(frame_idx, fps)
                        pts_time_source = "synthetic_literal_fps_interval"
                    coordinate_audit = _audit_candidate_coordinate(
                        frame_idx, pts_time, base_frames
                    )
                    selector = candidate.selector
                    origin = "supplemental"
                    audit_status = candidate.audit_status
                    frame_idx_source = "decoder_counter"
                else:  # pragma: no cover - set union construction makes this impossible
                    raise AssertionError("combined frame has no source")
                keyframe_id = build_keyframe_id(video_id, frame_idx)
                keyframe_rows.append(
                    KeyframeRow(
                        keyframe_id=keyframe_id,
                        video_id=video_id,
                        keyframe_index=ordinal,
                        frame_idx=frame_idx,
                        pts_time=pts_time,
                        timestamp_ms=pts_time_to_timestamp_ms(pts_time),
                        shot_id=shot_id,
                    )
                )
                image_bytes = image.read_bytes()
                width, height = read_jpeg_size(image_bytes)
                all_asset_rows.append(
                    AssetRecord(
                        keyframe_id=keyframe_id,
                        asset_type=AssetType.IMAGE,
                        asset_path=str(image.resolve()),
                        width=width,
                        height=height,
                        byte_size=len(image_bytes),
                    )
                )
                provenance.append(
                    ProvenanceRow(
                        keyframe_id=keyframe_id,
                        video_id=video_id,
                        frame_idx=frame_idx,
                        pts_time=pts_time,
                        ordinal=ordinal,
                        origin=origin,
                        selector=selector,
                        shot_id=shot_id,
                        parent_bundle=parent_bundle,
                        run_id=run_id,
                        decoder_fingerprint=decoder_fingerprint,
                        config_hash=config_hash,
                        image_sha256=sha256_file(image),
                        phash_distance=phash_distance_by_key.get((video_id, frame_idx)),
                        embedding_similarity=None,
                        audit_status=audit_status,
                        decoder_pts_time=decoder_pts_time,
                        frame_idx_source=frame_idx_source,
                        pts_time_source=pts_time_source,
                        **coordinate_audit,
                    )
                )
            relational.upsert_keyframes(keyframe_rows)
        problems = relational.verify_mapping()
        if problems:
            raise ValueError(f"augmented RelationalMap contract failed: {problems[:5]}")
    finally:
        relational.close()

    asset_store.add(all_asset_rows)
    asset_store.save()
    atomic_write_jsonl(
        output / "augmentation_manifest.jsonl",
        (row.model_dump(mode="json") for row in provenance),
    )
    atomic_write_jsonl(output / "dedup_dropped.jsonl", dropped)

    readiness = _build_optional_indexes(
        output=output,
        base_vector_store=resolve(paths["vector_store"]),
        base_inverted_store=resolve(paths["inverted_store"]),
        delta_vectors=Path(delta_vectors_path).resolve() if delta_vectors_path else None,
        delta_inverted=Path(delta_inverted_path).resolve() if delta_inverted_path else None,
        embed_dim=int(raw_core["retrieval"]["embed_dim"]),
    )

    snapshot_config = json.loads(json.dumps(raw_core))
    snapshot_paths = snapshot_config["paths"]
    snapshot_paths["relational_db"] = str(relational_path)
    snapshot_paths["asset_store"] = str(output / "stores" / "asset")
    snapshot_paths["vector_store"] = str(output / "stores" / "vector")
    snapshot_paths["inverted_store"] = str(output / "stores" / "inverted")
    snapshot_config_path = output / "snapshot-config.yaml"
    snapshot_config_path.write_text(
        yaml.safe_dump(snapshot_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    supplemental = [row for row in provenance if row.origin == "supplemental"]
    manifest = {
        "schema_version": 1,
        "bundle_id": output.name,
        "parent_bundle": parent_bundle,
        "run_id": run_id,
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "btc_base_immutable": True,
        "origin_counts": {
            "btc": len(provenance) - len(supplemental),
            "supplemental": len(supplemental),
        },
        "dedup_dropped": len(dropped),
        "guardrail_ratio": config.guardrail_ratio,
        "guardrail_pass": len(supplemental)
        <= math.floor((len(provenance) - len(supplemental)) * config.guardrail_ratio),
        "stores": {
            "relational": str(relational_path),
            "asset": str(output / "stores" / "asset"),
            "vector": str(output / "stores" / "vector"),
            "inverted": str(output / "stores" / "inverted"),
        },
        "artifact_hashes": {
            "relational_db": sha256_file(relational_path),
            "asset_manifest": sha256_file(output / "stores" / "asset" / "manifest.json"),
            "augmentation_manifest": sha256_file(output / "augmentation_manifest.jsonl"),
            "dedup_dropped": sha256_file(output / "dedup_dropped.jsonl"),
            "snapshot_config": sha256_file(snapshot_config_path),
        },
        "readiness": readiness,
        "rollback": {
            "base_config": str(Path(core_config_path).resolve()),
            "augmented_config": str(snapshot_config_path),
            "method": "atomic snapshot pointer switch; no live row deletion",
        },
        "evidence": {
            "bundle_contract": "MEASURED",
            "dedup": "MEASURED",
            "gate2_score": "UNVERIFIED",
        },
    }
    manifest["bundle_hash"] = sha256_json(manifest)
    atomic_write_json(output / "bundle-manifest.json", manifest)
    return manifest


def activate_snapshot(pointer_path: str | Path, config_path: str | Path) -> dict[str, Any]:
    pointer = Path(pointer_path).resolve()
    config = Path(config_path).resolve()
    if not config.is_file():
        raise ValueError(f"snapshot config does not exist: {config}")
    previous = None
    history: list[str] = []
    if pointer.is_file():
        current = json.loads(pointer.read_text(encoding="utf-8"))
        previous = current.get("active_config")
        history = list(current.get("history", []))
    if previous and previous != str(config):
        history.append(previous)
    payload = {
        "active_config": str(config),
        "previous_config": previous,
        "history": history,
        "pointer_hash": sha256_json({"active_config": str(config), "history": history}),
    }
    atomic_write_json(pointer, payload)
    return payload


def rollback_snapshot(pointer_path: str | Path) -> dict[str, Any]:
    pointer = Path(pointer_path).resolve()
    if not pointer.is_file():
        raise ValueError(f"snapshot pointer does not exist: {pointer}")
    current = json.loads(pointer.read_text(encoding="utf-8"))
    history = list(current.get("history", []))
    previous = current.get("previous_config") or (history[-1] if history else None)
    if not previous:
        raise ValueError("snapshot pointer has no rollback target")
    if history and history[-1] == previous:
        history.pop()
    payload = {
        "active_config": previous,
        "previous_config": history[-1] if history else None,
        "history": history,
        "pointer_hash": sha256_json({"active_config": previous, "history": history}),
    }
    atomic_write_json(pointer, payload)
    return payload
