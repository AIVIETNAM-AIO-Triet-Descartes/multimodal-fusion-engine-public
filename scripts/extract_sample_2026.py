"""Copy a small, real AIC 2026 subset without mutating the source dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from mfe.stores.production.audit import audit_sample

REQUIRED_TREES = (
    "videos",
    "keyframes",
    "clip-features",
    "map-keyframes",
    "objects",
    "media-info",
)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_files(source: Path, video_ids: list[str]) -> list[Path]:
    selected: list[Path] = []
    for tree in REQUIRED_TREES:
        root = source / tree
        if not root.is_dir():
            raise FileNotFoundError(f"required AIC source tree is missing: {root}")
        matches = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and any(
                path.stem == video_id or video_id in path.relative_to(root).parts
                for video_id in video_ids
            )
        ]
        if not matches:
            raise FileNotFoundError(f"no files for selected video IDs under {root}")
        selected.extend(matches)
    return sorted(set(selected))


def extract(
    source: Path,
    destination: Path,
    video_ids: list[str],
    *,
    batch: str,
    execute: bool,
) -> dict[str, object]:
    if destination.name != "sample_2026":
        raise ValueError("destination directory must be named exactly 'sample_2026'")
    source_resolved = source.resolve()
    destination_resolved = destination.resolve()
    if (
        source_resolved == destination_resolved
        or source_resolved in destination_resolved.parents
        or destination_resolved in source_resolved.parents
    ):
        raise ValueError("source and destination trees must not contain each other")
    if destination.exists():
        raise FileExistsError("destination already exists; refusing overwrite")
    if not video_ids or len(set(video_ids)) != len(video_ids):
        raise ValueError("video IDs must be non-empty and unique")
    files = select_files(source, video_ids)
    manifest: dict[str, object] = {
        "competition_year": 2026,
        "batch": batch,
        "source_root": str(source.resolve()),
        "selected_video_ids": video_ids,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dry_run": not execute,
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "schema_anomalies": ["PENDING_POST_COPY_AUDIT"],
        "files": [
            {
                "path": path.relative_to(source).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": checksum(path),
            }
            for path in files
        ],
    }
    if execute:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".sample_2026-staging-", dir=destination.parent
        ) as temporary:
            staging = Path(temporary)
            for path in files:
                target = staging / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            audit = audit_sample(staging)
            anomalies: list[str] = []
            if audit["frame_formula_mismatch_count"]:
                anomalies.append(
                    f"frame_formula_mismatch_count={audit['frame_formula_mismatch_count']}"
                )
            if audit["duplicate_frame_count"]:
                anomalies.append(f"duplicate_frame_count={audit['duplicate_frame_count']}")
            manifest["schema_anomalies"] = anomalies
            manifest["audit"] = audit
            (staging / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            staging.replace(destination)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--dest", "--destination", dest="destination", required=True, type=Path
    )
    parser.add_argument(
        "--video-ids", "--video-id", dest="video_ids", action="append", required=True
    )
    parser.add_argument("--batch", required=True)
    parser.add_argument(
        "--confirm-aic-2026-local-use",
        action="store_true",
        help="required to copy; confirms provenance and local-use permission were checked",
    )
    args = parser.parse_args()
    manifest = extract(
        args.source,
        args.destination,
        args.video_ids,
        batch=args.batch,
        execute=args.confirm_aic_2026_local_use,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
