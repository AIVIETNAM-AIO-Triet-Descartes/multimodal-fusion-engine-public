"""Canonical experiment paths and hash-verified transport publication."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from mfe.offline.augmentation.io import canonical_hash, sha256_file
from mfe.offline.transport import Transport

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ExperimentPaths:
    experiment_id: str

    @property
    def root(self) -> str:
        return f"/experiments/keyframe-augmentation/{self.experiment_id}"

    def remote(self, relative: str) -> str:
        clean = str(PurePosixPath(relative))
        if clean.startswith("../") or clean == "..":
            raise ValueError("experiment relative path may not escape its root")
        return f"{self.root}/{clean.lstrip('/')}"

    def marker(self, command: str, artifact_hash: str) -> str:
        return self.remote(f"_status/{command}/{artifact_hash}.done")


class ExperimentStore:
    """Publish only after transport round-trip hash verification."""

    def __init__(self, transport: Transport, experiment_id: str, workdir: str | Path) -> None:
        self.transport = transport
        self.paths = ExperimentPaths(experiment_id)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

    def publish(self, local: str | Path, relative: str) -> str:
        source = Path(local)
        expected = sha256_file(source)
        remote = self.paths.remote(relative)
        self.transport.put(source, remote)
        actual = self.transport.sha256(remote)
        if actual is not None and actual != expected:
            raise RuntimeError(f"transport hash mismatch for {remote}: {actual} != {expected}")
        marker_local = self.workdir / f"marker-{expected}.done"
        marker_local.write_text(
            json.dumps(
                {"relative": str(PurePosixPath(relative)).lstrip("/"), "sha256": expected},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        marker_remote = self.paths.marker(relative.replace("/", "_"), expected)
        self.transport.put(marker_local, marker_remote)
        marker_actual = self.transport.sha256(marker_remote)
        marker_expected = sha256_file(marker_local)
        marker_local.unlink(missing_ok=True)
        if marker_actual is not None and marker_actual != marker_expected:
            raise RuntimeError(
                f"transport marker hash mismatch for {marker_remote}: "
                f"{marker_actual} != {marker_expected}"
            )
        return expected

    def publish_many(self, pairs: list[tuple[str | Path, str]]) -> dict[str, str]:
        """Batch-upload when supported, then verify every stored byte hash."""

        if not pairs:
            return {}
        prepared = [
            (Path(local), self.paths.remote(relative), relative)
            for local, relative in pairs
        ]
        expected = {relative: sha256_file(local) for local, _, relative in prepared}
        if len(expected) != len(prepared):
            raise ValueError("publish_many contains duplicate relative paths")
        put_many = getattr(self.transport, "put_many", None)
        if callable(put_many):
            put_many([(local, remote) for local, remote, _ in prepared])
        else:
            for local, remote, _ in prepared:
                self.transport.put(local, remote)
        for _, remote, relative in prepared:
            actual = self.transport.sha256(remote)
            if actual is not None and actual != expected[relative]:
                raise RuntimeError(
                    f"transport hash mismatch for {remote}: "
                    f"{actual} != {expected[relative]}"
                )
        records = [
            {
                "relative": str(PurePosixPath(relative)).lstrip("/"),
                "sha256": expected[relative],
            }
            for _, _, relative in prepared
        ]
        records.sort(key=lambda record: record["relative"])
        batch_hash = canonical_hash(records)
        marker_local = self.workdir / f"batch-{batch_hash}.done"
        marker_local.write_text(
            json.dumps({"files": records}, sort_keys=True) + "\n", encoding="utf-8"
        )
        marker_remote = self.paths.marker("batch", batch_hash)
        self.transport.put(marker_local, marker_remote)
        marker_actual = self.transport.sha256(marker_remote)
        marker_expected = sha256_file(marker_local)
        marker_local.unlink(missing_ok=True)
        if marker_actual is not None and marker_actual != marker_expected:
            raise RuntimeError(
                f"transport marker hash mismatch for {marker_remote}: "
                f"{marker_actual} != {marker_expected}"
            )
        return expected

    def restore_prefix(
        self,
        relative: str = "",
        destination: str | Path | None = None,
        *,
        verify: bool = True,
    ) -> list[Path]:
        """Restore files below an experiment prefix into the local workdir.

        Kaggle notebooks lose ``/kaggle/working`` when a Version is restarted,
        while the Modal Volume is durable.  This helper is deliberately small:
        it only downloads files that are present remotely, writes through a
        temporary sibling, and optionally verifies the bytes against the
        checksum manifest written at publish time.  Existing local files are
        replaced atomically so a partially downloaded file can never be mistaken for a completed
        artifact by a later CLI stage.

        ``relative`` is always relative to the experiment root; callers cannot
        escape that root.  The returned paths are sorted for deterministic
        notebook summaries.
        """

        clean = str(PurePosixPath(relative)) if relative else ""
        if clean in {".", ".."} or clean.startswith("../"):
            raise ValueError("restore prefix may not escape experiment root")
        target_root = Path(destination) if destination is not None else self.workdir
        remote_prefix = self.paths.remote(clean) if clean else self.paths.root
        files = self.transport.listdir(remote_prefix)

        # A remote hash is not an independent checksum: hashing the bytes just
        # downloaded would merely prove that the download matches itself.  The
        # publish side writes a small, durable checksum manifest in ``_status``;
        # load those manifests first and compare restored artifacts against the
        # expected local hash.  Legacy plain-hash markers are tolerated but do
        # not claim verification because they lack the artifact relative path.
        expected_hashes: dict[str, str] = {}
        status_files = self.transport.listdir(self.paths.remote("_status"))
        marker_root = target_root / ".restore-markers"
        for number, marker in enumerate(status_files):
            marker_tmp = marker_root / f"{number:08d}.tmp"
            try:
                self.transport.get(marker, marker_tmp)
                try:
                    payload = json.loads(marker_tmp.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                records = payload.get("files") if isinstance(payload, dict) else None
                if records is None and isinstance(payload, dict):
                    records = [payload]
                if not isinstance(records, list):
                    continue
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    record_relative = str(record.get("relative", ""))
                    record_hash = str(record.get("sha256", ""))
                    if not record_relative or not _SHA256.fullmatch(record_hash):
                        continue
                    expected_hashes[str(PurePosixPath(record_relative)).lstrip("/")] = record_hash
            finally:
                marker_tmp.unlink(missing_ok=True)
        marker_root.rmdir() if marker_root.is_dir() and not any(marker_root.iterdir()) else None

        restored: list[Path] = []
        for remote in files:
            remote_path = PurePosixPath(remote)
            root_path = PurePosixPath(self.paths.root)
            try:
                relative_path = remote_path.relative_to(root_path)
            except ValueError as error:
                raise RuntimeError(
                    f"transport returned path outside experiment: {remote}"
                ) from error
            if ".." in relative_path.parts:
                raise RuntimeError(f"transport returned an escaping path: {remote}")
            local = target_root.joinpath(*relative_path.parts)
            try:
                local.resolve().relative_to(target_root.resolve())
            except ValueError as error:
                raise RuntimeError(f"restore path escapes destination: {local}") from error
            temporary = local.with_suffix(local.suffix + ".download")
            temporary.parent.mkdir(parents=True, exist_ok=True)
            self.transport.get(remote, temporary)
            if verify:
                relative_key = relative_path.as_posix()
                expected = expected_hashes.get(relative_key)
                if expected is not None and sha256_file(temporary) != expected:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(f"restore hash mismatch for {remote}")
            temporary.replace(local)
            restored.append(local)
        return sorted(restored)
