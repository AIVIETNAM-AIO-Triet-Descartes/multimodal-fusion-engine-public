"""Fail-closed fingerprint gate for workers that share one bundle.

The merge command detects divergences only after every expensive shard has been
written.  Parallel workers use this module to apply the same rules before they
write: numeric environment differences are fatal, provenance-only package
differences warn, and each artifact must use the reference model and params.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mfe.offline.bundle import (
    ARTIFACTS,
    NUMERIC_PACKAGES,
    OTHER_PACKAGES,
    BundleLayout,
    BundleWriter,
    StageStatus,
    freeze_environment,
)


class FingerprintMismatch(RuntimeError):
    """A worker would create a shard that ``merge_bundle.py`` must refuse."""


def _version(environment: dict[str, str], package: str) -> str:
    return environment.get(package) or "(khong cai)"


def _normalise_params(params: dict[str, Any] | None) -> str:
    return json.dumps(params or {}, sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class FingerprintContract:
    """Hard merge fields copied from one completed reference manifest."""

    reference_video: str
    reference_author: str
    environment: dict[str, str]
    artifacts: dict[str, dict[str, Any]]

    @classmethod
    def from_manifest(
        cls,
        manifest: dict[str, Any],
        *,
        reference_video: str,
        reference_author: str,
        bundle: str,
        required_artifacts: tuple[str, ...],
    ) -> FingerprintContract:
        if manifest.get("video_id") != reference_video:
            raise FingerprintMismatch(
                f"fingerprint reference sai video: {manifest.get('video_id')!r} != "
                f"{reference_video!r}"
            )
        if manifest.get("author") != reference_author:
            raise FingerprintMismatch(
                f"fingerprint reference sai author: {manifest.get('author')!r} != "
                f"{reference_author!r}"
            )
        if manifest.get("bundle") != bundle:
            raise FingerprintMismatch(
                f"fingerprint reference sai bundle: {manifest.get('bundle')!r} != {bundle!r}"
            )

        all_artifacts = manifest.get("artifacts") or {}
        missing = [name for name in required_artifacts if name not in all_artifacts]
        if missing:
            raise FingerprintMismatch(
                f"{reference_video} chua co fingerprint {', '.join(missing)}. "
                "Quang phai chay xong CA HAI tang tren video chuan truoc."
            )

        artifacts: dict[str, dict[str, Any]] = {}
        for name, body in all_artifacts.items():
            artifacts[name] = {
                "model": str(body.get("model", "")),
                "params": body.get("params") or {},
            }
        return cls(
            reference_video=reference_video,
            reference_author=reference_author,
            environment=dict(manifest.get("env") or {}),
            artifacts=artifacts,
        )

    def check_environment(self, current: dict[str, str] | None = None) -> list[str]:
        """Apply exactly merge.py's hard/warning package classification."""

        current = freeze_environment() if current is None else current
        hard = []
        for package in NUMERIC_PACKAGES:
            expected = _version(self.environment, package)
            actual = _version(current, package)
            if actual != expected:
                hard.append(f"{package}: Quang={expected}, worker={actual}")
        if hard:
            raise FingerprintMismatch(
                "env_lech - dung truoc khi ghi bundle:\n  " + "\n  ".join(hard)
            )

        warnings = []
        for package in OTHER_PACKAGES:
            expected = _version(self.environment, package)
            actual = _version(current, package)
            if actual != expected:
                warnings.append(
                    f"CANH BAO {package}: Quang={expected}, worker={actual}; van duoc merge"
                )
        return warnings

    def check_artifact(
        self, artifact: str, *, model: str, params: dict[str, Any] | None
    ) -> None:
        """Reject before upload when model/params differ from the reference."""

        expected = self.artifacts.get(artifact)
        if expected is None:
            raise FingerprintMismatch(
                f"model_lech - reference khong khoa artifact {artifact!r}"
            )
        expected_params = _normalise_params(expected.get("params"))
        actual_params = _normalise_params(params)
        if model != expected.get("model") or actual_params != expected_params:
            raise FingerprintMismatch(
                f"model_lech - {artifact}:\n"
                f"  Quang model={expected.get('model')!r} params={expected_params}\n"
                f"  worker model={model!r} params={actual_params}"
            )

    def check_existing_manifest(self, manifest: dict[str, Any]) -> None:
        """Protect fingerprints retained by ``write_part_manifest(merge=True)``.

        OCR2 artifacts are checked at the write boundary and may intentionally
        be replaced during a redo.  All older artifacts retained from notebooks
        02/03/04 must already match Quang; this catches differences such as
        ``window=next_keyframe`` versus ``window=fixed3`` before re-uploading the
        merged manifest.
        """

        for artifact, body in (manifest.get("artifacts") or {}).items():
            if artifact in self.artifacts and artifact not in {
                "paddle_ocr",
                "viet_ocr",
            }:
                self.check_artifact(
                    artifact,
                    model=str(body.get("model", "")),
                    params=body.get("params") or {},
                )


def load_fingerprint_contract(
    transport: Any,
    layout: BundleLayout,
    workdir: str | Path,
    *,
    reference_video: str,
    reference_author: str,
    required_artifacts: tuple[str, ...],
) -> FingerprintContract:
    """Fetch a completed reference manifest from the shared transport."""

    local = Path(workdir) / f"fingerprint-reference-{reference_video}.json"
    try:
        transport.get(layout.part_manifest(reference_video), local)
    except Exception as err:
        raise FingerprintMismatch(
            f"chua co fingerprint chuan {reference_video} cua Quang tren Volume. "
            "Khong duoc chay full/Modal truoc khi Quang hoan tat video chuan."
        ) from err
    try:
        manifest = json.loads(local.read_text(encoding="utf-8"))
    finally:
        local.unlink(missing_ok=True)
    return FingerprintContract.from_manifest(
        manifest,
        reference_video=reference_video,
        reference_author=reference_author,
        bundle=layout.bundle,
        required_artifacts=required_artifacts,
    )


def discover_fingerprint_contract(
    transport: Any,
    layout: BundleLayout,
    workdir: str | Path,
    *,
    reference_author: str,
    candidate_videos: list[str],
    required_artifacts: tuple[str, ...],
) -> FingerprintContract:
    """Use the first reference-author video that is already fully complete.

    This lets a second worker join an ingest that is already running.  It reads
    the author's marker directory once, then fetches only completed manifests;
    the running author does not need to stop, rerun, or publish a new lock.
    """

    required_stages = {ARTIFACTS[name].stage for name in required_artifacts}
    status = StageStatus.load(transport, layout, reference_author)
    completed = [
        video_id
        for video_id in candidate_videos
        if all(status.is_done(stage, video_id) for stage in required_stages)
    ]
    if not completed:
        raise FingerprintMismatch(
            f"{reference_author} chua co video nao xong CA HAI tang "
            f"{', '.join(sorted(required_stages))}. Cho video dau tien xong roi chay lai Step 0d; "
            "khong can dung worker cua Quang."
        )

    rejected: list[str] = []
    for video_id in completed:
        try:
            return load_fingerprint_contract(
                transport,
                layout,
                workdir,
                reference_video=video_id,
                reference_author=reference_author,
                required_artifacts=required_artifacts,
            )
        except FingerprintMismatch as err:
            rejected.append(f"{video_id}: {err}")
    raise FingerprintMismatch(
        "Quang co marker hoan tat nhung cac manifest fingerprint khong dung:\n  "
        + "\n  ".join(rejected)
    )


class FingerprintLockedWriter(BundleWriter):
    """BundleWriter that checks artifact hard fields before uploading a part."""

    def __init__(self, *args: Any, fingerprint: FingerprintContract, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fingerprint = fingerprint
        self._validated_existing_videos: set[str] = set()

    def _validate_video_before_upload(self, video_id: str) -> None:
        if video_id in self._validated_existing_videos:
            return
        remote = self.layout.part_manifest(video_id)
        if self.transport.exists(remote):
            local = self.workdir / f"fingerprint-existing-{video_id}.json"
            try:
                self.transport.get(remote, local)
                manifest = json.loads(local.read_text(encoding="utf-8"))
                self.fingerprint.check_existing_manifest(manifest)
            finally:
                local.unlink(missing_ok=True)
        self._validated_existing_videos.add(video_id)

    def write_text_part(
        self,
        artifact: str,
        video_id: str,
        mapping: dict[str, str],
        *,
        model: str,
        params: dict | None = None,
        facts: dict | None = None,
        elapsed_s: float | None = None,
    ) -> dict:
        # This must precede ``super().write_text_part``: that method uploads the
        # part and marker.  Detecting a retained ``window=fixed3`` afterwards
        # would leave a done marker without the matching manifest entry.
        self._validate_video_before_upload(video_id)
        self.fingerprint.check_artifact(artifact, model=model, params=params)
        return super().write_text_part(
            artifact,
            video_id,
            mapping,
            model=model,
            params=params,
            facts=facts,
            elapsed_s=elapsed_s,
        )

    def _validate_existing_manifest(self, manifest: dict[str, Any]) -> None:
        self.fingerprint.check_existing_manifest(manifest)
