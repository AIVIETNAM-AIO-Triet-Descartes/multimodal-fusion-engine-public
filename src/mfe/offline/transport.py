"""Getting bytes onto a Modal Volume — and a local stand-in for testing.

    docs/triet/ke-hoach-offline-ingest.md §5.4, §5.5, §6.5

Two implementations behind one protocol, and the second one is not a luxury:
`LocalDirTransport` lets the whole notebook pipeline run end to end on a laptop
against the 16 sample videos, so the parts that can be wrong — path layout,
resume logic, key validation, the merge — get exercised without spending a Kaggle
session or a Modal quota to find out. Only the model calls actually need Kaggle.

**What Modal gives us and what it does not.** Volume writes are last-write-wins on
a shared path, so §6.2 arranges for two members never to share one; Modal
recommends staying under 5 concurrent commits, so uploads here are sequential on
purpose and the notebook must not thread them. And whether `modal volume put`
validates a checksum is **not something we verified**, so `sha256()` exists and
`BundleWriter` reads the hash back itself rather than trusting the transfer.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_SHA_CHUNK = 1 << 20


@runtime_checkable
class Transport(Protocol):
    """The four operations the bundle protocol needs. Nothing more."""

    def put(self, local: Path, remote: str) -> None: ...

    def get(self, remote: str, local: Path) -> None: ...

    def listdir(self, prefix: str) -> list[str]:
        """Every file under `prefix`, recursively. Empty list if absent."""
        ...

    def sha256(self, remote: str) -> str | None:
        """Hash of the stored bytes, or `None` if this transport cannot say."""
        ...

    def exists(self, remote: str) -> bool:
        """Is there a **file** at this exact path?

        Separate from `listdir`, which answers a question about a *directory* and
        returns nothing for a file path — a difference that reads as "not there"
        and is wrong in the one way that matters.
        """
        ...


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Read `KEY=value` lines into a dict. No dependency, no interpolation.

    Values are **not** logged anywhere in this module, and `.env` is gitignored.
    A Modal token is a workspace-wide credential; a Drive refresh token does not
    expire at all.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def ensure_modal_credentials(dotenv_path: str | Path = ".env") -> str:
    """Put `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` into the environment.

    Three sources, in order — already-set environment, Kaggle Secrets, then
    `.env`. Environment first so a session can override without editing a file;
    Kaggle Secrets before `.env` because on Kaggle there is no `.env` and the
    secret store is the right place for a credential.

    Returns which source won, for the notebook to print. The *values* are never
    returned or logged.
    """
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return "environment"

    try:
        from kaggle_secrets import UserSecretsClient  # noqa: PLC0415

        secrets = UserSecretsClient()
        os.environ["MODAL_TOKEN_ID"] = secrets.get_secret("MODAL_TOKEN_ID")
        os.environ["MODAL_TOKEN_SECRET"] = secrets.get_secret("MODAL_TOKEN_SECRET")
        return "kaggle-secrets"
    except Exception:  # noqa: BLE001 - not on Kaggle, or the secret is not set
        pass

    values = load_dotenv(dotenv_path)
    missing = [k for k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET") if k not in values]
    if missing:
        raise RuntimeError(
            f"thieu {', '.join(missing)}. Dat trong Kaggle Secrets (khi chay tren "
            f"Kaggle) hoac trong {dotenv_path} (khi chay may minh). "
            "KHONG viet token vao o notebook."
        )
    os.environ["MODAL_TOKEN_ID"] = values["MODAL_TOKEN_ID"]
    os.environ["MODAL_TOKEN_SECRET"] = values["MODAL_TOKEN_SECRET"]
    return str(dotenv_path)


# ---------------------------------------------------------------------------
# Local — the one used by tests and by a dry run
# ---------------------------------------------------------------------------


class LocalDirTransport:
    """A directory pretending to be a Volume.

    Deliberately *not* a mock: it does real filesystem writes with the same path
    strings, so a layout bug or a resume bug shows up here exactly as it would on
    Modal. What it cannot reproduce is the network — partial uploads, and the
    last-write-wins race — which is why those are designed away in §6.2 rather
    than tested for.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, remote: str) -> Path:
        return self.root / remote.lstrip("/")

    def put(self, local: Path, remote: str) -> None:
        target = self._resolve(remote)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, target)

    def get(self, remote: str, local: Path) -> None:
        source = self._resolve(remote)
        if not source.is_file():
            raise FileNotFoundError(remote)
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local)

    def listdir(self, prefix: str) -> list[str]:
        base = self._resolve(prefix)
        if not base.is_dir():
            return []
        return sorted(
            "/" + path.relative_to(self.root).as_posix()
            for path in base.rglob("*")
            if path.is_file()
        )

    def exists(self, remote: str) -> bool:
        return self._resolve(remote).is_file()

    def sha256(self, remote: str) -> str | None:
        path = self._resolve(remote)
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(_SHA_CHUNK):
                digest.update(chunk)
        return digest.hexdigest()


# ---------------------------------------------------------------------------
# Modal
# ---------------------------------------------------------------------------


class ModalVolumeTransport:
    """A Modal Volume, driven from a client process (Kaggle, or a laptop).

    `read_file`, `listdir`, `batch_upload` and `remove_file` are all documented as
    usable from outside a Modal App, which is exactly our position: the notebook
    is a client, nothing runs *on* Modal here. No `commit()`/`reload()` either —
    those belong to code running inside a container with the volume mounted.
    """

    def __init__(
        self,
        volume_name: str,
        *,
        create_if_missing: bool = False,
        version: int | None = None,
        dotenv_path: str | Path = ".env",
    ) -> None:
        """`version=2` applies **only when the volume is created**.

        Modal fixes the version at creation and offers no in-place upgrade — the
        docs say plainly that there is no automated tool for it. So passing 2 at a
        name that already exists as v1 changes nothing; switching means a new
        volume and moving the data.

        Which version matters here for one measured reason: v1 has a hard limit of
        500,000 inodes, and the DAKE keyframe set for batch 1 lands near 648,000
        (§10.3). v2 removes the total-file limit and caps only per-directory at
        262,144 — our biggest directory is one video's keyframes, ~1,400 files.
        """
        self.credential_source = ensure_modal_credentials(dotenv_path)
        import modal  # noqa: PLC0415 - heavy, and only needed on the real path

        self._modal = modal
        self.volume_name = volume_name
        kwargs = {"create_if_missing": create_if_missing}
        if version is not None:
            kwargs["version"] = version
        self.volume = modal.Volume.from_name(volume_name, **kwargs)
        logger.info(
            "Modal volume %r, token tu %s", volume_name, self.credential_source
        )

    def put(self, local: Path, remote: str) -> None:
        # force=True because re-running a failed unit of work MUST be able to
        # overwrite the half-written file it left behind. Without it, resume
        # fails on exactly the file that needs replacing.
        with self.volume.batch_upload(force=True) as batch:
            batch.put_file(str(local), remote)

    def put_many(self, pairs: list[tuple[Path, str]]) -> None:
        """One batch for many files — the right call for a keyframe directory.

        A batch per file would be thousands of commits, and Modal asks for no
        more than ~5 concurrent; batching keeps it to one.
        """
        if not pairs:
            return
        with self.volume.batch_upload(force=True) as batch:
            for local, remote in pairs:
                batch.put_file(str(local), remote)

    def get(self, remote: str, local: Path) -> None:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        with Path(local).open("wb") as handle:
            for chunk in self.volume.read_file(remote.lstrip("/")):
                handle.write(chunk)

    def listdir(self, prefix: str) -> list[str]:
        try:
            entries = self.volume.listdir(prefix.lstrip("/") or "/", recursive=True)
        except Exception as err:  # noqa: BLE001 - absent prefix is a normal state
            logger.debug("listdir(%s): %s", prefix, err)
            return []
        out: list[str] = []
        for entry in entries:
            path = getattr(entry, "path", None) or str(entry)
            # FileEntry carries a type; directories must not look like markers.
            kind = getattr(entry, "type", None)
            if kind is not None and str(kind).lower().endswith("directory"):
                continue
            out.append("/" + str(path).lstrip("/"))
        return sorted(out)

    def exists(self, remote: str) -> bool:
        try:
            for _chunk in self.volume.read_file(remote.lstrip("/")):
                return True
        except Exception:  # noqa: BLE001 - absent is a normal state, not an error
            return False
        return True   # an empty file is still a file

    def sha256(self, remote: str) -> str | None:
        try:
            digest = hashlib.sha256()
            for chunk in self.volume.read_file(remote.lstrip("/")):
                digest.update(chunk)
            return digest.hexdigest()
        except Exception as err:  # noqa: BLE001
            logger.warning("khong doc lai duoc %s de doi chieu hash: %s", remote, err)
            return None


def build_transport(
    kind: str,
    *,
    volume: str = "aic2026",
    local_root: str = "bundle_local",
    version: int | None = None,
) -> Transport:
    """`kind` is `"modal"` or `"local"`. The notebook takes this from one variable."""
    if kind == "modal":
        return ModalVolumeTransport(volume, create_if_missing=True, version=version)
    if kind == "local":
        return LocalDirTransport(local_root)
    raise ValueError(f"transport khong biet: {kind!r}; dung 'modal' hoac 'local'")
