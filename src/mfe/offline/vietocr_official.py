"""Bootstrap the official VietOCR config and checkpoint for Kaggle producers.

VietOCR 0.3.x downloads its named config and checkpoint from ``vocr.vn`` at
runtime.  That host currently presents an expired TLS certificate in Kaggle,
so the normal config loader fails before the first frame is processed.  This
module keeps the provenance explicit: configs come from the official VietOCR
repository and the checkpoint is fetched only from the official ``vocr.vn``
host, then accepted only when its pinned size and SHA-256 match.

The certificate fallback is deliberately narrow.  It is used only for the
exact official checkpoint URL after normal TLS verification fails; no global
SSL settings are changed and no mirror is accepted.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

VIETOCR_CONFIG_NAME = "vgg_transformer"
VIETOCR_CONFIG_BASE_URL = (
    "https://raw.githubusercontent.com/pbcquoc/vietocr/master/config"
)
VIETOCR_WEIGHTS_URL = "https://vocr.vn/data/vietocr/vgg_transformer.pth"
VIETOCR_WEIGHTS_BYTES = 151_815_373
VIETOCR_WEIGHTS_SHA256 = (
    "380512193a8b6cbf6fad80deacdc9b6939d10d473d199892fc6408d13775ea59"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_request(url: str) -> urllib.request.Request:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"raw.githubusercontent.com", "vocr.vn"}:
        raise ValueError(f"Unexpected VietOCR source host: {url}")
    return urllib.request.Request(
        url,
        headers={"User-Agent": "mfe-kaggle-vietocr-official/1"},
    )


def _read_text(url: str) -> str:
    with urllib.request.urlopen(_official_request(url), timeout=120) as response:
        return response.read().decode("utf-8")


def _download_checkpoint(target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == VIETOCR_WEIGHTS_BYTES:
        digest = _sha256(target)
        if digest == VIETOCR_WEIGHTS_SHA256:
            return {
                "status": "CACHE_HIT",
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": digest,
                "url": VIETOCR_WEIGHTS_URL,
            }

    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)
    request = _official_request(VIETOCR_WEIGHTS_URL)
    used_certificate_fallback = False
    try:
        try:
            response = urllib.request.urlopen(request, timeout=300)
        except urllib.error.URLError as error:
            if not isinstance(error.reason, ssl.SSLCertVerificationError):
                raise
            # vocr.vn's certificate is expired in the Kaggle image.  Keep the
            # exception local to this exact official URL and verify the bytes.
            response = urllib.request.urlopen(
                request,
                timeout=300,
                context=ssl._create_unverified_context(),  # noqa: S323
            )
            used_certificate_fallback = True
        with response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        actual_size = partial.stat().st_size
        actual_sha256 = _sha256(partial)
        if actual_size != VIETOCR_WEIGHTS_BYTES or actual_sha256 != VIETOCR_WEIGHTS_SHA256:
            raise RuntimeError(
                "Official VietOCR checkpoint size/hash mismatch: "
                f"bytes={actual_size}, sha256={actual_sha256}"
            )
        partial.replace(target)
        return {
            "status": "DOWNLOADED",
            "path": str(target),
            "bytes": actual_size,
            "sha256": actual_sha256,
            "url": VIETOCR_WEIGHTS_URL,
            "certificate_fallback": used_certificate_fallback,
        }
    finally:
        partial.unlink(missing_ok=True)


def prepare_official_vietocr(workdir: Path) -> dict[str, Any]:
    """Materialize the official local VietOCR config and checkpoint."""

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    weights = workdir / "vgg_transformer.pth"
    weights_report = _download_checkpoint(weights)

    base = yaml.safe_load(
        _read_text(f"{VIETOCR_CONFIG_BASE_URL}/base.yml")
    )
    model = yaml.safe_load(
        _read_text(f"{VIETOCR_CONFIG_BASE_URL}/vgg-transformer.yml")
    )
    if not isinstance(base, dict) or not isinstance(model, dict):
        raise RuntimeError("Official VietOCR config is not a mapping")
    config = {**base, **model}
    config["pretrain"] = str(weights)
    config["weights"] = str(weights)
    config_path = workdir / "vgg-transformer-local.yml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    config_sha256 = _sha256(config_path)
    return {
        "status": "PASS",
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "config_sources": [
            f"{VIETOCR_CONFIG_BASE_URL}/base.yml",
            f"{VIETOCR_CONFIG_BASE_URL}/vgg-transformer.yml",
        ],
        "weights": weights_report,
    }


def install_local_vietocr_config(config_path: Path) -> None:
    """Make VietOCR's named loader use the verified local official config."""

    from vietocr.tool.config import Cfg  # noqa: PLC0415

    original_loader = Cfg.load_config_from_name

    def load_config_from_name(name: str):
        if name == VIETOCR_CONFIG_NAME:
            return Cfg.load_config_from_file(str(config_path))
        return original_loader(name)

    Cfg.load_config_from_name = staticmethod(load_config_from_name)


def bootstrap_and_install(workdir: Path) -> dict[str, Any]:
    """Prepare the checkpoint and install the loader patch in one call."""

    report = prepare_official_vietocr(workdir)
    install_local_vietocr_config(Path(report["config_path"]))
    return json.loads(json.dumps(report))
