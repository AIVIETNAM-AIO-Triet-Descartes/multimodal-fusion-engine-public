#!/usr/bin/env python3
"""Download the current MFE Modal volumes to a local /data mount.

The Modal CLI already supports recursive downloads, but this downloader keeps
completed files, limits RPC concurrency, retries individual files, and reports
aggregate progress.  Those properties matter for the several hundred thousand
small files in the batch-1 and batch-2 bundles.

Example:

    .venv/bin/python scripts/download_modal_volumes.py \
        --destination /data/multimodal-fusion-engine/modal \
        --workers 64

The default volume set includes the current batch-1 and batch-2 bundles plus
the older batch-2 and research volumes, preserving each Modal volume name in
the local directory tree.  Use ``--volumes`` to select a smaller set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfe.offline.transport import load_dotenv  # noqa: E402,I001


DEFAULT_VOLUMES = (
    "keyframes-btc",
    "keyframes-batch-2-v2",
    "keyframes-batch-2",
    "keyframe-augmentation-research",
)


@dataclass
class Progress:
    total_files: int = 0
    total_bytes: int = 0
    done_files: int = 0
    done_bytes: int = 0
    skipped_files: int = 0
    network_bytes: int = 0
    failed_files: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/data/multimodal-fusion-engine/modal"),
        help="Local root; each volume is stored below this directory.",
    )
    parser.add_argument(
        "--volumes",
        nargs="+",
        default=list(DEFAULT_VOLUMES),
        help="Modal volume names to download.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=128,
        help="Concurrent file downloads. 128 is tuned for the mounted /data host.",
    )
    parser.add_argument(
        "--block-workers",
        type=int,
        default=256,
        help="Concurrent block downloads shared by all files.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Retries for one file after a transient read failure.",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        default=Path(".env"),
        help="File containing MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download at most this many files per volume; useful for a smoke test.",
    )
    return parser.parse_args()


def configure_credentials(dotenv: Path) -> None:
    values = load_dotenv(dotenv)
    missing = [
        key
        for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
        if not os.environ.get(key) and not values.get(key)
    ]
    if missing:
        raise SystemExit(f"Missing Modal credential(s): {', '.join(missing)}")
    for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        os.environ.setdefault(key, values[key])


def safe_local_path(root: Path, remote_path: str) -> Path:
    relative = PurePosixPath(remote_path.lstrip("/"))
    if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"Unsafe Modal path: {remote_path!r}")
    return root.joinpath(*relative.parts)


def configure_logging(destination: Path) -> logging.Logger:
    destination.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mfe.modal_download")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(destination / "download.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


async def list_files(volume, *, limit: int | None = None):
    from modal.types import FileEntryType

    entries = []
    # The public synchronous facade exposes the underlying async callable as
    # ``_aio_func``.  This is the same path used by Modal's recursive CLI
    # downloader and lets all file reads share one event loop.
    async for entry in volume.iterdir._aio_func(volume, "/", recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        entries.append(entry)
        if limit is not None and len(entries) >= limit:
            break
    return entries


async def download_one(
    volume,
    entry,
    local: Path,
    progress: Progress,
    file_sem: asyncio.Semaphore,
    block_sem: asyncio.Semaphore,
    rpc_sem: asyncio.Semaphore,
    retries: int,
    logger: logging.Logger,
) -> None:
    async with file_sem:
        if local.is_file() and local.stat().st_size == entry.size:
            progress.skipped_files += 1
            progress.done_files += 1
            progress.done_bytes += entry.size
            return

        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(local.name + ".part")
        last_error: Exception | None = None

        for attempt in range(1, retries + 1):
            partial.unlink(missing_ok=True)
            try:
                with partial.open("wb") as handle:
                    def progress_cb(*, advance=0, **_kwargs):
                        progress.network_bytes += int(advance or 0)

                    received = await volume._read_file_into_fileobj._aio_func(
                        volume,
                        path=entry.path,
                        fileobj=handle,
                        download_semaphore=block_sem,
                        rpc_semaphore=rpc_sem,
                        progress_cb=progress_cb,
                    )
                if received != entry.size or partial.stat().st_size != entry.size:
                    raise OSError(
                        f"size mismatch: expected {entry.size}, received {received}, "
                        f"written {partial.stat().st_size}"
                    )
                os.replace(partial, local)
                progress.done_files += 1
                progress.done_bytes += entry.size
                return
            except Exception as error:  # noqa: BLE001 - retry the remote read
                last_error = error
                logger.warning(
                    "retry %s/%s %s: %s", attempt, retries, entry.path, error
                )
                if attempt < retries:
                    await asyncio.sleep(min(30.0, 2.0**attempt))

        progress.failed_files += 1
        raise RuntimeError(
            f"download failed after {retries} attempts: {entry.path}"
        ) from last_error


async def report_progress(
    volume_name: str,
    progress: Progress,
    started: float,
    finished: asyncio.Event,
) -> None:
    while not finished.is_set():
        elapsed = max(time.monotonic() - started, 0.001)
        rate = progress.network_bytes / elapsed / 2**20
        percent = (
            100.0 * progress.done_bytes / progress.total_bytes
            if progress.total_bytes
            else 100.0
        )
        print(
            f"[{volume_name}] {percent:6.2f}%  files {progress.done_files:,}/"
            f"{progress.total_files:,}  data {progress.done_bytes / 2**30:,.2f}/"
            f"{progress.total_bytes / 2**30:,.2f} GiB  network {rate:,.1f} MiB/s  "
            f"skipped {progress.skipped_files:,}  failed {progress.failed_files:,}",
            flush=True,
        )
        await asyncio.sleep(5)


async def download_volume(
    volume_name: str,
    destination: Path,
    *,
    workers: int,
    block_workers: int,
    retries: int,
    limit: int | None,
    logger: logging.Logger,
) -> None:
    import modal

    volume = modal.Volume.from_name(volume_name)
    print(f"[{volume_name}] listing remote files...", flush=True)
    entries = await list_files(volume, limit=limit)
    progress = Progress(
        total_files=len(entries),
        total_bytes=sum(entry.size for entry in entries),
    )
    volume_root = destination / volume_name
    volume_root.mkdir(parents=True, exist_ok=True)
    manifest = volume_root / ".modal-manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(
                json.dumps(
                    {"path": entry.path, "size": entry.size, "mtime": entry.mtime},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"[{volume_name}] {progress.total_files:,} files, "
        f"{progress.total_bytes / 2**30:,.2f} GiB -> {volume_root}",
        flush=True,
    )
    logger.info(
        "start volume=%s files=%s bytes=%s workers=%s block_workers=%s",
        volume_name,
        progress.total_files,
        progress.total_bytes,
        workers,
        block_workers,
    )

    file_sem = asyncio.Semaphore(workers)
    block_sem = asyncio.Semaphore(block_workers)
    rpc_sem = asyncio.Semaphore(min(128, workers))
    queue: asyncio.Queue = asyncio.Queue(maxsize=max(2 * workers, 1))
    failures: list[Exception] = []

    async def worker() -> None:
        while True:
            entry = await queue.get()
            if entry is None:
                queue.task_done()
                return
            try:
                await download_one(
                    volume,
                    entry,
                    safe_local_path(volume_root, entry.path),
                    progress,
                    file_sem,
                    block_sem,
                    rpc_sem,
                    retries,
                    logger,
                )
            except Exception as error:  # keep every failure visible; let cancellation through
                failures.append(error)
                print(f"[{volume_name}] ERROR {error}", file=sys.stderr, flush=True)
            finally:
                queue.task_done()

    tasks = [asyncio.create_task(worker()) for _ in range(workers)]
    finished = asyncio.Event()
    reporter = asyncio.create_task(
        report_progress(volume_name, progress, time.monotonic(), finished)
    )
    for entry in entries:
        await queue.put(entry)
    for _ in tasks:
        await queue.put(None)
    await queue.join()
    await asyncio.gather(*tasks)
    finished.set()
    await reporter

    logger.info(
        "finish volume=%s files=%s bytes=%s skipped=%s failures=%s",
        volume_name,
        progress.done_files,
        progress.done_bytes,
        progress.skipped_files,
        len(failures),
    )
    if failures:
        raise RuntimeError(f"{volume_name}: {len(failures)} file(s) failed")
    print(
        f"[{volume_name}] COMPLETE files={progress.done_files:,} "
        f"bytes={progress.done_bytes / 2**30:,.2f} GiB "
        f"skipped={progress.skipped_files:,}",
        flush=True,
    )


async def main_async(args: argparse.Namespace) -> None:
    configure_credentials(args.dotenv)
    args.destination.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.destination)
    print(
        f"Modal download root: {args.destination}\n"
        f"Volumes: {', '.join(args.volumes)}\n"
        f"workers={args.workers}, block_workers={args.block_workers}, retries={args.retries}",
        flush=True,
    )
    for volume_name in args.volumes:
        await download_volume(
            volume_name,
            args.destination,
            workers=args.workers,
            block_workers=args.block_workers,
            retries=args.retries,
            limit=args.limit,
            logger=logger,
        )


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.block_workers < 1 or args.retries < 1:
        raise SystemExit("workers, block-workers and retries must be positive")
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted; completed files are kept and .part files can be retried.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
