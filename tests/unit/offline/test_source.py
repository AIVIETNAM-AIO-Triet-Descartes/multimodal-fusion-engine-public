"""Locks the per-video fetch — mfe.offline.source.

Builds a real zip with STORED entries, serves it from the filesystem, and checks
that the bytes recovered by offset are byte-identical to what went in. The point
is the offset arithmetic: the central directory records where an entry *starts*,
but a variable-length local file header sits in front of the payload, and getting
that wrong prepends header bytes to an mp4 — which players tolerate and decoders
count differently, i.e. a wrong keyframe set rather than an error."""

from __future__ import annotations

import zipfile
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from mfe.offline import source
from mfe.offline.source import VideoEntry, fetch_video, load_catalogue


@pytest.fixture
def archive(tmp_path):
    """A zip with three STORED entries and known contents."""
    payloads = {
        "L21_V001": b"first-video-" + bytes(range(256)) * 40,
        "L21_V002": b"second" + bytes(range(200)) * 10,
        "L21_V003": bytes(range(256)) * 100,
    }
    path = tmp_path / "Videos_L21_a.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as handle:
        for name, blob in payloads.items():
            handle.writestr(f"video/{name}.mp4", blob)

    entries = {}
    with zipfile.ZipFile(path) as handle:
        for info in handle.infolist():
            stem = info.filename.split("/")[-1].removesuffix(".mp4")
            entries[stem] = VideoEntry(
                stem, path.name, path.as_uri(), info.header_offset, info.file_size
            )
    return payloads, entries


class TestFetchVideo:
    def test_recovers_the_exact_bytes(self, archive, tmp_path):
        payloads, entries = archive
        for name, blob in payloads.items():
            out = fetch_video(entries[name], tmp_path / f"{name}.mp4")
            assert out.read_bytes() == blob

    def test_skips_the_local_file_header(self, archive, tmp_path):
        """Reading from `header_offset` directly would prepend ~40 bytes of zip
        header. The recovered payload must start where the payload starts."""
        payloads, entries = archive
        out = fetch_video(entries["L21_V001"], tmp_path / "a.mp4")
        assert out.read_bytes().startswith(b"first-video-")

    def test_deletes_a_short_read_instead_of_keeping_it(self, archive, tmp_path):
        """A truncated video is not a smaller video: the decoder just produces
        fewer frames, so the keyframe set comes out quietly wrong. Better to have
        no file and re-run than a plausible one.

        Note what this check can and cannot do. It catches a transfer that ended
        early. It does **not** catch a wrong length in the middle of an archive —
        over-reading there simply grabs the next entry's bytes and the length comes
        out right. Wrong *offsets* are caught by the header magic instead, which is
        why both checks exist.
        """
        _, entries = archive
        # The last entry, so over-reading runs off the end of the file rather than
        # into the next entry.
        last = max(entries.values(), key=lambda e: e.header_offset)
        lying = VideoEntry(last.video_id, last.zip_name, last.url,
                           last.header_offset, last.size + 10_000)
        dest = tmp_path / "short.mp4"
        with pytest.raises(RuntimeError, match="cat cut"):
            fetch_video(lying, dest)
        assert not dest.exists()

    def test_rejects_an_offset_that_is_not_a_header(self, archive, tmp_path):
        _, entries = archive
        good = entries["L21_V003"]
        wrong = VideoEntry(good.video_id, good.zip_name, good.url,
                           good.header_offset + 7, good.size)
        with pytest.raises(RuntimeError, match="local file header"):
            fetch_video(wrong, tmp_path / "x.mp4")

    def test_rejects_server_that_ignores_range_before_keeping_archive(
        self, monkeypatch, tmp_path
    ):
        dest = tmp_path / "whole-archive.zip"
        observed = {}

        def fake_run(command, **_kwargs):
            observed["command"] = command
            output = Path(command[command.index("-o") + 1])
            output.write_bytes(b"x" * 10)
            return CompletedProcess(command, 0, stdout=b"200", stderr=b"")

        monkeypatch.setattr(source.subprocess, "run", fake_run)
        monkeypatch.setattr(source.time, "sleep", lambda _seconds: None)
        with pytest.raises(RuntimeError, match="bo qua HTTP Range"):
            source._curl_range("https://example.invalid/archive.zip", 10, 19, dest)

        assert not dest.exists()
        assert observed["command"][observed["command"].index("--max-filesize") + 1] == "10"

    def test_retries_an_intermittent_ignored_range(self, monkeypatch, tmp_path):
        dest = tmp_path / "video.mp4"
        attempts = []

        def fake_run(command, **_kwargs):
            attempts.append(command)
            output = Path(command[command.index("-o") + 1])
            output.write_bytes(b"x" * 10)
            status = b"200" if len(attempts) == 1 else b"206"
            return CompletedProcess(command, 0, stdout=status, stderr=b"")

        monkeypatch.setattr(source.subprocess, "run", fake_run)
        monkeypatch.setattr(source.time, "sleep", lambda _seconds: None)
        source._curl_range("https://example.invalid/archive.zip", 10, 19, dest)

        assert len(attempts) == 2
        assert dest.read_bytes() == b"x" * 10


class TestCatalogue:
    def test_the_shipped_catalogue_covers_batch_1(self):
        catalogue = load_catalogue()
        assert len(catalogue) == 873
        entry = catalogue["L21_V001"]
        assert entry.zip_name == "Videos_L21_a.zip"
        assert entry.url.startswith("https://aic-data.ledo.io.vn/")
        assert entry.size > 0

    def test_every_entry_lies_inside_its_archive(self):
        """Regression for the ZIP64 bug: slicing the trailer from the plain EOCD
        dropped the ZIP64 record, and `zipfile` then added a bogus offset to every
        entry. Archives under 4 GB looked fine; the other eleven pointed past the
        end of the file."""
        from pathlib import Path

        import yaml

        book = yaml.safe_load(Path("config/zips_batch1.yaml").read_text(encoding="utf-8"))
        for name, body in book["zips"].items():
            for video_id, (offset, size) in body["entries"].items():
                assert offset + size <= body["bytes"], f"{name}/{video_id}"
