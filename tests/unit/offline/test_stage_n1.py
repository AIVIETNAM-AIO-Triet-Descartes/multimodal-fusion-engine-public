"""Executes `stage_n1_keyframes` end to end against a stub OpenCV.

The decode path is the one part of the pipeline that cannot run on a laptop, and
"cannot run" quietly became "was never executed" — the function was shipped
having only ever been type-checked. A stub `cv2` does not prove the real decoder
behaves, but it does prove everything around it: the two-pass structure, the
`grab`/`retrieve` split, the mapping written, the contract the mapping must
satisfy, the cache re-use, and the rule that no marker appears unless the images
did. Those are the parts that would otherwise be discovered 175 videos in.
"""

from __future__ import annotations

import csv
import sys
import types

import numpy as np
import pytest

from mfe.offline.bundle import BundleLayout, BundleWriter, StageStatus
from mfe.offline.stages import StageContext, read_keyframes, stage_n1_keyframes
from mfe.offline.transport import LocalDirTransport

VIDEO = "L21_V001"


class FakeCapture:
    """Enough of `cv2.VideoCapture` for N1, and it counts what was decoded."""

    def __init__(self, frames: int, fps: float, stats: dict):
        self.frames = frames
        self.fps = fps
        self.stats = stats
        self.index = 0
        self.released = False

    def isOpened(self):  # noqa: N802 - matches the OpenCV name
        return True

    def get(self, _prop):
        return self.fps

    def read(self):
        if self.index >= self.frames:
            return False, None
        self.stats["read"] += 1
        self.index += 1
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def grab(self):
        if self.index >= self.frames:
            return False
        self.stats["grab"] += 1
        self.index += 1
        return True

    def retrieve(self):
        self.stats["retrieve"] += 1
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def fake_cv2(monkeypatch):
    stats = {"read": 0, "grab": 0, "retrieve": 0, "imwrite": 0, "imencode": 0}
    state = {"frames": 3000, "fps": 29.969999}

    module = types.ModuleType("cv2")
    module.IMWRITE_JPEG_QUALITY = 1
    module.CAP_PROP_FPS = 5
    module.INTER_AREA = 3
    module.VideoCapture = lambda _path: FakeCapture(state["frames"], state["fps"], stats)

    rng = np.random.default_rng(0)

    def imencode(_ext, _frame, _params):
        stats["imencode"] += 1
        # A size curve with real structure: a step at frame 1500 so the selection
        # has something to find, plus noise so it is not degenerate.
        base = 40_000 if stats["imencode"] <= 1500 else 65_000
        return True, np.zeros(int(base + rng.normal(0, 2000)), dtype=np.uint8)

    def imwrite(path, _frame):
        stats["imwrite"] += 1
        from pathlib import Path

        Path(path).write_bytes(b"\xff\xd8fake-jpeg")
        return True

    module.imencode = imencode
    module.imwrite = imwrite
    module.resize = lambda frame, _size, interpolation=None: frame

    monkeypatch.setitem(sys.modules, "cv2", module)
    return types.SimpleNamespace(stats=stats, state=state)


@pytest.fixture
def ctx(tmp_path):
    transport = LocalDirTransport(tmp_path / "volume")
    layout = BundleLayout("aic_dake_r002_b1_v1")
    writer = BundleWriter(transport, layout, "triet", tmp_path / "work")
    return StageContext(
        VIDEO, tmp_path / f"{VIDEO}.mp4", [], writer, layout, tmp_path / "work",
        params={"rho": 0.015},
    )


class TestStageN1:
    def test_produces_a_mapping_its_own_reader_accepts(self, fake_cv2, ctx, tmp_path):
        """The end-to-end contract, and the one that used to fail.

        `read_keyframes` enforces `frame_idx == floor(pts_time * fps)` on every
        row. With `pts_time = round(index/fps, 4)` this rejected ~50% of rows at
        29.97fps — after the decode had already been paid for.
        """
        marker = stage_n1_keyframes(ctx)

        local = tmp_path / "back.csv"
        ctx.writer.transport.get(ctx.layout.map_keyframes(VIDEO), local)
        keyframes = read_keyframes(local, tmp_path / "work" / "keyframes" / VIDEO)
        assert len(keyframes) == marker["keyframes"]

    def test_rounds_fps_to_the_value_the_dataset_states(self, fake_cv2, ctx, tmp_path):
        """OpenCV reports 29.969999 for a 30000/1001 stream; the dataset says
        29.97, and `floor(pts*fps)` matches BTC on 100% of rows only with the
        stated number (41% with the exact rational)."""
        marker = stage_n1_keyframes(ctx)
        assert marker["fps"] == 29.97
        assert marker["fps_raw"] == pytest.approx(29.969999)

    def test_second_pass_decodes_only_the_selected_frames(self, fake_cv2, ctx):
        """`grab()` walks, `retrieve()` reconstructs. Two full decodes would make
        N1 twice as expensive as it needs to be, on the stage that is already the
        most expensive part of the DAKE pass."""
        marker = stage_n1_keyframes(ctx)
        stats = fake_cv2.stats
        assert stats["read"] == 3000            # pass 1 reads everything
        assert stats["grab"] >= marker["keyframes"]
        assert stats["retrieve"] == marker["keyframes"]
        assert stats["retrieve"] < stats["grab"] / 10

    def test_honours_rho_and_the_gap_bound(self, fake_cv2, ctx):
        from mfe.offline.dake import MAX_GAP_SECONDS

        marker = stage_n1_keyframes(ctx)
        assert marker["ratio"] >= ctx.params["rho"]
        assert marker["max_gap_s"] <= MAX_GAP_SECONDS + 0.001
        assert marker["source_frames"] == 3000

    def test_writes_one_image_per_selected_frame(self, fake_cv2, ctx):
        marker = stage_n1_keyframes(ctx)
        assert fake_cv2.stats["imwrite"] == marker["keyframes"]
        assert marker["images"] == marker["keyframes"]

    def test_reuses_the_size_curve_instead_of_decoding_again(self, fake_cv2, ctx, tmp_path):
        """Cache on the Volume, not just on disk: a fresh Kaggle session starts
        with an empty working dir, so a local-only cache never survives the exact
        situation it exists for. Sweeping `rho` later depends on this too (§3.4).
        """
        stage_n1_keyframes(ctx)
        first = fake_cv2.stats["read"]

        # Simulate the next session: same Volume, working directory wiped.
        (tmp_path / "work" / f"frame_sizes_{VIDEO}.npy").unlink()
        fake_cv2.stats["read"] = 0
        marker = stage_n1_keyframes(ctx)

        assert first == 3000
        assert fake_cv2.stats["read"] == 0      # pass 1 skipped entirely
        assert marker["sizes_reused"] is True

    def test_no_marker_when_the_decoder_stops_early(self, fake_cv2, ctx):
        """A short write is not a partial result. `clip-features` is an (N, D)
        array indexed by row order, so a mapping claiming more rows than there are
        images is a bundle whose files disagree — worse than no bundle."""
        original = fake_cv2.state["frames"]

        def truncating_capture(_path):
            capture = FakeCapture(original, fake_cv2.state["fps"], fake_cv2.stats)
            real_grab = capture.grab

            def grab():
                # Die part way through the second pass only.
                if fake_cv2.stats["retrieve"] >= 3:
                    return False
                return real_grab()

            capture.grab = grab
            return capture

        sys.modules["cv2"].VideoCapture = truncating_capture
        with pytest.raises(RuntimeError, match="chi ghi duoc"):
            stage_n1_keyframes(ctx)

        status = StageStatus.load(ctx.writer.transport, ctx.layout, "triet")
        assert not status.is_done("n1_keyframes", VIDEO)

    def test_mapping_columns_match_the_btc_schema(self, fake_cv2, ctx, tmp_path):
        stage_n1_keyframes(ctx)
        local = tmp_path / "cols.csv"
        ctx.writer.transport.get(ctx.layout.map_keyframes(VIDEO), local)
        with local.open(encoding="utf-8") as handle:
            reader = csv.reader(handle)
            assert next(reader) == ["n", "pts_time", "fps", "frame_idx"]
            rows = list(reader)
        assert [int(r[0]) for r in rows] == list(range(1, len(rows) + 1))


class TestDecodeBackend:
    """Which decoder ran is recorded, because the accelerator setting does not
    imply the GPU did anything."""

    def test_falls_back_to_cpu_when_nvdec_is_absent(self, fake_cv2, ctx, monkeypatch):
        """An untested GPU path must not be able to fail a run the tested CPU
        path would have finished. `auto` probes and falls back.
        """
        monkeypatch.setattr("mfe.offline.dake.cuda_decode_available", lambda: False)
        marker = stage_n1_keyframes(ctx)
        assert marker["decode_backend"] == "cpu"
        assert fake_cv2.stats["read"] == 3000

    def test_records_nvdec_when_the_probe_says_yes(self, fake_cv2, ctx, monkeypatch, tmp_path):
        """Selecting `nvdec` must be visible in the marker: a timing without it
        cannot be compared against one with it."""
        monkeypatch.setattr("mfe.offline.dake.cuda_decode_available", lambda: True)

        frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3000)]
        monkeypatch.setattr(
            "mfe.offline.dake.decode_frames_nvdec", lambda path, max_side=320: iter(frames)
        )
        marker = stage_n1_keyframes(ctx)
        assert marker["decode_backend"] == "nvdec"
        assert fake_cv2.stats["read"] == 0     # duong CPU khong chay

    def test_explicit_cpu_overrides_the_probe(self, fake_cv2, ctx, monkeypatch):
        monkeypatch.setattr("mfe.offline.dake.cuda_decode_available", lambda: True)
        ctx.params["decode_backend"] = "cpu"
        assert stage_n1_keyframes(ctx)["decode_backend"] == "cpu"
