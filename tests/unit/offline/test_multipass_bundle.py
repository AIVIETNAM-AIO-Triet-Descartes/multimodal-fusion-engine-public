"""Locks the two things that change when one bundle is filled by several passes.

The BTC bundle is written by three notebooks — `02` imports the keyframes, `03`
adds SigLIP and the two OCR sources, `04` adds subtitles and CLAP — over the same
videos, into the same `aic_btc_b1_v1`. That is a shape the earlier single-pass
code did not have, and it breaks two assumptions that were safe before:

1. a stage may load its model once and be called for hundreds of videos, and
2. `part-manifest/{video}.json` is written more than once for the same video.

Both failures are silent. The first only costs time (measured at 5-16 hours per
member of pure model loading); the second loses data while every marker still
reports success.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mfe.offline.bundle import BundleLayout, BundleWriter, load_shard_plan
from mfe.offline.stages import STAGES, cached_model, unload_models
from mfe.offline.transport import LocalDirTransport

VIDEO = "L21_V001"


@pytest.fixture
def writer(tmp_path):
    transport = LocalDirTransport(tmp_path / "volume")
    layout = BundleLayout("aic_btc_b1_v1")
    return BundleWriter(transport, layout, "triet", tmp_path / "work")


def read_manifest(writer, tmp_path, video_id=VIDEO) -> dict:
    local = tmp_path / "read_back.json"
    writer.transport.get(writer.layout.part_manifest(video_id), local)
    return json.loads(local.read_text(encoding="utf-8"))


def marker(artifact: str, rows: int) -> dict:
    return {"artifact": artifact, "video_id": VIDEO, "stage": "x", "rows": rows}


class TestModelCache:
    def setup_method(self):
        unload_models()

    def teardown_method(self):
        unload_models()

    def test_builds_once_and_reuses(self):
        calls = []

        def build():
            calls.append(1)
            return object()

        first = cached_model("siglip", build)
        second = cached_model("siglip", build)
        assert first is second
        assert len(calls) == 1

    def test_different_checkpoints_do_not_collide(self):
        a = cached_model("siglip", lambda: "A")
        b = cached_model("qwen", lambda: "B")
        assert (a, b) == ("A", "B")

    def test_unload_forces_a_rebuild(self):
        """`unload_models` is what a video-major notebook calls between stages —
        it must genuinely let go, or §4.2's VRAM argument stops holding."""
        calls = []
        cached_model("siglip", lambda: calls.append(1))
        unload_models()
        cached_model("siglip", lambda: calls.append(1))
        assert len(calls) == 2

    def test_a_miss_is_the_only_thing_that_calls_build(self):
        """The lambda in each stage closes over `from_pretrained`, so calling it
        on a hit would download and initialise the model anyway — the cache would
        look present and buy nothing."""
        cached_model("siglip", lambda: "first")
        assert cached_model("siglip", lambda: pytest.fail("build chay tren cache hit")) == "first"


class TestManifestMerge:
    def test_a_second_pass_erases_the_first_without_merge(self, writer, tmp_path):
        """The behaviour being guarded against, asserted so it stays deliberate."""
        writer.write_part_manifest(VIDEO, [marker("map_keyframes", 307)], keyframes=307)
        writer.write_part_manifest(VIDEO, [marker("siglip_vectors", 307)], keyframes=307)
        assert set(read_manifest(writer, tmp_path)["artifacts"]) == {"siglip_vectors"}

    def test_merge_keeps_what_earlier_passes_recorded(self, writer, tmp_path):
        writer.write_part_manifest(VIDEO, [marker("map_keyframes", 307)], keyframes=307)
        writer.write_part_manifest(
            VIDEO, [marker("siglip_vectors", 307), marker("ocr_text", 41)],
            keyframes=307, merge=True,
        )
        writer.write_part_manifest(
            VIDEO, [marker("subtitles", 260)], keyframes=307, merge=True
        )
        assert set(read_manifest(writer, tmp_path)["artifacts"]) == {
            "map_keyframes", "siglip_vectors", "ocr_text", "subtitles",
        }

    def test_a_rerun_of_one_stage_replaces_only_that_artifact(self, writer, tmp_path):
        writer.write_part_manifest(
            VIDEO, [marker("map_keyframes", 307), marker("ocr_text", 41)], keyframes=307
        )
        writer.write_part_manifest(VIDEO, [marker("ocr_text", 44)], keyframes=307, merge=True)
        artifacts = read_manifest(writer, tmp_path)["artifacts"]
        assert artifacts["ocr_text"]["rows"] == 44
        assert artifacts["map_keyframes"]["rows"] == 307

    def test_merge_works_when_there_is_nothing_to_merge_with(self, writer, tmp_path):
        """First pass of the first video: no manifest exists yet, and a missing
        one is the normal case rather than an error."""
        writer.write_part_manifest(VIDEO, [marker("map_keyframes", 307)],
                                   keyframes=307, merge=True)
        assert set(read_manifest(writer, tmp_path)["artifacts"]) == {"map_keyframes"}

    def test_no_scratch_file_is_left_behind(self, writer, tmp_path):
        writer.write_part_manifest(VIDEO, [marker("a", 1)], keyframes=1, merge=True)
        writer.write_part_manifest(VIDEO, [marker("b", 1)], keyframes=1, merge=True)
        assert list((tmp_path / "work").glob("prev_manifest_*.json")) == []


class TestThreePassesAgreeOnOwnership:
    """The invariant that makes the three-notebook split safe.

    `part-manifest/{video}.json` records exactly one `author`, and
    `merge.check_ownership` refuses a shard whose author is not the member the
    plan assigns. So if two passes over the same bundle split the videos
    differently, each video gets its author rewritten by whichever pass ran last
    and the merge refuses the whole bundle.

    This is not hypothetical: BTC groups keyframes and videos into different
    archives, and splitting by the video archives gives `bao` 148 videos that
    overlap his 295 keyframe-archive videos in **zero** places. Pass `04` reads
    videos, so splitting it that way is the obvious move — and it is wrong.
    It gets away with the keyframe split because it fetches each video by HTTP
    Range, which makes archive grouping irrelevant to bandwidth.
    """

    PLANS = [
        "config/ingest_shards_batch1_btc.yaml",
        "config/ingest_shards_batch1_btc_images.yaml",
        "config/ingest_shards_batch1_btc_video.yaml",
        # Lượt 4 — hai nguồn OCR thay chỗ `n5_ocr_qwen`.
        "config/ingest_shards_batch1_btc_ocr2.yaml",
        # Lượt 5 — caption + chữ màn hình bằng Qwen2.5-VL qua vLLM.
        "config/ingest_shards_batch1_btc_caption.yaml",
        # Lượt 6 — nhúng chính caption đó bằng BGE-M3. Đọc part từ Volume, không
        # đọc ảnh, nên nó KHÔNG chia lát theo `PART` như lượt 5.
        "config/ingest_shards_batch1_btc_bge.yaml",
    ]

    def test_every_pass_assigns_the_same_videos_to_the_same_member(self):
        plans = [load_shard_plan(p) for p in self.PLANS]
        first = plans[0]
        # The BGE pass is intentionally archive-split into 14 workers so the
        # CPU encoder can finish within a session.  Its ownership need only be
        # complete and disjoint; the five earlier passes retain the two-member
        # assignment because they share the vLLM/GPU work partition.
        for plan in plans[1:-1]:
            assert plan.bundle == first.bundle
            assert plan.members == first.members
        bge = plans[-1]
        assert bge.bundle == first.bundle
        assert sorted(bge.videos) == sorted(first.videos)
        assert len(bge.videos) == len(set(bge.videos))

    def test_the_passes_together_cover_every_stage(self):
        stages = [s for p in self.PLANS for s in load_shard_plan(p).stages]
        assert stages == [
            "n0_import_btc",
            "n4_siglip", "n5_ocr_easyocr",
            "n2_subtitles", "n3_audio",
            "n5_ocr_paddle", "n5_ocr_vietocr",
            "n6_caption_qwen", "n6_ocr_qwen25",
            "n6_bge_m3",
        ]
        # n4_objects: BTC ships them, imported by pass 1.
        assert "n4_objects" not in stages
        # `n6_recap` (ReCap qua Gemini/Qwen2-VL) XOÁ HẲN 25-08-2026 — nhóm đổi
        # sang `n6_caption_qwen`. Khẳng định ở đây chứ không chỉ xoá khỏi yaml,
        # cùng lý do như `n5_ocr_qwen` bên dưới.
        assert "n6_recap" not in stages
        assert "n6_recap" not in STAGES
        # `n5_ocr_qwen` BỎ HẲN 21-08-2026 — 10,8 giây/ảnh, ~532 giờ GPU cho
        # batch 1. `n5_ocr_paddle` + `n5_ocr_vietocr` thay chỗ. Khẳng định ở đây
        # chứ không chỉ xoá khỏi danh sách trên: `plan.stages` là hợp đồng mà
        # `merge_bundle.py` đọc để đòi đủ mảnh, nên tầng lỡ quay lại yaml là
        # merge lại đòi 873 mảnh `sviptr_ocr` không tồn tại.
        assert "n5_ocr_qwen" not in stages

    def test_moi_tang_trong_ke_hoach_deu_CHAY_duoc(self):
        """`plan.stages` là hợp đồng `merge_bundle.py` đọc để đòi đủ mảnh.

        Một tên tầng còn trong yaml mà không còn trong `STAGES` là một lượt gộp
        đòi 873 mảnh không ai sinh được — và nó chỉ nói ra ở bước gộp.
        """
        for path in self.PLANS:
            for stage in load_shard_plan(path).stages:
                assert stage in STAGES, f"{path}: {stage} khong co trong STAGES"

    def test_luot_6_KHONG_chia_theo_PART(self):
        """Lượt 6 đọc caption từ Volume, không tải `Keyframes_*.zip`.

        Cắt lát tồn tại để tránh tải cùng một gói zip nhiều lần. Ở đây không có
        gói nào để tải, nên `PART` chỉ là một cần gạt sai được mà không lợi gì —
        và notebook 09 cố ý không có nó.
        """
        notebook = json.loads(
            Path("notebooks/btc/09-btc-bge-m3-embed.ipynb").read_text(encoding="utf-8")
        )
        # CHI cac o CODE. Markdown cua notebook 09 co nhac `NUM_PARTS` — de giai
        # thich vi sao no KHONG co, va do la thu dang giu chu khong dang cam.
        code = "".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "NUM_PARTS" not in code
        assert "Keyframes_" not in code

    def test_every_assigned_video_can_actually_be_fetched(self):
        """Pass `04` needs a Range entry per video, and it reads the assignment
        from a *different* file than the offsets. Nothing else checks that the
        two files cover the same 873 videos."""
        from mfe.offline.source import load_catalogue

        plan = load_shard_plan("config/ingest_shards_batch1_btc_video.yaml")
        catalogue = load_catalogue("config/zips_batch1.yaml")
        assert [v for v in plan.videos if v not in catalogue] == []


class TestZipSizesAreKnown:
    """`zip_bytes` is what turns "wget said OK" into a checked claim.

    `wget -c` resumes with an HTTP `Range` request. If the host answers `200`
    with the whole body instead of `206` with the tail — which a throttling or
    misconfigured server does — wget appends that body after the bytes already
    on disk. The archive is then larger than the real one and corrupt from the
    join, and nothing in the download reports a problem. The only signal is the
    size, and it only exists because `make_zip_inventory.py` measured it.
    """

    @pytest.mark.parametrize(
        "config",
        [
            "config/ingest_shards_batch1_btc.yaml",
            "config/ingest_shards_batch1_btc_images.yaml",
        ],
    )
    def test_every_archive_a_member_downloads_has_a_size(self, config):
        plan = load_shard_plan(config)
        for member in plan.members:
            for name, _ in plan.zips_for(member):
                assert plan.zip_bytes.get(name, 0) > 0, f"{name} khong co kich thuoc"

    def test_the_sizes_match_the_inventory_they_came_from(self):
        import yaml

        book = yaml.safe_load(open("config/zips_keyframes_btc.yaml", encoding="utf-8"))
        plan = load_shard_plan("config/ingest_shards_batch1_btc.yaml")
        assert plan.zip_bytes == {n: b["bytes"] for n, b in book["zips"].items()}


class TestNotebookCoDuGoiDeDayLenVolume:
    """Mọi notebook đẩy part lên Modal Volume phải khai `modal` trong requirements.

    Đây là một lỗi ĐÃ XẢY RA (25-08-2026), và nó đắt vì nó nổ MUỘN:
    `build_transport("modal", …)` gọi `import modal` bên trong
    `ModalVolumeTransport.__init__`, tức ở ô Step 0b — **sau** khi notebook đã
    clone repo, cài xong vLLM và nạp xong kế hoạch. Người chạy mất vài phút
    quota GPU để nhận một `ModuleNotFoundError` lẽ ra phải thấy từ đầu.

    Nó lọt lưới vì notebook 06 khai **hai** file requirements và `modal` nằm ở
    file kia, nên `…-paddle-vietocr.txt` thiếu mà không ai thấy. Notebook 08 và
    09 chỉ khai một file, và ở đó chỗ thiếu lộ ra ngay.

    Nên phép kiểm là trên HỢP của mọi file notebook đó cài, không phải trên từng
    file — đúng hình dạng thật của cái bẫy.
    """

    @staticmethod
    def _code(notebook: Path) -> str:
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        return "".join(
            "".join(c["source"]) for c in cells if c["cell_type"] == "code"
        )

    @pytest.mark.parametrize(
        "notebook", sorted(Path("notebooks/btc").glob("*.ipynb")), ids=lambda p: p.name
    )
    def test_notebook_dung_modal_thi_phai_cai_modal(self, notebook):
        import re

        code = self._code(notebook)
        if not re.search(r'TRANSPORT\s*=\s*"modal"', code):
            pytest.skip("notebook nay khong day len Volume")

        names = re.findall(r'"(requirements-kaggle[\w.-]+\.txt)"', code)
        assert names, f"{notebook.name}: dung modal ma khong cai requirements nao"

        declared = "\n".join(
            (Path("notebooks") / name).read_text(encoding="utf-8")
            for name in names
            if (Path("notebooks") / name).is_file()
        )
        assert re.search(r"^modal\b", declared, re.M), (
            f"{notebook.name} đẩy part lên Modal Volume nhưng không file nào trong "
            f"{names} khai `modal`. Ô Step 0b sẽ chết bằng ModuleNotFoundError."
        )
