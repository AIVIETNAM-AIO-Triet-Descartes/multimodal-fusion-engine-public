"""Khoa hai tang OCR de do: `n5_ocr_paddle` va `n5_ocr_vietocr`.

May nay khong co paddle lan vietocr, va se khong bao gio co — nen model bi thay
bang stub. Cai duoc do o day khong phai chat luong doc chu (do la viec cua ban
than lan chay tren GPU), ma la phan CUA TA quanh model:

* doc duoc ca hai the he API cua PaddleOCR — sai cho nay cho ra **file rong**
  chu khong ném loi, dung kieu hong ma §8 cua handoff ke lai;
* xep hop chu thanh thu tu doc duoc, giong nhau o ca hai tang, vi chuoi ghep ra
  la thu di vao BM25 va mot phep so bi nhieu boi thu tu la phep so khong doc duoc;
* VietOCR nhan ANH DA CAT, khong nhan ca khung hinh — dua nham thi no van tra ve
  mot chuoi, khong ném gi;
* marker ghi dung cho: do do duoc tung video vao `facts`, cau hinh vao `params`.

Cac gia tri khang dinh o day deu la TINH CHAT (tong khop, khong trung, dung thu
tu), khong phai con so doan truoc — §8.7.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from mfe.offline.bundle import BundleLayout, BundleWriter, StageStatus
from mfe.offline.stages import (
    Keyframe,
    StageContext,
    _quad_to_box,
    build_paddle_ocr,
    group_text_lines,
    join_text_lines,
    normalize_paddle_result,
    stage_n5_ocr_paddle,
    stage_n5_ocr_vietocr,
    unload_models,
)
from mfe.offline.transport import LocalDirTransport

VIDEO = "L21_V001"


def quad(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# ---------------------------------------------------------------------------
# Doc ket qua PaddleOCR
# ---------------------------------------------------------------------------


class TestNormalizePaddleResult:
    def test_it_reads_the_3x_mapping_shape(self):
        page = {
            "rec_texts": ["HTV9", "06:30"],
            "rec_scores": [0.99, 0.81],
            "rec_polys": [quad(10, 10, 60, 30), quad(70, 12, 130, 32)],
        }
        out = normalize_paddle_result([page])
        assert [text for _box, text, _score in out] == ["HTV9", "06:30"]
        assert out[0][0] == (10, 10, 60, 30)
        assert out[1][2] == pytest.approx(0.81)

    def test_it_reads_the_2x_nested_list_shape(self):
        page = [
            [quad(10, 10, 60, 30), ("HTV9", 0.99)],
            [quad(70, 12, 130, 32), ("06:30", 0.81)],
        ]
        out = normalize_paddle_result([page])
        assert [text for _box, text, _score in out] == ["HTV9", "06:30"]
        assert out[0][0] == (10, 10, 60, 30)

    def test_it_unwraps_the_res_envelope_3x_sometimes_adds(self):
        page = {"res": {"rec_texts": ["A"], "rec_scores": [1.0], "rec_polys": [quad(0, 0, 5, 5)]}}
        assert [t for _b, t, _s in normalize_paddle_result([page])] == ["A"]

    def test_it_falls_back_to_dt_polys_when_rec_polys_is_absent(self):
        page = {"rec_texts": ["A"], "dt_polys": [quad(1, 2, 3, 4)]}
        assert normalize_paddle_result([page])[0][0] == (1, 2, 3, 4)

    def test_an_image_with_no_text_is_empty_not_an_error(self):
        """`[None]` la cau tra loi that cua 2.x cho mot anh khong co chu."""
        assert normalize_paddle_result([None]) == []
        assert normalize_paddle_result([]) == []
        assert normalize_paddle_result(None) == []

    def test_a_text_without_a_polygon_is_dropped_rather_than_mispositioned(self):
        """Khong co hop thi khong xep duoc vao dong nao. Gan dai cho no mot hop
        gia se lam no nhay len dau trang ma khong ai thay."""
        page = {"rec_texts": ["A", "B"], "rec_polys": [quad(0, 0, 5, 5)]}
        assert len(normalize_paddle_result([page])) == 1

    def test_a_missing_score_defaults_to_confident_not_to_zero(self):
        """Mac dinh 0.0 se bi `rec_score_min` loc sach ngay khi ai do dat nguong."""
        page = {"rec_texts": ["A"], "rec_polys": [quad(0, 0, 5, 5)]}
        assert normalize_paddle_result([page])[0][2] == pytest.approx(1.0)


class TestQuadToBox:
    def test_it_rounds_outwards_so_no_stroke_is_clipped(self):
        assert _quad_to_box(quad(10.7, 20.2, 60.1, 30.9)) == (10, 20, 61, 31)

    def test_it_accepts_eight_flat_numbers(self):
        assert _quad_to_box([10, 20, 60, 20, 60, 30, 10, 30]) == (10, 20, 60, 30)

    def test_a_rotated_quad_becomes_its_bounding_box(self):
        assert _quad_to_box([[10, 22], [60, 20], [61, 30], [11, 32]]) == (10, 20, 61, 32)


# ---------------------------------------------------------------------------
# Thu tu doc
# ---------------------------------------------------------------------------


class TestGroupTextLines:
    def test_boxes_on_one_line_come_back_left_to_right(self):
        boxes = [(200, 10, 260, 30), (10, 12, 60, 32), (100, 11, 160, 31)]
        assert group_text_lines(boxes) == [[1, 2, 0]]

    def test_two_lines_stay_two_lines_top_first(self):
        boxes = [(10, 100, 60, 120), (10, 10, 60, 30)]
        assert group_text_lines(boxes) == [[1], [0]]

    def test_the_band_scales_with_box_height_not_with_pixels(self):
        """Cung mot bo cuc, phong to 10 lan, phai ra cung mot cach chia dong."""
        small = [(10, 10, 60, 30), (100, 12, 160, 32), (10, 60, 60, 80)]
        large = [(x * 10, y * 10, a * 10, b * 10) for x, y, a, b in small]
        assert group_text_lines(small) == group_text_lines(large) == [[0, 1], [2]]

    def test_a_staircase_of_boxes_does_not_swallow_the_page(self):
        """So voi hop DAU dong chu khong voi hop vua them: so voi hop vua them thi
        mot chuoi lech dan moi cai vai pixel se troi khong gioi han."""
        boxes = [(0, 0, 40, 20), (50, 8, 90, 28), (100, 16, 140, 36), (150, 24, 190, 44)]
        assert len(group_text_lines(boxes)) > 1

    def test_no_index_is_lost_or_repeated(self):
        boxes = [(i * 7 % 91, i * 13 % 140, i * 7 % 91 + 40, i * 13 % 140 + 18) for i in range(25)]
        flat = [i for line in group_text_lines(boxes) for i in line]
        assert sorted(flat) == list(range(25))

    def test_no_boxes_is_no_lines(self):
        assert group_text_lines([]) == []


class TestJoinTextLines:
    def test_within_a_line_space_between_lines_newline(self):
        boxes = [(10, 10, 60, 30), (100, 12, 160, 32), (10, 100, 60, 120)]
        assert join_text_lines(boxes, ["HTV9", "06:30", "BAN TIN"]) == "HTV9 06:30\nBAN TIN"

    def test_blank_recognitions_do_not_leave_double_spaces_or_empty_lines(self):
        boxes = [(10, 10, 60, 30), (100, 12, 160, 32), (10, 100, 60, 120)]
        assert join_text_lines(boxes, ["HTV9", "   ", ""]) == "HTV9"


# ---------------------------------------------------------------------------
# Khoi tao PaddleOCR qua hai the he tham so
# ---------------------------------------------------------------------------


SUPERSET = {
    "lang": "vi",
    "device": "gpu",
    "use_gpu": True,
    "show_log": False,
    "use_angle_cls": False,
    "use_textline_orientation": False,
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
}


class TestBuildPaddleOcr:
    """Ten tham so phai duoc chon TRUOC khi khoi tao, va chi khoi tao MOT lan.

    Do duoc tren Kaggle: PaddleOCR 3.x khoi tao PaddleX o pham vi tien trinh, nen
    lan dung thu hai — du lan dau da ném loi — chet bang `RuntimeError: PDX has
    already been initialized`. Ban dau tien cua ham nay thu roi bo tham so bi tu
    choi rop thu lai; no bien mot loi doc duoc thanh mot loi che mat loi that.
    """

    def _install(self, monkeypatch, version: str, signature_names=None):
        calls: list[dict] = []

        if signature_names is None:
            def factory(**kwargs):
                calls.append(dict(kwargs))
                return types.SimpleNamespace(kwargs=dict(kwargs))
        else:
            # Mot chu ky doc duoc — tuc KHONG co `**kwargs`, xem
            # `test_a_var_keyword_signature_disables_the_second_filter`.
            source = "def factory({}):\n    pass".format(
                ", ".join(f"{name}=None" for name in signature_names)
            )
            namespace: dict = {}
            exec(compile(source, "<fake>", "exec"), namespace)  # noqa: S102
            template = namespace["factory"]

            def factory(**kwargs):
                calls.append(dict(kwargs))
                return types.SimpleNamespace(kwargs=dict(kwargs))

            factory.__signature__ = __import__("inspect").signature(template)

        module = types.ModuleType("paddleocr")
        module.PaddleOCR = factory
        monkeypatch.setitem(sys.modules, "paddleocr", module)
        monkeypatch.setattr(
            "mfe.offline.stages._package_version",
            lambda name: version if name == "paddleocr" else "0",
        )
        return calls

    def test_a_3x_install_gets_the_3x_names_only(self, monkeypatch):
        calls = self._install(monkeypatch, "3.1.0")
        build_paddle_ocr(**SUPERSET)
        assert set(calls[0]) == {
            "lang", "device", "use_textline_orientation",
            "use_doc_orientation_classify", "use_doc_unwarping",
        }

    def test_a_2x_install_gets_the_2x_names_only(self, monkeypatch):
        calls = self._install(monkeypatch, "2.7.3")
        build_paddle_ocr(**SUPERSET)
        assert set(calls[0]) == {"lang", "use_gpu", "show_log", "use_angle_cls"}

    def test_it_constructs_exactly_once(self, monkeypatch):
        """Lan dung thu hai la cai chet bang PDX-already-initialized."""
        calls = self._install(monkeypatch, "3.1.0")
        build_paddle_ocr(**SUPERSET)
        assert len(calls) == 1

    def test_an_unknown_version_is_treated_as_the_newest(self, monkeypatch):
        calls = self._install(monkeypatch, "?")
        build_paddle_ocr(**SUPERSET)
        assert "device" in calls[0] and "use_gpu" not in calls[0]

    def test_a_readable_signature_narrows_further(self, monkeypatch):
        """Ban 3.x nao do bo mot tham so ma bang cua ta van khai."""
        calls = self._install(monkeypatch, "3.1.0", signature_names=["lang", "device"])
        build_paddle_ocr(**SUPERSET)
        assert set(calls[0]) == {"lang", "device"}

    def test_a_var_keyword_signature_disables_the_second_filter(self, monkeypatch):
        """PaddleOCR 3.x nhan `**kwargs`, nen chu ky khong noi len duoc gi.

        Loc theo mot chu ky `**kwargs` se cho ra tap RONG va bo het tham so —
        nen khi doc duoc `**kwargs`, bang theo phien ban la thu duy nhat con lai.
        """
        calls = self._install(monkeypatch, "3.1.0")  # factory nhan **kwargs
        build_paddle_ocr(**SUPERSET)
        assert "lang" in calls[0] and "device" in calls[0]

    def test_force_bypasses_both_filters(self, monkeypatch):
        """Duong thoat cho mot ban co ten khac ca hai bang."""
        calls = self._install(monkeypatch, "3.1.0", signature_names=["lang", "device"])
        build_paddle_ocr(force={"ten_la": 1}, **SUPERSET)
        assert calls[0]["ten_la"] == 1

    def test_a_real_construction_error_is_not_hidden(self, monkeypatch):
        module = types.ModuleType("paddleocr")

        def factory(**kwargs):  # noqa: ARG001
            raise RuntimeError("model file is corrupt")

        module.PaddleOCR = factory
        monkeypatch.setitem(sys.modules, "paddleocr", module)
        with pytest.raises(RuntimeError, match="corrupt"):
            build_paddle_ocr(**SUPERSET)


# ---------------------------------------------------------------------------
# Hai tang, chay tren stub
# ---------------------------------------------------------------------------


class FakePaddle:
    """PaddleOCR 3.x gia: tra ve ket qua da dat san theo tung anh."""

    def __init__(self, pages: dict[str, list]) -> None:
        self.pages = pages
        self.asked: list[str] = []

    def predict(self, path: str) -> list:
        self.asked.append(path)
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        return [self.pages.get(name, {"rec_texts": [], "rec_polys": []})]


class FakeVietOCR:
    """VietOCR gia. Ghi lai KICH THUOC anh nhan duoc — do la thu de sai nhat."""

    def __init__(self, reply=None) -> None:
        self.sizes: list[tuple[int, int]] = []
        self.batch_calls = 0
        self.reply = reply or (lambda image: f"viet:{image.size[0]}")

    def predict_batch(self, images):
        self.batch_calls += 1
        self.sizes.extend(image.size for image in images)
        return [self.reply(image) for image in images]


class FakeImage:
    def __init__(self, size=(640, 360), box=None) -> None:
        self.size = size
        self.box = box

    def convert(self, _mode):
        return self

    def crop(self, box):
        return FakeImage((box[2] - box[0], box[3] - box[1]), box)


@pytest.fixture
def probe_stubs(monkeypatch):
    """Cai `PIL` va `vietocr` gia; tra `paddleocr` qua cache model."""
    pil = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    image_mod.open = lambda path: FakeImage()  # noqa: ARG005
    # Giong Pillow >= 10: co LANCZOS, KHONG co ANTIALIAS. Do la moi truong that
    # tren Kaggle, va la thu lam tang nay chet truoc khi co shim.
    image_mod.LANCZOS = 1
    pil.Image = image_mod
    for name, module in (("PIL", pil), ("PIL.Image", image_mod)):
        monkeypatch.setitem(sys.modules, name, module)

    def install(paddle: FakePaddle, viet: FakeVietOCR | None = None) -> None:
        paddle_mod = types.ModuleType("paddleocr")
        paddle_mod.PaddleOCR = lambda **kwargs: paddle  # noqa: ARG005
        monkeypatch.setitem(sys.modules, "paddleocr", paddle_mod)

        if viet is not None:
            config_mod = types.ModuleType("vietocr.tool.config")
            config_mod.Cfg = types.SimpleNamespace(
                load_config_from_name=lambda name: {"predictor": {}, "name": name}
            )
            predictor_mod = types.ModuleType("vietocr.tool.predictor")
            predictor_mod.Predictor = lambda config: viet  # noqa: ARG005
            root = types.ModuleType("vietocr")
            tool = types.ModuleType("vietocr.tool")
            for name, module in (
                ("vietocr", root),
                ("vietocr.tool", tool),
                ("vietocr.tool.config", config_mod),
                ("vietocr.tool.predictor", predictor_mod),
            ):
                monkeypatch.setitem(sys.modules, name, module)

    unload_models()  # cache model la toan cuc; dung stub cua lan truoc la sai
    yield install
    unload_models()


def make_ctx(tmp_path, count: int, **params) -> StageContext:
    keyframes = [
        Keyframe(
            n=i + 1,
            pts_time=i * 2.0,
            fps=25.0,
            frame_idx=i * 50,
            image=tmp_path / f"{i + 1:03d}.jpg",
        )
        for i in range(count)
    ]
    transport = LocalDirTransport(tmp_path / "volume")
    layout = BundleLayout("aic_btc_b1_v1")
    writer = BundleWriter(transport, layout, "triet", tmp_path / "work")
    return StageContext(
        VIDEO, tmp_path / "unused.mp4", keyframes, writer, layout, tmp_path / "work",
        params=params,
    )


def read_part(ctx: StageContext, artifact: str) -> dict:
    local = ctx.workdir / "readback.json"
    ctx.writer.transport.get(ctx.layout.part(artifact, VIDEO), local)
    return json.loads(local.read_text(encoding="utf-8"))


class TestPaddleStage:
    def test_it_writes_one_key_per_keyframe_that_had_text(self, tmp_path, probe_stubs):
        probe_stubs(
            FakePaddle(
                {
                    "001.jpg": {"rec_texts": ["HTV9"], "rec_polys": [quad(10, 10, 60, 30)]},
                    "002.jpg": {"rec_texts": [], "rec_polys": []},
                }
            )
        )
        ctx = make_ctx(tmp_path, 2)
        marker = stage_n5_ocr_paddle(ctx)
        assert read_part(ctx, "paddle_ocr") == {f"{VIDEO}#000000": "HTV9"}
        assert marker["rows"] == 1
        assert marker["silent_images"] == 1
        assert marker["images"] == 2

    def test_the_payload_is_keyed_by_frame_idx_not_by_ordinal(self, tmp_path, probe_stubs):
        """`L21_V001#000050` la frame 50 cua video goc, khong phai keyframe thu 50.
        Nham la nop sai khung hinh, va khong co gi ném loi."""
        probe_stubs(
            FakePaddle(
                {f"{i:03d}.jpg": {"rec_texts": ["x"], "rec_polys": [quad(0, 0, 9, 9)]}
                 for i in (1, 2)}
            )
        )
        ctx = make_ctx(tmp_path, 2)
        stage_n5_ocr_paddle(ctx)
        assert set(read_part(ctx, "paddle_ocr")) == {f"{VIDEO}#000000", f"{VIDEO}#000050"}

    def test_it_joins_in_reading_order_not_in_detector_order(self, tmp_path, probe_stubs):
        probe_stubs(
            FakePaddle(
                {
                    "001.jpg": {
                        "rec_texts": ["DUOI", "PHAI", "TRAI"],
                        "rec_polys": [
                            quad(10, 100, 60, 120),
                            quad(100, 10, 160, 30),
                            quad(10, 12, 60, 32),
                        ],
                    }
                }
            )
        )
        ctx = make_ctx(tmp_path, 1)
        stage_n5_ocr_paddle(ctx)
        assert read_part(ctx, "paddle_ocr")[f"{VIDEO}#000000"] == "TRAI PHAI\nDUOI"

    def test_rec_score_min_filters_and_is_counted(self, tmp_path, probe_stubs):
        probe_stubs(
            FakePaddle(
                {
                    "001.jpg": {
                        "rec_texts": ["chac", "doan"],
                        "rec_scores": [0.95, 0.20],
                        "rec_polys": [quad(10, 10, 60, 30), quad(100, 12, 160, 32)],
                    }
                }
            )
        )
        ctx = make_ctx(tmp_path, 1, rec_score_min=0.5)
        marker = stage_n5_ocr_paddle(ctx)
        assert read_part(ctx, "paddle_ocr")[f"{VIDEO}#000000"] == "chac"
        assert marker["dropped_low_score"] == 1
        assert marker["boxes"] == 2

    def test_onednn_is_off_by_default(self, tmp_path, probe_stubs, monkeypatch):
        """Do duoc tren Kaggle CPU: nhanh oneDNN cua paddlepaddle 3.x ném
        `ConvertPirAttribute2RuntimeAttribute not support`. Tat no la doi cham
        lay chay duoc, va no phai la mac dinh chu khong phai mot co ai do nho bat.
        """
        seen: dict = {}
        paddle_mod = types.ModuleType("paddleocr")

        def factory(**kwargs):
            seen.update(kwargs)
            return FakePaddle({"001.jpg": {"rec_texts": [], "rec_polys": []}})

        paddle_mod.PaddleOCR = factory
        monkeypatch.setitem(sys.modules, "paddleocr", paddle_mod)
        stage_n5_ocr_paddle(make_ctx(tmp_path, 1))
        assert seen["enable_mkldnn"] is False

    def test_measurements_go_in_facts_and_settings_in_params(self, tmp_path, probe_stubs):
        """`check_fingerprints` tu choi ca bundle khi `params` lech giua cac video,
        nen mot con so khac nhau tung video de trong `params` la mot phep tu choi
        chac chan o buoc gop."""
        probe_stubs(FakePaddle({"001.jpg": {"rec_texts": ["a"], "rec_polys": [quad(0, 0, 9, 9)]}}))
        marker = stage_n5_ocr_paddle(make_ctx(tmp_path, 1, lang="vi"))
        assert set(marker["params"]) == {"lang", "rec_score_min", "api", "order"}
        assert marker["seconds_per_image"] >= 0


class TestVietOcrStage:
    def _paddle(self) -> FakePaddle:
        return FakePaddle(
            {
                "001.jpg": {
                    "rec_texts": ["paddle doc cai nay", "va cai nay"],
                    "rec_polys": [quad(10, 10, 60, 30), quad(100, 12, 160, 32)],
                }
            }
        )

    def test_vietocr_gets_crops_never_the_whole_keyframe(self, tmp_path, probe_stubs):
        """VietOCR khong co bo phat hien: dua ca khung hinh 640x360 vao no thi no
        van tra ve MOT chuoi, khong ném gi — nen chi test moi thay duoc."""
        viet = FakeVietOCR()
        probe_stubs(self._paddle(), viet)
        stage_n5_ocr_vietocr(make_ctx(tmp_path, 1, crop_pad=0))
        assert viet.sizes == [(50, 20), (60, 20)]
        assert (640, 360) not in viet.sizes

    def test_the_text_written_is_vietocrs_not_paddles(self, tmp_path, probe_stubs):
        probe_stubs(self._paddle(), FakeVietOCR(reply=lambda image: f"V{image.size[0]}"))
        ctx = make_ctx(tmp_path, 1, crop_pad=0)
        stage_n5_ocr_vietocr(ctx)
        written = read_part(ctx, "viet_ocr")[f"{VIDEO}#000000"]
        assert written == "V50 V60"
        assert "paddle doc cai nay" not in written

    def test_crop_padding_widens_the_box_and_is_clamped_to_the_frame(
        self, tmp_path, probe_stubs
    ):
        viet = FakeVietOCR()
        probe_stubs(
            FakePaddle({"001.jpg": {"rec_texts": ["goc"], "rec_polys": [quad(0, 0, 50, 20)]}}),
            viet,
        )
        stage_n5_ocr_vietocr(make_ctx(tmp_path, 1, crop_pad=4))
        # x0/y0 kep ve 0 chu khong thanh -4; x1/y1 duoc noi ra du 4.
        assert viet.sizes == [(54, 24)]

    def test_a_sliver_box_is_skipped_and_counted(self, tmp_path, probe_stubs):
        """Mot hop cao 1 pixel khong phai dong chu, nhung VietOCR van tra ve mot
        chuoi cho no — va chuoi do di thang vao index."""
        viet = FakeVietOCR()
        probe_stubs(
            FakePaddle(
                {
                    "001.jpg": {
                        "rec_texts": ["that", "vun"],
                        "rec_polys": [quad(10, 10, 60, 30), quad(100, 12, 101, 13)],
                    }
                }
            ),
            viet,
        )
        marker = stage_n5_ocr_vietocr(make_ctx(tmp_path, 1, crop_pad=0))
        assert len(viet.sizes) == 1
        assert marker["crops_skipped"] == 1
        assert marker["boxes"] == 2

    def test_an_image_with_no_detection_costs_no_recognition_call(
        self, tmp_path, probe_stubs
    ):
        viet = FakeVietOCR()
        probe_stubs(FakePaddle({"001.jpg": {"rec_texts": [], "rec_polys": []}}), viet)
        marker = stage_n5_ocr_vietocr(make_ctx(tmp_path, 1))
        assert viet.batch_calls == 0
        assert marker["rows"] == 0
        assert marker["silent_images"] == 1

    def test_it_falls_back_to_one_call_per_crop_without_predict_batch(
        self, tmp_path, probe_stubs
    ):
        """`predict_batch` khong co trong moi ban vietocr; thieu no phai cham hon,
        khong phai hong."""

        class OldVietOCR:
            def __init__(self) -> None:
                self.calls = 0

            def predict(self, image):
                self.calls += 1
                return f"V{image.size[0]}"

        viet = OldVietOCR()
        probe_stubs(self._paddle(), viet)
        marker = stage_n5_ocr_vietocr(make_ctx(tmp_path, 1, crop_pad=0))
        assert viet.calls == 2
        assert marker["batched"] is False

    def test_the_second_stage_does_not_pay_for_paddle_again(self, tmp_path, probe_stubs):
        """Hai tang chay CUNG pipeline tren CUNG bo anh — no chi duoc chay MOT lan.

        Ban dau tang VietOCR goi lai ca pipeline chi de lay hop, va tang do duoc
        ghi la "phan bi vut la phan re". Do la mot GIA DINH. Do thuc te: tren CPU,
        `n5_ocr_paddle` + `n5_ocr_vietocr` cho mot video 307 anh chay hon 25 phut,
        va mot nua thoi gian do la pipeline chay lan thu hai.

        Test nay TUNG khang dinh `len(paddle.asked) == 2` — no khoa dung cai lang
        phi do lai.
        """
        paddle = self._paddle()
        probe_stubs(paddle, FakeVietOCR())
        ctx = make_ctx(tmp_path, 1, crop_pad=0)
        stage_n5_ocr_paddle(ctx)
        stage_n5_ocr_vietocr(ctx)
        assert len(paddle.asked) == 1

    def test_changing_paddle_settings_invalidates_the_cached_detections(
        self, tmp_path, probe_stubs
    ):
        """Dung lai hop cu sau khi doi tham so la chay mot cau hinh khac voi cai
        marker se ghi — im lang, va khong the phat hien tu file ket qua."""
        paddle = self._paddle()
        probe_stubs(paddle, FakeVietOCR())
        stage_n5_ocr_paddle(make_ctx(tmp_path, 1))
        stage_n5_ocr_paddle(make_ctx(tmp_path, 1, lang="en"))
        assert len(paddle.asked) == 2

    def test_moving_to_another_video_drops_the_previous_ones_boxes(
        self, tmp_path, probe_stubs
    ):
        """Cache giu MOT video mot luc — do la mot chan bo nho.

        Vong lap that di qua ~400 video x ~300 keyframe. Giu lai video truoc
        khong cuu duoc lan goi nao (hai tang luon chay lien tiep tren cung mot
        video), no chi tich lai cho den khi phien Kaggle het RAM sau nhieu gio.
        """
        from mfe.offline.stages import _DETECTIONS

        paddle = self._paddle()
        probe_stubs(paddle, FakeVietOCR())
        stage_n5_ocr_paddle(make_ctx(tmp_path, 1))
        first = dict(_DETECTIONS)

        other = make_ctx(tmp_path, 1)
        other.video_id = "L21_V002"
        stage_n5_ocr_paddle(other)

        assert len(_DETECTIONS) == len(first)   # khong cong don

    def test_unloading_models_drops_the_detections_too(self, tmp_path, probe_stubs):
        """Hop gan voi doi tuong model da sinh ra chung."""
        paddle = self._paddle()
        probe_stubs(paddle, FakeVietOCR())
        stage_n5_ocr_paddle(make_ctx(tmp_path, 1))
        unload_models()
        stage_n5_ocr_paddle(make_ctx(tmp_path, 1))
        assert len(paddle.asked) == 2

    def test_limit_caps_the_images_and_says_so_in_the_marker(self, tmp_path, probe_stubs):
        """`limit` la cong cu cua phep DO. Mot lan chay bi cat khong duoc trong
        giong mot lan chay du: `rows` thap co the la "bo doc im lang" hoac "moi
        chay 2/10 anh", va hai cai do doi hoi hai phan ung nguoc nhau."""
        paddle = FakePaddle(
            {f"{i:03d}.jpg": {"rec_texts": ["x"], "rec_polys": [quad(0, 0, 40, 20)]}
             for i in range(1, 11)}
        )
        probe_stubs(paddle, FakeVietOCR())
        marker = stage_n5_ocr_paddle(make_ctx(tmp_path, 10, limit=2))
        assert len(paddle.asked) == 2
        assert marker["images"] == 2
        assert marker["keyframes_total"] == 10

    def test_all_detection_finishes_before_vietocr_is_loaded(self, tmp_path, probe_stubs):
        """Hai pha, khong xen ke — va thu tu nay la ca ly do sua.

        Paddle va torch cung mang oneDNN/OpenMP rieng. Lan chay 19-08 chet ngay o
        anh dau cua tang nay bang mot loi NOI BO cua Paddle, trong khi cung phep
        phat hien do vua chay tron ven o `n5_ocr_paddle`. Xen ke `_paddle_read`
        voi mot lan nap torch la khac biet duy nhat giua hai tang.
        """
        events: list[str] = []

        class LoggingPaddle(FakePaddle):
            def predict(self, path):
                events.append("detect")
                return super().predict(path)

        class LoggingVietOCR(FakeVietOCR):
            def predict_batch(self, images):
                events.append("recognize")
                return super().predict_batch(images)

        paddle = LoggingPaddle(
            {f"{i:03d}.jpg": {"rec_texts": ["a"], "rec_polys": [quad(0, 0, 40, 20)]}
             for i in (1, 2, 3)}
        )
        viet = LoggingVietOCR()
        probe_stubs(paddle, viet)
        # `load_vietocr` chi chay khi cached_model tra loi truot -> ghi vao events
        stage_n5_ocr_vietocr(make_ctx(tmp_path, 3, crop_pad=0))

        assert events == ["detect", "detect", "detect", "recognize", "recognize", "recognize"]

    def test_it_restores_the_antialias_constant_pillow_10_removed(
        self, tmp_path, probe_stubs
    ):
        """vietocr 0.3.x goi `Image.ANTIALIAS`, thu Pillow 10 da XOA.

        Do duoc 19-08-2026: tang nay chay Paddle xong 486 giay roi moi ngã bang
        `module 'PIL.Image' has no attribute 'ANTIALIAS'`.

        Gan lai la khoi phuc CHINH XAC chu khong phai thay the gan dung: o Pillow
        <= 9 hai ten do la cung mot hang so. Duong con lai — ha Pillow xuong < 10 —
        doi mot loi cua vietocr lay mot loi cua torchvision/SigLIP.
        """
        import sys as _sys

        probe_stubs(self._paddle(), FakeVietOCR())
        image_mod = _sys.modules["PIL.Image"]
        assert not hasattr(image_mod, "ANTIALIAS")   # tinh trang truoc khi chay

        marker = stage_n5_ocr_vietocr(make_ctx(tmp_path, 1, crop_pad=0))

        assert image_mod.ANTIALIAS == image_mod.LANCZOS
        assert marker["params"]["pillow_antialias_shim"] is True

    def test_the_two_phases_are_timed_apart(self, tmp_path, probe_stubs):
        probe_stubs(self._paddle(), FakeVietOCR())
        marker = stage_n5_ocr_vietocr(make_ctx(tmp_path, 1, crop_pad=0))
        assert marker["detect_seconds"] >= 0
        assert marker["recognize_seconds"] >= 0
        assert marker["detect_seconds"] + marker["recognize_seconds"] == pytest.approx(
            marker["elapsed_s"], abs=0.05
        )

    def test_the_two_artifacts_land_in_their_own_folders(self, tmp_path, probe_stubs):
        probe_stubs(self._paddle(), FakeVietOCR())
        ctx = make_ctx(tmp_path, 1, crop_pad=0)
        stage_n5_ocr_paddle(ctx)
        stage_n5_ocr_vietocr(ctx)
        root = tmp_path / "volume" / "bundle" / "aic_btc_b1_v1" / "parts"
        assert (root / "paddle_ocr" / f"{VIDEO}.json").is_file()
        assert (root / "viet_ocr" / f"{VIDEO}.json").is_file()

    def test_each_stage_marks_itself_done_separately(self, tmp_path, probe_stubs):
        """Phien dut giua hai tang phai chay tiep dung tang con lai."""
        probe_stubs(self._paddle(), FakeVietOCR())
        ctx = make_ctx(tmp_path, 1, crop_pad=0)
        stage_n5_ocr_paddle(ctx)
        status = StageStatus.load(ctx.writer.transport, ctx.layout, "triet")
        assert status.is_done("n5_ocr_paddle", VIDEO)
        assert not status.is_done("n5_ocr_vietocr", VIDEO)
