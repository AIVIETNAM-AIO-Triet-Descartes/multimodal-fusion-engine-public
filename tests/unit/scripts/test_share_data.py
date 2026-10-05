"""Gói dữ liệu gửi thành viên — file phải rơi ĐÚNG CHỖ sau khi giải nén.

Đây là phần duy nhất của dự án đi ra khỏi máy Lead bằng tay, và nó hỏng theo
kiểu tệ nhất: **im lặng**. Lệch một tầng thư mục thì giao diện vẫn mở, chỉ là
truy hồi ra 0 kết quả và mọi dòng ghi "không tra được frame_id" — trông y hệt
lỗi cấu hình, và người nhận sẽ đi sửa `.env`.

Nên test ở đây đóng đúng hai đinh:

1. Đường dẫn trong zip là **tương đối so với gốc repo**, và giải nén ra đúng chỗ
   đó — kiểm bằng cách gói ở một gốc rồi bung ở một gốc KHÁC.
2. `verify` **đếm dữ liệu**, không đọc log: thiếu một file phẳng là báo thiếu
   đúng tên file đó.
"""

from __future__ import annotations

import sqlite3
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

import share_data  # noqa: E402


def _build_repo(root: Path, *, flat_files=None, videos=873, keyframes=176707) -> None:
    """Một gốc repo giả có đủ ba phần `REQUIRED`."""
    vectors = root / "stores/flat/data/vectors"
    vectors.mkdir(parents=True)
    for name in (flat_files if flat_files is not None else share_data.FLAT_FILES):
        (vectors / name).write_bytes(f"noi dung cua {name}".encode())

    maps = root / "data/btc_b1/map-keyframes"
    maps.mkdir(parents=True)
    for i in range(800):
        (maps / f"L21_V{i:03d}.csv").write_text(
            "n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n", encoding="utf-8"
        )

    db_path = root / "stores/btc_b1/relational/app.db"
    db_path.parent.mkdir(parents=True)
    con = sqlite3.connect(db_path)
    con.execute("create table videos (id integer)")
    con.execute("create table keyframes (id integer)")
    con.executemany("insert into videos values (?)", [(i,) for i in range(videos)])
    con.executemany("insert into keyframes values (?)", [(i,) for i in range(keyframes)])
    con.commit()
    con.close()


@pytest.fixture
def lead(tmp_path, monkeypatch):
    """Máy Lead: có đủ dữ liệu."""
    root = tmp_path / "may_lead"
    root.mkdir()
    _build_repo(root)
    monkeypatch.setattr(share_data, "ROOT", root)
    return root


def test_giai_nen_ra_DUNG_CHO_o_mot_goc_repo_khac(tmp_path, lead, monkeypatch):
    """Đinh số một: gói ở gốc này, bung ở gốc kia, đường dẫn vẫn đúng.

    Gói và bung ở CÙNG một gốc thì test vẫn xanh kể cả khi `arcname` mang đường
    dẫn tuyệt đối của máy Lead — nên nó phải là hai gốc khác nhau.
    """
    archive = tmp_path / "mfe-data.zip"
    share_data.pack(archive)

    ban = tmp_path / "may_thanh_vien"
    ban.mkdir()
    monkeypatch.setattr(share_data, "ROOT", ban)
    assert share_data.unpack(archive) == 0

    vectors = ban / "stores/flat/data/vectors"
    assert sorted(p.name for p in vectors.iterdir()) == sorted(share_data.FLAT_FILES)
    assert (ban / "stores/btc_b1/relational/app.db").is_file()
    assert len(list((ban / "data/btc_b1/map-keyframes").glob("*.csv"))) == 800


def test_zip_khong_chua_duong_dan_tuyet_doi(tmp_path, lead):
    """Explorer *Extract All* và `zf.extractall` chỉ an toàn khi zip sạch.

    Một `arcname` tuyệt đối (`D:/Projects/...`) hay `..` là lối ghi ra ngoài
    repo, và `unpack` từ chối — nhưng tốt hơn là nó không bao giờ được sinh ra.
    """
    archive = tmp_path / "mfe-data.zip"
    share_data.pack(archive)
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert names
    for name in names:
        assert not name.startswith(("/", "..")), name
        assert ":" not in name, name
        assert "\\" not in name, name


def test_moi_file_phang_deu_vao_zip(tmp_path, lead):
    """`pack` gom bằng glob, nhưng bảng kê phải khớp — nếu không `verify` nói dối."""
    archive = tmp_path / "mfe-data.zip"
    share_data.pack(archive)
    with zipfile.ZipFile(archive) as zf:
        inside = {n.rsplit("/", 1)[-1] for n in zf.namelist()}
    for name in share_data.FLAT_FILES:
        assert name in inside, f"{name} khong vao zip"


def test_thieu_mot_file_phang_thi_verify_BAO_DUNG_TEN(tmp_path, monkeypatch, capsys):
    """Đinh số hai. Một gói thiếu arm phải bị bắt lúc CÀI, không phải lúc thi.

    Đây chính là chỗ bảng `FLAT_FILES` phải theo kịp kho: quên cập nhật nó thì
    gói thiếu bốn arm vẫn được báo "đủ".
    """
    root = tmp_path / "thieu"
    root.mkdir()
    _build_repo(root, flat_files=[f for f in share_data.FLAT_FILES if f != "qwen_ocr.json"])
    monkeypatch.setattr(share_data, "ROOT", root)

    assert share_data.verify() != 0
    assert "qwen_ocr.json" in capsys.readouterr().out


def test_app_db_sai_so_luong_thi_verify_TU_CHOI(tmp_path, monkeypatch, capsys):
    """177.321 là bản CHƯA gộp 614 cặp trùng `frame_idx` (CLAUDE.md §2).

    Một file đúng kích thước vẫn có thể là bản sai; chỉ đếm mới biết.
    """
    root = tmp_path / "db_cu"
    root.mkdir()
    _build_repo(root, keyframes=177321)
    monkeypatch.setattr(share_data, "ROOT", root)

    assert share_data.verify() != 0
    assert "176707" in capsys.readouterr().out


def test_bang_ke_phu_moi_arm_dense_va_text(tmp_path):
    """`FLAT_FILES` phải phủ mọi nguồn mà `flat_arms` biết tra.

    Thêm một arm mà quên tên file ở đây = gói gửi đi thiếu đúng arm đó, và
    người nhận chỉ biết khi mở giao diện ra thấy nó xám.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from mfe.online.flat_arms import DENSE_FILE, SOURCE_FILE  # noqa: PLC0415

    # `dense_captions.json` là file Bảo XUẤT RA, không tầng nào sinh — nó hợp lệ
    # khi vắng mặt, xem CLAUDE.md §4.
    wanted = (set(DENSE_FILE.values()) | set(SOURCE_FILE.values())) - {
        "dense_captions.json"
    }
    missing = wanted - set(share_data.FLAT_FILES)
    assert not missing, f"FLAT_FILES thieu: {sorted(missing)}"
