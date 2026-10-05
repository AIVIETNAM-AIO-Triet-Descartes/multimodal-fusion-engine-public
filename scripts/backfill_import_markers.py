"""Ghi marker + manifest cho những video ĐÃ có đủ dữ liệu trên Volume.

    python scripts/backfill_import_markers.py --member quang            # chi kiem
    python scripts/backfill_import_markers.py --member quang --write    # ghi that

Vì sao cần
-----------
`StageStatus.load(transport, layout, MEMBER)` chỉ đọc `/_status/{MEMBER}/`, và đó
là **cố ý** — một người xong không được làm người khác bỏ qua việc. Nhưng bảng
phân công đã đổi từ 5 người sang 2, nên việc chạy dưới tên `triet`/`bao`/`dat`
trở thành vô hình với chủ mới: notebook 02 báo "còn lại N" và nạp lại từ đầu.

Đo ngày 20-08-2026 trên `keyframes-btc`: **873/873 video đã đủ** 177.321 ảnh và
177.321 file objects, khớp từng video với `map-keyframes`. Thứ thật sự thiếu là
marker và manifest cho 215 video. Chạy lại notebook 02 sinh ra chúng — bằng cách
nạp lại ~38.000 tấm ảnh y hệt từng byte. Script này ghi hai file nhỏ đó, và không
đụng vào một tấm ảnh nào.

Vì sao nó ĐẾM chứ không đóng dấu
----------------------------------
Marker là **điểm chốt** (§5.2) đúng vì một lượt upload đứt giữa chừng để lại thư
mục **có thật mà thiếu ảnh**: `tests/unit/offline/test_bundle.py` có hẳn một test
tên "a half-written part without a marker is not done". Một script đóng dấu chỉ
vì thư mục tồn tại sẽ phá đúng cái tính chất đó, và phá theo hướng không ai nhìn
thấy — bundle trông đầy đủ, `verify_bundle.py` gật đầu, và thiếu hụt chỉ lộ ra ở
điểm số.

Nên trước mỗi lần ghi, ba con số phải bằng nhau:

    số ảnh trong keyframes/{video}/  ==  số file objects/{video}/  ==  số dòng map-keyframes

và **tập số thứ tự `n`** đọc từ tên file phải trùng đúng tập `n` trong CSV. Kiểm
tập chứ không chỉ kiểm số lượng: đủ số mà lệch một ordinal thì `read_object_labels`
gán nhãn của khung hình này cho khung hình khác, và không có gì phía sau phát hiện.

Marker nó ghi KHÔNG nói dối
-----------------------------
`model` và `params` chép nguyên của tầng thật vì chúng mô tả **dữ liệu** — dữ
liệu này đúng là của BTC, bất kể ai chuyển nó lên. Nhưng `bytes` và `elapsed_s`
thì bỏ hẳn: script này không thực hiện lượt chuyển nào, và điền một con số vào đó
là bịa ra một phép đo. Thay vào đó marker mang `backfilled: true` cùng câu đã
kiểm được những gì.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mfe.core.cli import setup_console  # noqa: E402
from mfe.offline.bundle import BundleLayout, BundleWriter, load_shard_plan  # noqa: E402
from mfe.offline.transport import build_transport  # noqa: E402

STAGE = "n0_import_btc"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="backfill_import_markers", description=__doc__)
    parser.add_argument("--member", required=True, help="Phai co trong bang phan cong")
    parser.add_argument("--shards", default="config/ingest_shards_batch1_btc.yaml")
    parser.add_argument("--bundle", default="aic_btc_b1_v1")
    parser.add_argument("--volume", default="keyframes-btc")
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Ghi that. Khong co co nay thi chi kiem va bao cao.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Doc song song khi kiem")
    parser.add_argument(
        "--write-workers",
        type=int,
        default=4,
        help="Ghi song song. De thap: Modal khuyen khong qua ~5 commit dong thoi.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = tat ca")
    return parser


# ---------------------------------------------------------------------------
# Kiem
# ---------------------------------------------------------------------------


def ordinals_under(transport, root: str, suffixes: tuple[str, ...]) -> dict[str, set[int]]:
    """`{video_id: {n, …}}` đọc từ TÊN FILE dưới `root`.

    Giữ tập `n` chứ không giữ số đếm, vì phép kiểm cần trả lời "đúng những khung
    hình nào" chứ không phải "đúng bao nhiêu khung hình".
    """
    key = root.rstrip("/").split("/")[-1]
    out: dict[str, set[int]] = defaultdict(set)
    for remote in transport.listdir(root):
        parts = remote.strip("/").split("/")
        if key not in parts:
            continue
        index = parts.index(key)
        if len(parts) <= index + 2:
            continue
        name = parts[-1]
        if not name.lower().endswith(suffixes):
            continue
        stem = Path(name).stem
        if stem.isdigit():
            out[parts[index + 1]].add(int(stem))
    return out


def read_map(transport, layout: BundleLayout, video_id: str, tmp: Path):
    """`(tập n, fps)` từ `map-keyframes/{video}.csv`, hoặc `(None, None)`."""
    local = tmp / f"{video_id}.csv"
    try:
        transport.get(layout.map_keyframes(video_id), local)
        rows = list(csv.DictReader(local.read_text(encoding="utf-8").splitlines()))
    except Exception:
        return None, None
    finally:
        local.unlink(missing_ok=True)
    if not rows:
        return None, None
    try:
        ordinals = {int(row["n"]) for row in rows}
        fps = float(rows[0]["fps"])
    except (KeyError, ValueError):
        return None, None
    return ordinals, fps


def audit(transport, layout, videos, images, objects, workers: int):
    """`(dat, hong)` — hỏng luôn kèm lý do, và không lý do nào được đoán."""
    ok: list[dict] = []
    bad: list[tuple[str, str]] = []

    def one(video_id: str, tmp: Path):
        mapped, fps = read_map(transport, layout, video_id, tmp)
        if mapped is None:
            return video_id, None, "khong doc duoc map-keyframes"
        img, obj = images.get(video_id, set()), objects.get(video_id, set())
        if not img:
            return video_id, None, "khong co anh nao"
        if img != mapped:
            return video_id, None, (
                f"anh lech map-keyframes: {len(img)} anh / {len(mapped)} dong, "
                f"thieu {sorted(mapped - img)[:5]}, thua {sorted(img - mapped)[:5]}"
            )
        if obj != mapped:
            return video_id, None, (
                f"objects lech: {len(obj)} file / {len(mapped)} dong, "
                f"thieu {sorted(mapped - obj)[:5]}"
            )
        return video_id, {"keyframes": len(mapped), "fps": fps}, None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, (video_id, facts, reason) in enumerate(
                pool.map(lambda v: one(v, root), videos), 1
            ):
                if facts is None:
                    bad.append((video_id, reason or "?"))
                else:
                    ok.append({"video_id": video_id, **facts})
                if index % 25 == 0 or index == len(videos):
                    print(f"  ... kiem {index}/{len(videos)}")
    return ok, bad


# ---------------------------------------------------------------------------


def main() -> None:
    setup_console()
    args = build_parser().parse_args()

    plan = load_shard_plan(args.shards)
    layout = BundleLayout(args.bundle)
    transport = build_transport(
        "modal", volume=args.volume, local_root="bundle_local", version=args.version
    )
    mine = plan.videos_for(args.member)  # nem loi neu member khong co trong bang

    marked = {
        Path(remote).stem
        for remote in transport.listdir(f"/_status/{args.member}/{STAGE}")
        if remote.endswith(".json")
    }
    todo = [video for video in mine if video not in marked]
    if args.limit:
        todo = todo[: args.limit]

    print(f"bundle : {args.bundle} -> volume {args.volume} (v{args.version})")
    print(f"nguoi  : {args.member} — {len(mine)} video, da co marker {len(mine) - len(todo)}")
    print(f"can ghi: {len(todo)}\n")
    if not todo:
        print("Khong co gi de ghi.")
        return

    print("liet ke keyframes/ va objects/ ...")
    images = ordinals_under(transport, f"{layout.root}/keyframes", IMAGE_SUFFIXES)
    objects = ordinals_under(transport, f"{layout.root}/objects", (".json",))
    print(f"  {len(images)} video co anh · {len(objects)} video co objects\n")

    print(f"doi chieu {len(todo)} video voi map-keyframes ...")
    ok, bad = audit(transport, layout, todo, images, objects, args.workers)

    print(f"\n{'=' * 72}")
    print(f"DAT  : {len(ok)}  (anh == objects == map-keyframes, dung tung ordinal)")
    print(f"HONG : {len(bad)}")
    for video_id, reason in bad[:20]:
        print(f"  {video_id:<12} {reason}")
    if len(bad) > 20:
        print(f"  ... va {len(bad) - 20} video nua")
    if bad:
        print("\nNhung video HONG phai chay lai notebook 02 that — du lieu cua chung")
        print("chua day du, va do dung la truong hop marker sinh ra de chan.")

    if not args.write:
        print(f"\n--write chua bat: khong ghi gi. Se ghi marker + manifest cho {len(ok)} video.")
        return
    if not ok:
        print("\nKhong co video nao dat — khong ghi gi.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        writer = BundleWriter(transport, layout, args.member, Path(tmp))
        done = Counter()

        def write_one(item: dict) -> tuple[str, str | None]:
            try:
                marker = writer.mark_done(
                    STAGE,
                    item["video_id"],
                    keyframes=item["keyframes"],
                    images=item["keyframes"],
                    objects=item["keyframes"],
                    fps=item["fps"],
                    # Chép của tầng thật: chúng mô tả DỮ LIỆU, và dữ liệu này đúng
                    # là của BTC bất kể ai chuyển nó lên Volume.
                    model="BTC cap san (khong trich)",
                    params={"source": "aic-data.ledo.io.vn"},
                    # `bytes` va `elapsed_s` BO HAN: script nay khong chuyen gi ca.
                    backfilled=True,
                    verified=(
                        "anh == objects == so dong map-keyframes, khop tung ordinal; "
                        "khong nap lai anh"
                    ),
                )
                writer.write_part_manifest(
                    item["video_id"], [marker], keyframes=item["keyframes"], merge=True
                )
                return item["video_id"], None
            except Exception as err:  # noqa: BLE001 - bao cao tung video
                return item["video_id"], f"{type(err).__name__}: {err}"

        print(f"\nghi marker + manifest cho {len(ok)} video ...")
        with ThreadPoolExecutor(max_workers=args.write_workers) as pool:
            for index, (video_id, err) in enumerate(pool.map(write_one, ok), 1):
                done["hong" if err else "xong"] += 1
                if err:
                    print(f"  ! {video_id}: {err}")
                if index % 25 == 0 or index == len(ok):
                    print(f"  ... ghi {index}/{len(ok)}")

    print(f"\nxong {done['xong']} · hong {done['hong']}")
    print(f"Kiem lai:  python scripts/verify_bundle.py --shards {args.shards} "
          f"--member {args.member}")


if __name__ == "__main__":
    main()
