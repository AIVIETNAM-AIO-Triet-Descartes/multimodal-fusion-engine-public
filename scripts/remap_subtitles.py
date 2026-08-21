"""Dựng lại `parts/subtitles/` từ segment thô, KHÔNG chạy lại Whisper.

    python scripts/remap_subtitles.py --member quang            # chi kiem
    python scripts/remap_subtitles.py --member quang --write    # ghi that

Vì sao
-------
Đo trên Volume 20-08-2026, quy tắc `next_keyframe` áp lên chính segment thô đang
có, so với file `parts/subtitles/` đang có:

    trong  L21_V001   quy tac phu 298/307 (97%)   file hien co  298 (97%)   khop
    quang  L22_V001   quy tac phu 294/298 (99%)   file hien co   93 (31%)
    quang  L26_V398   quy tac phu 161/176 (91%)   file hien co   34 (19%)
    quang  L27_V005   quy tac phu 258/336 (77%)   file hien co   43 (13%)

Bên Trọng khớp chính xác. Bên Quang mất 60–70% ánh xạ mà **chính segment thô của
bạn ấy đã hỗ trợ** — `build_keyframe_subtitles()` trong notebook 04 dùng bisect
sai, chính tác giả xác nhận. Đây không phải lệch để chấp nhận, đây là dữ liệu bị
đánh rơi.

Vá được rẻ vì `subtitles_segments/{video}.json` còn đủ **873/873**: chữ đã có,
chỉ ánh xạ sang keyframe là sai. Không GPU, không tải video, không Whisper.

Quy tắc dùng ở đây
-------------------
CHÉP NGUYÊN của `stage_n2_subtitles`, không phải bản dựng lại:

    low, high = pts_time(n), pts_time(n+1)      # keyframe cuoi: high = 1e12
    text = " ".join(s.text for s in spans if s.end > low and s.start < high)

Chọn quy tắc của repo chứ không phải `±3s` mà notebook 04 định làm: repo đã cân
nhắc và loại ±3s (§4.5 — cửa sổ cố định vừa hụt vừa đếm trùng tuỳ chỗ). Quan
trọng hơn, dùng đúng một quy tắc cho cả hai người thì `subtitles.params.window`
thôi là lệch, thay vì đổi một lệch cũ lấy một lệch mới.

Ba thứ nó KHÔNG sửa được
--------------------------
**Độ mịn segment.** Trọng 2,4s/segment, Quang 22–32s — chênh 10 lần, sinh ra từ
chính lượt chạy Whisper (`BatchedInferencePipeline`, batch 4) và đã đóng băng
trong segment thô. Sau remap, mỗi keyframe bên Quang mang cả khối 30 giây: đủ
coverage nhưng định vị thời gian mờ hơn. Lệch đó ở lại `accepted_divergences.yaml`.

**Lối giải mã.** `beam_size`, `temperature`, `condition_on_previous_text` nằm
trong chữ, không nằm trong ánh xạ. Remap không đụng tới.

**File của Trọng.** Không ghi vào. Đo rồi: file bạn ấy đã khớp quy tắc, remap lại
cho ra đúng byte cũ — chạy vào đó là rủi ro không đổi lấy gì.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mfe.contracts.ids import build_keyframe_id  # noqa: E402
from mfe.core.cli import setup_console  # noqa: E402
from mfe.offline.bundle import BundleLayout, BundleWriter, load_shard_plan  # noqa: E402
from mfe.offline.transport import build_transport  # noqa: E402

#: Cùng ngưỡng dừng của `stage_n2_subtitles`. Đây chính là cái chốt đáng lẽ bắt
#: được lỗi này ngay lượt chạy đầu: một video phiên âm ra hàng chục segment mà
#: rơi vào gần như không keyframe nào thì hoặc sai cửa sổ, hoặc sai kênh audio.
MAX_EMPTY_RATIO = 0.30

#: Nơi cất bản cũ. Nằm dưới `parts/` nhưng KHÔNG phải tên artifact, nên
#: `merge._list_parts` bỏ qua — nó chỉ nhận thư mục có trong `ARTIFACTS`.
BACKUP_DIR = "subtitles_pre_remap"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="remap_subtitles", description=__doc__)
    parser.add_argument("--member", required=True)
    parser.add_argument("--params", default="config/quang_nb04_params.json")
    parser.add_argument("--shards", default="config/ingest_shards_batch1_btc.yaml")
    parser.add_argument("--bundle", default="aic_btc_b1_v1")
    parser.add_argument("--volume", default="keyframes-btc")
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--model",
        default="faster-whisper large-v3",
        help="Ghi vao marker moi. Mac dinh la ten chinh tac cua repo.",
    )
    return parser


def spans_of(body) -> list[dict]:
    """Segment thô, chấp nhận cả hai hình dạng đã thấy trên Volume.

    `stage_n2_subtitles` ghi `{"language": …, "segments": [...]}`; notebook 04
    của Quang ghi thẳng một list. Cùng ba khoá `start`/`end`/`text`.
    """
    rows = body if isinstance(body, list) else (body or {}).get("segments") or []
    return [
        {"start": float(r["start"]), "end": float(r["end"]), "text": str(r["text"]).strip()}
        for r in rows
    ]


def remap(spans: list[dict], keyframes: list[tuple[float, int]], video_id: str) -> dict[str, str]:
    """`{keyframe_id: text}` theo quy tắc `next_keyframe`, chép của tầng thật."""
    mapping: dict[str, str] = {}
    for index, (pts, frame_idx) in enumerate(keyframes):
        nxt = keyframes[index + 1][0] if index + 1 < len(keyframes) else 1e12
        text = " ".join(s["text"] for s in spans if s["end"] > pts and s["start"] < nxt)
        if text.strip():
            mapping[build_keyframe_id(video_id, frame_idx)] = text
    return mapping


def prepare(transport, layout, video_id: str, tmp: Path):
    """`(mapping, facts)` hoặc `(None, lý do)`. Chưa ghi gì."""
    seg_local, map_local = tmp / f"s_{video_id}.json", tmp / f"m_{video_id}.csv"
    try:
        transport.get(layout.subtitle_segments(video_id), seg_local)
        spans = spans_of(json.loads(seg_local.read_text(encoding="utf-8")))
    except Exception as err:  # noqa: BLE001
        return None, f"khong doc duoc segment tho: {type(err).__name__}"
    finally:
        seg_local.unlink(missing_ok=True)
    if not spans:
        return None, "segment tho rong — Whisper khong ra chu nao"

    try:
        transport.get(layout.map_keyframes(video_id), map_local)
        rows = list(csv.DictReader(map_local.read_text(encoding="utf-8").splitlines()))
        keyframes = [(float(r["pts_time"]), int(r["frame_idx"])) for r in rows]
    except Exception as err:  # noqa: BLE001
        return None, f"khong doc duoc map-keyframes: {type(err).__name__}"
    finally:
        map_local.unlink(missing_ok=True)
    if not keyframes:
        return None, "map-keyframes rong"

    mapping = remap(spans, keyframes, video_id)
    empty_ratio = 1 - len(mapping) / len(keyframes)
    if empty_ratio > MAX_EMPTY_RATIO:
        # Cùng lý do như trong tầng thật: ném ra chứ không cảnh báo. Nếu sau khi
        # remap mà vẫn rỗng quá ngưỡng thì vấn đề KHÔNG phải ánh xạ, và ghi đè
        # file cũ bằng một file cũng sai là làm hỏng thêm.
        return None, (
            f"van rong {empty_ratio:.0%} sau remap du co {len(spans)} segment "
            "— khong phai loi anh xa, dung ghi de"
        )
    return mapping, {
        "segments": len(spans),
        "keyframes": len(keyframes),
        "empty_ratio": round(empty_ratio, 4),
    }


def main() -> None:
    setup_console()
    args = build_parser().parse_args()

    params = json.loads(Path(args.params).read_text(encoding="utf-8"))["subtitles"]
    # Ánh xạ giờ là của repo, nên `window` phải nói đúng điều đó.
    params = {k: v for k, v in params.items() if not k.startswith("//")}
    params["window"] = "next_keyframe"

    plan = load_shard_plan(args.shards)
    layout = BundleLayout(args.bundle)
    transport = build_transport(
        "modal", volume=args.volume, local_root="bundle_local", version=args.version
    )
    mine = plan.videos_for(args.member)
    if args.limit:
        mine = mine[: args.limit]

    print(f"nguoi : {args.member} — {len(mine)} video")
    print(f"model : {args.model}")
    print(f"params: {json.dumps(params, ensure_ascii=False)}\n")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ready: list[tuple[str, dict, dict]] = []
        problems: Counter = Counter()
        gained: list[tuple[str, int, int]] = []

        print(f"tinh lai anh xa cho {len(mine)} video ...")

        def one(video_id: str):
            mapping, facts = prepare(transport, layout, video_id, root)
            if mapping is None:
                return video_id, None, facts
            # Đọc file cũ CHỈ để báo cáo chênh lệch — không dùng để quyết định gì.
            old_local = root / f"o_{video_id}.json"
            before = None
            try:
                transport.get(layout.part("subtitles", video_id), old_local)
                body = json.loads(old_local.read_text(encoding="utf-8"))
                before = sum(1 for v in body.values() if str(v).strip())
            except Exception:  # noqa: BLE001 - chua co file cu la truong hop hop le
                before = None
            finally:
                old_local.unlink(missing_ok=True)
            facts["rows_before"] = before
            return video_id, mapping, facts

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, (video_id, mapping, facts) in enumerate(pool.map(one, mine), 1):
                if mapping is None:
                    problems[str(facts)] += 1
                else:
                    ready.append((video_id, mapping, facts))
                    if facts.get("rows_before") is not None:
                        gained.append((video_id, facts["rows_before"], len(mapping)))
                if index % 50 == 0 or index == len(mine):
                    print(f"  ... {index}/{len(mine)}", flush=True)

        print(f"\nsan sang : {len(ready)}/{len(mine)}")
        if gained:
            before = sum(b for _, b, _ in gained)
            after = sum(a for _, _, a in gained)
            print(f"dong co chu: {before} -> {after}  (+{after - before}, "
                  f"gap {after / max(1, before):.1f} lan)")
            worst = sorted(gained, key=lambda r: r[1] - r[2])[:5]
            for video_id, b, a in worst:
                print(f"    {video_id:<12} {b:>4} -> {a:>4}")
        if problems:
            print("\nVUONG (khong ghi de nhung video nay):")
            for reason, count in problems.most_common():
                print(f"  {count:>4}  {reason}")

        if not args.write:
            print(f"\n--write chua bat: khong ghi gi. Se ghi de {len(ready)} file.")
            return
        if not ready:
            print("\nKhong co video nao san sang.")
            return

        writer = BundleWriter(transport, layout, args.member, root)
        done = Counter()

        def write_one(item):
            video_id, mapping, facts = item
            try:
                # 1. Sao lưu bản cũ TRƯỚC — và KHÔNG ĐÈ bản sao lưu đã có.
                #
                # Lượt chạy 20-08 chết giữa chừng ở video thứ 311. Chạy lại mà
                # sao lưu vô điều kiện thì 311 video ấy sẽ bị "sao lưu" bằng
                # chính bản ĐÃ REMAP, ghi đè bản gốc — và bản gốc là thứ duy
                # nhất không dựng lại được. Một script có thể phải chạy lại,
                # nên bước sao lưu phải chịu được việc đó.
                remote_backup = f"{layout.root}/parts/{BACKUP_DIR}/{video_id}.json"
                backup = root / f"b_{video_id}.json"
                try:
                    if not transport.exists(remote_backup):
                        transport.get(layout.part("subtitles", video_id), backup)
                        transport.put(backup, remote_backup)
                except Exception as err:  # noqa: BLE001
                    return video_id, f"sao luu that bai, KHONG ghi de: {err}"
                finally:
                    backup.unlink(missing_ok=True)

                # 2. Đường ghi của production: kiểm payload, tải lên, đóng marker
                #    đúng hình dạng `mark_done` — nên marker mới có ca `artifact`
                #    lan `sha256`, thu ma marker viet tay cua notebook 04 khong co.
                marker = writer.write_text_part(
                    "subtitles",
                    video_id,
                    mapping,
                    model=args.model,
                    params=params,
                    facts={
                        "remapped": True,
                        "text_from": "notebook 04 cua quang (Whisper khong chay lai)",
                        "rule": "next_keyframe, chep tu stage_n2_subtitles",
                        "rows_before_remap": facts.get("rows_before"),
                        "segments": facts["segments"],
                        "backup": f"parts/{BACKUP_DIR}/{video_id}.json",
                    },
                )
                writer.write_part_manifest(
                    video_id, [marker], keyframes=facts["keyframes"], merge=True,
                    # Script nay KHONG chay Whisper — chu la chu cu, chi anh xa
                    # sang keyframe la moi. Ghi env cua may nay vao `stages` la
                    # khai rang faster-whisper chay o day, ma no khong he duoc cai.
                    record_env=False,
                )
                return video_id, None
            except Exception as err:  # noqa: BLE001
                return video_id, f"{type(err).__name__}: {err}"

        print(f"\nghi de {len(ready)} file (sao luu -> parts/{BACKUP_DIR}/) ...")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, (video_id, err) in enumerate(pool.map(write_one, ready), 1):
                done["hong" if err else "xong"] += 1
                if err:
                    print(f"  ! {video_id}: {err}", flush=True)
                if index % 50 == 0 or index == len(ready):
                    print(f"  ... {index}/{len(ready)}", flush=True)

    print(f"\nxong {done['xong']} · hong {done['hong']}")
    print(f"Kiem lai:  python scripts/check_shard_drift.py --volume {args.volume} --workers 4")


if __name__ == "__main__":
    main()
