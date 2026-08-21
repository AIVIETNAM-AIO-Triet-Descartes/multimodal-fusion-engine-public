"""Dựng lại mục `artifacts` trong manifest cho những mảnh ĐÃ có file trên Volume.

    python scripts/rebuild_artifact_manifest.py --member quang \
        --artifacts subtitles,audio_features --params config/quang_nb04_params.json
    ... thêm --write để ghi thật

Vì sao cần
-----------
`write_part_manifest` chỉ đưa vào manifest những marker **có khoá `artifact`**.
Marker do `mark_done` sinh ra thì có; marker do một bản cài đặt khác viết tay thì
không nhất thiết. Quang chạy notebook 04 bằng bản của bạn ấy, marker mang
`whisper_model`/`clap_model`/`num_segments` chứ không mang `artifact`/`model`/
`params`/`sha256` — nên 469 mảnh `subtitles` và `audio_features` có file thật
trên Volume mà **không mảnh nào được manifest nhắc tới**.

Hệ quả không phải mất dữ liệu: `merge_bundle.py` đọc mảnh từ `_list_parts`, tức
file thật, nên chúng vẫn được gộp. Hệ quả là **mất phép kiểm**:
`check_fingerprints` so `model`/`params` **qua manifest**, nên nó chỉ so 404 mảnh
của Trọng rồi báo "không có lệch nào" — trong khi một nửa bundle chưa từng được
so. Đó đúng là kiểu hỏng mà phép kiểm này sinh ra để bắt.

`params` phải do người điền, không được đoán
----------------------------------------------
`rows`, `bytes`, `sha256` đọc được từ chính file trên Volume. `model` đọc được từ
marker. Nhưng **`params` thì chưa từng được ghi ở đâu cả** — và nó là thứ
`check_fingerprints` so gắt nhất.

Nên script này **bắt buộc** một file `--params` do người vận hành điền từ chính
cấu hình lượt chạy đó. Suy `params` ra từ dữ liệu là bịa: đọc file phiên âm rồi
đoán ngược ra `beam_size` là không thể, và một con số đoán ở đây trông y hệt một
con số đo được — trong khi cả phép kiểm này tồn tại chỉ để phân biệt hai thứ đó.

Tham số nào KHÔNG biết thì ghi `"khong ghi lai"`, đừng bỏ trống và đừng đoán. Ghi
như vậy sẽ khiến `check_fingerprints` từ chối, và **từ chối là đúng** — ta thật
sự không biết hai bên đo cùng một cách. Muốn merge chạy tiếp thì Lead ký duyệt
trong `config/accepted_divergences.yaml`, ở đó lý do được ghi lại và đi theo
bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mfe.core.cli import setup_console  # noqa: E402
from mfe.offline.bundle import (  # noqa: E402
    ARTIFACTS,
    BundleLayout,
    BundleWriter,
    load_shard_plan,
)
from mfe.offline.transport import build_transport  # noqa: E402

#: Marker của bản cài đặt khác không dùng khoá `model`. Đây là những chỗ đã thấy
#: tên model thật nằm — tra theo thứ tự, và **không** có bước đoán cuối cùng.
MODEL_KEYS = {
    "subtitles": ("model", "whisper_model", "asr_model"),
    "audio_features": ("model", "clap_model", "audio_model"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rebuild_artifact_manifest", description=__doc__)
    parser.add_argument("--member", required=True)
    parser.add_argument("--artifacts", required=True, help="Ngan bang phay")
    parser.add_argument("--params", required=True, help="JSON {artifact: {params}}")
    parser.add_argument("--shards", default="config/ingest_shards_batch1_btc.yaml")
    parser.add_argument("--bundle", default="aic_btc_b1_v1")
    parser.add_argument("--volume", default="keyframes-btc")
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--write", action="store_true", help="Khong co thi chi kiem")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--write-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def model_of(marker: dict, artifact: str) -> str | None:
    for key in MODEL_KEYS.get(artifact, ("model",)):
        value = marker.get(key)
        if value:
            return str(value)
    return None


def inspect(transport, layout, video_id: str, artifact: str, member: str, tmp: Path):
    """Đọc marker + file thật, trả về mọi thứ trừ `params`.

    Trả `None` kèm lý do khi thiếu — thiếu marker hoặc thiếu file đều là trạng
    thái phải chạy lại tầng đó, không phải chỗ để điền bừa.
    """
    stage = ARTIFACTS[artifact].stage
    marker_local = tmp / f"mk_{artifact}_{video_id}.json"
    try:
        transport.get(layout.marker(member, stage, video_id), marker_local)
        marker = json.loads(marker_local.read_text(encoding="utf-8"))
    except Exception:
        return None, "khong co marker"
    finally:
        marker_local.unlink(missing_ok=True)

    model = model_of(marker, artifact)
    if model is None:
        return None, f"marker khong co ten model (khoa da thu: {MODEL_KEYS[artifact]})"

    part_local = tmp / f"pt_{artifact}_{video_id}"
    try:
        transport.get(layout.part(artifact, video_id), part_local)
        raw = part_local.read_bytes()
    except Exception:
        return None, "khong co file manh tren volume"
    try:
        if ARTIFACTS[artifact].kind == "text":
            body = json.loads(raw.decode("utf-8"))
            rows, empty = len(body), sum(1 for v in body.values() if not str(v).strip())
        else:
            import numpy as np  # noqa: PLC0415

            with np.load(part_local, allow_pickle=False) as payload:
                rows, empty = len(payload["ids"]), 0
    except Exception as err:  # noqa: BLE001
        return None, f"khong doc duoc manh: {type(err).__name__}: {err}"
    finally:
        part_local.unlink(missing_ok=True)

    return {
        "artifact": artifact,
        "model": model,
        "rows": rows,
        "empty": empty,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rebuilt_from": "marker + file tren volume",
    }, None


def main() -> None:
    setup_console()
    args = build_parser().parse_args()

    params_all = json.loads(Path(args.params).read_text(encoding="utf-8"))
    artifacts = [a.strip() for a in args.artifacts.split(",") if a.strip()]
    for artifact in artifacts:
        if artifact not in ARTIFACTS:
            raise SystemExit(f"{artifact!r} khong co trong ARTIFACTS")
        if artifact not in params_all:
            raise SystemExit(
                f"{args.params} thieu params cho {artifact!r}. Params chua tung duoc "
                "ghi o dau ca — phai lay tu cau hinh luot chay, khong duoc doan."
            )

    plan = load_shard_plan(args.shards)
    layout = BundleLayout(args.bundle)
    transport = build_transport(
        "modal", volume=args.volume, local_root="bundle_local", version=args.version
    )
    mine = plan.videos_for(args.member)
    if args.limit:
        mine = mine[: args.limit]

    print(f"nguoi    : {args.member} — {len(mine)} video")
    print(f"artifact : {', '.join(artifacts)}")
    for artifact in artifacts:
        print(f"  {artifact}: params = {json.dumps(params_all[artifact], ensure_ascii=False)}")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plans: dict[str, list[dict]] = {}
        problems: Counter = Counter()

        for artifact in artifacts:
            print(f"kiem {artifact} tren {len(mine)} video ...")
            found: list[dict] = []

            def one(video_id: str, artifact=artifact):
                facts, reason = inspect(
                    transport, layout, video_id, artifact, args.member, root
                )
                return video_id, facts, reason

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for index, (video_id, facts, reason) in enumerate(pool.map(one, mine), 1):
                    if facts is None:
                        problems[f"{artifact}: {reason}"] += 1
                    else:
                        found.append({"video_id": video_id, **facts})
                    if index % 50 == 0 or index == len(mine):
                        print(f"  ... {index}/{len(mine)}")
            plans[artifact] = found
            models = Counter(item["model"] for item in found)
            print(f"  dung duoc {len(found)}/{len(mine)} · model: {dict(models)}\n")

        if problems:
            print("VUONG:")
            for reason, count in problems.most_common():
                print(f"  {count:>4}  {reason}")
            print()

        if not args.write:
            total = sum(len(v) for v in plans.values())
            print(f"--write chua bat: se ghi {total} muc artifact vao manifest.")
            return

        # Gộp theo video: một video ghi manifest MỘT lần cho cả hai artifact.
        # `write_part_manifest` đọc lại manifest cũ mỗi lần gọi, nên gọi hai lần
        # cho một video là hai vòng mạng thừa và một cửa sổ để ghi đè lẫn nhau.
        by_video: dict[str, list[dict]] = {}
        for artifact, items in plans.items():
            for item in items:
                marker = {k: v for k, v in item.items() if k != "video_id"}
                marker["params"] = params_all[artifact]
                by_video.setdefault(item["video_id"], []).append(marker)

        writer = BundleWriter(transport, layout, args.member, root)
        done = Counter()

        def keyframe_count(video_id: str) -> int:
            """Số keyframe manifest ĐANG ghi, đọc lại để giữ nguyên.

            `write_part_manifest` ghi `counts.keyframes` từ tham số truyền vào —
            nó không gộp trường này như gộp `artifacts`. Truyền 0 vào đây sẽ xoá
            trắng con số mà `n0_import_btc` đã đo, và xoá theo kiểu không ai thấy:
            manifest vẫn hợp lệ, chỉ nói rằng video có 0 keyframe.
            """
            local = root / f"cnt_{video_id}.json"
            try:
                transport.get(layout.part_manifest(video_id), local)
                return int(json.loads(local.read_text(encoding="utf-8"))["counts"]["keyframes"])
            except Exception:
                # Không có manifest cũ: dùng số dòng của chính mảnh vừa đọc —
                # một mảnh có một dòng mỗi keyframe.
                return max(m.get("rows", 0) for m in by_video[video_id])
            finally:
                local.unlink(missing_ok=True)

        def write_one(pair):
            video_id, markers = pair
            try:
                writer.write_part_manifest(
                    video_id, markers, keyframes=keyframe_count(video_id), merge=True,
                    # Dung lai muc manifest tu marker + file da co: khong model
                    # nao chay o day, nen khong co moi truong nao de khai.
                    record_env=False,
                )
                return video_id, None
            except Exception as err:  # noqa: BLE001
                return video_id, f"{type(err).__name__}: {err}"

        print(f"ghi manifest cho {len(by_video)} video ...")
        with ThreadPoolExecutor(max_workers=args.write_workers) as pool:
            for index, (video_id, err) in enumerate(pool.map(write_one, by_video.items()), 1):
                done["hong" if err else "xong"] += 1
                if err:
                    print(f"  ! {video_id}: {err}")
                if index % 50 == 0 or index == len(by_video):
                    print(f"  ... {index}/{len(by_video)}")

    print(f"\nxong {done['xong']} · hong {done['hong']}")
    print("Kiem lai:  python scripts/check_shard_drift.py --volume " + args.volume)


if __name__ == "__main__":
    main()
