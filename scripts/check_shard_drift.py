"""Đọc `parts/manifest/` trên Volume và trả lời: các mảnh có LỆCH nhau không.

    python scripts/check_shard_drift.py --volume keyframes-btc
    python scripts/check_shard_drift.py --volume keyframes-btc --artifact subtitles

Vì sao cần script này chứ không đợi `merge_bundle.py`
------------------------------------------------------
`merge_bundle.py` đã kiểm đúng chuyện này — nhưng nó **chỉ chạy được khi đã đủ
mảnh cho mọi video**, và nó từ chối cả bundle nếu thiếu. Nghĩa là câu trả lời
"môi trường có lệch không" chỉ đến sau khi cả 873 video đã chạy xong: hàng chục
giờ GPU, và nếu lệch thì phải chạy lại phần đã lệch.

Script này chạy **giữa chừng**, trên bất kỳ số mảnh nào đã có, và nó gọi CHÍNH
`merge.check_fingerprints` chứ không viết lại phép so — nên thứ nó báo đúng là
thứ sẽ chặn lúc gộp, không phải một bản mô phỏng gần đúng.

Lệch ở đâu thì hệ quả khác nhau
---------------------------------
* **`env_lech`** — `merge_bundle.py` **TỪ CHỐI**. Hai bản `faster-whisper` phiên
  âm cùng một đoạn audio ra hai chuỗi khác nhau, và bundle thành hỗn hợp hai phép
  đo mà không có gì downstream phân biệt được. So **theo từng tầng**, và chỉ
  những gói `STAGE_PACKAGES` khai là có thể đổi số của chính tầng đó: manifest
  BTC do bốn notebook cùng ghi, nên `env` ở cấp manifest là của lượt ghi cuối —
  so hai manifest với nhau là đang so hai notebook chứ không phải hai phép đo.
* Manifest ghi **trước 20-08-2026** không có `stages`. Chúng được báo là **chưa
  kiểm**, không phải đã đạt: chạy lại một tầng bất kỳ sẽ ghi `stages` cho video
  đó.
* **`model_lech`** — cùng lý do, ở mức `model`/`params` của từng artifact.
* **`OTHER_PACKAGES`** (`modal`, `pillow`…) — chỉ **cảnh báo**: chúng không đổi
  con số.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mfe.core.cli import setup_console  # noqa: E402
from mfe.offline.bundle import (  # noqa: E402
    NUMERIC_PACKAGES,
    OTHER_PACKAGES,
    STAGE_PACKAGES,
    BundleLayout,
)
from mfe.offline.merge import MergeReport, check_fingerprints  # noqa: E402
from mfe.offline.transport import build_transport  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="check_shard_drift", description=__doc__)
    parser.add_argument("--volume", default="keyframes-btc")
    parser.add_argument("--bundle", default="aic_btc_b1_v1")
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--artifact", default=None, help="Chi xem mot artifact")
    parser.add_argument("--workers", type=int, default=8, help="So luong doc song song")
    parser.add_argument("--out", default="eval/results/shard_drift.json")
    return parser


def load_manifests(transport, layout: BundleLayout, workers: int = 8) -> dict[str, dict]:
    """Mọi `parts/manifest/{video}.json` đang có trên Volume.

    Đọc **song song**: mỗi manifest chỉ vài KB, nên nút thắt là độ trễ vòng mạng
    chứ không phải băng thông. Tuần tự thì 658 file mất ~40 phút và không ai đợi
    một phép kiểm lâu thế — nó sẽ bị bỏ qua, mà bỏ qua là quay lại đúng chỗ phát
    hiện lệch sau khi đã chạy xong cả batch.

    Số luồng để nhỏ có chủ ý. Modal khuyên không quá ~5 commit đồng thời; đó là
    luật cho GHI, còn đây là ĐỌC — nhưng cùng một hạ tầng, nên không tham.
    """
    from concurrent.futures import ThreadPoolExecutor

    root = f"{layout.root}/parts/manifest"
    remotes = sorted(r for r in transport.listdir(root) if r.endswith(".json"))
    out: dict[str, dict] = {}
    done = 0

    def fetch(remote: str, tmp: Path) -> tuple[str, dict | None]:
        video_id = Path(remote).stem
        local = tmp / f"{video_id}.json"
        try:
            transport.get(remote, local)
            return video_id, json.loads(local.read_text(encoding="utf-8"))
        except Exception as err:  # noqa: BLE001
            print(f"  (bo qua {video_id}: {type(err).__name__}: {err})")
            return video_id, None

    with tempfile.TemporaryDirectory() as tmp:
        root_tmp = Path(tmp)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for video_id, body in pool.map(lambda r: fetch(r, root_tmp), remotes):
                done += 1
                if body is not None:
                    out[video_id] = body
                if done % 50 == 0 or done == len(remotes):
                    print(f"  ... doc {done}/{len(remotes)}")
    return out


def main() -> None:
    setup_console()
    args = build_parser().parse_args()

    layout = BundleLayout(args.bundle)
    transport = build_transport(
        "modal", volume=args.volume, local_root="bundle_local", version=args.version
    )

    print(f"volume {args.volume} · bundle {args.bundle}")
    print("doc parts/manifest/ ...")
    manifests = load_manifests(transport, layout, workers=args.workers)
    if not manifests:
        raise SystemExit("khong co manifest nao — chua ai chay xong video nao")
    print(f"{len(manifests)} manifest\n")

    # --- ai lam gi -------------------------------------------------------
    by_author = Counter(m.get("author", "?") for m in manifests.values())
    print("theo nguoi:")
    for author, count in sorted(by_author.items()):
        print(f"  {author:<8} {count:>4} video")

    artifacts = Counter(
        name for m in manifests.values() for name in m.get("artifacts", {})
    )
    print("\ntheo artifact:")
    for name, count in sorted(artifacts.items()):
        print(f"  {name:<16} {count:>4} video")

    # --- moi truong tung TANG --------------------------------------------
    #
    # Day moi la don vi dung. Mot manifest BTC duoc bon notebook ghi chung, nen
    # `env` o cap manifest la cua LUOT GHI CUOI — so hai manifest voi nhau la
    # dang so hai notebook, khong phai hai phep do.
    print("\nmoi truong theo TANG (chi goi co the doi so cua chinh tang do):")
    per_stage: dict[str, dict[str, set[str]]] = {}
    no_stages = 0
    for manifest in manifests.values():
        stages = manifest.get("stages") or {}
        if not stages:
            no_stages += 1
            continue
        for stage, body in stages.items():
            env = (body or {}).get("env") or {}
            slot = per_stage.setdefault(stage, {})
            for package in STAGE_PACKAGES.get(stage, NUMERIC_PACKAGES):
                slot.setdefault(package, set()).add(env.get(package) or "(khong cai)")

    if not per_stage:
        print("  (khong manifest nao ghi `stages` — xem canh bao o duoi)")
    for stage in sorted(per_stage):
        print(f"  {stage}:")
        if not per_stage[stage]:
            print("    (khong chay model nao — khong co goi nao de so)")
        for package, values in sorted(per_stage[stage].items()):
            mark = "  <-- LECH" if len(values) > 1 else ""
            print(f"    {package:<22} {', '.join(sorted(values))}{mark}")
    if no_stages:
        print(f"\n  {no_stages}/{len(manifests)} manifest KHONG ghi `stages` "
              "(ghi truoc 20-08-2026) — moi truong cua chung la CHUA KIEM.")

    # --- moi truong tung nguoi (cap manifest, chi de tham khao) -----------
    print("\nmoi truong o cap manifest — cua LUOT GHI CUOI, khong phai cua tung tang:")
    env_by_author: dict[str, dict[str, set]] = {}
    for manifest in manifests.values():
        author = manifest.get("author", "?")
        slot = env_by_author.setdefault(author, {})
        for package, version in (manifest.get("env") or {}).items():
            slot.setdefault(package, set()).add(version)

    packages = sorted(
        {p for slot in env_by_author.values() for p in slot}
        & set(NUMERIC_PACKAGES + OTHER_PACKAGES)
    )
    authors = sorted(env_by_author)
    width = max((len(p) for p in packages), default=10)
    header = f"  {'goi':<{width}} " + " ".join(f"{a:<18}" for a in authors)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for package in packages:
        cells = []
        for author in authors:
            values = env_by_author[author].get(package, {"(khong cai)"})
            cells.append(",".join(sorted(values)))
        distinct = {c for c in cells if c}
        mark = "  <-- LECH" if len(distinct) > 1 else ""
        kind = "NUMERIC" if package in NUMERIC_PACKAGES else "khac"
        print(f"  {package:<{width}} " + " ".join(f"{c:<18}" for c in cells)
              + f"  [{kind}]{mark}")

    # --- lech co CHAM vao so nao khong ------------------------------------
    #
    # Cau hoi that su dang sau `env_lech`. Mot video chi qua `n0_import_btc`
    # (nap keyframe, KHONG chay model nao) thi ban dung torch cua no khong tinh
    # ra con so nao — bao "lech torch" o day la dung ve hinh thuc va vo nghia ve
    # noi dung. Chi khi mot artifact DO MODEL SINH RA nam cung manifest voi mot
    # phien ban lech thi moi dang lo.
    numeric_artifacts = {
        "siglip_vectors", "dense_vectors", "audio_features",
        "ocr_text", "sviptr_ocr", "paddle_ocr", "viet_ocr",
        "subtitles", "objects_text", "dense_captions",
    }
    # Chi con y nghia cho manifest CU (khong co `stages`): voi chung, `env` cap
    # manifest la bang chung duy nhat con lai, va cau hoi dung la "lech do co
    # cham vao artifact nao khong".
    print("\nlech o cap manifest CO cham vao artifact do model sinh khong:")
    for package in ("torch", "faster-whisper", "transformers", "easyocr"):
        values: dict[str, set[str]] = {}
        for manifest in manifests.values():
            value = (manifest.get("env") or {}).get(package) or "(khong cai)"
            touched = set(manifest.get("artifacts", {})) & numeric_artifacts
            values.setdefault(value, set()).update(touched)
        if len(values) < 2:
            continue
        print(f"  {package}:")
        for value, touched in sorted(values.items()):
            n = sum(
                1
                for m in manifests.values()
                if ((m.get("env") or {}).get(package) or "(khong cai)") == value
            )
            label = ", ".join(sorted(touched)) if touched else "KHONG artifact model nao"
            print(f"    {value:<22} {n:>4} video · {label}")

    # --- chinh phep kiem cua merge ---------------------------------------
    print("\n" + "=" * 72)
    print("chay CHINH `merge.check_fingerprints` tren nhung manifest nay:")
    report = MergeReport(bundle=args.bundle)
    check_fingerprints(manifests, report)

    refusals = [r for r in report.refusals]
    if args.artifact:
        refusals = [r for r in refusals if args.artifact in r.detail]
    if not refusals:
        print("  KHONG co lech nao — merge se khong tu choi vi ly do nay.")
    else:
        print(f"  {len(refusals)} LY DO TU CHOI:\n")
        for refusal in refusals:
            print(f"  [{refusal.kind}] {refusal.detail[:400]}")
            print()
    for warning in report.warnings:
        print(f"  (canh bao) {warning}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "volume": args.volume,
                "bundle": args.bundle,
                "manifests": len(manifests),
                "by_author": dict(by_author),
                "by_artifact": dict(artifacts),
                "env_by_author": {
                    a: {p: sorted(v) for p, v in slot.items()}
                    for a, slot in env_by_author.items()
                },
                "refusals": [
                    {"kind": r.kind, "detail": r.detail} for r in report.refusals
                ],
                "warnings": report.warnings,
                # Mot dong moi video: du de tra loi "video nao, cua ai, moi
                # truong gi, co artifact gi" ma khong phai doc lai Volume.
                "env_by_stage": {
                    stage: {p: sorted(v) for p, v in slot.items()}
                    for stage, slot in sorted(per_stage.items())
                },
                "manifests_without_stages": no_stages,
                "videos": {
                    video_id: {
                        "author": m.get("author"),
                        "artifacts": sorted(m.get("artifacts", {})),
                        "stages": {
                            s: (b or {}).get("author")
                            for s, b in sorted((m.get("stages") or {}).items())
                        },
                        "env": {
                            k: v
                            for k, v in (m.get("env") or {}).items()
                            if k in NUMERIC_PACKAGES
                        },
                    }
                    for video_id, m in sorted(manifests.items())
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nBao cao: {out}")


if __name__ == "__main__":
    main()
