"""Check a bundle before ingest reads it — §6.4, §6.5, §5.4.

    python scripts/verify_bundle.py --transport modal --volume aic2026
    python scripts/verify_bundle.py --transport local --local-root bundle_local --member triet

Runs *before* `ingest_sota_text.py` for one reason: that script treats a missing
file as a legitimate state, because four of its five sources had no producer for
most of this project. So a bundle short a whole tier loads clean and scores badly,
and nothing in the pipeline says why. This script is the thing that says why.

Three checks the merge cannot do, because they are about the bytes on the Volume
rather than the payloads inside them:

* **hash vs marker** — a marker claims a sha256; the stored file must still have
  it. `modal volume put` is not verified to validate checksums (§5.4), and a file
  overwritten by a later, half-finished run keeps its old marker.
* **author vs assignment** — within one Modal workspace there is no write boundary
  between members (§6.5), so this detects what it cannot prevent.
* **progress** — what each member has finished, which is also the answer to
  "can we merge yet?".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mfe.core.cli import setup_console
from mfe.offline.bundle import ARTIFACTS, BundleLayout, load_shard_plan
from mfe.offline.transport import build_transport


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verify_bundle", description=__doc__)
    parser.add_argument("--shards", default="config/ingest_shards.yaml")
    parser.add_argument("--transport", choices=("modal", "local"), default="modal")
    parser.add_argument("--volume", default="aic2026")
    parser.add_argument("--local-root", default="bundle_local")
    parser.add_argument(
        "--member", default=None, help="Chi kiem mot nguoi. Mac dinh: tat ca."
    )
    parser.add_argument(
        "--check-hashes",
        action="store_true",
        help="Tai lai tung manh de doi chieu sha256 voi marker. Cham nhung la phep "
        "kiem duy nhat bat duoc file bi ghi de dang do.",
    )
    parser.add_argument("--workdir", default=None)
    return parser


def main() -> None:  # noqa: PLR0912, PLR0915 - a report, read top to bottom
    setup_console()
    args = build_parser().parse_args()

    plan = load_shard_plan(args.shards)
    layout = BundleLayout(plan.bundle)
    transport = build_transport(args.transport, volume=args.volume, local_root=args.local_root)
    workdir = Path(args.workdir or Path("eval/merge") / plan.bundle / "verify")
    workdir.mkdir(parents=True, exist_ok=True)

    members = [args.member] if args.member else sorted(plan.members)
    problems: list[str] = []

    # -- progress ----------------------------------------------------------
    wanted_stages = [s for s in plan.stages if s in {a.stage for a in ARTIFACTS.values()}]
    print(f"bundle {plan.bundle} · {len(wanted_stages)} tang co artifact\n")
    print(f"{'nguoi':<8} {'video':>6} {'xong':>6} {'can':>6}  con thieu")
    print("-" * 64)

    markers: dict[tuple[str, str, str], dict] = {}
    for member in members:
        videos = plan.videos_for(member)
        done: set[tuple[str, str]] = set()
        for remote in transport.listdir(layout.marker_dir(member)):
            pieces = remote.strip("/").split("/")
            if len(pieces) < 2 or not pieces[-1].endswith(".json"):
                continue
            stage, video_id = pieces[-2], pieces[-1][: -len(".json")]
            if stage.startswith("_"):
                continue
            done.add((stage, video_id))
            local = workdir / f"marker_{member}_{stage}_{video_id}.json"
            transport.get(remote, local)
            markers[(member, stage, video_id)] = json.loads(
                local.read_text(encoding="utf-8")
            )

        need = {(s, v) for s in wanted_stages for v in videos}
        missing = sorted(need - done)
        label = "-" if not missing else f"{len(missing)} viec"
        print(f"{member:<8} {len(videos):>6} {len(need & done):>6} {len(need):>6}  {label}")
        if missing:
            for stage, video_id in missing[:6]:
                print(f"{'':<30}{stage} / {video_id}")
            if len(missing) > 6:
                print(f"{'':<30}… con {len(missing) - 6} nua")

    # -- author vs assignment ---------------------------------------------
    for (member, stage, video_id), marker in markers.items():
        owner = plan.owner_of(video_id)
        if owner != member:
            problems.append(
                f"sai_tac_gia: {member} cam moc {stage}/{video_id}, "
                f"nhung video do giao cho {owner}"
            )
        if marker.get("author") != member:
            problems.append(
                f"marker_lech: {member}/{stage}/{video_id} khai author="
                f"{marker.get('author')!r}"
            )

    # -- hash vs marker ----------------------------------------------------
    if args.check_hashes:
        print("\ndoi chieu sha256 marker <-> file tren volume …")
        checked = mismatched = 0
        for (member, stage, video_id), marker in sorted(markers.items()):
            artifact = marker.get("artifact")
            claimed = marker.get("sha256")
            if not artifact or not claimed:
                continue
            stored = transport.sha256(layout.part(artifact, video_id))
            checked += 1
            if stored is None:
                problems.append(
                    f"thieu_file: marker {member}/{stage}/{video_id} co nhung "
                    f"{layout.part(artifact, video_id)} khong doc duoc"
                )
            elif stored != claimed:
                mismatched += 1
                problems.append(
                    f"hash_lech: {artifact}/{video_id} — marker {claimed[:12]}, "
                    f"file {stored[:12]}. File bi ghi de sau khi cam moc; "
                    "chay lai viec do."
                )
        print(f"  {checked} manh, {mismatched} lech")

    # -- verdict -----------------------------------------------------------
    if problems:
        print(f"\nCO VAN DE — {len(problems)}:")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)

    total_done = len(markers)
    total_need = sum(len(plan.videos_for(m)) * len(wanted_stages) for m in members)
    print(f"\nkhong thay van de. {total_done}/{total_need} viec da xong.")
    if total_done < total_need:
        print("Chua du de gop — merge_bundle.py se tu choi vi thieu manh.")


if __name__ == "__main__":
    main()
