"""Merge the five members' query sets into one dev and one held-out file.

    python -m mfe.eval.merge_queryset          # validate + write both files
    python -m mfe.eval.merge_queryset --check  # validate only, write nothing

    eval/dev/{member}.jsonl       ->  eval/queryset_dev.jsonl
    eval/heldout/{member}.jsonl   ->  eval/queryset_heldout.jsonl
    eval/manifests/{member}.json  ->  which videos each member owns

Five people writing ground truth on five disjoint samples has exactly one
failure mode that matters, and it is silent: a query that names a video the
author does not own. It parses, it merges, and it scores 0.0 forever, because
the video is not in that person's index and never will be. So the merge checks
every `gt_video_id` against the author's own manifest and refuses the whole run
if one is wrong.

**Nothing is dropped, ever.** A bad line stops the merge and gets reported by
name. Skipping it would quietly shrink the query set — and the count is what
docs/00 §1 gates on, so a silently shorter set reads as a passing run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from mfe.core.cli import setup_console
from mfe.eval.official import AnyQuery, TRAKEQuery, load_official_queryset

DEV_DIR = Path("eval/dev")
HELDOUT_DIR = Path("eval/heldout")
MANIFEST_DIR = Path("eval/manifests")
DEV_OUT = Path("eval/queryset_dev.jsonl")
HELDOUT_OUT = Path("eval/queryset_heldout.jsonl")


class MergeError(Exception):
    """One or more problems that must be fixed by a person, not by this script."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(f"{len(problems)} problem(s)")


def read_manifests(directory: Path | None = None) -> dict[str, set[str]]:
    """member -> the video ids they own, from `prepare_sample_2026.py`.

    The default is read at call time, not bound into the signature: a default
    of `MANIFEST_DIR` would freeze the path at import and quietly ignore any
    later override, which is how a caller ends up merging an empty directory
    and being told nobody has submitted anything.
    """
    directory = directory if directory is not None else MANIFEST_DIR
    owned: dict[str, set[str]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        videos = payload.get("videos") or []
        if not videos:
            raise MergeError([f"{path}: manifest has no 'videos'"])
        owned[path.stem] = set(videos)
    return owned


def check_no_shared_videos(owned: dict[str, set[str]]) -> list[str]:
    """Two members owning the same video defeats the point of the split.

    Named rather than resolved: whose sample is wrong is a question for the two
    people involved, and picking one automatically would silently discard
    ground truth somebody spent an afternoon writing.
    """
    problems: list[str] = []
    members = sorted(owned)
    for i, first in enumerate(members):
        for second in members[i + 1 :]:
            shared = sorted(owned[first] & owned[second])
            if shared:
                problems.append(
                    f"{first} and {second} both claim {len(shared)} video(s): "
                    f"{', '.join(shared[:6])}{'…' if len(shared) > 6 else ''}. "
                    f"One of them re-ran prepare_sample_2026.py with the wrong "
                    f"MFE_MEMBER."
                )
    return problems


def check_query(
    query: AnyQuery, member: str, split: str, owned: dict[str, set[str]]
) -> tuple[list[str], list[str]]:
    """Check one line. Returns (errors, warnings), each reported all at once.

    The split matters. An **error** means the query cannot be scored correctly
    — a wrong video, a swapped range — and blocks the merge. A **warning** means
    it scores, just less informatively; blocking on those would stop 114 good
    queries because one is missing an optional field.
    """
    problems: list[str] = []
    warnings: list[str] = []
    where = f"eval/{split}/{member}.jsonl :: {query.query_id}"

    if query.author is not None and query.author != member:
        problems.append(f"{where}: author is '{query.author}' but the file is {member}'s")
    if query.split is not None and query.split != split:
        problems.append(f"{where}: split says '{query.split}' but it sits in eval/{split}/")
    if not query.query_id.startswith(f"{member}_"):
        problems.append(
            f"{where}: query_id should start with '{member}_' so ids stay unique on merge"
        )

    videos = owned.get(member)
    if videos is None:
        problems.append(f"{where}: no manifest at {MANIFEST_DIR / f'{member}.json'}")
    elif query.gt_video_id not in videos:
        problems.append(
            f"{where}: gt_video_id '{query.gt_video_id}' is not in {member}'s sample. "
            f"This query can never be answered — the video is not in their index."
        )

    try:
        query.ground_truth()
    except Exception as err:  # noqa: BLE001 — report, do not classify
        problems.append(f"{where}: ground truth is invalid ({err})")

    try:
        if query.ground_truth_strict() is None:
            warnings.append(
                f"{where}: no anchor frame — scored on the wide window only. "
                f"Add gt_anchor_frame (KIS/QA) or gt_anchor_frames (TRAKE); it is "
                f"the keyframe you already looked at, so it costs nothing."
            )
    except Exception as err:  # noqa: BLE001 — a *broken* anchor is an error
        problems.append(f"{where}: strict window is invalid ({err})")

    return problems, warnings


def collect(
    directory: Path, split: str, owned: dict[str, set[str]]
) -> tuple[list[AnyQuery], list[str], list[str]]:
    """Load and validate every member's file. Returns (queries, errors, warnings)."""
    queries: list[AnyQuery] = []
    problems: list[str] = []
    warnings: list[str] = []
    for path in sorted(directory.glob("*.jsonl")):
        member = path.stem
        try:
            lines = load_official_queryset(path)
        except Exception as err:  # noqa: BLE001 — a malformed line is a person's job
            problems.append(f"{path}: {err}")
            continue
        for query in lines:
            errors, warns = check_query(query, member, split, owned)
            problems.extend(errors)
            warnings.extend(warns)
        queries.extend(lines)
    return queries, problems, warnings


def check_duplicate_ids(queries: list[AnyQuery]) -> list[str]:
    counts = Counter(query.query_id for query in queries)
    return [
        f"query_id '{query_id}' appears {count} times across the merged set"
        for query_id, count in sorted(counts.items())
        if count > 1
    ]


def write(queries: list[AnyQuery], destination: Path) -> None:
    destination.write_text(
        "\n".join(
            query.model_dump_json(exclude_none=True, exclude_defaults=False)
            for query in queries
        )
        + "\n",
        encoding="utf-8",
    )


def _summarise(label: str, queries: list[AnyQuery]) -> None:
    by_type = Counter(query.type for query in queries)
    by_author = Counter(query.author or "?" for query in queries)
    no_split = sum(
        1 for q in queries if isinstance(q, TRAKEQuery) and not q.event_texts
    )
    print(f"\n{label}: {len(queries)} câu")
    print("  theo dạng  : " + ", ".join(f"{k} {v}" for k, v in sorted(by_type.items())))
    print("  theo người : " + ", ".join(f"{k} {v}" for k, v in sorted(by_author.items())))
    if no_split:
        print(f"  TRAKE chưa tách event_texts: {no_split} (sẽ tìm bằng nguyên câu)")


def run(check_only: bool = False) -> int:
    owned = read_manifests()
    if not owned:
        print(f"Không thấy manifest nào trong {MANIFEST_DIR}/ — chưa ai nộp bộ của mình.")
        return 2

    problems = check_no_shared_videos(owned)
    dev, dev_problems, dev_warnings = collect(DEV_DIR, "dev", owned)
    heldout, heldout_problems, heldout_warnings = collect(HELDOUT_DIR, "heldout", owned)
    problems += dev_problems + heldout_problems
    problems += check_duplicate_ids(dev + heldout)
    warnings = dev_warnings + heldout_warnings

    if problems:
        print(f"\nDỪNG — {len(problems)} lỗi, không ghi file nào:\n")
        for problem in problems:
            print(f"  - {problem}")
        print("\nSửa xong chạy lại. Script không tự bỏ dòng lỗi.")
        return 1

    if warnings:
        print(f"\nCẢNH BÁO — {len(warnings)} chỗ vẫn gộp được nhưng nên sửa:\n")
        for warning in warnings[:20]:
            print(f"  ! {warning}")
        if len(warnings) > 20:
            print(f"  … và {len(warnings) - 20} cảnh báo nữa")

    _summarise("DEV", dev)
    _summarise("HELD-OUT", heldout)

    if check_only:
        print("\n--check: hợp lệ, chưa ghi file.")
        return 0

    write(dev, DEV_OUT)
    write(heldout, HELDOUT_OUT)
    print(f"\nĐã ghi {DEV_OUT} và {HELDOUT_OUT}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mfe-merge-queryset",
        description="Validate and merge the five members' query sets.",
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate only; write nothing."
    )
    return parser


def main() -> None:
    setup_console()
    args = build_parser().parse_args()
    try:
        sys.exit(run(check_only=args.check))
    except MergeError as err:
        for problem in err.problems:
            print(f"  - {problem}")
        sys.exit(1)


if __name__ == "__main__":
    main()
