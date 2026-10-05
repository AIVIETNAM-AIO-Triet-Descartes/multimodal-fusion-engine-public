"""So hai báo cáo `mfe.eval.official` trên bộ dev — dùng cho A/B B0 so với B1.

Vì sao cần một script chứ không đọc hai con số tổng
----------------------------------------------------
Hai lượt chạy chênh nhau 0,005 có thể là "khá hơn đều" hoặc là "3 câu lên, 3 câu
xuống". Hai chuyện đó dẫn tới hai quyết định khác nhau, và con số tổng không
phân biệt được. Cột **lên / xuống / đứng yên** là thứ phân biệt.

`mfe.eval.official` chỉ ghi `per_query` cho bộ **dev** — heldout cố ý lược đi
(kỷ luật hộp đen). Nên script này chỉ chạy được trên dev, và đó là đúng.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path, window: str) -> dict[str, dict]:
    """`{query_id: QueryScore}` cho một trong hai cửa sổ.

    `QueryOutcome` là `{"wide": {...}, "strict": {...}}` — cùng câu trả lời, hai
    cái thước. `strict` là `None` với câu không có anchor, và ở đây nó thành một
    hàng RỖNG chứ không bị bỏ: bỏ đi là hai lượt chạy có thể còn khác số câu.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("per_query")
    if rows is None:
        raise SystemExit(
            f"{path}: khong co `per_query`. Day la bao cao HELDOUT — "
            "no co y luoc di, va A/B theo tung cau khong doc duoc tu no."
        )
    out: dict[str, dict] = {}
    for row in rows:
        query_id = row["wide"]["query_id"]
        out[query_id] = row.get(window) or {}
    return out


def score(row: dict, key: str) -> float:
    value = row.get(key)
    return 0.0 if value is None else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="Bao cao nen (B0)")
    parser.add_argument("--b", type=Path, required=True, help="Bao cao so (B1)")
    parser.add_argument("--label-a", default="B0")
    parser.add_argument("--label-b", default="B1")
    parser.add_argument(
        "--window",
        choices=("wide", "strict"),
        default="wide",
        help="RONG = span canh, doc lap voi cach trich keyframe — day la cua so "
        "de so khi doi cach trich. HEP = cua so quanh anchor, va anchor LUON la "
        "mot keyframe nen do thua khong bao gio lo ra o do.",
    )
    parser.add_argument(
        "--key",
        default="final_score",
        help="Truong diem trong QueryScore (mac dinh final_score)",
    )
    parser.add_argument("--top", type=int, default=15, help="So cau in ra moi chieu")
    args = parser.parse_args()

    a, b = load(args.a, args.window), load(args.b, args.window)
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    if only_a or only_b:
        raise SystemExit(
            "hai bao cao KHONG cung bo cau: "
            f"chi co o {args.label_a} {len(only_a)}, chi co o {args.label_b} {len(only_b)}. "
            "So hai bo cau khac nhau la so hai he thong khac nhau."
        )

    deltas = []
    for query_id in sorted(a):
        delta = score(b[query_id], args.key) - score(a[query_id], args.key)
        deltas.append((delta, query_id))

    up = [d for d in deltas if d[0] > 1e-9]
    down = [d for d in deltas if d[0] < -1e-9]
    same = len(deltas) - len(up) - len(down)
    mean_a = sum(score(a[q], args.key) for q in a) / len(a)
    mean_b = sum(score(b[q], args.key) for q in b) / len(b)

    print(f"cua so {args.window.upper()} · truong {args.key} · {len(deltas)} cau\n")
    print(f"  {args.label_a:<6}{mean_a:>10.4f}")
    print(f"  {args.label_b:<6}{mean_b:>10.4f}")
    print(f"  {'Δ':<6}{mean_b - mean_a:>+10.4f}\n")
    print(f"  len {len(up)}  ·  xuong {len(down)}  ·  dung yen {same}")

    if not up and not down:
        print(
            "\n  KHONG cau nao doi. Hai lo hang ra cung ket qua tren bo nay — "
            "phep do nay khong noi duoc gi ve chung, ke ca dau lan am."
        )
        return

    for title, rows, reverse in (
        (f"LEN nhieu nhat ({args.label_b} > {args.label_a})", up, True),
        (f"XUONG nhieu nhat ({args.label_b} < {args.label_a})", down, False),
    ):
        if not rows:
            continue
        print(f"\n  {title}")
        for delta, query_id in sorted(rows, reverse=reverse)[: args.top]:
            print(
                f"    {query_id:<24}{score(a[query_id], args.key):>7.3f}"
                f" -> {score(b[query_id], args.key):>6.3f}   {delta:>+7.3f}"
            )


if __name__ == "__main__":
    main()
