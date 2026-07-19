"""Sprint 1 exit criteria, checked automatically — docs/01 §2, docs/00 §1.

The harness prints PASS/FAIL per criterion. The point of automating it: "Done"
should be something the team runs, not something they argue about at review.

docs/00 §1, verbatim:

  - Offline chạy trên một thư mục video bất kỳ bằng một lệnh, đổ ra đủ 4 kho.
  - Online nhận query text -> trả top-k keyframe -> mở đúng video/timestamp.
  - Mapping ID<->frame<->timestamp đúng 100% trên tập mẫu.
  - Mọi submit sai schema bị validator chặn.
  - Chạy được >= 30 query nội bộ không crash, có log tái hiện.
  - Có số liệu Recall@K, TTCA, latency, throughput ghi lại.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

#: docs/00 §1 and docs/context/buoi-1.md §9 both say >= 30.
MIN_QUERIES_WITHOUT_CRASH = 30

#: docs/00 §1: mapping must be 100% on the sample. Not 99%.
REQUIRED_MAPPING_ACCURACY = 1.0


class CriterionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    passed: bool
    detail: str


def check_all(run_summary: dict) -> list[CriterionResult]:
    """Evaluate every criterion against a harness run.

    Owner: Vỹ Triết. TODO(Vỹ Triết): implement the checks below.
    """
    raise NotImplementedError("TODO(Vỹ Triết): evaluate all criteria -> results")


def check_queries_without_crash(run_summary: dict) -> CriterionResult:
    """>= 30 queries completed without a crash, with a reproducible log."""
    raise NotImplementedError("TODO(Vỹ Triết)")


def check_mapping_accuracy(run_summary: dict) -> CriterionResult:
    """Mapping 100% correct on the sample.

    TODO(Vỹ Triết): source this from `RelationalMap.verify_mapping()` rather than
    from query outcomes. A query set can pass while the map is wrong — it only
    touches the keyframes the queries happen to hit.
    """
    raise NotImplementedError("TODO(Vỹ Triết)")


def check_invalid_submits_blocked(run_summary: dict) -> CriterionResult:
    """Every malformed submit was blocked by the validator.

    TODO(Vỹ Triết): this needs deliberately bad submits in the run — a set of known
    violations, one per rule in docs/05 §3. A run where nothing was wrong proves
    the fence works only in the sense that an untested fence is standing.
    """
    raise NotImplementedError("TODO(Vỹ Triết)")


def check_metrics_recorded(run_summary: dict) -> CriterionResult:
    """Recall@K, TTCA, latency, throughput all present.

    TODO(Vỹ Triết): throughput comes from `eval/throughput.json` (Quốc Đạt, docs/02 §5),
    not from this run.
    """
    raise NotImplementedError("TODO(Vỹ Triết)")
