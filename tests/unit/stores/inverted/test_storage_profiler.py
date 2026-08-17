"""Unit tests for StorageProfiler and BenchmarkResult."""

from __future__ import annotations

import time

import pytest

from mfe.stores.production.profiler import StorageProfiler


def test_profiler_benchmark_callable_success() -> None:
    def dummy_query():
        time.sleep(0.001)
        return "ok"

    res = StorageProfiler.benchmark_callable("dummy_test", dummy_query, iterations=10, warmup=2)
    assert res.operation == "dummy_test"
    assert res.total_queries == 10
    assert res.successful_queries == 10
    assert res.failed_queries == 0
    assert res.p50_ms > 0.0
    assert res.qps > 0.0
    assert res.min_ms <= res.p50_ms <= res.p90_ms <= res.p95_ms <= res.p99_ms <= res.max_ms
    assert res.min_ms <= res.mean_ms <= res.max_ms


def test_profiler_benchmark_callable_failure_handling() -> None:
    def faulty_query():
        raise RuntimeError("simulated failure")

    res = StorageProfiler.benchmark_callable("faulty_test", faulty_query, iterations=5, warmup=1)
    assert res.operation == "faulty_test"
    assert res.total_queries == 5
    assert res.successful_queries == 0
    assert res.failed_queries == 5


def test_profiler_invalid_iterations() -> None:
    with pytest.raises(ValueError, match="iterations must be positive"):
        StorageProfiler.benchmark_callable("test", lambda: None, iterations=0)

