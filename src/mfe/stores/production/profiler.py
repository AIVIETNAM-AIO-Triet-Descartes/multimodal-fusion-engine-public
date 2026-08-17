"""Performance benchmark and latency profiler for Milvus and Elasticsearch stores.

Measures p50, p90, p95, p99 latencies, queries per second (QPS), and error rates.
Essential for validating that query latency remains within competition constraints (<500ms).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: str
    total_queries: int = Field(ge=0)
    successful_queries: int = Field(ge=0)
    failed_queries: int = Field(ge=0)
    total_time_seconds: float = Field(ge=0.0)
    qps: float = Field(ge=0.0)
    p50_ms: float = Field(ge=0.0)
    p90_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)
    p99_ms: float = Field(ge=0.0)
    min_ms: float = Field(ge=0.0)
    max_ms: float = Field(ge=0.0)
    mean_ms: float = Field(ge=0.0)


class StorageProfiler:
    """Benchmark runner for measuring storage backend latency and throughput."""

    @staticmethod
    def benchmark_callable(
        operation_name: str,
        query_fn: Callable[[], Any],
        *,
        iterations: int = 50,
        warmup: int = 5,
    ) -> BenchmarkResult:
        """Run benchmark iterations on a zero-argument callable query function."""
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        # Warmup iterations
        for _ in range(warmup):
            try:
                query_fn()
            except Exception:
                pass

        latencies_ms: list[float] = []
        success_count = 0
        fail_count = 0

        t_start = time.perf_counter()
        for _ in range(iterations):
            t0 = time.perf_counter()
            try:
                query_fn()
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
                success_count += 1
            except Exception:
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
                fail_count += 1
        t_end = time.perf_counter()

        total_time = max(1e-6, t_end - t_start)
        qps = float(success_count / total_time)

        arr = np.array(latencies_ms) if latencies_ms else np.zeros(1)
        return BenchmarkResult(
            operation=operation_name,
            total_queries=iterations,
            successful_queries=success_count,
            failed_queries=fail_count,
            total_time_seconds=float(total_time),
            qps=qps,
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            mean_ms=float(np.mean(arr)),
        )
