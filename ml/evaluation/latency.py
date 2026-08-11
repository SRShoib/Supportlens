"""CPU single-request latency benchmark (SPEC §3: "classification < 150 ms",
"measured and reported" — not a hard gate, but every number here must come
from an actual timed run, never a guess, per CLAUDE.md rule #5).

Generic over any Predictor (baseline or transformer) so the same helper
produces comparable numbers for the M3 comparison report.
"""

import time
from dataclasses import dataclass
from statistics import mean, median
from typing import Any

from ml.inference.base import Predictor


@dataclass(frozen=True)
class LatencyResult:
    n_runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


def benchmark_latency(
    predictor: Predictor[Any], text: str, *, warmup: int = 3, runs: int = 20
) -> LatencyResult:
    """Single-text predict() calls, timed one at a time (SPEC's budget is
    per-request, not batch throughput). warmup discards first-call overhead
    (lazy CUDA/MKL init, tokenizer caching) that a real warm server never
    pays per request."""
    for _ in range(warmup):
        predictor.predict([text])

    samples_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        predictor.predict([text])
        samples_ms.append((time.perf_counter() - start) * 1000)

    samples_ms.sort()
    p95_index = min(int(0.95 * (runs - 1)), runs - 1)
    return LatencyResult(
        n_runs=runs,
        mean_ms=mean(samples_ms),
        p50_ms=median(samples_ms),
        p95_ms=samples_ms[p95_index],
        max_ms=max(samples_ms),
    )
