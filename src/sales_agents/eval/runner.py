"""A generic runner for use cases whose gold label is not hot, warm or cold.

Same rules as the lead harness: a failed call is a missing prediction that stays
in the denominator, and a run whose token estimate is above the cap never starts.
"""

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sales_agents.eval.harness import TokenCapExceeded


@dataclass
class RunResult:
    n: int
    predictions: list[Any]  # None where the call failed
    errors: list[tuple[str, str]] = field(default_factory=list)
    cost: int = 0
    latency_p50: float = 0.0
    estimated_tokens: int = 0


def run_cases(
    cases: Sequence,
    fn: Callable[[Any], Any],
    *,
    case_id: Callable[[Any], str],
    token_cap: int,
    estimated_tokens: int = 0,
    count_tokens: Callable[[], int] | None = None,
) -> RunResult:
    if estimated_tokens > token_cap:
        raise TokenCapExceeded(
            f"estimated {estimated_tokens} tokens exceeds the cap of {token_cap}; run a smaller subset"
        )
    predictions, errors, latencies = [], [], []
    for case in cases:
        start = time.perf_counter()
        try:
            predictions.append(fn(case))
        except Exception as exc:  # any failure is a wrong prediction, not a dropped case
            predictions.append(None)
            errors.append((case_id(case), f"{type(exc).__name__}: {exc}"))
        latencies.append(time.perf_counter() - start)
    return RunResult(
        n=len(cases),
        predictions=predictions,
        errors=errors,
        cost=count_tokens() if count_tokens else 0,
        latency_p50=statistics.median(latencies) if latencies else 0.0,
        estimated_tokens=estimated_tokens,
    )
