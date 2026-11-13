"""Shared helper for live (real model) measurements.

Counts the tokens a run really used through LangChain's usage callback, and runs
the harness behind the token guard. Callers pass an estimate made before the run.
"""

from collections.abc import Callable

from langchain_core.callbacks import get_usage_metadata_callback

from sales_agents.eval.harness import EvalReport, run_eval
from sales_agents.eval.tokens import estimate_tokens
from sales_agents.schemas import LabelledLead, Lead, QualificationResult

SUBSET_TOKEN_LIMIT = 1_000_000  # global rule: first run on a subset below 1M tokens


def measure(
    fn: Callable[[Lead], QualificationResult],
    cases: list[LabelledLead],
    *,
    prompt_tokens: int,
    calls_per_case: int,
    token_cap: int,
    overhead_per_case: int = 0,
) -> EvalReport:
    """Run fn over the cases with the token guard and real usage counting."""
    estimate = estimate_tokens(
        cases, prompt_tokens=prompt_tokens, calls_per_case=calls_per_case, overhead_per_case=overhead_per_case
    )
    with get_usage_metadata_callback() as usage:
        return run_eval(
            cases,
            fn,
            token_cap=token_cap,
            estimated_tokens=estimate,
            count_tokens=lambda: sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values()),
        )
