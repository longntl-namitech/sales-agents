import math

from sales_agents.schemas import LabelledLead

OUTPUT_TOKENS_PER_CALL = 150  # a label plus a one-sentence reason, with headroom

# Measured 2026-11-13 with gpt-4.1-mini on 60 leads through the tool-based Supervisor: 303,464
# tokens against 28,335 for the plain Qualifier, so about 4,600 more per lead over 3 model
# calls. The Deep Agents harness adds its own prompt and tool schemas to every model call.
# The outbound path makes about 6 calls, so estimate it as three times this. The first
# Supervisor design (LLM subagents) cost 10,600 per lead.
HARNESS_OVERHEAD_TOKENS = 5_000


def approx_tokens(text: str) -> int:
    """Rough rule of thumb, about four characters per token for English."""
    return math.ceil(len(text) / 4)


def estimate_tokens(
    cases: list[LabelledLead], prompt_tokens: int, calls_per_case: int, overhead_per_case: int = 0
) -> int:
    """Upper-leaning estimate of input plus output tokens for a whole run."""
    total = 0
    for case in cases:
        case_text = case.lead.model_dump_json() + case.company.model_dump_json()
        per_call = prompt_tokens + approx_tokens(case_text) + OUTPUT_TOKENS_PER_CALL
        total += per_call * calls_per_case + overhead_per_case
    return total
