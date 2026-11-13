import math

from sales_agents.schemas import LabelledLead

OUTPUT_TOKENS_PER_CALL = 150  # a label plus a one-sentence reason, with headroom


def approx_tokens(text: str) -> int:
    """Rough rule of thumb, about four characters per token for English."""
    return math.ceil(len(text) / 4)


def estimate_tokens(cases: list[LabelledLead], prompt_tokens: int, calls_per_case: int) -> int:
    """Upper-leaning estimate of input plus output tokens for a whole run."""
    total = 0
    for case in cases:
        case_text = case.lead.model_dump_json() + case.company.model_dump_json()
        per_call = prompt_tokens + approx_tokens(case_text) + OUTPUT_TOKENS_PER_CALL
        total += per_call * calls_per_case
    return total
