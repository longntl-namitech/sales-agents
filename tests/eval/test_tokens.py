from sales_agents.datagen import generate_leads, template_text
from sales_agents.eval.tokens import OUTPUT_TOKENS_PER_CALL, estimate_tokens

CASES = generate_leads(10, seed=1, write_text=template_text)


def test_estimate_counts_input_and_output_tokens():
    base = estimate_tokens(CASES, prompt_tokens=0, calls_per_case=1)

    # Output is counted, and the lead text is counted on top of it.
    assert base > len(CASES) * OUTPUT_TOKENS_PER_CALL


def test_estimate_scales_with_prompt_and_calls():
    base = estimate_tokens(CASES, prompt_tokens=0, calls_per_case=1)

    assert estimate_tokens(CASES, prompt_tokens=500, calls_per_case=2) == 2 * (
        base + 500 * len(CASES)
    )
