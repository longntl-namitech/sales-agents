"""Measures the real Qualifier. Run on purpose, never by default:

    uv run pytest -m llm -s tests/agents/test_qualifier_llm.py

The default run excludes the `llm` marker. The guard below refuses to start when
the token estimate is above the subset limit or the configured cap.
"""

import os
from pathlib import Path

import pytest
from fakes import scripted, verdict
from langchain_core.callbacks import get_usage_metadata_callback

from sales_agents.agents.baseline import make_rule_baseline
from sales_agents.agents.qualifier import SYSTEM_PROMPT, qualify
from sales_agents.config import get_chat_model, get_settings
from sales_agents.datagen import load_leads
from sales_agents.eval.harness import format_table, run_eval
from sales_agents.eval.tokens import approx_tokens, estimate_tokens

DATASET = Path(__file__).parents[2] / "datasets" / "leads_seed42.jsonl"
SUBSET_TOKEN_LIMIT = 1_000_000  # global rule: first run on a subset below 1M tokens


def measure(model, cases, token_cap):
    """Run the Qualifier over the cases, with the token guard and usage counting."""
    estimate = estimate_tokens(cases, prompt_tokens=approx_tokens(SYSTEM_PROMPT), calls_per_case=1)
    with get_usage_metadata_callback() as usage:
        report = run_eval(
            cases,
            lambda lead: qualify(lead, model=model),
            token_cap=token_cap,
            estimated_tokens=estimate,
            count_tokens=lambda: sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values()),
        )
    return report


def test_measure_wiring_with_fake_model():
    # Offline check that the live path is wired correctly, costs nothing.
    cases = load_leads(DATASET)[:3]
    model = scripted(verdict("hot"), verdict("warm"), verdict("cold"))

    report = measure(model, cases, token_cap=100_000)

    assert report.n == 3 and report.errors == []


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY") and not Path(".env").exists(), reason="no API key")
def test_real_qualifier_row_next_to_baseline():
    settings = get_settings()
    cases = load_leads(DATASET)
    cap = min(settings.eval_token_cap, SUBSET_TOKEN_LIMIT)

    model = get_chat_model(settings)
    qualifier_report = measure(model, cases, token_cap=cap)
    baseline_report = run_eval(cases, make_rule_baseline(cases), token_cap=cap)

    print()
    print(f"estimated tokens: {qualifier_report.estimated_tokens}, used: {qualifier_report.cost}")
    print(format_table({"rule baseline": baseline_report, f"LLM {settings.model}": qualifier_report}))
