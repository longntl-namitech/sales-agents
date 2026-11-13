"""Measures what enrichment adds to the Qualifier. Run on purpose, never by default:

    uv run pytest -m llm -s tests/agents/test_researcher_llm.py
"""

import os
from pathlib import Path

import pytest
from fakes import scripted, verdict

from sales_agents.agents.qualifier import SYSTEM_PROMPT, qualify
from sales_agents.agents.researcher import research
from sales_agents.config import get_chat_model, get_settings
from sales_agents.datagen import load_leads
from sales_agents.eval.harness import format_table
from sales_agents.eval.live import SUBSET_TOKEN_LIMIT, measure
from sales_agents.eval.report import from_eval_report, save_measurement
from sales_agents.eval.tokens import approx_tokens

DATASET = Path(__file__).parents[2] / "datasets" / "leads_seed42.jsonl"


def run_with_and_without(model, cases, token_cap):
    prompt_tokens = approx_tokens(SYSTEM_PROMPT)
    plain = measure(
        lambda lead: qualify(lead, model=model),
        cases,
        prompt_tokens=prompt_tokens,
        calls_per_case=1,
        token_cap=token_cap,
    )
    enriched = measure(
        lambda lead: qualify(lead, research(lead), model=model),
        cases,
        prompt_tokens=prompt_tokens,
        calls_per_case=1,
        token_cap=token_cap,
    )
    return plain, enriched


def test_comparison_wiring_with_fake_model():
    cases = load_leads(DATASET)[:2]
    model = scripted(verdict("hot"), verdict("warm"), verdict("hot"), verdict("warm"))

    plain, enriched = run_with_and_without(model, cases, token_cap=100_000)

    assert plain.n == enriched.n == 2 and plain.errors == enriched.errors == []


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY") and not Path(".env").exists(), reason="no API key")
def test_enrichment_effect_on_hot_precision_and_recall():
    settings = get_settings()
    cases = load_leads(DATASET)
    cap = min(settings.eval_token_cap, SUBSET_TOKEN_LIMIT)

    plain, enriched = run_with_and_without(get_chat_model(settings), cases, token_cap=cap)

    print()
    print(format_table({"no enrichment": plain, "with enrichment": enriched}))
    print(f"delta precision {enriched.precision - plain.precision:+.3f}, delta recall {enriched.recall - plain.recall:+.3f}")
    save_measurement(from_eval_report("Qualifier + Researcher", "Researcher", enriched, note=f"enrichment lifts hot precision by {enriched.precision - plain.precision:+.3f} over the plain Qualifier"))
