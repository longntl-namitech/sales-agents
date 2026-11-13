"""D1: what does going through the Supervisor cost? Run on purpose:

    uv run pytest -m llm -s tests/agents/test_supervisor_llm.py

Compares the plain Qualifier (with enrichment) against the Supervisor path over
the same leads. The estimate includes the measured harness overhead.
"""

import os
from pathlib import Path

import pytest
from fakes import scripted, tool_call, verdict
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from sales_agents.agents.qualifier import SYSTEM_PROMPT, qualify
from sales_agents.agents.researcher import research
from sales_agents.agents.supervisor import LeadDesk, build_supervisor, run_lead
from sales_agents.config import Settings, get_chat_model, get_settings
from sales_agents.datagen import load_leads
from sales_agents.eval.harness import format_table
from sales_agents.eval.live import SUBSET_TOKEN_LIMIT, measure
from sales_agents.eval.report import from_eval_report, save_measurement
from sales_agents.eval.tokens import HARNESS_OVERHEAD_TOKENS, approx_tokens

DATASET = Path(__file__).parents[2] / "datasets" / "leads_seed42.jsonl"
N_LEADS = 30
SUBAGENT_OVERHEAD_FACTOR = 3  # each subagent is its own Deep Agent, so assume about three times the tool design's overhead


def compare(plain_model, agent, desk, cases, token_cap):
    prompt_tokens = approx_tokens(SYSTEM_PROMPT)
    plain = measure(
        lambda lead: qualify(lead, research(lead), model=plain_model),
        cases,
        prompt_tokens=prompt_tokens,
        calls_per_case=1,
        token_cap=token_cap,
    )
    supervised = measure(
        lambda lead: run_lead(lead, f"eval-{lead.lead_id}", agent=agent, desk=desk),
        cases,
        prompt_tokens=prompt_tokens,
        calls_per_case=1,
        token_cap=token_cap,
        overhead_per_case=SUBAGENT_OVERHEAD_FACTOR * HARNESS_OVERHEAD_TOKENS,
    )
    return plain, supervised


def test_comparison_wiring_with_fake_models():
    cases = load_leads(DATASET)[:1]
    lead_id = cases[0].lead.lead_id
    models = {
        "supervisor": scripted(
            tool_call("task", {"description": f"Research lead {lead_id}", "subagent_type": "researcher"}, "c1"),
            tool_call("task", {"description": f"Qualify lead {lead_id}", "subagent_type": "qualifier"}, "c2"),
            AIMessage(content="done"),
        ),
        "researcher": scripted(AIMessage(content="no record")),
        "qualifier": scripted(
            tool_call("submit_qualification", {"lead_id": lead_id, "label": "cold", "reason": "x"}), AIMessage(content="Final label: cold")
        ),
    }
    desk = LeadDesk()
    agent = build_supervisor(Settings(), MemorySaver(), desk, models=models)

    plain, supervised = compare(scripted(verdict("cold")), agent, desk, cases, token_cap=1_000_000)

    assert plain.n == supervised.n == 1 and supervised.errors == []


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY") and not Path(".env").exists(), reason="no API key")
def test_plain_qualifier_vs_supervisor_cost_and_latency():
    settings = get_settings()
    cases = load_leads(DATASET)[:N_LEADS]  # a subset, so the estimate stays under 1M tokens
    cap = min(settings.eval_token_cap, SUBSET_TOKEN_LIMIT)

    desk = LeadDesk()
    plain, supervised = compare(
        get_chat_model(settings), build_supervisor(settings, MemorySaver(), desk), desk, cases, token_cap=cap
    )

    print()
    print(f"estimated tokens: plain {plain.estimated_tokens}, supervisor {supervised.estimated_tokens}")
    print(format_table({"plain + enrichment": plain, "supervisor": supervised}))
    save_measurement(from_eval_report("Supervisor (subagents)", "Supervisor", supervised, note=f"first {N_LEADS} of the 60 leads, a Researcher subagent then a Qualifier subagent"))
