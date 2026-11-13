"""The full outbound path with a real model, the mock CRM behind MCP and the scripted approver.

    uv run pytest -m llm -s tests/agents/test_outbound_llm.py

Checks the Task 8 promise: the CRM holds exactly the drafts the approver approved,
and nothing it rejected or the guardrail blocked.
"""

import asyncio
import os
import time
from datetime import date
from pathlib import Path

import pytest
from fakes import scripted, tool_call, verdict
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from sales_agents.agents.qualifier import SYSTEM_PROMPT as QUALIFIER_PROMPT
from sales_agents.agents.supervisor import LeadDesk, build_supervisor, run_outbound, scripted_decision
from sales_agents.agents.writer import SYSTEM_PROMPT as WRITER_PROMPT
from sales_agents.config import Settings, get_settings
from sales_agents.crm_mock.store import CrmStore
from sales_agents.datagen import load_leads
from sales_agents.eval.live import SUBSET_TOKEN_LIMIT
from sales_agents.eval.report import Measurement, save_measurement
from sales_agents.eval.tokens import HARNESS_OVERHEAD_TOKENS, approx_tokens, estimate_tokens
from sales_agents.tools.crm import build_crm_tools, list_crm_records, load_crm_tools, store_tools
from sales_agents.tools.kb import load_docs

DATASET = Path(__file__).parents[2] / "datasets" / "leads_seed42.jsonl"
PER_LABEL = 4


def pick(per_label):
    cases = load_leads(DATASET)
    return [c for label in ("hot", "warm", "cold") for c in [c for c in cases if c.gold == label][:per_label]]


async def run_batch(cases, make_agent, raw_tools, docs):
    """Run each lead through the outbound path. Returns what the approver decided and what the CRM holds."""
    decide = scripted_decision(docs)
    decisions = {}  # lead_id -> (decision type, draft text) for every draft shown to the approver

    def logged(request):
        decision = decide(request)
        decisions[request["args"]["lead_id"]] = (decision["type"], request["args"]["draft_text"])
        return decision

    agent, desk = make_agent(build_crm_tools(raw_tools, docs))
    for case in cases:
        await run_outbound(agent, desk, case.lead, f"outbound-{case.lead.lead_id}", decide=logged)
    return decisions, await list_crm_records(raw_tools)


def test_batch_wiring_with_fake_models():
    case = pick(1)[0]  # a hot lead
    store = CrmStore()
    good = 'Hi Dana,\n\nPipelane routes each qualified lead to a rep by territory and segment rules.\n\nReply "unsubscribe" and I will stop.'
    steps = [
        tool_call("research_lead", {"lead_id": case.lead.lead_id}, "c1"),
        tool_call("qualify_lead", {"lead_id": case.lead.lead_id}, "c2"),
        tool_call("upsert_lead", {"lead_id": case.lead.lead_id, "label": "hot", "notes": "x"}, "c3"),
        tool_call("write_draft", {"lead_id": case.lead.lead_id}, "c4"),
        tool_call("record_outreach", {"lead_id": case.lead.lead_id, "draft_text": good}, "c5"),
        AIMessage(content="done"),
    ]

    def make_agent(crm_tools):
        desk = LeadDesk(
            models={"qualifier": scripted(verdict("hot")), "writer": scripted(tool_call("Draft", {"subject": "", "body": good}))}
        )
        agent = build_supervisor(Settings(), MemorySaver(), desk, models={"supervisor": scripted(*steps)}, crm_tools=crm_tools)
        return agent, desk

    decisions, records = asyncio.run(run_batch([case], make_agent, store_tools(store), load_docs()))

    assert decisions[case.lead.lead_id][0] == "approve"
    assert [r["outreach"] for r in records] == [good]


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY") and not Path(".env").exists(), reason="no API key")
def test_crm_holds_exactly_the_approved_drafts(tmp_path):
    settings = get_settings()
    cases = pick(PER_LABEL)
    estimate = estimate_tokens(
        cases,
        prompt_tokens=approx_tokens(QUALIFIER_PROMPT) + approx_tokens(WRITER_PROMPT) + 400,
        calls_per_case=2,
        overhead_per_case=3 * HARNESS_OVERHEAD_TOKENS,
    )
    assert estimate < min(settings.eval_token_cap, SUBSET_TOKEN_LIMIT)

    async def scenario():
        raw_tools = await load_crm_tools(tmp_path / "crm.sqlite3")

        def make_agent(crm_tools):
            desk = LeadDesk(settings=settings)
            return build_supervisor(settings, MemorySaver(), desk, crm_tools=crm_tools), desk

        return await run_batch(cases, make_agent, raw_tools, load_docs())

    start = time.perf_counter()
    with get_usage_metadata_callback() as usage:
        decisions, records = asyncio.run(scenario())
    seconds = time.perf_counter() - start
    tokens = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())

    approved = {lead_id for lead_id, (kind, _) in decisions.items() if kind == "approve"}
    recorded = {r["lead_id"] for r in records if r["outreach"]}
    by_gold = {c.lead.lead_id: c.gold for c in cases}
    print()
    print(f"leads {len(cases)}, tokens {tokens} (estimated {estimate}), drafts shown {len(decisions)}, approved {len(approved)}, recorded {len(recorded)}")
    print("labels saved in the CRM vs gold:", [(r["lead_id"], r["label"], by_gold[r["lead_id"]]) for r in records])
    exact = recorded == approved and all(r["outreach"] == decisions[r["lead_id"]][1] for r in records if r["outreach"])
    label_accuracy = sum(r["label"] == by_gold[r["lead_id"]] for r in records) / len(cases)
    save_measurement(
        Measurement(
            name="Outbound path (Supervisor, MCP CRM, scripted approver)",
            use_case="Outbound outreach",
            n=len(cases),
            metrics={"CRM holds exactly the approved drafts": float(exact), "label accuracy in CRM": label_accuracy},
            tokens=tokens,
            latency_p50=seconds / len(cases),
            date=date.today().isoformat(),
            note=f"{len(approved)} of {len(decisions)} drafts approved",
        )
    )
    assert recorded == approved
    assert all(r["outreach"] == decisions[r["lead_id"]][1] for r in records if r["outreach"])
