import asyncio

from sales_agents.crm_mock.store import CrmStore
from sales_agents.tools.crm import build_crm_tools, store_tools
from sales_agents.tools.kb import load_docs

GOOD = 'Hi Dana,\n\nPipelane routes each qualified lead to a rep by territory and segment rules.\n\nReply "unsubscribe" and I will stop.'
BAD = 'Hi Dana,\n\nPipelane gives customers 40% more booked meetings.\n\nReply "unsubscribe" and I will stop.'


def tools_for(store):
    return {t.name: t for t in build_crm_tools(store_tools(store), load_docs())}


def test_record_outreach_writes_a_clean_draft():
    store = CrmStore()

    asyncio.run(tools_for(store)["record_outreach"].ainvoke({"lead_id": "lead-1", "draft_text": GOOD}))

    assert store.get_lead("lead-1")["outreach"] == GOOD


def test_record_outreach_refuses_a_draft_the_guardrail_blocks():
    store = CrmStore()

    answer = asyncio.run(tools_for(store)["record_outreach"].ainvoke({"lead_id": "lead-1", "draft_text": BAD}))

    assert "BLOCKED" in answer and "40%" in answer
    assert store.get_lead("lead-1") is None


def test_agent_sees_one_record_outreach_and_the_upsert_tool():
    assert set(tools_for(CrmStore())) == {"upsert_lead", "record_outreach"}
