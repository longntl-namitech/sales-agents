import asyncio

import pytest

from sales_agents.crm_mock.store import CrmStore
from sales_agents.schemas import QualificationResult
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


def gated_tools(store, results):
    return {t.name: t for t in build_crm_tools(store_tools(store), load_docs(), results=results.get)}


def test_with_results_the_label_comes_from_the_qualifier_not_the_model():
    store, results = CrmStore(), {"lead-1": QualificationResult(label="warm", reason="one signal")}

    asyncio.run(gated_tools(store, results)["upsert_lead"].ainvoke({"lead_id": "lead-1", "label": "hot", "notes": "x"}))

    record = store.get_lead("lead-1")
    assert record["label"] == "warm" and record["notes"] == "one signal"


def test_an_unqualified_lead_is_not_saved():
    store = CrmStore()

    answer = asyncio.run(gated_tools(store, {})["upsert_lead"].ainvoke({"lead_id": "lead-1", "label": "hot", "notes": ""}))

    assert "BLOCKED" in answer and store.get_lead("lead-1") is None


@pytest.mark.parametrize("label", ["warm", "cold"])
def test_only_a_hot_lead_can_get_a_draft_recorded(label):
    store, results = CrmStore(), {"lead-1": QualificationResult(label=label, reason="r")}

    answer = asyncio.run(gated_tools(store, results)["record_outreach"].ainvoke({"lead_id": "lead-1", "draft_text": GOOD}))

    assert "BLOCKED" in answer and store.get_lead("lead-1") is None


def test_a_hot_lead_can_get_a_clean_draft_recorded():
    store, results = CrmStore(), {"lead-1": QualificationResult(label="hot", reason="r")}

    asyncio.run(gated_tools(store, results)["record_outreach"].ainvoke({"lead_id": "lead-1", "draft_text": GOOD}))

    assert store.get_lead("lead-1")["outreach"] == GOOD


def test_agent_sees_one_record_outreach_and_the_upsert_tool():
    assert set(tools_for(CrmStore())) == {"upsert_lead", "record_outreach"}
