from pathlib import Path

import pytest
from fakes import scripted, tool_call, verdict
from langchain_core.messages import AIMessage

from sales_agents.cli import run_outbound_cli, terminal_decision
from sales_agents.crm_mock.store import CrmStore
from sales_agents.datagen import load_leads
from sales_agents.tools.crm import store_tools

DATASET = Path(__file__).parents[1] / "datasets" / "leads_seed42.jsonl"
LEAD = next(c.lead for c in load_leads(DATASET) if c.gold == "hot")
GOOD = 'Hi Dana,\n\nPipelane routes each qualified lead to a rep by territory and segment rules.\n\nReply "unsubscribe" and I will stop.'


def models():
    lead_id = LEAD.lead_id
    return {
        "supervisor": scripted(
            tool_call("research_lead", {"lead_id": lead_id}, "c1"),
            tool_call("qualify_lead", {"lead_id": lead_id}, "c2"),
            tool_call("upsert_lead", {"lead_id": lead_id, "label": "hot", "notes": "gates pass"}, "c3"),
            tool_call("write_draft", {"lead_id": lead_id}, "c4"),
            tool_call("record_outreach", {"lead_id": lead_id, "draft_text": GOOD}, "c5"),
            AIMessage(content="done"),
        ),
        "qualifier": scripted(verdict("hot", "gates pass")),
        "writer": scripted(tool_call("Draft", {"subject": "", "body": GOOD})),
    }


def run(decide, store=None):
    store = store or CrmStore()
    return run_outbound_cli(
        LEAD.lead_id, decide=decide, dataset=DATASET, raw_tools=store_tools(store), models=models(), out=lambda *a: None
    )


def test_an_approved_draft_ends_up_in_the_crm_record():
    record = run(lambda request: {"type": "approve"})

    assert record["label"] == "hot" and record["outreach"] == GOOD


def test_a_rejected_draft_leaves_the_record_without_outreach():
    record = run(lambda request: {"type": "reject", "message": "no"})

    assert record["label"] == "hot" and record["outreach"] is None


def test_an_unknown_lead_id_is_a_clear_error():
    with pytest.raises(SystemExit, match="unknown lead id"):
        run_outbound_cli("nope", decide=lambda r: {"type": "approve"}, dataset=DATASET, raw_tools=store_tools(CrmStore()))


REQUEST = {"name": "record_outreach", "args": {"lead_id": "x", "draft_text": GOOD}}


def answers(*given):
    items = iter(given)
    return lambda prompt="": next(items)


def test_terminal_approve():
    assert terminal_decision(REQUEST, ask=answers("a"), out=lambda *a: None) == {"type": "approve"}


def test_terminal_reject_carries_the_reason():
    decision = terminal_decision(REQUEST, ask=answers("r", "too salesy"), out=lambda *a: None)

    assert decision == {"type": "reject", "message": "too salesy"}


def test_terminal_edit_sends_the_edited_text_to_the_same_tool():
    decision = terminal_decision(REQUEST, ask=answers("e", "Hello, edited."), out=lambda *a: None)

    assert decision["type"] == "edit"
    assert decision["edited_action"] == {"name": "record_outreach", "args": {"lead_id": "x", "draft_text": "Hello, edited."}}


def test_terminal_asks_again_on_an_unknown_answer():
    assert terminal_decision(REQUEST, ask=answers("maybe", "a"), out=lambda *a: None) == {"type": "approve"}
