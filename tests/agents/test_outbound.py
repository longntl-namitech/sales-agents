import asyncio

from fakes import scripted, tool_call, verdict
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from sales_agents.agents.supervisor import LeadDesk, build_supervisor, run_outbound, scripted_decision
from sales_agents.config import Settings
from sales_agents.crm_mock.store import CrmStore
from sales_agents.schemas import Lead
from sales_agents.tools.crm import build_crm_tools, store_tools
from sales_agents.tools.kb import load_docs

LEAD = Lead(lead_id="lead-1", company_name="Quill Dynamics", contact_role="VP Sales", notes="n/a", source="outbound")
GOOD = 'Hi Dana,\n\nPipelane routes each qualified lead to a rep by territory and segment rules.\n\nReply "unsubscribe" and I will stop.'
EDITED = GOOD.replace("Hi Dana", "Hello Dana")
BAD = 'Hi Dana,\n\nPipelane gives customers 40% more booked meetings.\n\nReply "unsubscribe" and I will stop.'


def build(store, draft_text, label="hot"):
    """The Supervisor model is scripted to follow the outbound steps. The writer returns draft_text."""
    steps = [
        tool_call("research_lead", {"lead_id": "lead-1"}, "c1"),
        tool_call("qualify_lead", {"lead_id": "lead-1"}, "c2"),
        tool_call("upsert_lead", {"lead_id": "lead-1", "label": label, "notes": "gates pass"}, "c3"),
    ]
    if label == "hot":
        steps += [
            tool_call("write_draft", {"lead_id": "lead-1"}, "c4"),
            tool_call("record_outreach", {"lead_id": "lead-1", "draft_text": draft_text}, "c5"),
        ]
    steps.append(AIMessage(content="done"))
    desk = LeadDesk(
        models={
            "qualifier": scripted(verdict(label, "gates pass")),
            "writer": scripted(tool_call("Draft", {"subject": "", "body": draft_text})),
        }
    )
    crm_tools = build_crm_tools(store_tools(store), load_docs())
    agent = build_supervisor(
        Settings(), MemorySaver(), desk, models={"supervisor": scripted(*steps)}, crm_tools=crm_tools
    )
    return agent, desk


def run(agent, desk, decide):
    return asyncio.run(run_outbound(agent, desk, LEAD, "t1", decide=decide))


def approve(request):
    return {"type": "approve"}


def reject(request):
    return {"type": "reject", "message": "not good enough"}


def edit_to(text):
    def decide(request):
        args = {**request["args"], "draft_text": text}
        return {"type": "edit", "edited_action": {"name": request["name"], "args": args}}

    return decide


def test_approved_draft_is_recorded_after_the_label():
    store = CrmStore()

    run(*build(store, GOOD), approve)

    record = store.get_lead("lead-1")
    assert record["label"] == "hot" and record["outreach"] == GOOD


def test_rejected_draft_not_recorded():
    store = CrmStore()

    run(*build(store, GOOD), reject)

    record = store.get_lead("lead-1")
    assert record["label"] == "hot" and record["outreach"] is None


def test_edited_draft_records_edited_text():
    store = CrmStore()

    run(*build(store, GOOD), edit_to(EDITED))

    assert store.get_lead("lead-1")["outreach"] == EDITED


def test_edited_draft_is_checked_again_by_the_guardrail():
    store = CrmStore()

    run(*build(store, GOOD), edit_to(BAD))

    assert store.get_lead("lead-1")["outreach"] is None


def test_guardrail_failure_blocks_even_if_approved():
    store = CrmStore()

    run(*build(store, BAD), approve)

    assert store.get_lead("lead-1")["outreach"] is None


def test_a_warm_lead_is_saved_with_its_label_and_no_draft_is_asked_for():
    store = CrmStore()

    run(*build(store, GOOD, label="warm"), approve)

    record = store.get_lead("lead-1")
    assert record["label"] == "warm" and record["outreach"] is None


def test_scripted_decision_follows_the_checklist():
    decide = scripted_decision(load_docs())

    assert decide({"name": "record_outreach", "args": {"lead_id": "x", "draft_text": GOOD}})["type"] == "approve"
    assert decide({"name": "record_outreach", "args": {"lead_id": "x", "draft_text": BAD}})["type"] == "reject"
