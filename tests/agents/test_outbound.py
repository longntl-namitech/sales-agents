import asyncio
from pathlib import Path

from fakes import scripted, tool_call
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from sales_agents.agents.supervisor import LeadDesk, build_supervisor, run_outbound, scripted_decision
from sales_agents.config import Settings
from sales_agents.crm_mock.store import CrmStore
from sales_agents.datagen import load_leads
from sales_agents.tools.crm import build_crm_tools, store_tools
from sales_agents.tools.kb import load_docs

DATASET = Path(__file__).parents[2] / "datasets" / "leads_seed42.jsonl"
LEAD = next(c.lead for c in load_leads(DATASET) if c.gold == "hot")
LEAD_ID = LEAD.lead_id
GOOD = 'Hi Dana,\n\nPipelane routes each qualified lead to a rep by territory and segment rules.\n\nReply "unsubscribe" and I will stop.'
EDITED = GOOD.replace("Hi Dana", "Hello Dana")
BAD = 'Hi Dana,\n\nPipelane gives customers 40% more booked meetings.\n\nReply "unsubscribe" and I will stop.'


def task(subagent, call_id):
    return tool_call("task", {"description": f"Handle lead {LEAD_ID}", "subagent_type": subagent}, call_id)


def build(store, draft_text, label="hot", model_label=None):
    """Every role is a scripted fake. `label` is what the Qualifier submits. `model_label` is what the
    CRM subagent tries to save, to check that code and not the model decides the label."""
    crm_steps = [
        tool_call("upsert_lead", {"lead_id": LEAD_ID, "label": model_label or label, "notes": "made up"}, "u1"),
        AIMessage(content="saved"),
    ]
    supervisor = [task("researcher", "c1"), task("qualifier", "c2"), task("crm_agent", "c3")]
    if label == "hot":
        supervisor += [task("writer", "c4"), task("crm_agent", "c5")]
        crm_steps += [
            tool_call("get_draft", {"lead_id": LEAD_ID}, "g1"),
            tool_call("record_outreach", {"lead_id": LEAD_ID, "draft_text": draft_text}, "r1"),
            AIMessage(content="recorded"),
        ]
    models = {
        "supervisor": scripted(*supervisor, AIMessage(content="done")),
        "researcher": scripted(tool_call("research_lead", {"lead_id": LEAD_ID}), AIMessage(content="facts")),
        "qualifier": scripted(
            tool_call("get_lead_context", {"lead_id": LEAD_ID}, "q1"),
            tool_call("submit_qualification", {"lead_id": LEAD_ID, "label": label, "reason": "gates pass"}, "q2"),
            AIMessage(content=f"Final label: {label}"),
        ),
        "writer": scripted(
            tool_call("get_lead_facts", {"lead_id": LEAD_ID}, "w1"),
            tool_call("save_draft", {"lead_id": LEAD_ID, "subject": "", "body": draft_text}, "w2"),
            AIMessage(content="saved"),
        ),
        "crm_agent": scripted(*crm_steps),
    }
    desk = LeadDesk()
    crm_tools = build_crm_tools(store_tools(store), load_docs(), results=desk.results.get)
    agent = build_supervisor(Settings(), MemorySaver(), desk, models=models, crm_tools=crm_tools)
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

    record = store.get_lead(LEAD_ID)
    assert record["label"] == "hot" and record["outreach"] == GOOD


def test_rejected_draft_not_recorded():
    store = CrmStore()

    run(*build(store, GOOD), reject)

    record = store.get_lead(LEAD_ID)
    assert record["label"] == "hot" and record["outreach"] is None


def test_edited_draft_records_edited_text():
    store = CrmStore()

    run(*build(store, GOOD), edit_to(EDITED))

    assert store.get_lead(LEAD_ID)["outreach"] == EDITED


def test_edited_draft_is_checked_again_by_the_guardrail():
    store = CrmStore()

    run(*build(store, GOOD), edit_to(BAD))

    assert store.get_lead(LEAD_ID)["outreach"] is None


def test_guardrail_failure_blocks_even_if_approved():
    store = CrmStore()

    run(*build(store, BAD), approve)

    assert store.get_lead(LEAD_ID)["outreach"] is None


def test_a_warm_lead_is_saved_with_its_label_and_no_draft_is_asked_for():
    store = CrmStore()

    run(*build(store, GOOD, label="warm"), approve)

    record = store.get_lead(LEAD_ID)
    assert record["label"] == "warm" and record["outreach"] is None


def test_the_crm_subagent_cannot_save_a_label_the_qualifier_did_not_give():
    store = CrmStore()

    run(*build(store, GOOD, label="warm", model_label="hot"), approve)

    record = store.get_lead(LEAD_ID)
    assert record["label"] == "warm" and record["notes"] == "gates pass"


def test_scripted_decision_follows_the_checklist():
    decide = scripted_decision(load_docs())

    assert decide({"name": "record_outreach", "args": {"lead_id": "x", "draft_text": GOOD}})["type"] == "approve"
    assert decide({"name": "record_outreach", "args": {"lead_id": "x", "draft_text": BAD}})["type"] == "reject"
