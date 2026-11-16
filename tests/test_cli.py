from pathlib import Path

import pytest
from fakes import scripted, tool_call
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

    def task(subagent, call_id):
        return tool_call("task", {"description": f"Handle lead {lead_id}", "subagent_type": subagent}, call_id)

    return {
        "supervisor": scripted(
            task("researcher", "c1"), task("qualifier", "c2"), task("crm_agent", "c3"), task("writer", "c4"), task("crm_agent", "c5"),
            AIMessage(content="done"),
        ),
        "researcher": scripted(tool_call("research_lead", {"lead_id": lead_id}), AIMessage(content="facts")),
        "qualifier": scripted(
            tool_call("submit_qualification", {"lead_id": lead_id, "label": "hot", "reason": "gates pass"}), AIMessage(content="Final label: hot")
        ),
        "writer": scripted(
            tool_call("save_draft", {"lead_id": lead_id, "subject": "", "body": GOOD}, "w1"), AIMessage(content="saved")
        ),
        "crm_agent": scripted(
            tool_call("upsert_lead", {"lead_id": lead_id}, "u1"),
            AIMessage(content="saved"),
            tool_call("get_draft", {"lead_id": lead_id}, "g1"),
            tool_call("record_outreach", {"lead_id": lead_id, "draft_text": GOOD}, "r1"),
            AIMessage(content="recorded"),
        ),
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


def test_a_lead_with_a_draft_from_an_earlier_run_is_flagged():
    store = CrmStore()
    store.record_outreach(LEAD.lead_id, "an old draft")
    printed = []

    run_outbound_cli(
        LEAD.lead_id, decide=lambda r: {"type": "reject", "message": "no"}, dataset=DATASET,
        raw_tools=store_tools(store), models=models(), out=lambda *a: printed.append(" ".join(map(str, a))),
    )

    assert any("earlier run" in line for line in printed)


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


def test_the_web_command_serves_the_demo_app(monkeypatch):
    from sales_agents import cli

    served = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: served.update(app=app, **kw))

    cli.main(["web", "--port", "8123"])

    assert served["port"] == 8123 and served["app"].title == "sales-agents demo"


def telegram_env(monkeypatch, *, token="123:abc", chats="1"):
    monkeypatch.setenv("SALES_AGENTS_TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setenv("SALES_AGENTS_TELEGRAM_ALLOWED_CHATS", chats)


def test_the_telegram_command_refuses_to_start_without_an_allow_list(monkeypatch):
    from telegram.ext import Application

    from sales_agents import cli

    polled = []
    monkeypatch.setattr(Application, "run_polling", lambda self, *a, **k: polled.append(1))
    telegram_env(monkeypatch, chats="")

    with pytest.raises(SystemExit, match="ALLOWED_CHATS"):
        cli.main(["telegram"])

    assert polled == []


def test_the_telegram_command_refuses_without_a_token(monkeypatch):
    from telegram.ext import Application

    from sales_agents import cli

    polled = []
    monkeypatch.setattr(Application, "run_polling", lambda self, *a, **k: polled.append(1))
    telegram_env(monkeypatch, token="")

    with pytest.raises(SystemExit, match="BOT_TOKEN"):
        cli.main(["telegram"])

    assert polled == []


def test_the_telegram_command_starts_polling_with_a_token_and_an_allow_list(monkeypatch):
    from telegram.ext import Application

    from sales_agents import cli

    polled = []
    monkeypatch.setattr(Application, "run_polling", lambda self, *a, **k: polled.append(1))
    telegram_env(monkeypatch)

    cli.main(["telegram"])

    assert polled == [1]


def test_build_application_registers_start_callback_and_text_handlers():
    from telegram.ext import CallbackQueryHandler, CommandHandler, MessageHandler

    from sales_agents.config import Settings
    from sales_agents.telegram_bot import build_application

    application = build_application(Settings(telegram_bot_token="123:abc", telegram_allowed_chats="1"))

    kinds = {type(handler) for handler in application.handlers[0]}
    assert kinds == {CommandHandler, CallbackQueryHandler, MessageHandler}
