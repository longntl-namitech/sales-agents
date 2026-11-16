import asyncio

from fakes import scripted, tool_call
from langchain_core.messages import AIMessage
from test_cli import DATASET, GOOD, LEAD, models

from sales_agents import telegram_front
from sales_agents.crm_mock.store import CrmStore
from sales_agents.datagen import load_leads
from sales_agents.runner import RunManager
from sales_agents.telegram_front import TelegramFront
from sales_agents.tools.crm import store_tools

CHAT = 7
COLD = next(c.lead for c in load_leads(DATASET) if c.gold == "cold")


def cold_models():
    def task(subagent, call_id):
        return tool_call("task", {"description": f"Handle lead {COLD.lead_id}", "subagent_type": subagent}, call_id)

    return {
        "supervisor": scripted(task("researcher", "c1"), task("qualifier", "c2"), task("crm_agent", "c3"), AIMessage(content="done")),
        "researcher": scripted(tool_call("research_lead", {"lead_id": COLD.lead_id}), AIMessage(content="facts")),
        "qualifier": scripted(
            tool_call("submit_qualification", {"lead_id": COLD.lead_id, "label": "cold", "reason": "gates fail"}),
            AIMessage(content="Final label: cold"),
        ),
        "writer": scripted(AIMessage(content="unused")),
        "crm_agent": scripted(tool_call("upsert_lead", {"lead_id": COLD.lead_id}, "u1"), AIMessage(content="saved")),
    }


def setup(*, allowed=frozenset({CHAT}), max_runs=5, model_set=None):
    store = CrmStore()
    runner = RunManager(dataset=DATASET, raw_tools=store_tools(store), models=model_set or models())
    sent = []

    async def send(chat_id, text, buttons=None):
        sent.append((chat_id, text, buttons))

    return TelegramFront(runner, send, allowed=allowed, max_runs=max_runs), sent, store


def data(sent, label):
    buttons = next(b for _, _, b in reversed(sent) if b)
    return next(d for name, d in buttons if name == label)


def drafted():
    front, sent, store = setup()
    asyncio.run(front.run_lead(CHAT, LEAD.lead_id))
    return front, sent, store


def test_a_hot_lead_sends_the_draft_with_three_buttons_once():
    _, sent, _ = drafted()

    with_buttons = [(text, b) for _, text, b in sent if b]
    assert len(with_buttons) == 1
    text, buttons = with_buttons[0]
    assert GOOD in text and [name for name, _ in buttons] == ["Approve", "Edit", "Reject"]


def test_approve_sends_a_result_line_and_the_crm_has_the_outreach():
    front, sent, store = drafted()

    asyncio.run(front.on_callback(CHAT, data(sent, "Approve")))

    assert "hot" in sent[-1][1] and "outreach recorded" in sent[-1][1]
    assert store.get_lead(LEAD.lead_id)["outreach"] == GOOD


def test_edit_then_text_records_the_edited_text():
    front, sent, store = drafted()
    edited = GOOD.replace("Hi Dana", "Hello Dana")

    asyncio.run(front.on_callback(CHAT, data(sent, "Edit")))
    asyncio.run(front.on_text(CHAT, edited))

    assert store.get_lead(LEAD.lead_id)["outreach"] == edited


def test_reject_leaves_no_outreach():
    front, sent, store = drafted()

    asyncio.run(front.on_callback(CHAT, data(sent, "Reject")))

    assert not store.get_lead(LEAD.lead_id).get("outreach") and "no outreach recorded" in sent[-1][1]


def test_a_second_tap_gets_no_longer_waiting_and_does_not_resume():
    front, sent, _ = drafted()
    approve = data(sent, "Approve")
    asyncio.run(front.on_callback(CHAT, approve))
    count = len(sent)

    asyncio.run(front.on_callback(CHAT, approve))

    assert len(sent) == count + 1 and "no longer waiting" in sent[-1][1]


def test_a_button_from_a_run_replaced_by_a_new_run_is_stale():
    front, sent, store = drafted()
    approve = data(sent, "Approve")

    asyncio.run(front.run_lead(CHAT, "nope"))
    asyncio.run(front.on_callback(CHAT, approve))

    assert "no longer waiting" in sent[-1][1] and not store.get_lead(LEAD.lead_id).get("outreach")


def test_plain_text_when_not_awaiting_an_edit_is_ignored_with_a_hint():
    front, sent, store = drafted()
    count = len(sent)

    asyncio.run(front.on_text(CHAT, "hello"))

    assert len(sent) == count + 1 and not store.get_lead(LEAD.lead_id).get("outreach")


def test_a_draft_longer_than_the_limit_is_split_without_error(monkeypatch):
    monkeypatch.setattr(telegram_front, "LIMIT", 40)

    _, sent, _ = drafted()

    assert all(len(text) <= 40 for _, text, _ in sent)
    chunks = -(-len(GOOD) // 40)
    assert "".join(text for _, text, _ in sent[-chunks:]) == GOOD
    assert [bool(b) for _, _, b in sent[-chunks:]] == [False] * (chunks - 1) + [True]


def test_start_shows_a_picker_with_every_lead():
    front, sent, _ = setup()

    asyncio.run(front.on_start(CHAT))

    buttons = sent[-1][2]
    assert len(buttons) == 60 and (LEAD.company_name, f"pick:{LEAD.lead_id}") in buttons


def test_picking_a_lead_runs_it_and_reports_each_step():
    front, sent, _ = setup()

    asyncio.run(front.on_callback(CHAT, f"pick:{LEAD.lead_id}"))

    said = " | ".join(text for _, text, _ in sent)
    assert "Researching" in said and "Qualifying" in said and GOOD in said


def test_a_cold_lead_ends_with_a_one_line_result_and_no_buttons():
    front, sent, _ = setup(model_set=cold_models())

    asyncio.run(front.run_lead(CHAT, COLD.lead_id))

    assert not any(b for _, _, b in sent) and sent[-1][1] == "Result: cold - no outreach recorded."


def test_a_chat_not_on_the_list_gets_no_reply_from_any_entry_point():
    front, sent, _ = setup()

    asyncio.run(front.on_start(99))
    asyncio.run(front.on_callback(99, f"pick:{LEAD.lead_id}"))
    asyncio.run(front.on_text(99, "hello"))
    asyncio.run(front.run_lead(99, LEAD.lead_id))

    assert sent == []


def test_a_group_chat_is_ignored():
    front, sent, _ = setup(allowed=frozenset({-100}))

    asyncio.run(front.on_start(-100))
    asyncio.run(front.run_lead(-100, LEAD.lead_id))

    assert sent == []


def test_the_run_cap_stops_further_runs_in_a_chat():
    front, sent, _ = setup(max_runs=1)
    asyncio.run(front.run_lead(CHAT, LEAD.lead_id))
    count = len(sent)

    asyncio.run(front.run_lead(CHAT, LEAD.lead_id))

    assert len(sent) == count + 1 and "limit" in sent[-1][1]
