import asyncio

from test_cli import DATASET, GOOD, LEAD, models

from sales_agents import telegram_front
from sales_agents.crm_mock.store import CrmStore
from sales_agents.runner import RunManager
from sales_agents.telegram_front import TelegramFront
from sales_agents.tools.crm import store_tools

CHAT = 7


def setup():
    store = CrmStore()
    runner = RunManager(dataset=DATASET, raw_tools=store_tools(store), models=models())
    sent = []

    async def send(chat_id, text, buttons=None):
        sent.append((chat_id, text, buttons))

    return TelegramFront(runner, send), sent, store


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

    assert all(len(text) <= 40 for _, text, _ in sent) and len(sent) > 1
    assert "".join(text for _, text, _ in sent) == GOOD
    assert [bool(b) for _, _, b in sent] == [False] * (len(sent) - 1) + [True]
