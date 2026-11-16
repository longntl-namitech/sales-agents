import asyncio
from types import SimpleNamespace

from telegram.error import BadRequest
from telegram.ext import CallbackQueryHandler

from sales_agents.config import Settings
from sales_agents.telegram_bot import build_application


def tap(data):
    calls = {"front": [], "cleared": 0}

    class Front:
        async def on_callback(self, chat_id, data):
            calls["front"].append(data)

    async def answer():
        pass

    async def clear(reply_markup=None):
        calls["cleared"] += 1

    application = build_application(Settings(telegram_bot_token="123:abc", telegram_allowed_chats="1"), front=Front())
    handler = next(h for h in application.handlers[0] if isinstance(h, CallbackQueryHandler))
    query = SimpleNamespace(answer=answer, data=data, edit_message_reply_markup=clear)
    asyncio.run(handler.callback(SimpleNamespace(effective_chat=SimpleNamespace(id=1), callback_query=query), None))
    return calls


def test_the_buttons_are_removed_once_the_draft_is_approved_or_rejected():
    assert tap("a:run")["cleared"] == 1 and tap("r:run")["cleared"] == 1


def test_the_buttons_stay_after_edit_and_on_the_lead_picker():
    assert tap("e:run")["cleared"] == 0 and tap("pick:lead-0001")["cleared"] == 0


def test_a_late_button_answer_error_does_not_drop_the_tap():
    taps = []

    class Front:
        async def on_callback(self, chat_id, data):
            taps.append((chat_id, data))

    async def too_old():
        raise BadRequest("Query is too old")

    application = build_application(Settings(telegram_bot_token="123:abc", telegram_allowed_chats="1"), front=Front())
    handler = next(h for h in application.handlers[0] if isinstance(h, CallbackQueryHandler))
    async def clear(reply_markup=None):
        pass

    query = SimpleNamespace(answer=too_old, data="a:run", edit_message_reply_markup=clear)
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=1), callback_query=query)

    asyncio.run(handler.callback(update, None))

    assert taps == [(1, "a:run")]
