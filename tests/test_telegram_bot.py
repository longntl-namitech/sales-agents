import asyncio
from types import SimpleNamespace

from telegram.error import BadRequest
from telegram.ext import CallbackQueryHandler

from sales_agents.config import Settings
from sales_agents.telegram_bot import build_application


def test_a_late_button_answer_error_does_not_drop_the_tap():
    taps = []

    class Front:
        async def on_callback(self, chat_id, data):
            taps.append((chat_id, data))

    async def too_old():
        raise BadRequest("Query is too old")

    application = build_application(Settings(telegram_bot_token="123:abc", telegram_allowed_chats="1"), front=Front())
    handler = next(h for h in application.handlers[0] if isinstance(h, CallbackQueryHandler))
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=1), callback_query=SimpleNamespace(answer=too_old, data="a:run"))

    asyncio.run(handler.callback(update, None))

    assert taps == [(1, "a:run")]
