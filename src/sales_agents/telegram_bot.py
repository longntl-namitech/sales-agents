"""The Telegram bot: long polling over python-telegram-bot, with all chat logic in `TelegramFront`.

    uv run python -m sales_agents telegram

Set SALES_AGENTS_TELEGRAM_BOT_TOKEN and SALES_AGENTS_TELEGRAM_ALLOWED_CHATS (comma-separated chat ids).
The bot refuses to start without either. A paused run lives in memory and is lost on restart.
"""

from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from sales_agents.config import Settings, allowed_chat_ids, get_settings
from sales_agents.runner import RunManager
from sales_agents.telegram_front import TelegramFront


def build_application(settings: Settings, front: TelegramFront | None = None, *, crm_db: str | Path = "crm.sqlite3") -> Application:
    """`front` lets tests stand in for the chat logic."""
    application = Application.builder().token(settings.telegram_bot_token.get_secret_value()).build()

    async def send(chat_id: int, text: str, buttons: list[tuple[str, str]] | None = None) -> None:
        markup = InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=data)] for label, data in buttons]) if buttons else None
        await application.bot.send_message(chat_id, text, reply_markup=markup)

    front = front or TelegramFront(RunManager(crm_db=crm_db), send, allowed_chat_ids(settings))

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await front.on_start(update.effective_chat.id)

    async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.callback_query.answer()
        await front.on_callback(update.effective_chat.id, update.callback_query.data)

    async def text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await front.on_text(update.effective_chat.id, update.message.text)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text))
    return application


def run_bot(crm_db: str | Path = "crm.sqlite3") -> None:
    settings = get_settings()
    allowed_chat_ids(settings)  # exits before anything starts when the allow-list is empty
    if settings.telegram_bot_token is None or not settings.telegram_bot_token.get_secret_value():
        raise SystemExit("SALES_AGENTS_TELEGRAM_BOT_TOKEN is not set. Create a bot with @BotFather and add the token.")
    build_application(settings, crm_db=crm_db).run_polling()
