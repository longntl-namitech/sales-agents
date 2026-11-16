from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SALES_AGENTS_", env_file=".env", extra="ignore"
    )

    model: str = "openai:gpt-4.1-mini"
    eval_token_cap: int = 2_000_000
    langfuse_enabled: bool = False
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY")
    )
    telegram_bot_token: SecretStr | None = None
    telegram_allowed_chats: str = ""


def get_settings() -> Settings:
    return Settings()


def allowed_chat_ids(settings: Settings) -> frozenset[int]:
    """The chats the Telegram bot answers. Empty or malformed means the bot does not start."""
    names = [part.strip() for part in settings.telegram_allowed_chats.split(",") if part.strip()]
    if not names:
        raise SystemExit("SALES_AGENTS_TELEGRAM_ALLOWED_CHATS is empty. Set it to a comma-separated list of chat ids.")
    try:
        return frozenset(int(name) for name in names)
    except ValueError:
        raise SystemExit(f"SALES_AGENTS_TELEGRAM_ALLOWED_CHATS must be chat ids, got {', '.join(names)!r}.") from None


def get_callbacks(settings: Settings) -> list:
    """LangChain callbacks for tracing. Empty unless Langfuse is switched on.

    Langfuse reads its own LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and
    LANGFUSE_HOST from the environment. The package is an optional extra.
    """
    if not settings.langfuse_enabled:
        return []
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError as exc:
        raise RuntimeError(
            "Tracing is enabled but langfuse is not installed. Run: uv sync --extra tracing"
        ) from exc
    return [CallbackHandler()]


def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.openai_api_key is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env or the environment."
        )
    return init_chat_model(
        settings.model, api_key=settings.openai_api_key.get_secret_value()
    )
