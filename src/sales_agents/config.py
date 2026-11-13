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


def get_settings() -> Settings:
    return Settings()


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
