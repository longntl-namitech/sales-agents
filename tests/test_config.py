import pytest
from langchain_core.language_models import BaseChatModel

from sales_agents.config import Settings, get_chat_model, get_settings

ENV_NAMES = (
    "OPENAI_API_KEY",
    "SALES_AGENTS_MODEL",
    "SALES_AGENTS_EVAL_TOKEN_CAP",
    "SALES_AGENTS_LANGFUSE_ENABLED",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    # Hermetic: no real env vars and no .env file from the working directory.
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def test_defaults_are_gpt_4_1_mini_and_2m_cap():
    settings = get_settings()

    assert settings.model == "openai:gpt-4.1-mini"
    assert settings.eval_token_cap == 2_000_000
    assert settings.langfuse_enabled is False


def test_env_overrides_model(monkeypatch):
    monkeypatch.setenv("SALES_AGENTS_MODEL", "openai:gpt-4.1")
    monkeypatch.setenv("SALES_AGENTS_EVAL_TOKEN_CAP", "500000")

    settings = Settings(_env_file=None)

    assert settings.model == "openai:gpt-4.1"
    assert settings.eval_token_cap == 500_000


def test_chat_model_requires_api_key():
    settings = Settings(_env_file=None)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_chat_model(settings)


def test_chat_model_built_when_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")

    model = get_chat_model(Settings(_env_file=None))

    assert isinstance(model, BaseChatModel)
