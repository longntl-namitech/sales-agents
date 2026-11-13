import sys

import pytest

from sales_agents.config import Settings, get_callbacks


def test_tracing_is_off_by_default():
    assert get_callbacks(Settings()) == []


def test_enabling_tracing_without_the_package_fails_with_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "langfuse", None)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", None)

    with pytest.raises(RuntimeError, match="langfuse"):
        get_callbacks(Settings(langfuse_enabled=True))
