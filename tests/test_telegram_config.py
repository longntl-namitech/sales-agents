import pytest

from sales_agents.config import Settings, allowed_chat_ids


def test_chat_ids_are_parsed_from_a_comma_separated_list():
    assert allowed_chat_ids(Settings(telegram_allowed_chats="1, 22")) == {1, 22}


@pytest.mark.parametrize("value", ["", "  ", " , "])
def test_an_empty_list_refuses_to_start(value):
    with pytest.raises(SystemExit, match="SALES_AGENTS_TELEGRAM_ALLOWED_CHATS"):
        allowed_chat_ids(Settings(telegram_allowed_chats=value))


def test_a_non_integer_entry_refuses_to_start():
    with pytest.raises(SystemExit, match="abc"):
        allowed_chat_ids(Settings(telegram_allowed_chats="1,abc"))
