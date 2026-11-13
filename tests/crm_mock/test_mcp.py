import asyncio

from sales_agents.tools.crm import list_crm_records, load_crm_tools, tool_text


def by_name(tools):
    return {t.name: t for t in tools}


def test_mcp_tools_listed(tmp_path):
    tools = asyncio.run(load_crm_tools(tmp_path / "crm.sqlite3"))

    assert set(by_name(tools)) == {"upsert_lead", "record_outreach", "list_records"}


def test_tool_text_reads_a_string_or_mcp_content_blocks():
    assert tool_text("plain") == "plain"
    assert tool_text([{"type": "text", "text": "[1, "}, {"type": "text", "text": "2]"}]) == "[1, 2]"
    assert tool_text(["a", "b"]) == "ab"


def test_list_crm_records_parses_what_an_mcp_tool_returns(tmp_path):
    async def scenario():
        tools = await load_crm_tools(tmp_path / "crm.sqlite3")
        await by_name(tools)["upsert_lead"].ainvoke({"lead_id": "lead-1", "label": "hot", "notes": "x"})
        return await list_crm_records(tools)

    records = asyncio.run(scenario())

    assert [(r["lead_id"], r["label"]) for r in records] == [("lead-1", "hot")]


def test_mcp_state_persists_across_sessions(tmp_path):
    path = tmp_path / "crm.sqlite3"

    async def scenario():
        first = by_name(await load_crm_tools(path))
        await first["upsert_lead"].ainvoke({"lead_id": "lead-1", "label": "hot", "notes": "gates pass"})
        second = by_name(await load_crm_tools(path))  # a new connection, a new server process
        return await second["list_records"].ainvoke({})

    listed = asyncio.run(scenario())

    assert "lead-1" in str(listed) and "hot" in str(listed)
