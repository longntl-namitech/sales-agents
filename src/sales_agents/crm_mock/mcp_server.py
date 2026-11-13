"""The mock CRM behind an MCP server, over stdio.

The tools call the FastAPI mock in process through an ASGI transport, so there is
no network. State lives in the sqlite file named by SALES_AGENTS_CRM_DB, which is
how it survives between the short-lived server processes a client starts.

    SALES_AGENTS_CRM_DB=crm.sqlite3 uv run python -m sales_agents.crm_mock.mcp_server
"""

import json
import os

import httpx
from mcp.server.fastmcp import FastMCP

from sales_agents.crm_mock.app import create_app
from sales_agents.crm_mock.store import CrmStore

mcp = FastMCP("mock-crm")
_client = httpx.AsyncClient(
    transport=httpx.ASGITransport(app=create_app(CrmStore(os.environ.get("SALES_AGENTS_CRM_DB", ":memory:")))),
    base_url="http://mock-crm",
)


@mcp.tool()
async def upsert_lead(lead_id: str, label: str, notes: str) -> str:
    """Create or update a lead record with its label and notes. The same lead id updates the same record."""
    response = await _client.put(f"/leads/{lead_id}", json={"label": label, "notes": notes})
    return json.dumps(response.json())


@mcp.tool()
async def record_outreach(lead_id: str, draft_text: str) -> str:
    """Record an approved outreach draft on the lead's record."""
    response = await _client.post(f"/leads/{lead_id}/outreach", json={"draft_text": draft_text})
    return json.dumps(response.json())


@mcp.tool()
async def list_records() -> str:
    """List every lead record in the CRM."""
    response = await _client.get("/leads")
    return json.dumps(response.json())


if __name__ == "__main__":
    mcp.run()
