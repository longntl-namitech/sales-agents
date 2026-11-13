"""CRM tools for the Supervisor.

`load_crm_tools` starts the mock CRM's MCP server and returns its tools.
`store_tools` offers the same three tools straight over a `CrmStore`, for tests and
offline evals that do not need a subprocess. `build_crm_tools` gives the agent
`upsert_lead` as is, and wraps `record_outreach` so the guardrail runs inside it.

That wrapper is where the safety rule lives. The human decision comes first, from
`interrupt_on` on the Supervisor. The wrapper then checks the text that was
actually approved, so an edited draft is checked too, and refuses on a violation.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from sales_agents.crm_mock.store import CrmStore
from sales_agents.guardrail import check_draft
from sales_agents.schemas import Draft, QualificationResult


async def load_crm_tools(db_path: str | Path) -> list[BaseTool]:
    client = MultiServerMCPClient(
        {
            "crm": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "sales_agents.crm_mock.mcp_server"],
                "env": {"SALES_AGENTS_CRM_DB": str(db_path)},
            }
        }
    )
    return await client.get_tools()


def tool_text(result) -> str:
    """The text of a tool result. MCP tools return a list of content blocks, in-process tools a string."""
    if isinstance(result, str):
        return result
    return "".join(block if isinstance(block, str) else block.get("text", "") for block in result)


async def list_crm_records(tools: list[BaseTool]) -> list[dict]:
    listing = next(t for t in tools if t.name == "list_records")
    return json.loads(tool_text(await listing.ainvoke({})))


def store_tools(store: CrmStore) -> list[BaseTool]:
    async def upsert_lead(lead_id: str, label: str, notes: str) -> str:
        """Create or update a lead record with its label and notes."""
        return json.dumps(store.upsert_lead(lead_id, label, notes))

    async def record_outreach(lead_id: str, draft_text: str) -> str:
        """Record an approved outreach draft on the lead's record."""
        return json.dumps(store.record_outreach(lead_id, draft_text))

    async def list_records() -> str:
        """List every lead record in the CRM."""
        return json.dumps(store.list_records())

    return [StructuredTool.from_function(coroutine=f) for f in (upsert_lead, record_outreach, list_records)]


def record_if_approved(
    store: CrmStore, lead_id: str, draft_text: str, docs: list[str], decision: str
) -> bool:
    """For code paths that run without the Supervisor: record a draft only if a human (or the scripted
    approver) approved it and the guardrail passes on the text that was approved. Returns whether it wrote."""
    if decision != "approve" or not check_draft(Draft(body=draft_text), docs).passed:
        return False
    store.record_outreach(lead_id, draft_text)
    return True


def build_crm_tools(
    raw_tools: list[BaseTool],
    docs: list[str],
    *,
    results: Callable[[str], QualificationResult | None] | None = None,
) -> list[BaseTool]:
    """With `results`, the label and notes written to the CRM come from the Qualifier's result for that
    lead and not from what the agent passes in, and only a hot lead can get a draft recorded."""
    raw = {t.name: t for t in raw_tools}

    @tool("record_outreach")
    async def record_outreach(lead_id: str, draft_text: str) -> str:
        """Record an outreach draft on the lead's CRM record. Refuses a draft that fails the guardrail."""
        if results is not None:
            result = results(lead_id)
            if result is None or result.label != "hot":
                return "BLOCKED, nothing was recorded: only a lead the Qualifier labelled hot can get a draft"
        verdict = check_draft(Draft(body=draft_text), docs)
        if not verdict.passed:
            return "BLOCKED by the guardrail, nothing was recorded: " + "; ".join(verdict.violations)
        return await raw["record_outreach"].ainvoke({"lead_id": lead_id, "draft_text": draft_text})

    if results is None:
        return [raw["upsert_lead"], record_outreach]

    @tool("upsert_lead")
    async def upsert_lead(lead_id: str, label: str = "", notes: str = "") -> str:
        """Save the lead's qualification to the CRM. The label and reason come from the Qualifier's result."""
        result = results(lead_id)
        if result is None:
            return "BLOCKED, nothing was saved: this lead has not been qualified yet"
        return await raw["upsert_lead"].ainvoke({"lead_id": lead_id, "label": result.label, "notes": result.reason})

    return [upsert_lead, record_outreach]
