"""A demo chat page for the outbound path: pick a lead, watch the subagents work, decide on the draft.

    uv run python -m sales_agents web

Each run streams its steps to the browser. A hot lead pauses at record_outreach and the page shows
Approve, Edit and Reject. The decision resumes the same run, and only then does the guardrail check the
text and the mock CRM record it. Nothing is ever sent.
"""

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from sales_agents.agents.supervisor import LeadDesk, _opening_message, build_supervisor
from sales_agents.config import get_callbacks, get_settings
from sales_agents.datagen import load_leads
from sales_agents.tools.crm import build_crm_tools, list_crm_records, load_crm_tools, tool_text
from sales_agents.tools.kb import load_docs

DATASET = Path(__file__).parents[2] / "datasets" / "leads_seed42.jsonl"
PAGE = Path(__file__).parent / "static" / "index.html"
SHOWN_RESULTS = {"upsert_lead", "record_outreach"}


class Decision(BaseModel):
    type: str
    text: str | None = None
    message: str | None = None


class StartRun(BaseModel):
    lead_id: str


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _messages(update) -> list:
    return update.get("messages", []) if isinstance(update, dict) and isinstance(update.get("messages"), list) else []


def create_app(*, dataset: Path = DATASET, crm_db: str | Path = "crm.sqlite3", raw_tools: list | None = None, models: dict | None = None) -> FastAPI:
    """`raw_tools` and `models` let tests stand in for MCP and the model."""
    app = FastAPI(title="sales-agents demo")
    leads = {c.lead.lead_id: c.lead for c in load_leads(dataset)}
    runs: dict[str, dict] = {}
    cache: dict = {}

    async def tools() -> list:
        if "tools" not in cache:
            cache["tools"] = raw_tools if raw_tools is not None else await load_crm_tools(crm_db)
        return cache["tools"]

    async def stream(run: dict, agent_input) -> AsyncIterator[str]:
        agent, config = run["agent"], run["config"]
        paused = False
        async for _, chunk in agent.astream(agent_input, config=config, stream_mode="updates", subgraphs=True):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    request = update[0].value["action_requests"][0]
                    run["request"] = request
                    if paused:
                        continue
                    paused = True
                    yield _sse({"type": "approval", "draft_text": request["args"]["draft_text"]})
                    continue
                for message in _messages(update):
                    if isinstance(message, AIMessage):
                        for call in message.tool_calls:
                            if call["name"] == "task":
                                yield _sse({"type": "step", "agent": call["args"].get("subagent_type", "")})
                    elif isinstance(message, ToolMessage) and message.name in SHOWN_RESULTS:
                        yield _sse({"type": "tool", "name": message.name, "text": tool_text(message.content)})
        if not paused:
            record = next((r for r in await list_crm_records(await tools()) if r["lead_id"] == run["lead_id"]), None)
            yield _sse({"type": "result", "record": record})

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE.read_text()

    @app.get("/api/leads")
    def list_leads() -> list[dict]:
        return [
            {"lead_id": lead.lead_id, "company_name": lead.company_name, "contact_role": lead.contact_role, "source": lead.source}
            for lead in leads.values()
        ]

    @app.post("/api/runs")
    async def start(body: StartRun) -> StreamingResponse:
        if body.lead_id not in leads:
            raise HTTPException(404, f"unknown lead id {body.lead_id!r}")
        lead = leads[body.lead_id]
        desk = LeadDesk()
        crm_tools = build_crm_tools(await tools(), load_docs(), results=desk.results.get)
        agent = build_supervisor(get_settings(), MemorySaver(), desk, models=models, crm_tools=crm_tools)
        run_id = uuid.uuid4().hex[:8]
        config = {"configurable": {"thread_id": run_id}, "callbacks": get_callbacks(get_settings())}
        run = {"agent": agent, "config": config, "lead_id": lead.lead_id}
        runs[run_id] = run
        desk.add(lead)

        async def body_stream() -> AsyncIterator[str]:
            yield _sse({"type": "started", "run_id": run_id, "lead_id": lead.lead_id, "company_name": lead.company_name})
            async for line in stream(run, _opening_message(lead)):
                yield line

        return StreamingResponse(body_stream(), media_type="text/event-stream")

    @app.post("/api/runs/{run_id}/decision")
    async def decide(run_id: str, body: Decision) -> StreamingResponse:
        run = runs.get(run_id)
        if run is None or "request" not in run:
            raise HTTPException(404, f"no run {run_id!r} waiting for a decision")
        request = run.pop("request")
        if body.type == "edit":
            edited = {"name": request["name"], "args": {**request["args"], "draft_text": body.text or ""}}
            decision = {"type": "edit", "edited_action": edited}
        elif body.type == "reject":
            decision = {"type": "reject", "message": body.message or "rejected by the reviewer"}
        else:
            decision = {"type": "approve"}
        return StreamingResponse(stream(run, Command(resume={"decisions": [decision]})), media_type="text/event-stream")

    return app
