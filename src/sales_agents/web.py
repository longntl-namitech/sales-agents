"""A demo chat page for the outbound path: pick a lead, watch the subagents work, decide on the draft.

    uv run python -m sales_agents web

Each run streams its steps to the browser. A hot lead pauses at record_outreach and the page shows
Approve, Edit and Reject. The decision resumes the same run, and only then does the guardrail check the
text and the mock CRM record it. Nothing is ever sent. The run itself lives in `sales_agents.runner`.
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from sales_agents.runner import DATASET, RunManager

PAGE = Path(__file__).parent / "static" / "index.html"


class Decision(BaseModel):
    type: str
    text: str | None = None
    message: str | None = None


class StartRun(BaseModel):
    lead_id: str


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def _lines(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    async for event in events:
        yield _sse(event)


def create_app(*, dataset: Path = DATASET, crm_db: str | Path = "crm.sqlite3", raw_tools: list | None = None, models: dict | None = None) -> FastAPI:
    """`raw_tools` and `models` let tests stand in for MCP and the model."""
    app = FastAPI(title="sales-agents demo")
    runner = RunManager(dataset=dataset, crm_db=crm_db, raw_tools=raw_tools, models=models)

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE.read_text()

    @app.get("/api/leads")
    def list_leads() -> list[dict]:
        return runner.leads()

    @app.post("/api/runs")
    async def start(body: StartRun) -> StreamingResponse:
        try:
            events = await runner.start(body.lead_id)
        except KeyError:
            raise HTTPException(404, f"unknown lead id {body.lead_id!r}") from None
        return StreamingResponse(_lines(events), media_type="text/event-stream")

    @app.post("/api/runs/{run_id}/decision")
    async def decide(run_id: str, body: Decision) -> StreamingResponse:
        try:
            events = runner.decide(run_id, body.model_dump())
        except KeyError:
            raise HTTPException(404, f"no run {run_id!r} waiting for a decision") from None
        return StreamingResponse(_lines(events), media_type="text/event-stream")

    return app
