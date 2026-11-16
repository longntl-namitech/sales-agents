"""Runs one lead through the Supervisor and yields its steps as events. The web page and Telegram share it.

A hot lead pauses at record_outreach: the stream ends with one `approval` event, and `decide` resumes the
same run with the reviewer's decision. Only then does the guardrail check the text and the mock CRM record it.
Nothing is ever sent.
"""

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from sales_agents.agents.supervisor import LeadDesk, _opening_message, build_supervisor
from sales_agents.config import get_callbacks, get_settings
from sales_agents.datagen import load_leads
from sales_agents.tools.crm import build_crm_tools, list_crm_records, load_crm_tools, tool_text
from sales_agents.tools.kb import load_docs

DATASET = Path(__file__).parents[2] / "datasets" / "leads_seed42.jsonl"
SHOWN_RESULTS = {"upsert_lead", "record_outreach"}


def _messages(update) -> list:
    return update.get("messages", []) if isinstance(update, dict) and isinstance(update.get("messages"), list) else []


class RunManager:
    """`raw_tools` and `models` let tests stand in for MCP and the model."""

    def __init__(self, *, dataset: Path = DATASET, crm_db: str | Path = "crm.sqlite3", raw_tools: list | None = None, models: dict | None = None):
        self._leads = {c.lead.lead_id: c.lead for c in load_leads(dataset)}
        self._crm_db = crm_db
        self._raw_tools = raw_tools
        self._models = models
        self._runs: dict[str, dict] = {}
        self._tools_cache: list | None = None

    async def _tools(self) -> list:
        if self._tools_cache is None:
            self._tools_cache = self._raw_tools if self._raw_tools is not None else await load_crm_tools(self._crm_db)
        return self._tools_cache

    def leads(self) -> list[dict]:
        return [
            {"lead_id": lead.lead_id, "company_name": lead.company_name, "contact_role": lead.contact_role, "source": lead.source}
            for lead in self._leads.values()
        ]

    async def start(self, lead_id: str) -> AsyncIterator[dict]:
        """Raises KeyError for an unknown lead before any event is produced."""
        if lead_id not in self._leads:
            raise KeyError(lead_id)
        lead = self._leads[lead_id]
        desk = LeadDesk()
        crm_tools = build_crm_tools(await self._tools(), load_docs(), results=desk.results.get)
        agent = build_supervisor(get_settings(), MemorySaver(), desk, models=self._models, crm_tools=crm_tools)
        run_id = uuid.uuid4().hex[:8]
        config = {"configurable": {"thread_id": run_id}, "callbacks": get_callbacks(get_settings())}
        run = {"agent": agent, "config": config, "lead_id": lead.lead_id}
        self._runs[run_id] = run
        desk.add(lead)

        async def events() -> AsyncIterator[dict]:
            yield {"type": "started", "run_id": run_id, "lead_id": lead.lead_id, "company_name": lead.company_name}
            async for event in self._stream(run, _opening_message(lead)):
                yield event

        return events()

    def decide(self, run_id: str, decision: dict) -> AsyncIterator[dict]:
        """Raises KeyError when the run is unknown or not waiting for a decision."""
        run = self._runs.get(run_id)
        if run is None or "request" not in run:
            raise KeyError(run_id)
        request = run.pop("request")
        if decision["type"] == "edit":
            edited = {"name": request["name"], "args": {**request["args"], "draft_text": decision.get("text") or ""}}
            resume = {"type": "edit", "edited_action": edited}
        elif decision["type"] == "reject":
            resume = {"type": "reject", "message": decision.get("message") or "rejected by the reviewer"}
        else:
            resume = {"type": "approve"}
        return self._stream(run, Command(resume={"decisions": [resume]}))

    def drop(self, run_id: str) -> None:
        self._runs.pop(run_id, None)

    async def _stream(self, run: dict, agent_input) -> AsyncIterator[dict]:
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
                    yield {"type": "approval", "draft_text": request["args"]["draft_text"]}
                    continue
                for message in _messages(update):
                    if isinstance(message, AIMessage):
                        for call in message.tool_calls:
                            if call["name"] == "task":
                                yield {"type": "step", "agent": call["args"].get("subagent_type", "")}
                    elif isinstance(message, ToolMessage) and message.name in SHOWN_RESULTS:
                        yield {"type": "tool", "name": message.name, "text": tool_text(message.content)}
        if not paused:
            record = next((r for r in await list_crm_records(await self._tools()) if r["lead_id"] == run["lead_id"]), None)
            yield {"type": "result", "record": record}
