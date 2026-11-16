"""The chat side of the Telegram front end: one chat, one run, a person deciding on the draft.

It talks to a `RunManager` and to an injected `send`, so it needs no Telegram to be tested. A hot lead pauses
at record_outreach: the draft arrives with Approve, Edit and Reject buttons. Only private chats on the
allow-list are answered. Nothing is ever sent to a lead.
"""

import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable

from sales_agents.runner import RunManager

log = logging.getLogger(__name__)

LIMIT = 4096  # Telegram's maximum message length

Send = Callable[[int, str, list[tuple[str, str]] | None], Awaitable[None]]

STEPS = {
    "researcher": "Researching the lead...",
    "qualifier": "Qualifying the lead...",
    "writer": "Writing the draft...",
    "crm_agent": "Updating the CRM...",
}


class TelegramFront:
    def __init__(self, runner: RunManager, send: Send, allowed: frozenset[int], max_runs: int = 5):
        self._runner = runner
        self._send = send
        self._allowed = allowed
        self._max_runs = max_runs
        self._chats: dict[int, dict] = {}
        self._started: dict[int, int] = {}

    def _answers(self, chat_id: int) -> bool:
        """Private chats have positive ids in Telegram, groups and channels negative ones."""
        return chat_id > 0 and chat_id in self._allowed

    async def _say(self, chat_id: int, text: str, buttons: list[tuple[str, str]] | None = None) -> None:
        parts = [text[i : i + LIMIT] for i in range(0, len(text), LIMIT)] or [""]
        for part in parts[:-1]:
            await self._send(chat_id, part, None)
        await self._send(chat_id, parts[-1], buttons)

    async def on_start(self, chat_id: int) -> None:
        if not self._answers(chat_id):
            return
        buttons = [(lead["company_name"], f"pick:{lead['lead_id']}") for lead in self._runner.leads()]
        await self._say(chat_id, "Pick a lead to run.", buttons)

    async def run_lead(self, chat_id: int, lead_id: str) -> None:
        """Starts a run. A chat has one run in flight, so an earlier pending run is dropped."""
        if not self._answers(chat_id):
            return
        if self._started.get(chat_id, 0) >= self._max_runs:
            await self._say(chat_id, "Run limit reached for this chat.")
            return
        earlier = self._chats.pop(chat_id, None)
        if earlier:
            self._runner.drop(earlier["run_id"])
        try:
            events = await self._runner.start(lead_id)
        except KeyError:
            await self._say(chat_id, f"Unknown lead {lead_id}.")
            return
        self._started[chat_id] = self._started.get(chat_id, 0) + 1
        await self._consume(chat_id, events)

    async def on_callback(self, chat_id: int, data: str) -> None:
        if not self._answers(chat_id):
            return
        kind, _, rest = data.partition(":")
        if kind == "pick":
            await self.run_lead(chat_id, rest)
            return
        state = self._chats.get(chat_id)
        if kind not in ("a", "e", "r") or not state or state["run_id"] != rest or not state["pending"]:
            await self._say(chat_id, "This run is no longer waiting.")
            return
        if kind == "e":
            state["awaiting_edit"] = True
            await self._say(chat_id, "Reply with the new draft text.")
            return
        decision = {"type": "approve"} if kind == "a" else {"type": "reject"}
        await self._decide(chat_id, state, decision)

    async def on_text(self, chat_id: int, text: str) -> None:
        if not self._answers(chat_id):
            return
        state = self._chats.get(chat_id)
        if not state or not state["pending"] or not state["awaiting_edit"]:
            await self._say(chat_id, "Nothing is waiting for text. Use the buttons on the draft.")
            return
        await self._decide(chat_id, state, {"type": "edit", "text": text})

    async def _decide(self, chat_id: int, state: dict, decision: dict) -> None:
        state["pending"] = state["awaiting_edit"] = False
        try:
            events = self._runner.decide(state["run_id"], decision)
        except KeyError:
            await self._say(chat_id, "This run is no longer waiting.")
            return
        await self._consume(chat_id, events)

    async def _consume(self, chat_id: int, events: AsyncIterator[dict]) -> None:
        try:
            async for event in events:
                await self._handle(chat_id, event)
        except Exception as exc:
            log.exception("run failed")
            state = self._chats.pop(chat_id, None)
            if state:
                self._runner.drop(state["run_id"])
            await self._say(chat_id, f"The run failed: {type(exc).__name__}. Pick a lead to try again.")

    async def _handle(self, chat_id: int, event: dict) -> None:
        if event["type"] == "started":
            self._chats[chat_id] = {"run_id": event["run_id"], "pending": False, "awaiting_edit": False}
            await self._say(chat_id, f"Running {event['company_name']}...")
        elif event["type"] == "step":
            await self._say(chat_id, STEPS.get(event["agent"], f"{event['agent']}..."))
        elif event["type"] == "tool":
            await self._say(chat_id, _tool_line(event["name"], event["text"]))
        elif event["type"] == "approval":
            state = self._chats[chat_id]
            state["pending"] = True
            run_id = state["run_id"]
            buttons = [("Approve", f"a:{run_id}"), ("Edit", f"e:{run_id}"), ("Reject", f"r:{run_id}")]
            await self._say(chat_id, event["draft_text"], buttons)
        elif event["type"] == "result":
            await self._say(chat_id, _result_line(event["record"]))


def _tool_line(name: str, text: str) -> str:
    """A short line for a CRM tool result. A reviewer's edit puts a note before the tool's own answer."""
    blocked = re.search(r"(?:^|\n)(BLOCKED.*)", text, re.S)
    if blocked:
        return blocked.group(1)
    if name == "record_outreach":
        return "Guardrail passed. The draft is recorded in the mock CRM."
    try:
        record = json.loads(text)
    except ValueError:
        return text
    return f"CRM: saved as {str(record.get('label', '?')).upper()}.\n{record.get('notes') or ''}".rstrip()


def _result_line(record: dict | None) -> str:
    if not record:
        return "Result: no CRM record."
    recorded = "outreach recorded" if record.get("outreach") else "no outreach recorded"
    return f"Result: {record['label']} - {recorded}."
