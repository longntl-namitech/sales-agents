"""The chat side of the Telegram front end: one chat, one run, a person deciding on the draft.

It talks to a `RunManager` and to an injected `send`, so it needs no Telegram to be tested. A hot lead pauses
at record_outreach: the draft arrives with Approve, Edit and Reject buttons. Nothing is ever sent to a lead.
"""

from collections.abc import AsyncIterator, Awaitable, Callable

from sales_agents.runner import RunManager

LIMIT = 4096  # Telegram's maximum message length

Send = Callable[[int, str, list[tuple[str, str]] | None], Awaitable[None]]


class TelegramFront:
    def __init__(self, runner: RunManager, send: Send):
        self._runner = runner
        self._send = send
        self._chats: dict[int, dict] = {}

    async def _say(self, chat_id: int, text: str, buttons: list[tuple[str, str]] | None = None) -> None:
        parts = [text[i : i + LIMIT] for i in range(0, len(text), LIMIT)] or [""]
        for part in parts[:-1]:
            await self._send(chat_id, part, None)
        await self._send(chat_id, parts[-1], buttons)

    async def run_lead(self, chat_id: int, lead_id: str) -> None:
        """Starts a run. A chat has one run in flight, so an earlier pending run is dropped."""
        earlier = self._chats.pop(chat_id, None)
        if earlier:
            self._runner.drop(earlier["run_id"])
        try:
            events = await self._runner.start(lead_id)
        except KeyError:
            await self._say(chat_id, f"Unknown lead {lead_id}.")
            return
        await self._consume(chat_id, events)

    async def on_callback(self, chat_id: int, data: str) -> None:
        kind, _, run_id = data.partition(":")
        state = self._chats.get(chat_id)
        if kind not in ("a", "e", "r") or not state or state["run_id"] != run_id or not state["pending"]:
            await self._say(chat_id, "This run is no longer waiting.")
            return
        if kind == "e":
            state["awaiting_edit"] = True
            await self._say(chat_id, "Reply with the new draft text.")
            return
        decision = {"type": "approve"} if kind == "a" else {"type": "reject"}
        await self._decide(chat_id, state, decision)

    async def on_text(self, chat_id: int, text: str) -> None:
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
        async for event in events:
            if event["type"] == "started":
                self._chats[chat_id] = {"run_id": event["run_id"], "pending": False, "awaiting_edit": False}
            elif event["type"] == "approval":
                state = self._chats[chat_id]
                state["pending"] = True
                run_id = state["run_id"]
                buttons = [("Approve", f"a:{run_id}"), ("Edit", f"e:{run_id}"), ("Reject", f"r:{run_id}")]
                await self._say(chat_id, event["draft_text"], buttons)
            elif event["type"] == "result":
                await self._say(chat_id, _result_line(event["record"]))


def _result_line(record: dict | None) -> str:
    if not record:
        return "Result: no CRM record."
    recorded = "outreach recorded" if record.get("outreach") else "no outreach recorded"
    return f"Result: {record['label']} - {recorded}."
