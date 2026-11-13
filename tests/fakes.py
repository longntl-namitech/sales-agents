from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic import Field


class FakeToolCallingModel(GenericFakeChatModel):
    """Replays scripted messages and records every prompt it was sent.

    GenericFakeChatModel has no bind_tools, which structured output needs, so it
    is overridden to return self. Structured output works when the scripted
    message calls a tool named after the schema class.
    """

    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, *args, **kwargs)


def scripted(*messages) -> FakeToolCallingModel:
    return FakeToolCallingModel(messages=iter(messages))


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def verdict(label: str, reason: str = "scripted") -> AIMessage:
    """A model turn that answers with a QualificationResult."""
    return tool_call("QualificationResult", {"label": label, "reason": reason})
