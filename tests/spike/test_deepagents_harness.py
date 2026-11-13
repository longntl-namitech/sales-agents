"""Task 1 spike: what does the Deep Agents harness actually do?

Every model here is a scripted fake, so these tests run offline and cost no tokens.

Versions tested: deepagents 0.7.37, langchain 1.5.0, langgraph 1.2.21.

FINDINGS
1. Fake models: GenericFakeChatModel has no bind_tools, which Deep Agents always
   calls, so FakeToolCallingModel below overrides it.
2. Plain Python functions work as tools. The model "decides" through the
   scripted tool_calls on an AIMessage.
3. response_format: the answer is result["structured_response"]. The model must
   call a tool named after the schema class (here "Verdict").
4. Human approval: with interrupt_on, invoke returns a result containing
   "__interrupt__" and the tool has NOT run. Resume with
   Command(resume={"decisions": [...]}), the same thread_id and a checkpointer.
   A reject decision skips the tool and the agent gets a ToolMessage instead.
5. No checkpointer: the first invoke does not raise. It returns the interrupt
   and the tool does not run (fails closed). Resuming raises RuntimeError
   "Cannot use Command(resume=...) without checkpointer".
6. Subagents: the parent calls the built-in "task" tool with description and
   subagent_type. The subagent sees description as its only user message. The
   parent receives the subagent's final message as the ToolMessage content.

ANSWERS TO THE THREE OPEN QUESTIONS
Q1. Compiled subagent: CompiledSubAgent(name=..., description=...,
    runnable=<compiled graph>), see test_compiled_subagent_runs.
Q2. response_format works on a subagent: the parent receives the structured
    answer as a JSON string in the ToolMessage, see
    test_subagent_response_format_returns_json_to_parent.
Q3. MCP tools: langchain_mcp_adapters.client.MultiServerMCPClient (checked on
    langchain-mcp-adapters 0.3.2, its get_tools is async). That version has no
    MCPAdapter class, so the name on the Deep Agents tools page did not match.
    langchain-mcp-adapters is not a project dependency yet, add it in Task 8.
"""

import json
from typing import Literal

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from deepagents import CompiledSubAgent, create_deep_agent


class FakeToolCallingModel(GenericFakeChatModel):
    """Replays scripted messages in order.

    GenericFakeChatModel does not implement bind_tools, but Deep Agents always
    calls it, so we override it. The scripted tool calls are all the "model"
    ever does, which makes every test deterministic.
    """

    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*messages):
    return FakeToolCallingModel(messages=iter(messages))


def tool_call(name, args, call_id="call_1"):
    """A model turn that asks the agent to run one tool."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def ask(text):
    return {"messages": [("user", text)]}


class Verdict(BaseModel):
    label: Literal["hot", "warm", "cold"]
    reason: str


def test_agent_calls_custom_tool():
    calls = []

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        calls.append((a, b))
        return a + b

    model = scripted(tool_call("add", {"a": 2, "b": 3}), AIMessage(content="It is 5"))
    agent = create_deep_agent(model=model, tools=[add])

    result = agent.invoke(ask("what is 2 + 3?"))

    assert calls == [(2, 3)]
    assert result["messages"][-1].content == "It is 5"


def test_response_format_returns_schema_object():
    # With a schema, the model "answers" by calling a tool named after the schema.
    model = scripted(
        tool_call("Verdict", {"label": "hot", "reason": "fits the rubric"}),
    )
    agent = create_deep_agent(model=model, response_format=Verdict)

    result = agent.invoke(ask("qualify this lead"))

    assert result["structured_response"] == Verdict(
        label="hot", reason="fits the rubric"
    )


def make_gated_agent(model, sent, checkpointer):
    def send_email(to: str) -> str:
        """Send an email."""
        sent.append(to)
        return "sent"

    return create_deep_agent(
        model=model,
        tools=[send_email],
        interrupt_on={"send_email": True},
        checkpointer=checkpointer,
    )


def test_interrupt_on_pauses_then_approve_resumes():
    sent = []
    model = scripted(
        tool_call("send_email", {"to": "a@example.com"}), AIMessage(content="done")
    )
    agent = make_gated_agent(model, sent, MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    paused = agent.invoke(ask("email a"), config=config)

    assert "__interrupt__" in paused
    assert sent == []  # the tool has not run yet

    resumed = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}), config=config
    )

    assert sent == ["a@example.com"]
    assert resumed["messages"][-1].content == "done"


def test_reject_decision_skips_tool():
    sent = []
    model = scripted(
        tool_call("send_email", {"to": "a@example.com"}), AIMessage(content="ok")
    )
    agent = make_gated_agent(model, sent, MemorySaver())
    config = {"configurable": {"thread_id": "t2"}}

    agent.invoke(ask("email a"), config=config)
    resumed = agent.invoke(
        Command(resume={"decisions": [{"type": "reject", "message": "no"}]}),
        config=config,
    )

    assert sent == []
    tool_messages = [m for m in resumed["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages, "the agent should be told the call was rejected"


def test_resume_without_checkpointer_fails():
    sent = []
    model = scripted(
        tool_call("send_email", {"to": "a@example.com"}), AIMessage(content="done")
    )
    agent = make_gated_agent(model, sent, checkpointer=None)
    config = {"configurable": {"thread_id": "t3"}}

    # Without a checkpointer the first call does not raise. It returns an
    # interrupt and the tool has not run, so it fails closed.
    paused = agent.invoke(ask("email a"), config=config)
    assert "__interrupt__" in paused

    # But the pause cannot be resumed, so nothing can ever be approved.
    with pytest.raises(RuntimeError, match="checkpointer"):
        agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)

    assert sent == []


def test_subagent_receives_delegated_task():
    parent = scripted(
        tool_call(
            "task", {"description": "find the CEO", "subagent_type": "researcher"}
        ),
        AIMessage(content="parent done"),
    )
    researcher_model = scripted(AIMessage(content="The CEO is Ada"))
    agent = create_deep_agent(
        model=parent,
        subagents=[
            {
                "name": "researcher",
                "description": "Finds facts about companies.",
                "system_prompt": "You research companies.",
                "model": researcher_model,
            }
        ],
    )

    result = agent.invoke(ask("who runs Acme?"))

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[-1].content == "The CEO is Ada"


def test_compiled_subagent_runs():
    inner = create_deep_agent(model=scripted(AIMessage(content="compiled says hi")))
    parent = scripted(
        tool_call("task", {"description": "say hi", "subagent_type": "compiled"}),
        AIMessage(content="parent done"),
    )
    agent = create_deep_agent(
        model=parent,
        subagents=[
            CompiledSubAgent(
                name="compiled", description="A prebuilt graph.", runnable=inner
            )
        ],
    )

    result = agent.invoke(ask("use the compiled agent"))

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[-1].content == "compiled says hi"


def test_subagent_response_format_returns_json_to_parent():
    parent = scripted(
        tool_call("task", {"description": "qualify", "subagent_type": "qualifier"}),
        AIMessage(content="parent done"),
    )
    qualifier_model = scripted(tool_call("Verdict", {"label": "cold", "reason": "gate"}))
    agent = create_deep_agent(
        model=parent,
        subagents=[
            {
                "name": "qualifier",
                "description": "Qualifies a lead.",
                "system_prompt": "You qualify leads.",
                "model": qualifier_model,
                "response_format": Verdict,
            }
        ],
    )

    result = agent.invoke(ask("qualify the lead"))

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert json.loads(tool_messages[-1].content) == {"label": "cold", "reason": "gate"}
