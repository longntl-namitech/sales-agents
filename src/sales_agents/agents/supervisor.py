"""The Supervisor: the Deep Agent that calls each agent and gets its result back.

Agents are plain typed functions. The Supervisor reaches them through tools that
take only a lead id, and the data goes from the function to the function through
a shared `LeadDesk`, never through the Supervisor's own words. A first version
delegated to LLM subagents instead, and the Supervisor's written task
description dropped the company facts, so hot precision fell from 0.952 to 0.650
(see the README).

Two consequences. The lead's free-text notes never reach the Supervisor model.
And the label comes from the `qualify_lead` result, so the Supervisor owns no
business rule and cannot change one.
"""

from collections.abc import Callable, Mapping

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from deepagents import create_deep_agent
from sales_agents.agents.qualifier import qualify
from sales_agents.agents.researcher import research
from sales_agents.agents.writer import draft_outreach
from sales_agents.approver import scripted_approve
from sales_agents.config import Settings, get_callbacks, get_chat_model, get_settings
from sales_agents.schemas import CompanyRecord, Draft, Lead, QualificationResult

SUPERVISOR_PROMPT = """You coordinate lead qualification. You receive a lead id, a company name and a source. Do this:
1. Call research_lead with the lead id.
2. Call qualify_lead with the lead id.
3. Reply with the label and reason that qualify_lead returned.
Never decide the label yourself."""

OUTBOUND_PROMPT = """You coordinate outbound sales for one lead. You receive a lead id, a company name and a source. Do this:
1. Call research_lead with the lead id.
2. Call qualify_lead with the lead id.
3. Call upsert_lead with the lead id, the label, and the reason as notes.
4. Only if the label is hot: call write_draft with the lead id, then call record_outreach with the lead id and the exact draft text that write_draft returned, unchanged.
For a warm or cold lead, stop after upsert_lead and write no draft.
Never decide the label yourself and never edit a draft. If record_outreach reports that a draft was blocked or rejected, do not retry and do not write another draft. Say what happened and stop."""

APPROVAL = {"allowed_decisions": ["approve", "edit", "reject"]}


class LeadDesk:
    """Per-run state shared by the agent tools: the leads, what research found, what the Qualifier said."""

    def __init__(self, *, models: Mapping[str, BaseChatModel] | None = None, settings: Settings | None = None):
        self.leads: dict[str, Lead] = {}
        self.companies: dict[str, CompanyRecord | None] = {}
        self.results: dict[str, QualificationResult] = {}
        self._models = models or {}
        self._settings = settings

    def add(self, lead: Lead) -> None:
        self.leads[lead.lead_id] = lead

    def _lead(self, lead_id: str) -> Lead:
        if lead_id not in self.leads:
            raise ValueError(f"unknown lead id {lead_id!r}")
        return self.leads[lead_id]

    def _company(self, lead_id: str) -> CompanyRecord | None:
        """The researched company, looked up now if the Supervisor has not called research_lead yet.
        The Qualifier needs it, so that dependency is code rather than something the prompt must secure."""
        if lead_id not in self.companies:
            self.companies[lead_id] = research(self._lead(lead_id))
        return self.companies[lead_id]

    def _model(self, role: str) -> BaseChatModel:
        return self._models.get(role) or get_chat_model(self._settings or get_settings())

    def tools(self) -> list[BaseTool]:
        desk = self

        @tool
        def research_lead(lead_id: str) -> str:
            """Look up company size, industry, region and signals for the lead in the synthetic store."""
            record = desk._company(lead_id)
            return record.model_dump_json() if record else "no record found"

        @tool
        def qualify_lead(lead_id: str) -> str:
            """Apply the qualification rubric to the lead. Returns JSON with a label (hot, warm or cold) and a reason."""
            result = qualify(desk._lead(lead_id), desk._company(lead_id), model=desk._model("qualifier"))
            desk.results[lead_id] = result
            return result.model_dump_json()

        @tool
        def write_draft(lead_id: str) -> str:
            """Write the first outreach email for a researched lead. Returns the email text."""
            company = desk._company(lead_id)
            if company is None:
                return "cannot write a draft: there is no company record for this lead"
            return draft_outreach(desk._lead(lead_id), company, model=desk._model("writer")).text

        return [research_lead, qualify_lead, write_draft]


def build_supervisor(
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
    desk: LeadDesk,
    *,
    models: Mapping[str, BaseChatModel] | None = None,
    crm_tools: list[BaseTool] | None = None,
):
    """`models` overrides the "supervisor" model for tests. With `crm_tools` it is the outbound
    Supervisor: record_outreach then pauses for a human decision before it runs."""
    models = models or {}
    outbound = crm_tools is not None
    return create_deep_agent(
        model=models.get("supervisor") or get_chat_model(settings),
        system_prompt=OUTBOUND_PROMPT if outbound else SUPERVISOR_PROMPT,
        tools=[*desk.tools(), *(crm_tools or [])],
        interrupt_on={"record_outreach": APPROVAL} if outbound else None,
        checkpointer=checkpointer,
    )


def _opening_message(lead: Lead) -> dict:
    return {"messages": [("user", f"Lead {lead.lead_id}: {lead.company_name} ({lead.source})")]}


def run_lead(
    lead: Lead,
    thread_id: str,
    *,
    agent,
    desk: LeadDesk,
    settings: Settings | None = None,
) -> QualificationResult:
    settings = settings or get_settings()
    desk.add(lead)
    agent.invoke(
        _opening_message(lead),
        config={"configurable": {"thread_id": thread_id}, "callbacks": get_callbacks(settings)},
    )
    if lead.lead_id not in desk.results:
        raise ValueError("the supervisor finished without calling qualify_lead")
    return desk.results[lead.lead_id]


def scripted_decision(docs: list[str]) -> Callable[[dict], dict]:
    """The eval approver as a resume decision: approve a draft that passes the checklist, reject the rest."""

    def decide(request: dict) -> dict:
        draft = Draft(body=request["args"]["draft_text"])
        if scripted_approve(draft, docs) == "approve":
            return {"type": "approve"}
        return {"type": "reject", "message": "the scripted checklist rejected the draft"}

    return decide


async def run_outbound(
    agent,
    desk: LeadDesk,
    lead: Lead,
    thread_id: str,
    *,
    decide: Callable[[dict], dict],
    settings: Settings | None = None,
) -> dict:
    """Run one lead through the outbound path and answer each approval pause with `decide`.

    Resumes on the same thread_id, which the checkpointer needs to continue a paused run."""
    settings = settings or get_settings()
    desk.add(lead)
    config = {"configurable": {"thread_id": thread_id}, "callbacks": get_callbacks(settings)}
    result = await agent.ainvoke(_opening_message(lead), config=config)
    while result.get("__interrupt__"):
        requests = result["__interrupt__"][0].value["action_requests"]
        result = await agent.ainvoke(Command(resume={"decisions": [decide(r) for r in requests]}), config=config)
    return result
