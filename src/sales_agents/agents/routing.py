"""Inbound lead routing: qualify a form or demo request, then assign a rep by rules.

Routing is plain code over structured fields (the qualification label and the
company's region and industry), never over the form's free text, so text in a
form cannot choose its own rep.
"""

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from sales_agents.agents.qualifier import qualify
from sales_agents.agents.researcher import research
from sales_agents.crm_mock.store import CrmStore
from sales_agents.schemas import CompanyRecord, Label, Lead


class Rep(BaseModel):
    rep_id: str
    name: str
    regions: list[str]
    industries: list[str]


# One rep per region and industry group. Together they cover every target region and industry.
DEFAULT_REPS = [
    Rep(rep_id="rep-1", name="Ava", regions=["north america"], industries=["software", "fintech"]),
    Rep(rep_id="rep-2", name="Ben", regions=["north america"], industries=["healthcare", "logistics", "ecommerce"]),
    Rep(rep_id="rep-3", name="Chloe", regions=["europe"], industries=["software", "fintech", "ecommerce"]),
    Rep(rep_id="rep-4", name="Dev", regions=["europe"], industries=["healthcare", "logistics"]),
]


def route_lead(lead: Lead, label: Label, reps: list[Rep], company: CompanyRecord | None) -> Rep | None:
    """The first rep, in roster order, whose region and industry match. A cold lead, or one whose
    region or industry is unknown, goes to nobody. `lead` is accepted for the interface and is not read."""
    if label == "cold" or company is None or company.region is None or company.industry is None:
        return None
    region, industry = company.region.strip().lower(), company.industry.strip().lower()
    return next((r for r in reps if region in r.regions and industry in r.industries), None)


@dataclass
class InboundOutcome:
    label: Label
    rep: Rep | None


def process_inbound(
    lead: Lead, *, model: BaseChatModel, crm: CrmStore, reps: list[Rep] = DEFAULT_REPS
) -> InboundOutcome:
    company = research(lead)
    result = qualify(lead, company, model=model)
    rep = route_lead(lead, result.label, reps, company)
    note = f"assigned to {rep.name}" if rep else f"unassigned ({result.label})"
    crm.upsert_lead(lead.lead_id, result.label, f"{note}. {result.reason}")
    return InboundOutcome(result.label, rep)
