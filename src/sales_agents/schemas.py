from typing import Literal

from pydantic import BaseModel

Label = Literal["hot", "warm", "cold"]
LABELS: tuple[Label, ...] = ("hot", "warm", "cold")


class Lead(BaseModel):
    lead_id: str
    company_name: str
    contact_role: str | None
    notes: str
    source: Literal["outbound", "inbound"]


class CompanyRecord(BaseModel):
    """Enrichment data. Kept off Lead so the Researcher has something to add."""

    company_name: str
    employee_count: int | None
    industry: str | None
    region: str | None
    signals: list[str]


class QualificationResult(BaseModel):
    label: Label
    reason: str


class LabelledLead(BaseModel):
    lead: Lead
    company: CompanyRecord
    gold: Label
    ambiguous: bool
