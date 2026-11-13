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


ReplyCategory = Literal["interested", "not_interested", "question", "reschedule"]
REPLY_CATEGORIES: tuple[ReplyCategory, ...] = ("interested", "not_interested", "question", "reschedule")


class Reply(BaseModel):
    lead_id: str
    text: str


class ReplyClassification(BaseModel):
    category: ReplyCategory
    reason: str


class CrmRecord(BaseModel):
    """An aged CRM record, for lead reactivation."""

    lead_id: str
    company_name: str
    label: Label  # the last qualification
    days_since_contact: int
    history: str  # free text about earlier contact, written by people and prospects


class ReactivationDecision(BaseModel):
    worthwhile: bool
    reason: str


class Rfp(BaseModel):
    rfp_id: str
    title: str
    text: str


class Draft(BaseModel):
    """An outreach email. The subject may be empty, for text that arrives as one block."""

    subject: str = ""
    body: str

    @property
    def text(self) -> str:
        return f"{self.subject}\n\n{self.body}" if self.subject else self.body


class GuardrailResult(BaseModel):
    passed: bool
    violations: list[str]


class LabelledLead(BaseModel):
    lead: Lead
    company: CompanyRecord
    gold: Label
    ambiguous: bool
