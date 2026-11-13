"""Deterministic qualification rubric (Q12).

Fit (size, industry, region) and Buyer role are hard gates. Need and Intent are
scored from signal tokens. Only structured attributes are read, never free text,
so a lead cannot talk its way to a better label.

The thresholds below are the builder's calls, not decisions from the grill.
"""

from sales_agents.schemas import CompanyRecord, Label, Lead, QualificationResult

MIN_EMPLOYEES = 50
MAX_EMPLOYEES = 1000
TARGET_INDUSTRIES = frozenset({"software", "fintech", "healthcare", "logistics", "ecommerce"})
TARGET_REGIONS = frozenset({"north america", "europe"})
BUYER_ROLES = frozenset(
    {
        "vp sales",
        "head of sales",
        "director of sales operations",
        "chief revenue officer",
        "head of revenue operations",
    }
)

NEED_SIGNALS = frozenset({"manual_lead_triage", "hiring_sdrs", "crm_migration"})
INTENT_SIGNALS = frozenset({"demo_request", "pricing_page_visit", "competitor_evaluation"})
SIGNAL_CAP = 2  # each of Need and Intent contributes at most this much
HOT_MIN_SCORE = 3
WARM_MIN_SCORE = 1


def _norm(value: str) -> str:
    return value.strip().lower()


def _missing_fields(lead: Lead, company: CompanyRecord) -> list[str]:
    checks = {
        "employee count": company.employee_count,
        "industry": company.industry,
        "region": company.region,
        "contact role": lead.contact_role,
    }
    return [name for name, value in checks.items() if value is None]


def _failed_gates(lead: Lead, company: CompanyRecord) -> list[str]:
    failed = []
    if not MIN_EMPLOYEES <= company.employee_count <= MAX_EMPLOYEES:
        failed.append(f"company size {company.employee_count} is outside {MIN_EMPLOYEES}-{MAX_EMPLOYEES}")
    if _norm(company.industry) not in TARGET_INDUSTRIES:
        failed.append(f"industry '{company.industry}' is not a target industry")
    if _norm(company.region) not in TARGET_REGIONS:
        failed.append(f"region '{company.region}' is not a target region")
    if _norm(lead.contact_role) not in BUYER_ROLES:
        failed.append(f"role '{lead.contact_role}' is not a buyer role")
    return failed


def explain_lead(lead: Lead, company: CompanyRecord) -> QualificationResult:
    missing = _missing_fields(lead, company)
    if missing:
        return QualificationResult(
            label="cold", reason=f"Gate field missing: {', '.join(missing)}."
        )

    failed = _failed_gates(lead, company)
    if failed:
        return QualificationResult(label="cold", reason=f"Failed gate: {'; '.join(failed)}.")

    signals = {_norm(s) for s in company.signals}
    need = min(len(signals & NEED_SIGNALS), SIGNAL_CAP)
    intent = min(len(signals & INTENT_SIGNALS), SIGNAL_CAP)
    score = need + intent
    label: Label = (
        "hot" if score >= HOT_MIN_SCORE else "warm" if score >= WARM_MIN_SCORE else "cold"
    )
    return QualificationResult(
        label=label, reason=f"Gates passed, need {need} + intent {intent} = {score}."
    )


def label_lead(lead: Lead, company: CompanyRecord) -> Label:
    return explain_lead(lead, company).label
