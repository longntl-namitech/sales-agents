from collections.abc import Callable

from sales_agents.rubric import explain_lead
from sales_agents.schemas import LabelledLead, Lead, QualificationResult


def make_rule_baseline(cases: list[LabelledLead]) -> Callable[[Lead], QualificationResult]:
    """Applies the rubric to the full company data. This is the ceiling for any
    Qualifier, since gold labels come from the same rubric. It looks the company
    up by lead_id because the eval harness hands the agent only the Lead."""
    company_by_lead = {c.lead.lead_id: c.company for c in cases}

    def rule_baseline(lead: Lead) -> QualificationResult:
        return explain_lead(lead, company_by_lead[lead.lead_id])

    return rule_baseline
