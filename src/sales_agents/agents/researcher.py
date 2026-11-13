"""The Researcher: enriches a lead from the synthetic store.

It makes no model call. The store is a lookup, so a plain function is cheaper and
cannot invent facts. Its value is measured as the change in Qualifier accuracy
with and without this enrichment.
"""

from sales_agents.schemas import CompanyRecord, Lead
from sales_agents.tools.enrichment import lookup_company


def research(lead: Lead) -> CompanyRecord | None:
    return lookup_company(lead.company_name)
