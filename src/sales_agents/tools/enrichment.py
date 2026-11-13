"""The synthetic enrichment store behind the Researcher.

It reads company facts from the committed lead set and never touches the
network. An unknown company returns None, so nothing is invented. Missing fields
stay None.
"""

from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

from sales_agents.datagen import load_leads
from sales_agents.schemas import CompanyRecord

DEFAULT_STORE = Path(__file__).parents[3] / "datasets" / "leads_seed42.jsonl"


def _key(company_name: str) -> str:
    return company_name.strip().lower()


@lru_cache
def load_store(path: Path = DEFAULT_STORE) -> dict[str, CompanyRecord]:
    return {_key(c.company.company_name): c.company for c in load_leads(path)}


def lookup_company(company_name: str, store: dict[str, CompanyRecord] | None = None) -> CompanyRecord | None:
    """Look up a company in the synthetic store. Returns None when it is not there."""
    store = load_store() if store is None else store
    return store.get(_key(company_name))


@tool("lookup_company")
def lookup_company_tool(company_name: str) -> str:
    """Look up a company's size, industry, region and public signals in the synthetic store."""
    record = lookup_company(company_name)
    return record.model_dump_json() if record else "no record found"
