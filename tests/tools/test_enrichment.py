import socket

import pytest

from sales_agents.agents.researcher import research
from sales_agents.schemas import CompanyRecord, Lead
from sales_agents.tools.enrichment import load_store, lookup_company


def _lead(company_name: str) -> Lead:
    return Lead(lead_id="x", company_name=company_name, contact_role=None, notes="", source="outbound")


def test_known_company_returns_record():
    record = lookup_company("Quill Dynamics")

    assert isinstance(record, CompanyRecord)
    assert record.employee_count == 822 and record.industry == "software"


def test_lookup_ignores_case_and_surrounding_space():
    assert lookup_company("  quill dynamics ") == lookup_company("Quill Dynamics")


def test_unknown_company_returns_none_not_invented():
    assert lookup_company("Nonexistent Holdings") is None
    assert research(_lead("Nonexistent Holdings")) is None


def test_tool_only_reads_synthetic_store(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("enrichment must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    load_store.cache_clear()

    assert lookup_company("Quill Dynamics") is not None


def test_research_returns_the_stored_record():
    assert research(_lead("Quill Dynamics")) == lookup_company("Quill Dynamics")


def test_store_keeps_missing_fields_as_none_not_guessed():
    record = lookup_company("Quill Dynamics")  # region is null in the dataset

    assert record.region is None
