from collections import Counter

from sales_agents.datagen import AMBIGUOUS_SHARE, generate_leads, template_text
from sales_agents.rubric import label_lead


def test_same_seed_same_leads():
    a = generate_leads(30, seed=7, write_text=template_text)
    b = generate_leads(30, seed=7, write_text=template_text)
    assert a == b


def test_different_seed_different_leads():
    a = generate_leads(30, seed=7, write_text=template_text)
    b = generate_leads(30, seed=8, write_text=template_text)
    assert a != b


def test_all_three_labels_present():
    cases = generate_leads(30, seed=1, write_text=template_text)
    assert {c.gold for c in cases} == {"hot", "warm", "cold"}


def test_generator_includes_ambiguous_cases():
    cases = generate_leads(50, seed=1, write_text=template_text)
    assert sum(c.ambiguous for c in cases) == round(50 * AMBIGUOUS_SHARE)


def test_gold_equals_rubric_on_attributes():
    for case in generate_leads(60, seed=3, write_text=template_text):
        assert case.gold == label_lead(case.lead, case.company)


def test_notes_come_from_injected_writer():
    seen = []

    def write_text(attributes):
        seen.append(attributes)
        return f"NOTE for {attributes['company_name']}"

    cases = generate_leads(6, seed=1, write_text=write_text)

    assert len(seen) == 6
    assert all(c.lead.notes == f"NOTE for {c.lead.company_name}" for c in cases)


def test_lead_ids_and_company_names_are_unique():
    cases = generate_leads(200, seed=2, write_text=template_text)
    assert len({c.lead.lead_id for c in cases}) == 200
    assert len({c.lead.company_name for c in cases}) == 200


def test_both_sources_appear():
    cases = generate_leads(40, seed=1, write_text=template_text)
    assert set(Counter(c.lead.source for c in cases)) == {"outbound", "inbound"}


def test_some_leads_miss_a_gate_field():
    cases = generate_leads(60, seed=1, write_text=template_text)
    missing = [
        c
        for c in cases
        if c.lead.contact_role is None
        or c.company.employee_count is None
        or c.company.industry is None
        or c.company.region is None
    ]
    assert missing and all(c.gold == "cold" for c in missing)
