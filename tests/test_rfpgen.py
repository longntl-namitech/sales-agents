from sales_agents.agents.rfp import decide_from_facts
from sales_agents.rfpgen import INJECTION, generate_rfps


def test_same_seed_same_rfps():
    assert generate_rfps(seed=1) == generate_rfps(seed=1)


def test_the_set_has_bids_and_no_bids():
    labelled = generate_rfps(seed=42)

    assert sum(r.gold == "bid" for r in labelled) == 12 and sum(r.gold == "no_bid" for r in labelled) == 28


def test_the_gold_label_comes_from_the_true_facts_not_from_the_text():
    for r in generate_rfps(seed=42):
        assert r.gold == decide_from_facts(r.facts).label


def test_missing_budget_and_missing_deadline_cases_are_included_and_never_bid():
    labelled = generate_rfps(seed=42)

    missing = [r for r in labelled if r.facts.budget_usd is None or r.facts.deadline_days is None]
    assert len(missing) >= 6 and all(r.gold == "no_bid" for r in missing)


def test_injected_rfps_carry_the_injection_and_are_never_bid():
    injected = [r for r in generate_rfps(seed=42) if INJECTION in r.rfp.text]

    assert injected and all(r.gold == "no_bid" for r in injected)


def test_rfp_ids_are_unique():
    ids = [r.rfp.rfp_id for r in generate_rfps(seed=42)]

    assert len(ids) == len(set(ids))
