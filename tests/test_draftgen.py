from sales_agents.draftgen import APPROVER_ONLY_FAULTS, GUARDRAIL_FAULTS, generate_drafts


def test_same_seed_same_drafts():
    assert generate_drafts(seed=1) == generate_drafts(seed=1)


def test_every_fault_type_and_clean_drafts_are_present():
    faults = {p.fault for p in generate_drafts(seed=42)}

    assert faults == {None, *GUARDRAIL_FAULTS, *APPROVER_ONLY_FAULTS}


def test_clean_drafts_are_the_largest_group():
    planted = generate_drafts(seed=42)
    clean = sum(p.fault is None for p in planted)

    assert clean >= len(planted) // 3
