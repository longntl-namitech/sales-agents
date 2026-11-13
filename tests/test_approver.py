
from sales_agents.approver import MAX_BODY_WORDS, scripted_approve
from sales_agents.draftgen import generate_drafts
from sales_agents.tools.kb import load_docs

DOCS = load_docs()
PLANTED = generate_drafts(seed=42)


def test_approver_rejects_planted_faults():
    faulty = [p for p in PLANTED if p.fault is not None]

    assert faulty and all(scripted_approve(p.draft, DOCS) == "reject" for p in faulty)


def test_approver_approves_clean_drafts():
    clean = [p for p in PLANTED if p.fault is None]

    assert clean and all(scripted_approve(p.draft, DOCS) == "approve" for p in clean)


def test_long_body_is_rejected_even_when_every_claim_is_supported():
    clean = next(p for p in PLANTED if p.fault is None)
    padded = clean.draft.model_copy(update={"body": clean.draft.body + " Thanks." * MAX_BODY_WORDS})

    assert scripted_approve(padded, DOCS) == "reject"


def test_shouting_is_rejected():
    clean = next(p for p in PLANTED if p.fault is None)
    loud = clean.draft.model_copy(update={"subject": "THIS IS THE BEST TOOL YOU WILL EVER SEE!!!"})

    assert scripted_approve(loud, DOCS) == "reject"
