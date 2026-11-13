import pytest

from sales_agents.guardrail import check_draft
from sales_agents.schemas import Draft
from sales_agents.tools.kb import load_docs

DOCS = load_docs()
OPT_OUT = 'If you would rather not hear from us, reply "unsubscribe" and I will stop.'
CLEAN_BODY = (
    "Hi Dana,\n\nI saw your team is hiring SDRs. Pipelane triages inbound leads automatically "
    "and scores each one against the sales team's own qualification rules.\n\n" + OPT_OUT
)


def draft(body: str, subject: str = "Quick idea for your SDR team") -> Draft:
    return Draft(subject=subject, body=body)


def test_supported_claim_passes():
    result = check_draft(draft(CLEAN_BODY), DOCS)

    assert result.passed and result.violations == []


@pytest.mark.parametrize(
    "wording",
    [
        'Reply "unsubscribe" and I will stop.',
        "You can opt out at any time.",
        "If you would rather not hear from us, just say so.",
        "Reply STOP and I won't email again.",
        "Tell me if you no longer wish to receive these notes.",
        "Let me know if you don't want to hear from me.",
    ],
)
def test_opt_out_wording_variants_are_accepted(wording):
    assert check_draft(draft(f"Hi Dana,\n\nPipelane routes each qualified lead to a rep by territory and segment rules.\n\n{wording}"), DOCS).passed


def test_missing_opt_out_blocked():
    body = "Hi Dana,\n\nPipelane triages inbound leads automatically and scores each one."

    result = check_draft(draft(body), DOCS)

    assert not result.passed and any("opt-out" in v for v in result.violations)


def test_claim_not_in_docs_blocked_number():
    body = f"Hi Dana,\n\nPipelane gives customers 40% more booked meetings.\n\n{OPT_OUT}"

    result = check_draft(draft(body), DOCS)

    assert not result.passed and any("40%" in v for v in result.violations)


def test_claim_not_in_docs_blocked_capability():
    body = f"Hi Dana,\n\nPipelane also runs on-premise behind your firewall.\n\n{OPT_OUT}"

    result = check_draft(draft(body), DOCS)

    assert not result.passed and any("not supported" in v for v in result.violations)


def test_number_that_is_in_the_docs_passes():
    body = f"Hi Dana,\n\nThe Team plan costs $49 per seat per month, billed annually.\n\n{OPT_OUT}"

    assert check_draft(draft(body), DOCS).passed


def test_prohibited_promise_blocked():
    body = f"Hi Dana,\n\nPipelane triages inbound leads. We guarantee results.\n\n{OPT_OUT}"

    result = check_draft(draft(body), DOCS)

    assert not result.passed and any("prohibited" in v for v in result.violations)


def test_generic_sentences_about_the_lead_are_not_blocked():
    body = f"Hi Dana,\n\nCongrats on the new office in Berlin. Pipelane routes each qualified lead to a rep by territory and segment rules.\n\n{OPT_OUT}"

    assert check_draft(draft(body), DOCS).passed
