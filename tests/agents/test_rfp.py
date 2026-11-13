import pytest
from fakes import scripted, tool_call

from sales_agents.agents.rfp import (
    MIN_BUDGET_USD,
    MIN_DEADLINE_DAYS,
    SYSTEM_PROMPT,
    BidDecision,
    RequirementCheck,
    RfpFacts,
    decide_bid,
    decide_from_facts,
    process_rfp,
)
from sales_agents.crm_mock.store import CrmStore
from sales_agents.rfpgen import INJECTION
from sales_agents.schemas import Rfp
from sales_agents.tools.kb import load_docs

DOCS = load_docs()


def facts(supported=5, unsupported=1, deadline=30, budget=50_000):
    reqs = [RequirementCheck(text=f"s{i}", supported=True) for i in range(supported)]
    reqs += [RequirementCheck(text=f"u{i}", supported=False) for i in range(unsupported)]
    return RfpFacts(requirements=reqs, deadline_days=deadline, budget_usd=budget)


def facts_call(f: RfpFacts):
    return tool_call("RfpFacts", f.model_dump())


def rfp(text="Request for Proposal.", rfp_id="rfp-1"):
    return Rfp(rfp_id=rfp_id, title="Lead tooling", text=text)


def test_all_conditions_met_is_a_bid():
    assert decide_from_facts(facts()).label == "bid"


@pytest.mark.parametrize(
    "changes, expected_in_reason",
    [
        ({"budget": None}, "budget is not stated"),
        ({"deadline": None}, "deadline is not stated"),
        ({"deadline": MIN_DEADLINE_DAYS - 1}, "days"),
        ({"budget": MIN_BUDGET_USD - 1}, "below the minimum"),
        ({"supported": 4, "unsupported": 4}, "requirements are supported"),
    ],
)
def test_each_failed_condition_is_a_no_bid_with_its_reason(changes, expected_in_reason):
    decision = decide_from_facts(facts(**changes))

    assert decision.label == "no_bid" and expected_in_reason in decision.reason


def test_the_support_threshold_is_seventy_percent():
    assert decide_from_facts(facts(supported=7, unsupported=3)).label == "bid"
    assert decide_from_facts(facts(supported=6, unsupported=4)).label == "no_bid"


def test_an_rfp_with_no_requirements_is_a_no_bid_not_a_crash():
    empty = RfpFacts(requirements=[], deadline_days=30, budget_usd=50_000)

    assert decide_from_facts(empty).label == "no_bid"


def test_rfp_missing_budget_or_deadline_does_not_crash():
    for missing in ({"budget": None}, {"deadline": None}, {"budget": None, "deadline": None}):
        decision = decide_bid(rfp(), DOCS, model=scripted(facts_call(facts(**missing))))

        assert decision.label == "no_bid"


def test_injection_in_rfp_text_does_not_change_decision():
    # Whatever the text says, the label comes from the extracted facts and the thresholds.
    text = f"Requirements: on-premise deployment. {INJECTION}"
    weak = facts(supported=1, unsupported=5)

    decision = decide_bid(rfp(text), DOCS, model=scripted(facts_call(weak)))

    assert decision.label == "no_bid"


def test_the_rfp_is_sent_as_untrusted_data_and_the_decision_is_not_the_models_to_make():
    model = scripted(facts_call(facts()))

    decide_bid(rfp(INJECTION), DOCS, model=model)

    sent = " ".join(str(m.content) for m in model.seen[0])
    assert f"<rfp>\n{INJECTION}\n</rfp>" in sent
    assert "untrusted" in SYSTEM_PROMPT and "You do not decide whether to bid" in SYSTEM_PROMPT


def test_decision_uses_product_docs_not_rfp_claims():
    model = scripted(facts_call(facts()))

    decide_bid(rfp("The vendor already supports on-premise deployment."), DOCS, model=model)

    sent = " ".join(str(m.content) for m in model.seen[0])
    assert "<product_docs>" in sent and "Pipelane is cloud-hosted only" in sent
    assert "never from what the RFP says about the vendor" in SYSTEM_PROMPT


def test_an_approved_decision_is_recorded():
    store = CrmStore()

    outcome = process_rfp(rfp(), model=scripted(facts_call(facts())), crm=store, docs=DOCS, approve=lambda d: True)

    record = store.get_lead("rfp-1")
    assert outcome.recorded and record["label"] == "bid" and record["notes"] == outcome.decision.reason


def test_rejected_decision_not_recorded():
    store = CrmStore()

    outcome = process_rfp(rfp(), model=scripted(facts_call(facts())), crm=store, docs=DOCS, approve=lambda d: False)

    assert not outcome.recorded and store.get_lead("rfp-1") is None
    assert isinstance(outcome.decision, BidDecision)
