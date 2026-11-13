from fakes import scripted, tool_call

from sales_agents.agents.reactivation import (
    MIN_QUIET_DAYS,
    SYSTEM_PROMPT,
    baseline_select,
    reactivate,
    select_for_reactivation,
)
from sales_agents.crm_mock.store import CrmStore
from sales_agents.reactivationgen import INJECTION, generate_records
from sales_agents.schemas import CrmRecord
from sales_agents.tools.kb import load_docs

DOCS = load_docs()
GOOD = 'Hi Dana,\n\nPipelane routes each qualified lead to a rep by territory and segment rules.\n\nReply "unsubscribe" and I will stop.'
BAD = 'Hi Dana,\n\nPipelane gives customers 40% more booked meetings.\n\nReply "unsubscribe" and I will stop.'


def record(days=120, label="warm", history="We emailed twice and heard nothing.", lead_id="lead-1"):
    return CrmRecord(lead_id=lead_id, company_name="Acme", label=label, days_since_contact=days, history=history)


def judgement(worthwhile, reason="scripted"):
    return tool_call("ReactivationDecision", {"worthwhile": worthwhile, "reason": reason})


def draft(body):
    return tool_call("Draft", {"subject": "Checking in", "body": body})


def approve(draft_, docs):
    return "approve"


def test_same_seed_same_records():
    assert generate_records(seed=1) == generate_records(seed=1)


def test_the_set_has_worthwhile_and_not_worthwhile_records_and_the_day_boundary():
    labelled = generate_records(seed=42)

    assert sum(r.gold for r in labelled) == 12 and sum(not r.gold for r in labelled) == 28
    days = {r.record.days_since_contact for r in labelled}
    assert MIN_QUIET_DAYS in days and MIN_QUIET_DAYS - 1 in days


def test_injected_records_are_never_gold_worthwhile():
    injected = [r for r in generate_records(seed=42) if INJECTION in r.record.history]

    assert injected and not any(r.gold for r in injected)


def test_recent_active_lead_not_selected():
    model = scripted()  # any model call would exhaust the script

    assert select_for_reactivation([record(days=MIN_QUIET_DAYS - 1)], model=model) == []
    assert model.seen == []


def test_cold_lead_is_not_selected_without_asking_the_model():
    model = scripted()

    assert select_for_reactivation([record(label="cold")], model=model) == []
    assert model.seen == []


def test_the_model_reads_the_history_to_keep_or_drop_a_lead():
    records = [record(lead_id="keep"), record(lead_id="drop", history="Asked us to stop emailing them.")]

    selected = select_for_reactivation(records, model=scripted(judgement(True), judgement(False)))

    assert selected == ["keep"]


def test_history_is_sent_as_untrusted_data():
    model = scripted(judgement(False))

    select_for_reactivation([record(history=INJECTION)], model=model)

    sent = " ".join(str(m.content) for m in model.seen[0])
    assert f"<history>\n{INJECTION}\n</history>" in sent and "untrusted" in SYSTEM_PROMPT


def test_the_naive_baseline_ignores_history_and_keeps_a_lead_that_opted_out():
    opted_out = record(history="Asked us to stop emailing them.")

    assert baseline_select([opted_out, record(days=10)]) == [opted_out.lead_id]


def test_selected_drafts_pass_guardrail():
    store = CrmStore()

    outcome = reactivate([record()], model=scripted(judgement(True), draft(GOOD)), crm=store, docs=DOCS, approve=approve)

    assert (outcome.selected, outcome.drafted, outcome.passed_guardrail, outcome.recorded) == (["lead-1"], 1, 1, 1)
    assert store.get_lead("lead-1")["outreach"] == f"Checking in\n\n{GOOD}"


def test_a_draft_the_guardrail_blocks_is_not_recorded_even_if_approved():
    store = CrmStore()

    outcome = reactivate([record()], model=scripted(judgement(True), draft(BAD)), crm=store, docs=DOCS, approve=approve)

    assert (outcome.drafted, outcome.passed_guardrail, outcome.recorded) == (1, 0, 0)
    assert store.get_lead("lead-1") is None
