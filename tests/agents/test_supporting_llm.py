"""Measures the four supporting use cases with a real model. Run on purpose:

    uv run pytest -m llm -s tests/agents/test_supporting_llm.py

Each test estimates its tokens first and refuses to start above the subset limit. Each
saves a row in results/measurements.json. The offline tests check the wiring with fakes.
"""

import os
from datetime import date
from pathlib import Path

import pytest
from fakes import scripted, tool_call, verdict
from langchain_core.callbacks import get_usage_metadata_callback

from sales_agents.agents.qualifier import SYSTEM_PROMPT as QUALIFIER_PROMPT
from sales_agents.agents.reactivation import SYSTEM_PROMPT as JUDGE_PROMPT
from sales_agents.agents.reactivation import baseline_select, reactivate, select_for_reactivation
from sales_agents.agents.replies import STATUS_FOR, handle_reply
from sales_agents.agents.replies import SYSTEM_PROMPT as REPLY_PROMPT
from sales_agents.agents.rfp import SYSTEM_PROMPT as RFP_PROMPT
from sales_agents.agents.rfp import decide_bid
from sales_agents.agents.routing import DEFAULT_REPS, process_inbound, route_lead
from sales_agents.agents.writer import REACTIVATION_SYSTEM_PROMPT, REPLY_SYSTEM_PROMPT
from sales_agents.config import get_chat_model, get_settings
from sales_agents.crm_mock.store import CrmStore
from sales_agents.datagen import load_leads
from sales_agents.eval.live import SUBSET_TOKEN_LIMIT
from sales_agents.eval.metrics import accuracy, precision_recall
from sales_agents.eval.report import Measurement, save_measurement
from sales_agents.eval.runner import run_cases
from sales_agents.eval.tokens import OUTPUT_TOKENS_PER_CALL, approx_tokens
from sales_agents.reactivationgen import generate_records
from sales_agents.replygen import generate_replies
from sales_agents.rfpgen import generate_rfps
from sales_agents.tools.kb import load_docs

DATASET = Path(__file__).parents[2] / "datasets" / "leads_seed42.jsonl"
DOCS = load_docs()
DOCS_TOKENS = 700  # all ten product passages, generous


def tokens_used(usage):
    return sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())


def guard(estimate):
    cap = min(get_settings().eval_token_cap, SUBSET_TOKEN_LIMIT)
    assert estimate < cap, f"estimate {estimate} is above {cap}"


def measurement(name, use_case, result, metrics, note):
    return Measurement(
        name=name,
        use_case=use_case,
        n=result.n,
        metrics={**metrics, "errors": float(len(result.errors))},
        tokens=result.cost,
        latency_p50=result.latency_p50,
        date=date.today().isoformat(),
        note=note,
    )


# --- reply handling -------------------------------------------------------------------------


def run_replies(model, labelled, estimate):
    store = CrmStore()
    with get_usage_metadata_callback() as usage:
        result = run_cases(
            labelled,
            lambda item: handle_reply(item.reply, model=model, crm=store, docs=DOCS),
            case_id=lambda item: item.reply.lead_id,
            token_cap=SUBSET_TOKEN_LIMIT,
            estimated_tokens=estimate,
            count_tokens=lambda: tokens_used(usage),
        )
    gold = [item.gold for item in labelled]
    predicted = [o.category if o else None for o in result.predictions]
    crm_correct = [
        (store.get_lead(item.reply.lead_id) or {}).get("label") == STATUS_FOR[item.gold] for item in labelled
    ]
    tricky = [i for i, item in enumerate(labelled) if item.tricky]
    metrics = {
        "classification accuracy": accuracy(gold, predicted),
        "correct CRM update rate": sum(crm_correct) / len(labelled),
        "accuracy on unclear and injected": accuracy([gold[i] for i in tricky], [predicted[i] for i in tricky]),
    }
    return result, metrics


def test_reply_handling_wiring_with_fake_models():
    labelled = generate_replies(seed=42)[:2]
    script = []
    for item in labelled:
        script.append(tool_call("ReplyClassification", {"category": item.gold, "reason": "x"}))
        if item.gold != "not_interested":
            script.append(tool_call("Draft", {"subject": "", "body": "Hi."}))

    result, metrics = run_replies(scripted(*script), labelled, estimate=0)

    assert result.n == 2 and metrics["classification accuracy"] == 1.0


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY") and not Path(".env").exists(), reason="no API key")
def test_reply_handling_measured():
    labelled = generate_replies(seed=42)
    per_reply = approx_tokens(REPLY_PROMPT) + approx_tokens(REPLY_SYSTEM_PROMPT) + DOCS_TOKENS // 2 + 2 * OUTPUT_TOKENS_PER_CALL + 100
    estimate = len(labelled) * per_reply
    guard(estimate)

    result, metrics = run_replies(get_chat_model(get_settings()), labelled, estimate)

    print(f"\nreply handling: {metrics}, tokens {result.cost} (estimated {estimate}), errors {result.errors}")
    save_measurement(measurement("Reply handling", "Reply handling", result, metrics, "classify, update the CRM, draft, scripted approver"))


# --- inbound routing ------------------------------------------------------------------------


def run_inbound(model, cases, estimate):
    store = CrmStore()
    with get_usage_metadata_callback() as usage:
        result = run_cases(
            cases,
            lambda case: ("ok", getattr(process_inbound(case.lead, model=model, crm=store).rep, "rep_id", None)),
            case_id=lambda case: case.lead.lead_id,
            token_cap=SUBSET_TOKEN_LIMIT,
            estimated_tokens=estimate,
            count_tokens=lambda: tokens_used(usage),
        )

    def gold_rep(case):
        rep = route_lead(case.lead, case.gold, DEFAULT_REPS, case.company)
        return ("ok", rep.rep_id if rep else None)  # "no rep" is a legitimate answer, a failure is None

    return result, {"correct-rep rate": accuracy([gold_rep(c) for c in cases], result.predictions)}


def test_inbound_wiring_with_fake_model():
    cases = [c for c in load_leads(DATASET) if c.lead.source == "inbound"][:2]

    result, metrics = run_inbound(scripted(verdict("cold"), verdict("cold")), cases, estimate=0)

    assert result.n == 2 and 0.0 <= metrics["correct-rep rate"] <= 1.0


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY") and not Path(".env").exists(), reason="no API key")
def test_inbound_routing_measured():
    cases = [c for c in load_leads(DATASET) if c.lead.source == "inbound"]
    estimate = len(cases) * (approx_tokens(QUALIFIER_PROMPT) + 500 + OUTPUT_TOKENS_PER_CALL)
    guard(estimate)

    result, metrics = run_inbound(get_chat_model(get_settings()), cases, estimate)

    print(f"\ninbound routing: {metrics}, tokens {result.cost} (estimated {estimate}), errors {result.errors}")
    save_measurement(measurement("Inbound routing", "Inbound routing", result, metrics, "research, qualify, route by rules; latency is form to decision"))


# --- reactivation ---------------------------------------------------------------------------


def run_selection(model, labelled, estimate):
    with get_usage_metadata_callback() as usage:
        result = run_cases(
            labelled,
            lambda item: bool(select_for_reactivation([item.record], model=model)),
            case_id=lambda item: item.record.lead_id,
            token_cap=SUBSET_TOKEN_LIMIT,
            estimated_tokens=estimate,
            count_tokens=lambda: tokens_used(usage),
        )
    gold = [item.gold for item in labelled]
    precision, recall = precision_recall(gold, result.predictions, positive=True)
    return result, {"precision": precision, "recall": recall}


def test_reactivation_wiring_with_fake_model():
    labelled = generate_records(seed=42)[:3]

    result, metrics = run_selection(scripted(*[tool_call("ReactivationDecision", {"worthwhile": False, "reason": "x"})] * 3), labelled, 0)

    assert result.n == 3 and "precision" in metrics


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY") and not Path(".env").exists(), reason="no API key")
def test_reactivation_measured():
    labelled = generate_records(seed=42)
    model = get_chat_model(get_settings())
    judge = len(labelled) * (approx_tokens(JUDGE_PROMPT) + 100 + OUTPUT_TOKENS_PER_CALL)
    drafts = 14 * (approx_tokens(REACTIVATION_SYSTEM_PROMPT) + 400 + OUTPUT_TOKENS_PER_CALL)
    guard(2 * judge + drafts)  # selection runs twice: once for the metrics, once inside reactivate()

    result, metrics = run_selection(model, labelled, judge)
    baseline_pred = [item.record.lead_id in baseline_select([item.record]) for item in labelled]
    base_precision, base_recall = precision_recall([i.gold for i in labelled], baseline_pred, positive=True)

    store = CrmStore()
    with get_usage_metadata_callback() as usage:
        outcome = reactivate([i.record for i in labelled], model=model, crm=store, docs=DOCS)
    pass_rate = outcome.passed_guardrail / outcome.drafted if outcome.drafted else 0.0

    print(f"\nreactivation: LLM {metrics}, baseline precision {base_precision:.3f} recall {base_recall:.3f}")
    print(f"drafted {outcome.drafted}, guardrail passed {outcome.passed_guardrail}, recorded {outcome.recorded}, tokens (selection) {result.cost}, (reactivate) {tokens_used(usage)}")
    save_measurement(measurement("Reactivation selector (LLM)", "Lead reactivation", result, {**metrics, "guardrail pass rate on drafts": pass_rate}, "code gates on label and quiet days, model reads the history"))
    save_measurement(
        Measurement(
            name="Reactivation baseline (label and days only)",
            use_case="Lead reactivation",
            n=len(labelled),
            metrics={"precision": base_precision, "recall": base_recall},
            tokens=0,
            latency_p50=None,
            date=date.today().isoformat(),
            note="ignores the history text",
        )
    )


# --- RFP bid or no-bid ----------------------------------------------------------------------


def run_rfps(model, labelled, estimate):
    with get_usage_metadata_callback() as usage:
        result = run_cases(
            labelled,
            lambda item: decide_bid(item.rfp, DOCS, model=model).label,
            case_id=lambda item: item.rfp.rfp_id,
            token_cap=SUBSET_TOKEN_LIMIT,
            estimated_tokens=estimate,
            count_tokens=lambda: tokens_used(usage),
        )
    gold = [item.gold for item in labelled]
    precision, recall = precision_recall(gold, result.predictions, positive="bid")
    return result, {"precision (bid)": precision, "recall (bid)": recall, "accuracy": accuracy(gold, result.predictions)}


def test_rfp_wiring_with_fake_model():
    labelled = generate_rfps(seed=42)[:2]
    script = [tool_call("RfpFacts", item.facts.model_dump()) for item in labelled]

    result, metrics = run_rfps(scripted(*script), labelled, estimate=0)

    assert result.n == 2 and metrics["accuracy"] == 1.0  # the fake returns the true facts


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY") and not Path(".env").exists(), reason="no API key")
def test_rfp_bid_or_no_bid_measured():
    labelled = generate_rfps(seed=42)
    estimate = len(labelled) * (approx_tokens(RFP_PROMPT) + DOCS_TOKENS + 300 + 2 * OUTPUT_TOKENS_PER_CALL)
    guard(estimate)

    result, metrics = run_rfps(get_chat_model(get_settings()), labelled, estimate)

    print(f"\nRFP: {metrics}, tokens {result.cost} (estimated {estimate}), errors {result.errors}")
    save_measurement(measurement("RFP bid or no-bid", "RFP bid or no-bid", result, metrics, "model extracts facts, thresholds in code decide"))
