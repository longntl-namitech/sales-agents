"""Measures the real Writer against the guardrail and the scripted approver. Run on purpose:

    uv run pytest -m llm -s tests/agents/test_writer_llm.py

Also writes datasets/hand_review_sample.jsonl for the owner to label by hand.
"""

import os
from pathlib import Path

import pytest
from fakes import scripted, tool_call
from datetime import date

from langchain_core.callbacks import get_usage_metadata_callback

from sales_agents.agents.writer import SYSTEM_PROMPT, draft_outreach
from sales_agents.approver import scripted_approve
from sales_agents.config import get_chat_model, get_settings
from sales_agents.datagen import load_leads
from sales_agents.eval.hand_review import write_sample
from sales_agents.eval.live import SUBSET_TOKEN_LIMIT
from sales_agents.eval.report import Measurement, save_measurement
from sales_agents.eval.tokens import OUTPUT_TOKENS_PER_CALL, approx_tokens
from sales_agents.guardrail import check_draft
from sales_agents.tools.kb import load_docs

DATASET = Path(__file__).parents[2] / "datasets" / "leads_seed42.jsonl"
N_LEADS = 20
DOCS_TOKENS_PER_CALL = 400  # three retrieved passages, generous


def hot_cases(n):
    return [c for c in load_leads(DATASET) if c.gold == "hot"][:n]


def estimate(n):
    return n * (approx_tokens(SYSTEM_PROMPT) + DOCS_TOKENS_PER_CALL + OUTPUT_TOKENS_PER_CALL)


def run_writer(model, cases, sample_path):
    docs = load_docs()
    drafts = [(c.lead.lead_id, draft_outreach(c.lead, c.company, model=model)) for c in cases]
    passed = sum(check_draft(d, docs).passed for _, d in drafts)
    approved = sum(scripted_approve(d, docs) == "approve" for _, d in drafts)
    try:
        write_sample(drafts, docs, sample_path)
    except FileExistsError as exc:  # the owner has labelled the earlier sample, keep it
        print(exc)
    return drafts, passed, approved


def test_writer_wiring_with_fake_model(tmp_path):
    model = scripted(*[tool_call("Draft", {"subject": "s", "body": "Hi Dana."}) for _ in range(3)])

    drafts, passed, approved = run_writer(model, hot_cases(3), tmp_path / "sample.jsonl")

    assert len(drafts) == 3 and approved == 0  # no opt-out line, so the checklist rejects


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY") and not Path(".env").exists(), reason="no API key")
def test_real_writer_guardrail_pass_rate_and_acceptance():
    settings = get_settings()
    cases = hot_cases(N_LEADS)
    assert estimate(len(cases)) < min(settings.eval_token_cap, SUBSET_TOKEN_LIMIT)

    with get_usage_metadata_callback() as usage:
        drafts, passed, approved = run_writer(
            get_chat_model(settings), cases, Path(__file__).parents[2] / "datasets" / "hand_review_sample.jsonl"
        )
    tokens = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())

    docs = load_docs()
    save_measurement(
        Measurement(
            name="Writer + guardrail + approver",
            use_case="Outbound outreach",
            n=len(drafts),
            metrics={"guardrail pass rate": passed / len(drafts), "approver acceptance rate": approved / len(drafts)},
            tokens=tokens,
            latency_p50=None,
            date=date.today().isoformat(),
            note="drafts for gold-hot leads, scripted approver, not yet hand-reviewed",
        )
    )
    print()
    print(f"drafts {len(drafts)}, guardrail passed {passed}, approver accepted {approved}, tokens {tokens} (estimated {estimate(len(cases))})")
    for lead_id, draft in drafts:
        result = check_draft(draft, docs)
        if not result.passed:
            print(lead_id, result.violations)
