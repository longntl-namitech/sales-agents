from sales_agents.draftgen import GUARDRAIL_FAULTS, generate_drafts
from sales_agents.eval.outreach import score_outreach
from sales_agents.schemas import GuardrailResult
from sales_agents.tools.kb import load_docs

DOCS = load_docs()
PLANTED = generate_drafts(seed=42)


def test_scores_cover_all_four_measures():
    scores = score_outreach(PLANTED, DOCS)

    assert scores.n_clean + scores.n_faulty == len(PLANTED)
    for rate in (
        scores.guardrail_catch_rate,
        scores.guardrail_false_block_rate,
        scores.approver_accept_rate_clean,
        scores.approver_reject_rate_faulty,
    ):
        assert 0.0 <= rate <= 1.0


def test_a_guardrail_that_blocks_nothing_scores_zero_catch_rate():
    scores = score_outreach(PLANTED, DOCS, guardrail=lambda draft, docs: GuardrailResult(passed=True, violations=[]))

    assert scores.guardrail_catch_rate == 0.0


def test_catch_rate_counts_only_faults_the_guardrail_is_meant_to_catch():
    scores = score_outreach(PLANTED, DOCS)

    guarded = sum(p.fault in GUARDRAIL_FAULTS for p in PLANTED)
    assert scores.n_guardrail_faults == guarded
