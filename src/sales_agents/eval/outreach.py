"""Scores the guardrail and the scripted approver on the planted-fault drafts."""

from collections.abc import Callable
from dataclasses import dataclass

from sales_agents.approver import scripted_approve
from sales_agents.draftgen import GUARDRAIL_FAULTS, PlantedDraft
from sales_agents.guardrail import check_draft
from sales_agents.schemas import Draft, GuardrailResult


@dataclass
class OutreachScores:
    n_clean: int
    n_faulty: int
    n_guardrail_faults: int
    guardrail_catch_rate: float  # blocked, among faults the guardrail is meant to catch
    guardrail_false_block_rate: float  # blocked, among clean drafts
    approver_accept_rate_clean: float  # draft acceptance rate on clean drafts
    approver_reject_rate_faulty: float  # rejected, among every planted fault


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def score_outreach(
    planted: list[PlantedDraft],
    docs: list[str],
    *,
    guardrail: Callable[[Draft, list[str]], GuardrailResult] = check_draft,
) -> OutreachScores:
    clean = [p for p in planted if p.fault is None]
    faulty = [p for p in planted if p.fault is not None]
    guarded = [p for p in faulty if p.fault in GUARDRAIL_FAULTS]
    return OutreachScores(
        n_clean=len(clean),
        n_faulty=len(faulty),
        n_guardrail_faults=len(guarded),
        guardrail_catch_rate=_rate(sum(not guardrail(p.draft, docs).passed for p in guarded), len(guarded)),
        guardrail_false_block_rate=_rate(sum(not guardrail(p.draft, docs).passed for p in clean), len(clean)),
        approver_accept_rate_clean=_rate(sum(scripted_approve(p.draft, docs) == "approve" for p in clean), len(clean)),
        approver_reject_rate_faulty=_rate(sum(scripted_approve(p.draft, docs) == "reject" for p in faulty), len(faulty)),
    )


if __name__ == "__main__":
    from sales_agents.draftgen import generate_drafts
    from sales_agents.tools.kb import load_docs

    print(score_outreach(generate_drafts(seed=42), load_docs()))
