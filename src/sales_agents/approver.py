"""The scripted approver used in evals (Q14): a checklist, not a model.

Approve only when there is no claim outside the product docs, an opt-out is
present, and tone and length are within bounds. A hand-reviewed sample checks
that this checklist matches human judgment.
"""

import re
from typing import Literal

from sales_agents.guardrail import check_draft
from sales_agents.schemas import Draft

MAX_BODY_WORDS = 150
MAX_SUBJECT_CHARS = 80
MAX_EXCLAMATIONS = 1
MAX_SHOUTED_WORDS = 2  # words of 4+ letters written in capitals


def tone_and_length_issues(draft: Draft) -> list[str]:
    issues = []
    if len(draft.body.split()) > MAX_BODY_WORDS:
        issues.append(f"body is longer than {MAX_BODY_WORDS} words")
    if len(draft.subject) > MAX_SUBJECT_CHARS:
        issues.append(f"subject is longer than {MAX_SUBJECT_CHARS} characters")
    if draft.text.count("!") > MAX_EXCLAMATIONS:
        issues.append("too many exclamation marks")
    shouted = re.findall(r"\b[A-Z]{4,}\b", draft.text)
    if len(shouted) > MAX_SHOUTED_WORDS:
        issues.append("too many words in capitals")
    return issues


def scripted_approve(draft: Draft, docs: list[str]) -> Literal["approve", "reject"]:
    if not check_draft(draft, docs).passed or tone_and_length_issues(draft):
        return "reject"
    return "approve"
